from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class DQNConfig:
    hidden_layers: tuple[int, ...] = (
        256,
        128,
        64,
    )

    gamma: float = 0.95

    learning_rate: float = 0.001

    replay_capacity: int = 100_000
    batch_size: int = 64

    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_episodes: int = 100

    # Reconstruction assumptions.
    replay_warmup_transitions: int = 1000
    target_update_steps: int = 250


def linear_epsilon(
    episode_index: int,
    *,
    start: float = 1.0,
    end: float = 0.01,
    decay_episodes: int = 100,
) -> float:
    """
    Linear epsilon decay across training episodes.

    For 100 episodes:
        episode 0  -> 1.00
        episode 99 -> 0.01
    """

    if decay_episodes <= 1:
        return float(end)

    if episode_index <= 0:
        return float(start)

    if episode_index >= decay_episodes - 1:
        return float(end)

    progress = (
        episode_index
        / (decay_episodes - 1)
    )

    return float(
        start
        + progress * (end - start)
    )


class QNetwork(nn.Module):
    """
    Vanilla DQN network.

    Default manuscript architecture:

        input
          -> 256 ReLU
          -> 128 ReLU
          -> 64 ReLU
          -> linear Q-values
    """

    def __init__(
        self,
        *,
        input_dim: int,
        action_dim: int,
        hidden_layers: tuple[int, ...] = (
            256,
            128,
            64,
        ),
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError(
                "input_dim must be positive"
            )

        if action_dim <= 0:
            raise ValueError(
                "action_dim must be positive"
            )

        layers: list[nn.Module] = []

        previous = input_dim

        for hidden in hidden_layers:
            layers.append(
                nn.Linear(
                    previous,
                    hidden,
                )
            )

            layers.append(
                nn.ReLU()
            )

            previous = hidden

        layers.append(
            nn.Linear(
                previous,
                action_dim,
            )
        )

        self.network = nn.Sequential(
            *layers
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(x)


@dataclass
class ReplayTransition:
    state: np.ndarray

    action: int
    reward: float

    next_state: np.ndarray

    terminal: bool

    packed_next_mask: np.ndarray


class ReplayBuffer:
    """
    Experience replay with memory-conscious storage.

    States are stored as float16 and converted back to
    float32 during sampling.

    Action masks are bit-packed.

    These are storage optimizations only; neural-network
    computation remains float32.
    """

    def __init__(
        self,
        *,
        capacity: int,
        action_dim: int,
    ) -> None:

        if capacity <= 0:
            raise ValueError(
                "capacity must be positive"
            )

        if action_dim <= 0:
            raise ValueError(
                "action_dim must be positive"
            )

        self.capacity = capacity
        self.action_dim = action_dim

        self._data: deque[
            ReplayTransition
        ] = deque(
            maxlen=capacity
        )

    def __len__(self) -> int:
        return len(self._data)

    def append(
        self,
        *,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminal: bool,
        next_action_mask: np.ndarray,
    ) -> None:

        state = np.asarray(
            state,
            dtype=np.float32,
        )

        next_state = np.asarray(
            next_state,
            dtype=np.float32,
        )

        next_action_mask = np.asarray(
            next_action_mask,
            dtype=bool,
        )

        if next_action_mask.shape != (
            self.action_dim,
        ):
            raise ValueError(
                "next_action_mask has invalid shape"
            )

        packed_mask = np.packbits(
            next_action_mask.astype(
                np.uint8
            )
        )

        self._data.append(
            ReplayTransition(
                state=state.astype(
                    np.float16
                ),
                action=int(action),
                reward=float(reward),
                next_state=next_state.astype(
                    np.float16
                ),
                terminal=bool(terminal),
                packed_next_mask=packed_mask,
            )
        )

    def sample(
        self,
        *,
        batch_size: int,
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be positive"
            )

        if len(self._data) < batch_size:
            raise ValueError(
                "not enough replay transitions"
            )

        indices = rng.choice(
            len(self._data),
            size=batch_size,
            replace=False,
        )

        transitions = [
            self._data[int(index)]
            for index in indices
        ]

        states = np.stack(
            [
                transition.state
                for transition in transitions
            ]
        ).astype(
            np.float32
        )

        next_states = np.stack(
            [
                transition.next_state
                for transition in transitions
            ]
        ).astype(
            np.float32
        )

        actions = np.asarray(
            [
                transition.action
                for transition in transitions
            ],
            dtype=np.int64,
        )

        rewards = np.asarray(
            [
                transition.reward
                for transition in transitions
            ],
            dtype=np.float32,
        )

        terminals = np.asarray(
            [
                transition.terminal
                for transition in transitions
            ],
            dtype=bool,
        )

        next_masks = np.stack(
            [
                np.unpackbits(
                    transition.packed_next_mask
                )[
                    : self.action_dim
                ].astype(bool)
                for transition in transitions
            ]
        )

        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "terminals": terminals,
            "next_action_masks": (
                next_masks
            ),
        }


class DQNAgent:
    """
    Vanilla DQN agent with:

    - experience replay
    - target network
    - masked epsilon-greedy actions
    - Adam optimizer
    - Smooth-L1 loss

    Double-DQN, dueling networks and prioritized replay
    are intentionally NOT used because they are not part
    of the manuscript methodology.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        action_dim: int,
        seed: int,
        config: DQNConfig | None = None,
        device: str = "cpu",
    ) -> None:

        self.config = (
            config
            if config is not None
            else DQNConfig()
        )

        self.input_dim = input_dim
        self.action_dim = action_dim

        self.device = torch.device(
            device
        )

        self.rng = np.random.default_rng(
            seed
        )

        torch.manual_seed(seed)

        self.online_network = QNetwork(
            input_dim=input_dim,
            action_dim=action_dim,
            hidden_layers=(
                self.config.hidden_layers
            ),
        ).to(
            self.device
        )

        self.target_network = QNetwork(
            input_dim=input_dim,
            action_dim=action_dim,
            hidden_layers=(
                self.config.hidden_layers
            ),
        ).to(
            self.device
        )

        self.target_network.load_state_dict(
            self.online_network.state_dict()
        )

        self.target_network.eval()

        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(),
            lr=self.config.learning_rate,
        )

        self.loss_function = (
            nn.SmoothL1Loss()
        )

        self.replay_buffer = ReplayBuffer(
            capacity=(
                self.config.replay_capacity
            ),
            action_dim=action_dim,
        )

        self.optimization_steps = 0

    def epsilon(
        self,
        episode_index: int,
    ) -> float:
        return linear_epsilon(
            episode_index,
            start=(
                self.config.epsilon_start
            ),
            end=(
                self.config.epsilon_end
            ),
            decay_episodes=(
                self.config
                .epsilon_decay_episodes
            ),
        )

    def select_greedy_action(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
    ) -> int:

        observation = np.asarray(
            observation,
            dtype=np.float32,
        )

        action_mask = np.asarray(
            action_mask,
            dtype=bool,
        )

        if action_mask.shape != (
            self.action_dim,
        ):
            raise ValueError(
                "action_mask has invalid shape"
            )

        valid_actions = np.flatnonzero(
            action_mask
        )

        if len(valid_actions) == 0:
            raise ValueError(
                "no valid actions available"
            )

        with torch.no_grad():

            state_tensor = torch.as_tensor(
                observation,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)

            q_values = (
                self.online_network(
                    state_tensor
                )[0]
            )

            mask_tensor = torch.as_tensor(
                action_mask,
                dtype=torch.bool,
                device=self.device,
            )

            q_values = q_values.masked_fill(
                ~mask_tensor,
                float("-inf"),
            )

            return int(
                torch.argmax(
                    q_values
                ).item()
            )

    def select_action(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        *,
        epsilon: float,
    ) -> int:

        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(
                "epsilon must be between 0 and 1"
            )

        action_mask = np.asarray(
            action_mask,
            dtype=bool,
        )

        if action_mask.shape != (
            self.action_dim,
        ):
            raise ValueError(
                "action_mask has invalid shape"
            )

        valid_actions = np.flatnonzero(
            action_mask
        )

        if len(valid_actions) == 0:
            raise ValueError(
                "no valid actions available"
            )

        # Important:
        # epsilon == 0 must not advance the exploration RNG.
        if epsilon == 0.0:
            return self.select_greedy_action(
                observation,
                action_mask,
            )

        if self.rng.random() < epsilon:
            return int(
                self.rng.choice(
                    valid_actions
                )
            )

        return self.select_greedy_action(
            observation,
            action_mask,
        )
    def remember(
        self,
        *,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminal: bool,
        next_action_mask: np.ndarray,
    ) -> None:

        self.replay_buffer.append(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            terminal=terminal,
            next_action_mask=(
                next_action_mask
            ),
        )

    def can_optimize(self) -> bool:
        required = max(
            self.config.batch_size,
            self.config
            .replay_warmup_transitions,
        )

        return (
            len(self.replay_buffer)
            >= required
        )

    def synchronize_target(self) -> None:
        self.target_network.load_state_dict(
            self.online_network.state_dict()
        )

    def optimize(self) -> float | None:

        if not self.can_optimize():
            return None

        batch = self.replay_buffer.sample(
            batch_size=(
                self.config.batch_size
            ),
            rng=self.rng,
        )

        states = torch.as_tensor(
            batch["states"],
            dtype=torch.float32,
            device=self.device,
        )

        actions = torch.as_tensor(
            batch["actions"],
            dtype=torch.int64,
            device=self.device,
        )

        rewards = torch.as_tensor(
            batch["rewards"],
            dtype=torch.float32,
            device=self.device,
        )

        next_states = torch.as_tensor(
            batch["next_states"],
            dtype=torch.float32,
            device=self.device,
        )

        terminals = torch.as_tensor(
            batch["terminals"],
            dtype=torch.bool,
            device=self.device,
        )

        next_masks = torch.as_tensor(
            batch["next_action_masks"],
            dtype=torch.bool,
            device=self.device,
        )

        current_q = (
            self.online_network(
                states
            )
            .gather(
                1,
                actions.unsqueeze(1),
            )
            .squeeze(1)
        )

        with torch.no_grad():

            next_q_all = (
                self.target_network(
                    next_states
                )
            )

            masked_next_q = (
                next_q_all.masked_fill(
                    ~next_masks,
                    float("-inf"),
                )
            )

            has_valid_action = (
                next_masks.any(
                    dim=1
                )
            )

            next_q = torch.zeros(
                len(states),
                dtype=torch.float32,
                device=self.device,
            )

            bootstrap_mask = (
                (~terminals)
                & has_valid_action
            )

            if bootstrap_mask.any():

                next_q[
                    bootstrap_mask
                ] = (
                    masked_next_q[
                        bootstrap_mask
                    ]
                    .max(
                        dim=1
                    )
                    .values
                )

            target_q = (
                rewards
                + self.config.gamma
                * next_q
            )

        loss = self.loss_function(
            current_q,
            target_q,
        )

        self.optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        self.optimizer.step()

        self.optimization_steps += 1

        if (
            self.optimization_steps
            % self.config
            .target_update_steps
            == 0
        ):
            self.synchronize_target()

        return float(
            loss.detach().cpu().item()
        )