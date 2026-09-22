from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

from smart_waste.rl.dqn import DQNAgent
from smart_waste.rl.environment import SmartWasteRoutingEnv


@dataclass(frozen=True)
class EpisodeResult:
    episode: int
    environment_seed: int

    epsilon: float

    steps: int

    total_reward: float

    mean_loss: float | None
    final_loss: float | None
    optimization_steps: int

    distance_km: float
    fuel_litres: float

    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class TrainingResult:
    episodes: tuple[EpisodeResult, ...]

    def to_records(self) -> list[dict]:
        return [
            asdict(episode)
            for episode in self.episodes
        ]


def train_dqn(
    *,
    env: SmartWasteRoutingEnv,
    agent: DQNAgent,
    episodes: int,
    master_seed: int,
    episode_seed_fn: Callable[
        [int, int],
        int,
    ]
    | None = None,
) -> TrainingResult:
    """
    Train a DQN agent for a fixed number of episodes.

    The environment itself remains responsible for the physical
    routing state. This function only controls:

      - episode reset,
      - epsilon scheduling,
      - transition collection,
      - replay optimization,
      - per-episode metric collection.

    The primary reconstruction uses a fixed city/scenario across
    the 100 episodes so that episodic reward/loss convergence can
    be observed as described in the manuscript.
    """

    if episodes <= 0:
        raise ValueError(
            "episodes must be positive"
        )

    results: list[EpisodeResult] = []

    for episode in range(episodes):

        if episode_seed_fn is None:
            environment_seed = (
                master_seed + episode
            )
        else:
            environment_seed = (
                episode_seed_fn(
                    master_seed,
                    episode,
                )
            )

        observation, info = env.reset(
            seed=environment_seed
        )

        epsilon = agent.epsilon(
            episode
        )

        terminated = False
        truncated = False

        episode_reward = 0.0
        episode_losses: list[float] = []

        episode_optimization_start = (
            agent.optimization_steps
        )

        steps = 0

        while not (
            terminated
            or truncated
        ):

            action_mask = info[
                "action_mask"
            ]

            if not np.any(
                action_mask
            ):
                terminated = True
                break

            action = agent.select_action(
                observation,
                action_mask,
                epsilon=epsilon,
            )

            (
                next_observation,
                reward,
                terminated,
                truncated,
                next_info,
            ) = env.step(action)

            terminal_for_replay = (
                terminated
                or truncated
            )

            agent.remember(
                state=observation,
                action=action,
                reward=reward,
                next_state=next_observation,
                terminal=(
                    terminal_for_replay
                ),
                next_action_mask=(
                    next_info[
                        "action_mask"
                    ]
                ),
            )

            loss = agent.optimize()

            if loss is not None:
                if not np.isfinite(loss):
                    raise RuntimeError(
                        "non-finite DQN loss "
                        f"at episode {episode}"
                    )

                episode_losses.append(
                    loss
                )

            if not np.isfinite(
                reward
            ):
                raise RuntimeError(
                    "non-finite reward "
                    f"at episode {episode}"
                )

            episode_reward += reward

            observation = (
                next_observation
            )

            info = next_info

            steps += 1

        if episode_losses:
            mean_loss = float(
                np.mean(
                    episode_losses
                )
            )

            final_loss = float(
                episode_losses[-1]
            )
        else:
            mean_loss = None
            final_loss = None

        episode_optimization_steps = (
            agent.optimization_steps
            - episode_optimization_start
        )

        results.append(
            EpisodeResult(
                episode=episode,
                environment_seed=(
                    environment_seed
                ),
                epsilon=float(
                    epsilon
                ),
                steps=steps,
                total_reward=float(
                    episode_reward
                ),
                mean_loss=mean_loss,
                final_loss=final_loss,
                optimization_steps=(
                    episode_optimization_steps
                ),
                distance_km=float(
                    env.total_distance_km
                ),
                fuel_litres=float(
                    env.total_fuel_litres
                ),
                terminated=bool(
                    terminated
                ),
                truncated=bool(
                    truncated
                ),
            )
        )

    return TrainingResult(
        episodes=tuple(
            results
        )
    )