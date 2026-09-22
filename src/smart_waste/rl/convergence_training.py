from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

from smart_waste.rl.dqn import DQNAgent
from smart_waste.rl.environment import (
    SmartWasteRoutingEnv,
)
from smart_waste.rl.training import (
    EpisodeResult,
)


@dataclass(frozen=True)
class GreedyEvaluation:
    episode_completed: int

    steps: int
    total_reward: float

    distance_km: float
    fuel_litres: float

    simulated_time_hours: float

    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class ConvergenceCheck:
    episode_completed: int

    reward_slope: float
    loss_slope: float

    greedy_fuel_relative_change: (
        float | None
    )

    reward_stable: bool
    loss_stable: bool
    greedy_stable: bool
    natural_termination_ok: bool

    criteria_met: bool

    consecutive_stable_checks: int


@dataclass(frozen=True)
class ConvergenceTrainingResult:
    episodes: tuple[
        EpisodeResult,
        ...
    ]

    evaluations: tuple[
        GreedyEvaluation,
        ...
    ]

    convergence_checks: tuple[
        ConvergenceCheck,
        ...
    ]

    converged: bool
    stop_reason: str

    @property
    def episodes_completed(self) -> int:
        return len(
            self.episodes
        )

    def episode_records(
        self,
    ) -> list[dict]:
        return [
            asdict(item)
            for item in self.episodes
        ]

    def evaluation_records(
        self,
    ) -> list[dict]:
        return [
            asdict(item)
            for item in self.evaluations
        ]

    def convergence_records(
        self,
    ) -> list[dict]:
        return [
            asdict(item)
            for item
            in self.convergence_checks
        ]


GreedyEvaluator = Callable[
    [DQNAgent, int],
    GreedyEvaluation,
]


def linear_slope(
    values: list[float],
) -> float:

    if len(values) < 2:
        return float("nan")

    y = np.asarray(
        values,
        dtype=np.float64,
    )

    x = np.arange(
        len(y),
        dtype=np.float64,
    )

    return float(
        np.polyfit(
            x,
            y,
            1,
        )[0]
    )


def relative_change(
    current: float,
    previous: float,
) -> float:

    denominator = max(
        abs(previous),
        1.0e-12,
    )

    return float(
        abs(
            current - previous
        )
        / denominator
    )


def train_dqn_until_convergence(
    *,
    env: SmartWasteRoutingEnv,
    agent: DQNAgent,
    master_seed: int,

    minimum_episodes: int,
    maximum_episodes: int,

    window_size: int,
    patience_windows: int,

    reward_absolute_slope_threshold: float,
    loss_absolute_slope_threshold: float,

    greedy_evaluation_interval_episodes: int,
    greedy_relative_change_threshold: float,

    greedy_evaluator: GreedyEvaluator,

    require_natural_termination: bool = True,
) -> ConvergenceTrainingResult:

    if minimum_episodes <= 0:
        raise ValueError(
            "minimum_episodes must be positive"
        )

    if maximum_episodes < minimum_episodes:
        raise ValueError(
            "maximum_episodes must be >= minimum_episodes"
        )

    if window_size < 2:
        raise ValueError(
            "window_size must be >= 2"
        )

    if patience_windows <= 0:
        raise ValueError(
            "patience_windows must be positive"
        )

    if (
        greedy_evaluation_interval_episodes
        <= 0
    ):
        raise ValueError(
            "greedy evaluation interval must be positive"
        )

    if (
        reward_absolute_slope_threshold
        < 0.0
    ):
        raise ValueError(
            "reward slope threshold cannot be negative"
        )

    if (
        loss_absolute_slope_threshold
        < 0.0
    ):
        raise ValueError(
            "loss slope threshold cannot be negative"
        )

    if (
        greedy_relative_change_threshold
        < 0.0
    ):
        raise ValueError(
            "greedy change threshold cannot be negative"
        )

    episode_results: list[
        EpisodeResult
    ] = []

    evaluations: list[
        GreedyEvaluation
    ] = []

    checks: list[
        ConvergenceCheck
    ] = []

    consecutive_stable_checks = 0

    converged = False

    stop_reason = (
        "maximum_episodes_reached"
    )

    for episode in range(
        maximum_episodes
    ):

        environment_seed = (
            master_seed
            + episode
        )

        observation, info = env.reset(
            seed=environment_seed
        )

        epsilon = agent.epsilon(
            episode
        )

        terminated = False
        truncated = False

        total_reward = 0.0

        losses: list[float] = []

        optimization_start = (
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
            ) = env.step(
                action
            )

            agent.remember(
                state=observation,
                action=action,
                reward=reward,
                next_state=(
                    next_observation
                ),
                terminal=(
                    terminated
                    or truncated
                ),
                next_action_mask=(
                    next_info[
                        "action_mask"
                    ]
                ),
            )

            loss = agent.optimize()

            if loss is not None:

                if not np.isfinite(
                    loss
                ):
                    raise RuntimeError(
                        "non-finite DQN loss"
                    )

                losses.append(
                    float(loss)
                )

            if not np.isfinite(
                reward
            ):
                raise RuntimeError(
                    "non-finite reward"
                )

            total_reward += (
                reward
            )

            observation = (
                next_observation
            )

            info = next_info

            steps += 1

        if losses:

            mean_loss = float(
                np.mean(
                    losses
                )
            )

            final_loss = float(
                losses[-1]
            )

        else:

            mean_loss = None
            final_loss = None

        episode_result = EpisodeResult(
            episode=episode,
            environment_seed=(
                environment_seed
            ),
            epsilon=float(
                epsilon
            ),
            steps=steps,
            total_reward=float(
                total_reward
            ),
            mean_loss=mean_loss,
            final_loss=final_loss,
            optimization_steps=(
                agent.optimization_steps
                - optimization_start
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

        episode_results.append(
            episode_result
        )

        episodes_completed = (
            episode + 1
        )

        should_evaluate = (
            episodes_completed
            % greedy_evaluation_interval_episodes
            == 0
        )

        if not should_evaluate:
            continue

        evaluation = greedy_evaluator(
            agent,
            episodes_completed,
        )

        evaluations.append(
            evaluation
        )

        if (
            episodes_completed
            < minimum_episodes
        ):
            continue

        if (
            len(episode_results)
            < window_size
        ):
            continue

        window = episode_results[
            -window_size:
        ]

        reward_values = [
            item.total_reward
            for item in window
        ]

        reward_slope = linear_slope(
            reward_values
        )

        loss_values = [
            float(item.mean_loss)
            for item in window
            if item.mean_loss is not None
        ]

        loss_slope = linear_slope(
            loss_values
        )

        reward_stable = bool(
            np.isfinite(
                reward_slope
            )
            and abs(
                reward_slope
            )
            <= (
                reward_absolute_slope_threshold
            )
        )

        loss_stable = bool(
            np.isfinite(
                loss_slope
            )
            and abs(
                loss_slope
            )
            <= (
                loss_absolute_slope_threshold
            )
        )

        if len(
            evaluations
        ) >= 2:

            previous_evaluation = (
                evaluations[-2]
            )

            greedy_change = (
                relative_change(
                    evaluation.fuel_litres,
                    previous_evaluation.fuel_litres,
                )
            )

            greedy_stable = bool(
                greedy_change
                <= (
                    greedy_relative_change_threshold
                )
            )

        else:

            greedy_change = None
            greedy_stable = False

        if require_natural_termination:

            training_natural = all(
                item.terminated
                and not item.truncated
                for item in window
            )

            evaluation_natural = (
                evaluation.terminated
                and not evaluation.truncated
            )

            natural_termination_ok = (
                training_natural
                and evaluation_natural
            )

        else:

            natural_termination_ok = True

        criteria_met = (
            reward_stable
            and loss_stable
            and greedy_stable
            and natural_termination_ok
        )

        if criteria_met:

            consecutive_stable_checks += 1

        else:

            consecutive_stable_checks = 0

        check = ConvergenceCheck(
            episode_completed=(
                episodes_completed
            ),
            reward_slope=float(
                reward_slope
            ),
            loss_slope=float(
                loss_slope
            ),
            greedy_fuel_relative_change=(
                greedy_change
            ),
            reward_stable=(
                reward_stable
            ),
            loss_stable=(
                loss_stable
            ),
            greedy_stable=(
                greedy_stable
            ),
            natural_termination_ok=(
                natural_termination_ok
            ),
            criteria_met=(
                criteria_met
            ),
            consecutive_stable_checks=(
                consecutive_stable_checks
            ),
        )

        checks.append(
            check
        )

        if (
            consecutive_stable_checks
            >= patience_windows
        ):

            converged = True

            stop_reason = (
                "convergence_criteria_satisfied"
            )

            break

    return ConvergenceTrainingResult(
        episodes=tuple(
            episode_results
        ),
        evaluations=tuple(
            evaluations
        ),
        convergence_checks=tuple(
            checks
        ),
        converged=converged,
        stop_reason=stop_reason,
    )