from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from run_dqn_primary_training import (
    build_environment,
)

from smart_waste.rl.convergence_training import (
    GreedyEvaluation,
    train_dqn_until_convergence,
)

from smart_waste.rl.dqn import (
    DQNAgent,
    DQNConfig,
)


BASE_CONFIG = Path(
    "configs/base.yaml"
)

DEFAULT_CONFIG = Path(
    "configs/dqn_convergence.yaml"
)


def load_yaml(
    path: Path,
) -> dict:

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return yaml.safe_load(
            file
        )


def build_agent(
    *,
    env,
    protocol_cfg: dict,
) -> DQNAgent:

    dqn = protocol_cfg[
        "dqn"
    ]

    config = DQNConfig(
        hidden_layers=tuple(
            dqn[
                "hidden_layers"
            ]
        ),
        gamma=float(
            dqn["gamma"]
        ),
        learning_rate=float(
            dqn[
                "learning_rate"
            ]
        ),
        replay_capacity=int(
            dqn[
                "replay_capacity"
            ]
        ),
        batch_size=int(
            dqn[
                "batch_size"
            ]
        ),
        epsilon_start=float(
            dqn[
                "epsilon_start"
            ]
        ),
        epsilon_end=float(
            dqn[
                "epsilon_end"
            ]
        ),
        epsilon_decay_episodes=int(
            dqn[
                "epsilon_decay_episodes"
            ]
        ),
        replay_warmup_transitions=int(
            dqn[
                "replay_warmup_transitions"
            ]
        ),
        target_update_steps=int(
            dqn[
                "target_update_steps"
            ]
        ),
    )

    return DQNAgent(
        input_dim=(
            env.observation_space.shape[0]
        ),
        action_dim=(
            env.action_space.n
        ),
        seed=int(
            protocol_cfg[
                "protocol"
            ][
                "agent_seed"
            ]
        ),
        config=config,
        device="cpu",
    )


def save_model_snapshot(
    *,
    agent: DQNAgent,
    path: Path,
    episodes_completed: int,
    converged: bool | None = None,
    stop_reason: str | None = None,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "episodes_completed":
                int(
                    episodes_completed
                ),

            "online_network_state_dict":
                agent.online_network.state_dict(),

            "target_network_state_dict":
                agent.target_network.state_dict(),

            "optimizer_state_dict":
                agent.optimizer.state_dict(),

            "optimization_steps":
                int(
                    agent.optimization_steps
                ),

            "converged":
                converged,

            "stop_reason":
                stop_reason,
        },
        path,
    )


def evaluate_greedy(
    *,
    agent: DQNAgent,
    base_cfg: dict,
    protocol_cfg: dict,
    episodes_completed: int,
    checkpoint_directory: Path,
) -> GreedyEvaluation:

    env = build_environment(
        base_cfg=base_cfg,
        protocol_cfg=protocol_cfg,
    )

    seed = int(
        protocol_cfg[
            "protocol"
        ][
            "city_master_seed"
        ]
    )

    observation, info = env.reset(
        seed=seed
    )

    terminated = False
    truncated = False

    total_reward = 0.0
    steps = 0

    was_training = (
        agent.online_network.training
    )

    agent.online_network.eval()

    try:

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

            action = (
                agent.select_greedy_action(
                    observation,
                    action_mask,
                )
            )

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                action
            )

            total_reward += float(
                reward
            )

            steps += 1

    finally:

        if was_training:
            agent.online_network.train()

    result = GreedyEvaluation(
        episode_completed=(
            episodes_completed
        ),
        steps=steps,
        total_reward=float(
            total_reward
        ),
        distance_km=float(
            env.total_distance_km
        ),
        fuel_litres=float(
            env.total_fuel_litres
        ),
        simulated_time_hours=float(
            info[
                "simulated_time_hours"
            ]
        ),
        terminated=bool(
            terminated
        ),
        truncated=bool(
            truncated
        ),
    )

    checkpoint_path = (
        checkpoint_directory
        / (
            f"episode_"
            f"{episodes_completed:04d}.pt"
        )
    )

    save_model_snapshot(
        agent=agent,
        path=checkpoint_path,
        episodes_completed=(
            episodes_completed
        ),
    )

    print(
        f"[checkpoint {episodes_completed:4d}] "
        f"reward={result.total_reward:10.4f}  "
        f"distance={result.distance_km:10.4f} km  "
        f"fuel={result.fuel_litres:9.4f} L"
    )

    return result


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )

    args = parser.parse_args()

    base_cfg = load_yaml(
        BASE_CONFIG
    )

    protocol_cfg = load_yaml(
        args.config
    )

    protocol = protocol_cfg[
        "protocol"
    ]

    convergence = protocol_cfg[
        "convergence"
    ]

    outputs = protocol_cfg[
        "outputs"
    ]

    master_seed = int(
        protocol[
            "agent_seed"
        ]
    )

    np.random.seed(
        master_seed
    )

    torch.manual_seed(
        master_seed
    )

    torch.use_deterministic_algorithms(
        True
    )

    torch.set_num_threads(
        4
    )

    env = build_environment(
        base_cfg=base_cfg,
        protocol_cfg=protocol_cfg,
    )

    agent = build_agent(
        env=env,
        protocol_cfg=protocol_cfg,
    )

    checkpoint_directory = Path(
        outputs[
            "checkpoint_directory"
        ]
    )

    checkpoint_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 96)
    print(
        "CONVERGENCE-DRIVEN DQN TRAINING"
    )
    print("=" * 96)

    print(
        "Minimum episodes    :",
        convergence[
            "minimum_episodes"
        ],
    )

    print(
        "Maximum episodes    :",
        convergence[
            "maximum_episodes"
        ],
    )

    print(
        "Convergence window  :",
        convergence[
            "window_size"
        ],
    )

    print(
        "Patience windows    :",
        convergence[
            "patience_windows"
        ],
    )

    print(
        "Checkpoint interval :",
        convergence[
            "greedy_evaluation_interval_episodes"
        ],
    )

    print(
        "Epsilon decay       :",
        protocol_cfg[
            "dqn"
        ][
            "epsilon_decay_episodes"
        ],
        "episodes",
    )

    print(
        "Observation dim     :",
        env.observation_space.shape[0],
    )

    print(
        "Action dim          :",
        env.action_space.n,
    )

    print("-" * 96)

    def evaluator(
        current_agent: DQNAgent,
        episodes_completed: int,
    ) -> GreedyEvaluation:

        return evaluate_greedy(
            agent=current_agent,
            base_cfg=base_cfg,
            protocol_cfg=protocol_cfg,
            episodes_completed=(
                episodes_completed
            ),
            checkpoint_directory=(
                checkpoint_directory
            ),
        )

    result = train_dqn_until_convergence(
        env=env,
        agent=agent,
        master_seed=int(
            protocol[
                "city_master_seed"
            ]
        ),
        minimum_episodes=int(
            convergence[
                "minimum_episodes"
            ]
        ),
        maximum_episodes=int(
            convergence[
                "maximum_episodes"
            ]
        ),
        window_size=int(
            convergence[
                "window_size"
            ]
        ),
        patience_windows=int(
            convergence[
                "patience_windows"
            ]
        ),
        reward_absolute_slope_threshold=float(
            convergence[
                "reward_absolute_slope_threshold"
            ]
        ),
        loss_absolute_slope_threshold=float(
            convergence[
                "loss_absolute_slope_threshold"
            ]
        ),
        greedy_evaluation_interval_episodes=int(
            convergence[
                "greedy_evaluation_interval_episodes"
            ]
        ),
        greedy_relative_change_threshold=float(
            convergence[
                "greedy_relative_change_threshold"
            ]
        ),
        greedy_evaluator=evaluator,
        require_natural_termination=bool(
            convergence[
                "require_natural_termination"
            ]
        ),
    )

    episode_path = Path(
        outputs[
            "episode_metrics_csv"
        ]
    )

    greedy_path = Path(
        outputs[
            "greedy_evaluations_csv"
        ]
    )

    checks_path = Path(
        outputs[
            "convergence_checks_csv"
        ]
    )

    final_model_path = Path(
        outputs[
            "model_checkpoint"
        ]
    )

    protocol_snapshot_path = Path(
        outputs[
            "protocol_snapshot"
        ]
    )

    for path in (
        episode_path,
        greedy_path,
        checks_path,
        final_model_path,
        protocol_snapshot_path,
    ):
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    pd.DataFrame(
        result.episode_records()
    ).to_csv(
        episode_path,
        index=False,
    )

    pd.DataFrame(
        result.evaluation_records()
    ).to_csv(
        greedy_path,
        index=False,
    )

    pd.DataFrame(
        result.convergence_records()
    ).to_csv(
        checks_path,
        index=False,
    )

    save_model_snapshot(
        agent=agent,
        path=final_model_path,
        episodes_completed=(
            result.episodes_completed
        ),
        converged=(
            result.converged
        ),
        stop_reason=(
            result.stop_reason
        ),
    )

    shutil.copyfile(
        args.config,
        protocol_snapshot_path,
    )

    episodes_df = pd.DataFrame(
        result.episode_records()
    )

    evaluations_df = pd.DataFrame(
        result.evaluation_records()
    )

    print()
    print("=" * 96)
    print(
        "FINAL TRAINING SUMMARY"
    )
    print("=" * 96)

    print(
        "Episodes completed :",
        result.episodes_completed,
    )

    print(
        "Converged          :",
        result.converged,
    )

    print(
        "Stop reason        :",
        result.stop_reason,
    )

    if not episodes_df.empty:

        last50 = episodes_df.tail(
            50
        )

        print(
            "Last-50 reward mean:",
            f"{last50['total_reward'].mean():.6f}",
        )

        print(
            "Last-50 fuel mean  :",
            f"{last50['fuel_litres'].mean():.6f} L",
        )

    if not evaluations_df.empty:

        final_eval = (
            evaluations_df.iloc[-1]
        )

        print(
            "Final greedy reward:",
            f"{final_eval['total_reward']:.6f}",
        )

        print(
            "Final greedy dist. :",
            f"{final_eval['distance_km']:.6f} km",
        )

        print(
            "Final greedy fuel  :",
            f"{final_eval['fuel_litres']:.6f} L",
        )

    if result.convergence_checks:

        last_check = (
            result.convergence_checks[-1]
        )

        print("-" * 96)

        print(
            "Reward slope       :",
            f"{last_check.reward_slope:.8f}",
        )

        print(
            "Loss slope         :",
            f"{last_check.loss_slope:.8f}",
        )

        print(
            "Greedy fuel change :",
            last_check.greedy_fuel_relative_change,
        )

        print(
            "Stable checks      :",
            last_check.consecutive_stable_checks,
            "/",
            convergence[
                "patience_windows"
            ],
        )

    print("-" * 96)

    print(
        "Episode metrics     :",
        episode_path,
    )

    print(
        "Greedy evaluations  :",
        greedy_path,
    )

    print(
        "Convergence checks  :",
        checks_path,
    )

    print(
        "Final checkpoint    :",
        final_model_path,
    )

    print(
        "Protocol snapshot   :",
        protocol_snapshot_path,
    )

    print("=" * 96)


if __name__ == "__main__":
    main()