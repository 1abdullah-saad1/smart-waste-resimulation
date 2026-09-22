from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from smart_waste.rl.dqn import (
    DQNAgent,
    DQNConfig,
)
from run_dqn_primary_training import (
    build_environment,
)


BASE_CONFIG = Path(
    "configs/base.yaml"
)

PROTOCOL_CONFIG = Path(
    "configs/dqn_primary_reconstruction.yaml"
)

CHECKPOINT = Path(
    "results/models/dqn_primary_final.pt"
)

OUTPUT = Path(
    "results/raw/"
    "dqn_primary_greedy_evaluation.csv"
)


def load_yaml(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def main() -> None:

    base_cfg = load_yaml(
        BASE_CONFIG
    )

    protocol_cfg = load_yaml(
        PROTOCOL_CONFIG
    )

    env = build_environment(
        base_cfg=base_cfg,
        protocol_cfg=protocol_cfg,
    )

    dqn_cfg = protocol_cfg["dqn"]

    config = DQNConfig(
        hidden_layers=tuple(
            dqn_cfg["hidden_layers"]
        ),
        gamma=float(
            dqn_cfg["gamma"]
        ),
        learning_rate=float(
            dqn_cfg["learning_rate"]
        ),
        replay_capacity=int(
            dqn_cfg["replay_capacity"]
        ),
        batch_size=int(
            dqn_cfg["batch_size"]
        ),
        epsilon_start=float(
            dqn_cfg["epsilon_start"]
        ),
        epsilon_end=float(
            dqn_cfg["epsilon_end"]
        ),
        epsilon_decay_episodes=int(
            dqn_cfg[
                "epsilon_decay_episodes"
            ]
        ),
        replay_warmup_transitions=int(
            dqn_cfg[
                "replay_warmup_transitions"
            ]
        ),
        target_update_steps=int(
            dqn_cfg[
                "target_update_steps"
            ]
        ),
    )

    agent = DQNAgent(
        input_dim=(
            env.observation_space.shape[0]
        ),
        action_dim=(
            env.action_space.n
        ),
        seed=int(
            protocol_cfg["protocol"][
                "agent_seed"
            ]
        ),
        config=config,
        device="cpu",
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    agent.online_network.load_state_dict(
        checkpoint[
            "online_network_state_dict"
        ]
    )

    agent.target_network.load_state_dict(
        checkpoint[
            "target_network_state_dict"
        ]
    )

    agent.online_network.eval()
    agent.target_network.eval()

    observation, info = env.reset(
        seed=int(
            protocol_cfg["protocol"][
                "city_master_seed"
            ]
        )
    )

    terminated = False
    truncated = False

    total_reward = 0.0
    steps = 0

    step_records = []

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
            epsilon=0.0,
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

        total_reward += reward
        steps += 1

        step_records.append(
            {
                "step": steps,
                "selected_bin": action,
                "reward": reward,
                "candidate_count_after": (
                    next_info[
                        "candidate_count"
                    ]
                ),
                "step_distance_km": (
                    next_info[
                        "step_distance_km"
                    ]
                ),
                "step_fuel_litres": (
                    next_info[
                        "step_fuel_litres"
                    ]
                ),
                "total_distance_km": (
                    next_info[
                        "total_distance_km"
                    ]
                ),
                "total_fuel_litres": (
                    next_info[
                        "total_fuel_litres"
                    ]
                ),
                "simulated_time_hours": (
                    next_info[
                        "simulated_time_hours"
                    ]
                ),
            }
        )

        observation = next_observation
        info = next_info

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        step_records
    ).to_csv(
        OUTPUT,
        index=False,
    )

    print("=" * 88)
    print(
        "FROZEN PRIMARY DQN GREEDY EVALUATION"
    )
    print("=" * 88)

    print(
        f"Checkpoint episodes : "
        f"{checkpoint['episodes_completed']}"
    )

    print(
        f"Epsilon             : 0.0"
    )

    print(
        f"Steps               : {steps}"
    )

    print(
        f"Terminated          : {terminated}"
    )

    print(
        f"Truncated           : {truncated}"
    )

    print(
        f"Total reward        : "
        f"{total_reward:.6f}"
    )

    print(
        f"Total distance      : "
        f"{env.total_distance_km:.6f} km"
    )

    print(
        f"Total fuel          : "
        f"{env.total_fuel_litres:.6f} L"
    )

    print(
        f"Simulated time      : "
        f"{info['simulated_time_hours']:.6f} h"
    )

    print(
        f"Output              : {OUTPUT}"
    )

    print("=" * 88)


if __name__ == "__main__":
    main()