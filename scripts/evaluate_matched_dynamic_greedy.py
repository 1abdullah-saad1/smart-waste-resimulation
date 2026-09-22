from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from run_dqn_primary_training import (
    build_environment,
)


BASE_CONFIG = Path(
    "configs/base.yaml"
)

PROTOCOL_CONFIG = Path(
    "configs/dqn_primary_reconstruction.yaml"
)

OUTPUT = Path(
    "results/raw/"
    "matched_dynamic_greedy_evaluation.csv"
)


def load_yaml(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def euclidean(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    return float(
        np.linalg.norm(a - b)
    )


def incremental_dispatch_distance(
    env,
    bin_id: int,
) -> float:
    """
    Exact myopic dispatch cost under the same capacity policy
    used by SmartWasteRoutingEnv.

    If the active truck cannot carry the selected bin's waste,
    the cost includes:
        current truck -> depot -> selected bin

    Otherwise:
        current truck -> selected bin
    """

    truck_id = env.active_truck

    current_location = env.truck_xy[
        truck_id
    ]

    fill_fraction = (
        float(
            env.fill_percent[bin_id]
        )
        / 100.0
    )

    demand_tonnes = (
        env.full_bin_mass_tonnes
        * fill_fraction
    )

    current_load = float(
        env.truck_load_tonnes[
            truck_id
        ]
    )

    if (
        current_load
        + demand_tonnes
        > env.truck_capacity_tonnes
    ):
        return (
            euclidean(
                current_location,
                env.depot_xy,
            )
            + euclidean(
                env.depot_xy,
                env.bin_xy[bin_id],
            )
        )

    return euclidean(
        current_location,
        env.bin_xy[bin_id],
    )


def select_greedy_action(
    env,
    action_mask: np.ndarray,
) -> int:

    valid_actions = np.flatnonzero(
        action_mask
    )

    if len(valid_actions) == 0:
        raise RuntimeError(
            "no valid actions"
        )

    distances = np.asarray(
        [
            incremental_dispatch_distance(
                env,
                int(bin_id),
            )
            for bin_id in valid_actions
        ],
        dtype=np.float64,
    )

    # Stable deterministic tie-breaking:
    # valid_actions is sorted by bin ID and argmin
    # selects the first minimum.
    best_index = int(
        np.argmin(distances)
    )

    return int(
        valid_actions[
            best_index
        ]
    )


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

    records = []

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

        action = select_greedy_action(
            env,
            action_mask,
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

        steps += 1
        total_reward += reward

        records.append(
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
        records
    ).to_csv(
        OUTPUT,
        index=False,
    )

    print("=" * 92)
    print(
        "MATCHED-WORKLOAD DYNAMIC GREEDY EVALUATION"
    )
    print("=" * 92)

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

    print("=" * 92)


if __name__ == "__main__":
    main()