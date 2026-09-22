from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from smart_waste.core.city import City
from smart_waste.core.fill_dynamics import (
    generate_fill_rates,
)
from smart_waste.rl.dqn import (
    DQNAgent,
    DQNConfig,
)
from smart_waste.rl.environment import (
    RewardWeights,
    SmartWasteRoutingEnv,
)
from smart_waste.rl.training import (
    train_dqn,
)


DEFAULT_CONFIG = Path(
    "configs/dqn_primary_reconstruction.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the primary 1,000-bin DQN "
            "reconstruction experiment."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help=(
            "Optional episode override. "
            "Use only for integration testing."
        ),
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run a short integration check without "
            "overwriting primary experiment outputs."
        ),
    )

    return parser.parse_args()


def load_yaml(
    path: Path,
) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def calculate_full_bin_mass_tonnes(
    *,
    bin_volume_m3: float,
    waste_density_kg_per_m3: float,
) -> float:
    return (
        bin_volume_m3
        * waste_density_kg_per_m3
        / 1000.0
    )


def build_environment(
    *,
    base_cfg: dict,
    protocol_cfg: dict,
) -> SmartWasteRoutingEnv:

    city_seed = int(
        protocol_cfg["protocol"][
            "city_master_seed"
        ]
    )

    num_bins = int(
        protocol_cfg["environment"][
            "num_bins"
        ]
    )

    city = City.generate(
        area_km2=float(
            base_cfg["city"]["area_km2"]
        ),
        num_bins=num_bins,
        master_seed=city_seed,
        minimum_fill_percent=float(
            base_cfg["bins"]["initial_fill"][
                "minimum_percent"
            ]
        ),
        maximum_fill_percent=float(
            base_cfg["bins"]["initial_fill"][
                "maximum_percent"
            ]
        ),
    )

    fill_cfg = protocol_cfg[
        "fill_dynamics"
    ]

    fill_rates = generate_fill_rates(
        num_bins=num_bins,
        master_seed=city_seed,
        minimum_rate_percent_per_hour=float(
            fill_cfg[
                "minimum_rate_percent_per_hour"
            ]
        ),
        maximum_rate_percent_per_hour=float(
            fill_cfg[
                "maximum_rate_percent_per_hour"
            ]
        ),
        replicate_id=0,
    )

    # --------------------------------------------------
    # Primary nominal telemetry
    # --------------------------------------------------

    verified_mask = np.ones(
        num_bins,
        dtype=bool,
    )

    # --------------------------------------------------
    # Deterministic representative hazard
    #
    # The DQN environment receives already-computed,
    # verified hazard severity H_t. It does not attempt
    # to reconstruct the missing temperature-humidity
    # predictor here.
    # --------------------------------------------------

    hazard_cfg = protocol_cfg[
        "hazard"
    ]

    hazard_severity = np.zeros(
        num_bins,
        dtype=np.float64,
    )

    predicted_count = int(
        hazard_cfg[
            "predicted_combustion_risk_count"
        ]
    )

    emergency_count = int(
        hazard_cfg[
            "emergency_fire_count"
        ]
    )

    moisture_count = int(
        hazard_cfg[
            "excess_moisture_count"
        ]
    )

    total_hazards = (
        predicted_count
        + emergency_count
        + moisture_count
    )

    if total_hazards > num_bins:
        raise ValueError(
            "hazard counts exceed number of bins"
        )

    hazard_rng = np.random.default_rng(
        int(
            hazard_cfg[
                "selection_seed"
            ]
        )
    )

    hazard_ids = hazard_rng.choice(
        num_bins,
        size=total_hazards,
        replace=False,
    )

    cursor = 0

    predicted_ids = hazard_ids[
        cursor : cursor + predicted_count
    ]

    cursor += predicted_count

    emergency_ids = hazard_ids[
        cursor : cursor + emergency_count
    ]

    cursor += emergency_count

    moisture_ids = hazard_ids[
        cursor : cursor + moisture_count
    ]

    predicted_severity = float(
        hazard_cfg[
            "predicted_combustion_normalized_severity"
        ]
    )

    hazard_severity[
        predicted_ids
    ] = predicted_severity

    # Manuscript severity encoding:
    # emergency fire is the maximum normalized H_t.
    hazard_severity[
        emergency_ids
    ] = 1.0

    # Moisture = raw severity 1 / maximum raw severity 3.
    hazard_severity[
        moisture_ids
    ] = 1.0 / 3.0

    full_bin_mass_tonnes = (
        calculate_full_bin_mass_tonnes(
            bin_volume_m3=float(
                base_cfg["waste"][
                    "bin_volume_m3"
                ]
            ),
            waste_density_kg_per_m3=float(
                base_cfg["waste"][
                    "density"
                ][
                    "nominal_kg_per_m3"
                ]
            ),
        )
    )

    env_cfg = protocol_cfg[
        "environment"
    ]

    temporal_cfg = protocol_cfg[
        "temporal"
    ]

    reward_cfg = base_cfg[
        "rl"
    ][
        "reward"
    ]

    environment = SmartWasteRoutingEnv(
        bin_xy=city.coordinate_array(),
        depot_xy=np.asarray(
            city.depot_location,
            dtype=np.float64,
        ),
        initial_fill_percent=(
            city.fill_array()
        ),
        fill_rate_percent_per_hour=(
            fill_rates
        ),
        hazard_severity=(
            hazard_severity
        ),
        verified_mask=verified_mask,
        prediction_horizon_hours=float(
            temporal_cfg[
                "prediction_horizon_hours"
            ]
        ),
        decision_interval_hours=float(
            temporal_cfg[
                "decision_interval_hours"
            ]
        ),
        num_trucks=int(
            env_cfg[
                "num_trucks"
            ]
        ),
        truck_capacity_tonnes=float(
            env_cfg[
                "truck_capacity_tonnes"
            ]
        ),
        fuel_efficiency_km_per_litre=float(
            env_cfg[
                "fuel_efficiency_km_per_litre"
            ]
        ),
        full_bin_mass_tonnes=(
            full_bin_mass_tonnes
        ),
        near_full_threshold_percent=float(
            env_cfg[
                "near_full_threshold_percent"
            ]
        ),
        sla_limit_hours=float(
            env_cfg[
                "sla_limit_hours"
            ]
        ),
        max_steps=int(
            env_cfg[
                "max_steps"
            ]
        ),
        reward_weights=RewardWeights(
            collection=float(
                reward_cfg[
                    "collection_weight"
                ]
            ),
            distance=float(
                reward_cfg[
                    "distance_weight"
                ]
            ),
            hazard_delay=float(
                reward_cfg[
                    "hazard_delay_weight"
                ]
            ),
            sla_violation=float(
                reward_cfg[
                    "sla_violation_weight"
                ]
            ),
        ),
    )

    return environment


def build_agent(
    *,
    env: SmartWasteRoutingEnv,
    protocol_cfg: dict,
) -> DQNAgent:

    cfg = protocol_cfg["dqn"]

    dqn_config = DQNConfig(
        hidden_layers=tuple(
            int(value)
            for value
            in cfg[
                "hidden_layers"
            ]
        ),
        gamma=float(
            cfg["gamma"]
        ),
        learning_rate=float(
            cfg[
                "learning_rate"
            ]
        ),
        replay_capacity=int(
            cfg[
                "replay_capacity"
            ]
        ),
        batch_size=int(
            cfg[
                "batch_size"
            ]
        ),
        epsilon_start=float(
            cfg[
                "epsilon_start"
            ]
        ),
        epsilon_end=float(
            cfg[
                "epsilon_end"
            ]
        ),
        epsilon_decay_episodes=int(
            cfg[
                "epsilon_decay_episodes"
            ]
        ),
        replay_warmup_transitions=int(
            cfg[
                "replay_warmup_transitions"
            ]
        ),
        target_update_steps=int(
            cfg[
                "target_update_steps"
            ]
        ),
    )

    agent_seed = int(
        protocol_cfg[
            "protocol"
        ][
            "agent_seed"
        ]
    )

    return DQNAgent(
        input_dim=(
            env.observation_space.shape[0]
        ),
        action_dim=(
            env.action_space.n
        ),
        seed=agent_seed,
        config=dqn_config,
        device="cpu",
    )


def save_checkpoint(
    *,
    path: Path,
    agent: DQNAgent,
    protocol_cfg: dict,
    episodes_completed: int,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "episodes_completed": (
                episodes_completed
            ),
            "input_dim": (
                agent.input_dim
            ),
            "action_dim": (
                agent.action_dim
            ),
            "optimization_steps": (
                agent.optimization_steps
            ),
            "online_network_state_dict": (
                agent.online_network.state_dict()
            ),
            "target_network_state_dict": (
                agent.target_network.state_dict()
            ),
            "optimizer_state_dict": (
                agent.optimizer.state_dict()
            ),
            "protocol": protocol_cfg,
        },
        path,
    )


def print_training_summary(
    dataframe: pd.DataFrame,
    *,
    elapsed_seconds: float,
) -> None:

    print()
    print("=" * 92)
    print(
        "PRIMARY DQN TRAINING SUMMARY"
    )
    print("=" * 92)

    print(
        f"Episodes completed : "
        f"{len(dataframe)}"
    )

    print(
        f"Elapsed time       : "
        f"{elapsed_seconds:.2f} s"
    )

    print(
        f"Mean steps         : "
        f"{dataframe['steps'].mean():.2f}"
    )

    print(
        f"Mean reward        : "
        f"{dataframe['total_reward'].mean():.6f}"
    )

    print(
        f"Mean fuel          : "
        f"{dataframe['fuel_litres'].mean():.6f} L"
    )

    print(
        f"Mean distance      : "
        f"{dataframe['distance_km'].mean():.6f} km"
    )

    losses = dataframe[
        "mean_loss"
    ].dropna()

    if not losses.empty:
        print(
            f"Final mean loss    : "
            f"{losses.iloc[-1]:.6f}"
        )

    if len(dataframe) >= 10:

        first_10 = dataframe.iloc[
            :10
        ]

        last_10 = dataframe.iloc[
            -10:
        ]

        print()
        print("FIRST 10 EPISODES")

        print(
            f"  reward mean      : "
            f"{first_10['total_reward'].mean():.6f}"
        )

        print(
            f"  fuel mean        : "
            f"{first_10['fuel_litres'].mean():.6f} L"
        )

        print()
        print("LAST 10 EPISODES")

        print(
            f"  reward mean      : "
            f"{last_10['total_reward'].mean():.6f}"
        )

        print(
            f"  fuel mean        : "
            f"{last_10['fuel_litres'].mean():.6f} L"
        )

    print("=" * 92)


def main() -> None:

    args = parse_args()

    base_cfg = load_yaml(
        Path("configs/base.yaml")
    )

    protocol_cfg = load_yaml(
        args.config
    )

    configured_episodes = int(
        protocol_cfg[
            "protocol"
        ][
            "training_episodes"
        ]
    )

    if args.smoke:
        episodes = 2
    elif args.episodes is not None:
        if args.episodes <= 0:
            raise ValueError(
                "--episodes must be positive"
            )

        episodes = args.episodes
    else:
        episodes = (
            configured_episodes
        )

    # --------------------------------------------------
    # Reproducibility controls
    # --------------------------------------------------

    agent_seed = int(
        protocol_cfg[
            "protocol"
        ][
            "agent_seed"
        ]
    )

    np.random.seed(
        agent_seed
    )

    torch.manual_seed(
        agent_seed
    )

    torch.use_deterministic_algorithms(
        True
    )

    # CPU-only experiment.
    # Fixing the thread count makes the execution
    # environment explicit for reproducibility.
    torch.set_num_threads(4)

    env = build_environment(
        base_cfg=base_cfg,
        protocol_cfg=protocol_cfg,
    )

    agent = build_agent(
        env=env,
        protocol_cfg=protocol_cfg,
    )

    initial_observation, initial_info = (
        env.reset(
            seed=agent_seed
        )
    )

    parameter_count = sum(
        parameter.numel()
        for parameter
        in agent.online_network.parameters()
    )

    print("=" * 92)
    print(
        "PRIMARY DQN RECONSTRUCTION"
    )
    print("=" * 92)

    print(
        "Episodes            :",
        episodes,
    )

    print(
        "Observation dim     :",
        env.observation_space.shape[0],
    )

    print(
        "Action dim          :",
        env.action_space.n,
    )

    print(
        "Initial candidates  :",
        initial_info[
            "candidate_count"
        ],
    )

    print(
        "Network parameters  :",
        f"{parameter_count:,}",
    )

    print(
        "Torch threads       :",
        torch.get_num_threads(),
    )

    print(
        "CUDA available      :",
        torch.cuda.is_available(),
    )

    print(
        "Prediction horizon  :",
        protocol_cfg[
            "temporal"
        ][
            "prediction_horizon_hours"
        ],
        "h",
    )

    print(
        "Decision interval   :",
        protocol_cfg[
            "temporal"
        ][
            "decision_interval_hours"
        ],
        "h",
    )

    print(
        "Fill-rate max       :",
        protocol_cfg[
            "fill_dynamics"
        ][
            "maximum_rate_percent_per_hour"
        ],
        "%/h",
    )

    print("-" * 92)

    start_time = time.perf_counter()

    result = train_dqn(
        env=env,
        agent=agent,
        episodes=episodes,
        master_seed=agent_seed,
    )

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    dataframe = pd.DataFrame(
        result.to_records()
    )

    # --------------------------------------------------
    # Smoke mode: do not overwrite primary outputs.
    # --------------------------------------------------

    if args.smoke:

        output_csv = Path(
            "results/raw/"
            "dqn_primary_integration_smoke.csv"
        )

        checkpoint_path = Path(
            "results/models/"
            "dqn_primary_integration_smoke.pt"
        )

        protocol_snapshot = Path(
            "results/processed/"
            "dqn_primary_integration_smoke_protocol.yaml"
        )

    else:

        outputs_cfg = protocol_cfg[
            "outputs"
        ]

        output_csv = Path(
            outputs_cfg[
                "episode_metrics_csv"
            ]
        )

        checkpoint_path = Path(
            outputs_cfg[
                "model_checkpoint"
            ]
        )

        protocol_snapshot = Path(
            outputs_cfg[
                "protocol_snapshot"
            ]
        )

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        output_csv,
        index=False,
    )

    save_checkpoint(
        path=checkpoint_path,
        agent=agent,
        protocol_cfg=protocol_cfg,
        episodes_completed=episodes,
    )

    protocol_snapshot.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copyfile(
        args.config,
        protocol_snapshot,
    )

    print_training_summary(
        dataframe,
        elapsed_seconds=elapsed_seconds,
    )

    print()
    print(
        "Episode metrics:",
        output_csv,
    )

    print(
        "Checkpoint     :",
        checkpoint_path,
    )

    print(
        "Protocol       :",
        protocol_snapshot,
    )

    print()

    if args.smoke:
        print(
            "INTEGRATION SMOKE RUN PASSED"
        )
    else:
        print(
            "PRIMARY DQN TRAINING COMPLETED"
        )


if __name__ == "__main__":
    main()