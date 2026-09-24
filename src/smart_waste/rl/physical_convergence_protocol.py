from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import yaml


DEFAULT_PROTOCOL_PATH = Path(
    "configs/rl/"
    "dqn_physical_convergence.yaml"
)


class PhysicalConvergenceProtocolError(
    RuntimeError
):
    pass


def _canonical_json_bytes(
    value: object,
) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode(
        "utf-8"
    )


@dataclass(frozen=True)
class PhysicalConvergenceProtocol:
    schema_version: str
    protocol_version: str

    topologies: tuple[
        str,
        ...,
    ]

    dataset_manifest_sha256: str

    minimum_episodes: int
    maximum_episodes: int

    curriculum_period_episodes: int

    validation_interval_episodes: int

    training_window_episodes: int

    normalized_reward_slope_threshold: float
    normalized_loss_slope_threshold: float

    max_validation_fuel_relative_change: float
    max_validation_reward_relative_change: float

    patience_consecutive_checks: int

    master_agent_seed: int

    require_training_natural_completion: bool
    require_validation_natural_completion: bool

    validation_metric_stability_required: bool

    payload_sha256: str


def load_physical_convergence_protocol(
    path: Path = DEFAULT_PROTOCOL_PATH,
) -> PhysicalConvergenceProtocol:
    path = Path(
        path
    )

    payload = yaml.safe_load(
        path.read_text(
            encoding="utf-8"
        )
    )

    schema_version = payload[
        "schema_version"
    ]

    if schema_version not in {
        "dqn-physical-convergence-v1",
        "dqn-physical-convergence-v2",
        "dqn-physical-convergence-v3",
    }:
        raise PhysicalConvergenceProtocolError(
            "unexpected convergence protocol schema"
        )

    training = payload[
        "training"
    ]

    validation = payload[
        "validation"
    ]

    convergence = payload[
        "convergence"
    ]

    model_scope = payload[
        "model_scope"
    ]

    seeds = payload[
        "seeds"
    ]

    topologies = tuple(
        model_scope[
            "topologies"
        ]
    )

    if topologies != (
        "manhattan",
        "superblock",
        "hex",
        "radial_concentric",
    ):
        raise PhysicalConvergenceProtocolError(
            "unexpected frozen topology order"
        )

    minimum = int(
        training[
            "minimum_episodes"
        ]
    )

    maximum = int(
        training[
            "maximum_episodes"
        ]
    )

    period = int(
        training[
            "curriculum_period_episodes"
        ]
    )

    validation_interval = int(
        validation[
            "interval_episodes"
        ]
    )

    window = int(
        convergence[
            "training_window_episodes"
        ]
    )

    patience = int(
        convergence[
            "patience_consecutive_checks"
        ]
    )

    if minimum <= 0:
        raise PhysicalConvergenceProtocolError(
            "minimum episodes must be positive"
        )

    if maximum < minimum:
        raise PhysicalConvergenceProtocolError(
            "maximum episodes below minimum"
        )

    if period != 32:
        raise PhysicalConvergenceProtocolError(
            "frozen curriculum period must be 32"
        )

    if (
        validation_interval
        != period
    ):
        raise PhysicalConvergenceProtocolError(
            "validation must occur once per "
            "complete curriculum cycle"
        )

    if (
        window % period
        != 0
    ):
        raise PhysicalConvergenceProtocolError(
            "training window must contain an "
            "integer number of curriculum cycles"
        )

    if patience <= 0:
        raise PhysicalConvergenceProtocolError(
            "patience must be positive"
        )

    for name in (
        "normalized_reward_slope_threshold",
        "normalized_loss_slope_threshold",
    ):
        if float(
            convergence[
                name
            ]
        ) < 0.0:
            raise PhysicalConvergenceProtocolError(
                f"{name} cannot be negative"
            )

    if (
        schema_version
        == "dqn-physical-convergence-v1"
    ):
        fuel_change_threshold = float(
            convergence[
                "max_validation_fuel_relative_change"
            ]
        )

        reward_change_threshold = float(
            convergence[
                "max_validation_reward_relative_change"
            ]
        )

        validation_metric_stability_required = True

    else:
        fuel_change_threshold = float(
            convergence[
                "diagnostic_max_validation_fuel_relative_change"
            ]
        )

        reward_change_threshold = float(
            convergence[
                "diagnostic_max_validation_reward_relative_change"
            ]
        )

        validation_metric_stability_required = bool(
            convergence[
                "require_validation_metric_stability_for_convergence"
            ]
        )

    if fuel_change_threshold < 0.0:
        raise PhysicalConvergenceProtocolError(
            "validation fuel-change threshold "
            "cannot be negative"
        )

    if reward_change_threshold < 0.0:
        raise PhysicalConvergenceProtocolError(
            "validation reward-change threshold "
            "cannot be negative"
        )

    digest = hashlib.sha256(
        _canonical_json_bytes(
            payload
        )
    ).hexdigest()

    return PhysicalConvergenceProtocol(
        schema_version=(
            schema_version
        ),
        protocol_version=(
            str(
                payload[
                    "protocol_version"
                ]
            )
        ),
        topologies=topologies,
        dataset_manifest_sha256=(
            payload[
                "dataset"
            ][
                "manifest_sha256"
            ]
        ),
        minimum_episodes=minimum,
        maximum_episodes=maximum,
        curriculum_period_episodes=(
            period
        ),
        validation_interval_episodes=(
            validation_interval
        ),
        training_window_episodes=(
            window
        ),
        normalized_reward_slope_threshold=float(
            convergence[
                "normalized_reward_slope_threshold"
            ]
        ),
        normalized_loss_slope_threshold=float(
            convergence[
                "normalized_loss_slope_threshold"
            ]
        ),
        max_validation_fuel_relative_change=(
            fuel_change_threshold
        ),
        max_validation_reward_relative_change=(
            reward_change_threshold
        ),
        patience_consecutive_checks=(
            patience
        ),
        master_agent_seed=int(
            seeds[
                "master_agent_seed"
            ]
        ),
        require_training_natural_completion=bool(
            convergence[
                "require_all_training_episodes_natural_completion"
            ]
        ),
        require_validation_natural_completion=bool(
            convergence[
                "require_all_validation_runs_natural_completion"
            ]
        ),
        validation_metric_stability_required=(
            validation_metric_stability_required
        ),
        payload_sha256=digest,
    )
