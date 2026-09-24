from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Final

import numpy as np
import torch

from smart_waste.experiments.dqn_physical_training_cases import (
    DEFAULT_OUTPUT_PATH as PHYSICAL_CASE_MANIFEST_PATH,
    cases_for_topology,
)
from smart_waste.experiments.hazard_overlay import (
    CHALLENGE_ORDER,
)
from smart_waste.rl.dqn import (
    DQNAgent,
    DQNConfig,
)
from smart_waste.rl.physical_convergence_protocol import (
    load_physical_convergence_protocol,
)
from smart_waste.rl.physical_episode import (
    PhysicalDQNEpisodeResult,
    run_physical_dqn_episode,
)


TRAINING_TRACE_SCHEMA: Final = (
    "physical-dqn-convergence-training-v1"
)

DEFAULT_OUTPUT_ROOT = Path(
    "results/rl/dqn_physical"
)


class PhysicalDQNConvergenceTrainingError(
    RuntimeError
):
    pass


@dataclass(frozen=True)
class TrainingEpisodeRecord:
    episode_index: int

    training_slot_id: int
    challenge_id: str

    epsilon: float

    total_reward: float
    mean_loss: float | None
    final_loss: float | None

    optimization_steps: int

    completed_services: int
    transition_records: int
    replay_transitions: int

    reward_distance_km: float
    fleet_distance_km: float
    fuel_litres: float
    completion_time_hours: float

    natural_completion: bool


@dataclass(frozen=True)
class ValidationRunRecord:
    checkpoint_episode: int

    validation_slot_id: int
    challenge_id: str

    total_reward: float

    reward_distance_km: float
    fleet_distance_km: float
    fuel_litres: float
    completion_time_hours: float

    completed_services: int
    natural_completion: bool


@dataclass(frozen=True)
class ConvergenceCheckRecord:
    checkpoint_episode: int

    normalized_reward_slope: float
    normalized_loss_slope: float

    max_validation_fuel_relative_change: (
        float | None
    )

    max_validation_reward_relative_change: (
        float | None
    )

    reward_stable: bool
    loss_stable: bool
    validation_fuel_stable: bool
    validation_reward_stable: bool

    training_natural_completion_ok: bool
    validation_natural_completion_ok: bool

    criteria_met: bool
    consecutive_stable_checks: int


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


def _sha256_file(
    path: Path,
) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def _source_commit() -> str:
    return subprocess.check_output(
        [
            "git",
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()


def _require_clean_git_tree() -> None:
    output = subprocess.check_output(
        [
            "git",
            "status",
            "--porcelain",
        ],
        text=True,
    )

    if output.strip():
        raise PhysicalDQNConvergenceTrainingError(
            "Real DQN training requires a clean Git tree"
        )


def _derive_agent_seed(
    *,
    master_seed: int,
    topology_id: str,
) -> int:
    payload = (
        "physical-dqn-agent|"
        f"master={master_seed}|"
        f"topology={topology_id}"
    ).encode(
        "utf-8"
    )

    digest = hashlib.sha256(
        payload
    ).digest()

    return (
        int.from_bytes(
            digest[:8],
            byteorder="big",
            signed=False,
        )
        & 0x7FFFFFFF
    )


def _linear_slope(
    values: list[float],
) -> float:
    if len(
        values
    ) < 2:
        return float(
            "nan"
        )

    y = np.asarray(
        values,
        dtype=np.float64,
    )

    x = np.arange(
        len(
            y
        ),
        dtype=np.float64,
    )

    return float(
        np.polyfit(
            x,
            y,
            1,
        )[
            0
        ]
    )


def _normalized_slope(
    values: list[float],
) -> float:
    slope = _linear_slope(
        values
    )

    if not np.isfinite(
        slope
    ):
        return float(
            "nan"
        )

    scale = max(
        float(
            np.mean(
                np.abs(
                    np.asarray(
                        values,
                        dtype=np.float64,
                    )
                )
            )
        ),
        1.0e-12,
    )

    return float(
        slope
        / scale
    )


def _relative_change(
    current: float,
    previous: float,
) -> float:
    denominator = max(
        abs(
            previous
        ),
        1.0e-12,
    )

    return float(
        abs(
            current
            - previous
        )
        / denominator
    )


def _curriculum_condition(
    episode_index: int,
):
    condition_index = (
        episode_index
        % 32
    )

    slot_id = (
        condition_index
        // 4
    )

    challenge_id = (
        CHALLENGE_ORDER[
            condition_index
            % 4
        ]
    )

    return (
        slot_id,
        challenge_id,
    )


def _episode_to_training_record(
    *,
    episode_index: int,
    slot_id: int,
    challenge_id: str,
    result: PhysicalDQNEpisodeResult,
    optimization_steps: int,
) -> TrainingEpisodeRecord:
    losses = tuple(
        float(value)
        for value
        in result.optimization_losses
    )

    if losses:
        mean_loss = float(
            np.mean(
                losses
            )
        )

        final_loss = float(
            losses[
                -1
            ]
        )

    else:
        mean_loss = None
        final_loss = None

    return TrainingEpisodeRecord(
        episode_index=(
            episode_index
        ),
        training_slot_id=(
            slot_id
        ),
        challenge_id=(
            challenge_id
        ),
        epsilon=float(
            result.epsilon
        ),
        total_reward=float(
            result.total_reward
        ),
        mean_loss=mean_loss,
        final_loss=final_loss,
        optimization_steps=int(
            optimization_steps
        ),
        completed_services=int(
            result.completed_services
        ),
        transition_records=int(
            result.transition_records
        ),
        replay_transitions=int(
            result.replay_transitions
        ),
        reward_distance_km=float(
            result.reward_distance_km
        ),
        fleet_distance_km=float(
            result.fleet_distance_km
        ),
        fuel_litres=float(
            result.fuel_litres
        ),
        completion_time_hours=float(
            result.completion_time_hours
        ),
        natural_completion=bool(
            result.natural_completion
        ),
    )


def _run_validation(
    *,
    agent: DQNAgent,
    topology_id: str,
    checkpoint_episode: int,
) -> list[
    ValidationRunRecord
]:
    cases = (
        cases_for_topology(
            topology_id,
            split="validation",
        )
    )

    replay_before = len(
        agent.replay_buffer
    )

    optimization_before = (
        agent.optimization_steps
    )

    records = []

    for case in cases:
        for challenge_id in (
            CHALLENGE_ORDER
        ):
            result = (
                run_physical_dqn_episode(
                    snapshot=(
                        case.snapshot
                    ),
                    scenario_id=(
                        case.case_id
                    ),
                    scenario_sha256=(
                        case.case_sha256
                    ),
                    agent=agent,
                    episode_index=(
                        checkpoint_episode
                    ),
                    epsilon=0.0,
                    hazard_severity_by_bin=(
                        case
                        .hazard_severity_by_bin(
                            challenge_id
                        )
                    ),
                    optimize_after_transition=False,
                    record_replay=False,
                )
            )

            records.append(
                ValidationRunRecord(
                    checkpoint_episode=(
                        checkpoint_episode
                    ),
                    validation_slot_id=(
                        case.slot_id
                    ),
                    challenge_id=(
                        challenge_id
                    ),
                    total_reward=float(
                        result.total_reward
                    ),
                    reward_distance_km=float(
                        result.reward_distance_km
                    ),
                    fleet_distance_km=float(
                        result.fleet_distance_km
                    ),
                    fuel_litres=float(
                        result.fuel_litres
                    ),
                    completion_time_hours=float(
                        result.completion_time_hours
                    ),
                    completed_services=int(
                        result.completed_services
                    ),
                    natural_completion=bool(
                        result.natural_completion
                    ),
                )
            )

    if len(
        agent.replay_buffer
    ) != replay_before:
        raise PhysicalDQNConvergenceTrainingError(
            "validation mutated ReplayBuffer"
        )

    if (
        agent.optimization_steps
        != optimization_before
    ):
        raise PhysicalDQNConvergenceTrainingError(
            "validation mutated optimizer state"
        )

    return records


def _matched_validation_change(
    current: list[
        ValidationRunRecord
    ],
    previous: list[
        ValidationRunRecord
    ],
    *,
    field_name: str,
) -> float:
    previous_by_key = {
        (
            row.validation_slot_id,
            row.challenge_id,
        ): row
        for row in previous
    }

    changes = []

    for row in current:
        key = (
            row.validation_slot_id,
            row.challenge_id,
        )

        if key not in previous_by_key:
            raise PhysicalDQNConvergenceTrainingError(
                "validation checkpoints are not matched"
            )

        previous_row = (
            previous_by_key[
                key
            ]
        )

        changes.append(
            _relative_change(
                float(
                    getattr(
                        row,
                        field_name,
                    )
                ),
                float(
                    getattr(
                        previous_row,
                        field_name,
                    )
                ),
            )
        )

    return float(
        max(
            changes
        )
    )


def _load_physical_case_manifest_sha() -> str:
    payload = json.loads(
        Path(
            PHYSICAL_CASE_MANIFEST_PATH
        ).read_text(
            encoding="utf-8"
        )
    )

    return str(
        payload[
            "manifest_sha256"
        ]
    )


def _save_checkpoint(
    *,
    path: Path,
    agent: DQNAgent,
    metadata: dict,
) -> dict:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "metadata": metadata,
        "input_dim": (
            agent.input_dim
        ),
        "action_dim": (
            agent.action_dim
        ),
        "config": asdict(
            agent.config
        ),
        "online_network_state": (
            agent.online_network.state_dict()
        ),
        "target_network_state": (
            agent.target_network.state_dict()
        ),
        "optimizer_state": (
            agent.optimizer.state_dict()
        ),
        "optimization_steps": (
            agent.optimization_steps
        ),
        "numpy_rng_state": (
            agent.rng.bit_generator.state
        ),
    }

    torch.save(
        payload,
        path,
    )

    return {
        "checkpoint_file": (
            path.as_posix()
        ),
        "checkpoint_file_sha256": (
            _sha256_file(
                path
            )
        ),
    }


def train_topology_until_convergence(
    *,
    topology_id: str,
    output_root: Path = (
        DEFAULT_OUTPUT_ROOT
    ),
    require_clean_git_tree: bool = True,
) -> dict:
    protocol = (
        load_physical_convergence_protocol()
    )

    if (
        topology_id
        not in protocol.topologies
    ):
        raise ValueError(
            f"unknown topology_id={topology_id}"
        )

    if require_clean_git_tree:
        _require_clean_git_tree()

    source_commit = (
        _source_commit()
    )

    physical_case_manifest_sha = (
        _load_physical_case_manifest_sha()
    )

    training_cases = {
        case.slot_id: case
        for case in cases_for_topology(
            topology_id,
            split="training",
        )
    }

    if tuple(
        sorted(
            training_cases
        )
    ) != tuple(
        range(
            8
        )
    ):
        raise PhysicalDQNConvergenceTrainingError(
            "expected training slot IDs 0..7"
        )

    agent_seed = (
        _derive_agent_seed(
            master_seed=(
                protocol.master_agent_seed
            ),
            topology_id=(
                topology_id
            ),
        )
    )

    agent = DQNAgent(
        input_dim=3002,
        action_dim=1000,
        seed=agent_seed,
        config=DQNConfig(),
    )

    training_records: list[
        TrainingEpisodeRecord
    ] = []

    validation_checkpoints: list[
        list[
            ValidationRunRecord
        ]
    ] = []

    convergence_checks: list[
        ConvergenceCheckRecord
    ] = []

    consecutive_stable = 0
    converged = False

    stop_reason = (
        "maximum_episodes_reached"
    )

    for episode_index in range(
        protocol.maximum_episodes
    ):
        (
            slot_id,
            challenge_id,
        ) = _curriculum_condition(
            episode_index
        )

        case = training_cases[
            slot_id
        ]

        optimization_before = (
            agent.optimization_steps
        )

        result = (
            run_physical_dqn_episode(
                snapshot=(
                    case.snapshot
                ),
                scenario_id=(
                    case.case_id
                ),
                scenario_sha256=(
                    case.case_sha256
                ),
                agent=agent,
                episode_index=(
                    episode_index
                ),
                epsilon=None,
                hazard_severity_by_bin=(
                    case
                    .hazard_severity_by_bin(
                        challenge_id
                    )
                ),
                optimize_after_transition=True,
                record_replay=True,
            )
        )

        optimization_delta = (
            agent.optimization_steps
            - optimization_before
        )

        training_records.append(
            _episode_to_training_record(
                episode_index=(
                    episode_index
                ),
                slot_id=slot_id,
                challenge_id=(
                    challenge_id
                ),
                result=result,
                optimization_steps=(
                    optimization_delta
                ),
            )
        )

        episodes_completed = (
            episode_index
            + 1
        )

        if (
            episodes_completed
            % protocol.validation_interval_episodes
            != 0
        ):
            continue

        validation = (
            _run_validation(
                agent=agent,
                topology_id=(
                    topology_id
                ),
                checkpoint_episode=(
                    episodes_completed
                ),
            )
        )

        validation_checkpoints.append(
            validation
        )

        if (
            episodes_completed
            < protocol.minimum_episodes
        ):
            continue

        if (
            len(
                training_records
            )
            < protocol.training_window_episodes
        ):
            continue

        training_window = (
            training_records[
                -protocol.training_window_episodes:
            ]
        )

        reward_values = [
            row.total_reward
            for row
            in training_window
        ]

        loss_values = [
            float(
                row.mean_loss
            )
            for row
            in training_window
            if (
                row.mean_loss
                is not None
            )
        ]

        reward_slope = (
            _normalized_slope(
                reward_values
            )
        )

        loss_slope = (
            _normalized_slope(
                loss_values
            )
        )

        reward_stable = bool(
            np.isfinite(
                reward_slope
            )
            and abs(
                reward_slope
            )
            <= (
                protocol
                .normalized_reward_slope_threshold
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
                protocol
                .normalized_loss_slope_threshold
            )
        )

        training_natural_ok = (
            all(
                row.natural_completion
                for row
                in training_window
            )
            if (
                protocol
                .require_training_natural_completion
            )
            else True
        )

        validation_natural_ok = (
            all(
                row.natural_completion
                for row
                in validation
            )
            if (
                protocol
                .require_validation_natural_completion
            )
            else True
        )

        if (
            len(
                validation_checkpoints
            )
            >= 2
        ):
            previous_validation = (
                validation_checkpoints[
                    -2
                ]
            )

            fuel_change = (
                _matched_validation_change(
                    validation,
                    previous_validation,
                    field_name=(
                        "fuel_litres"
                    ),
                )
            )

            reward_change = (
                _matched_validation_change(
                    validation,
                    previous_validation,
                    field_name=(
                        "total_reward"
                    ),
                )
            )

            fuel_stable = bool(
                fuel_change
                <= (
                    protocol
                    .max_validation_fuel_relative_change
                )
            )

            validation_reward_stable = bool(
                reward_change
                <= (
                    protocol
                    .max_validation_reward_relative_change
                )
            )

        else:
            fuel_change = None
            reward_change = None

            fuel_stable = False
            validation_reward_stable = False

        criteria_met = bool(
            reward_stable
            and loss_stable
            and fuel_stable
            and validation_reward_stable
            and training_natural_ok
            and validation_natural_ok
        )

        if criteria_met:
            consecutive_stable += 1
        else:
            consecutive_stable = 0

        convergence_checks.append(
            ConvergenceCheckRecord(
                checkpoint_episode=(
                    episodes_completed
                ),
                normalized_reward_slope=float(
                    reward_slope
                ),
                normalized_loss_slope=float(
                    loss_slope
                ),
                max_validation_fuel_relative_change=(
                    fuel_change
                ),
                max_validation_reward_relative_change=(
                    reward_change
                ),
                reward_stable=(
                    reward_stable
                ),
                loss_stable=(
                    loss_stable
                ),
                validation_fuel_stable=(
                    fuel_stable
                ),
                validation_reward_stable=(
                    validation_reward_stable
                ),
                training_natural_completion_ok=(
                    training_natural_ok
                ),
                validation_natural_completion_ok=(
                    validation_natural_ok
                ),
                criteria_met=(
                    criteria_met
                ),
                consecutive_stable_checks=(
                    consecutive_stable
                ),
            )
        )

        print(
            f"[{topology_id}] "
            f"episode={episodes_completed} "
            f"reward_slope={reward_slope:.6g} "
            f"loss_slope={loss_slope:.6g} "
            f"fuel_change={fuel_change} "
            f"validation_reward_change={reward_change} "
            f"stable={consecutive_stable}/"
            f"{protocol.patience_consecutive_checks}",
            flush=True,
        )

        if (
            consecutive_stable
            >= (
                protocol
                .patience_consecutive_checks
            )
        ):
            converged = True

            stop_reason = (
                "convergence_criteria_satisfied"
            )

            break

    episodes_completed = len(
        training_records
    )

    output_dir = (
        Path(
            output_root
        )
        / topology_id
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = {
        "schema_version": (
            TRAINING_TRACE_SCHEMA
        ),
        "topology_id": (
            topology_id
        ),
        "source_commit": (
            source_commit
        ),
        "agent_seed": (
            agent_seed
        ),
        "protocol_sha256": (
            protocol.payload_sha256
        ),
        "dataset_manifest_sha256": (
            protocol
            .dataset_manifest_sha256
        ),
        "physical_case_manifest_sha256": (
            physical_case_manifest_sha
        ),
        "episodes_completed": (
            episodes_completed
        ),
        "converged": (
            converged
        ),
        "stop_reason": (
            stop_reason
        ),
        "final_optimization_steps": (
            agent.optimization_steps
        ),
        "final_replay_size": len(
            agent.replay_buffer
        ),
        "dqn_config": asdict(
            agent.config
        ),
    }

    checkpoint_info = (
        _save_checkpoint(
            path=(
                output_dir
                / "model_checkpoint.pt"
            ),
            agent=agent,
            metadata=metadata,
        )
    )

    training_payload = [
        asdict(
            row
        )
        for row
        in training_records
    ]

    validation_payload = [
        asdict(
            row
        )
        for checkpoint
        in validation_checkpoints
        for row
        in checkpoint
    ]

    convergence_payload = [
        asdict(
            row
        )
        for row
        in convergence_checks
    ]

    (
        output_dir
        / "training_trace.json"
    ).write_text(
        json.dumps(
            training_payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    (
        output_dir
        / "validation_trace.json"
    ).write_text(
        json.dumps(
            validation_payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    (
        output_dir
        / "convergence_checks.json"
    ).write_text(
        json.dumps(
            convergence_payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = {
        **metadata,
        **checkpoint_info,
        "training_trace_sha256": (
            _sha256_file(
                output_dir
                / "training_trace.json"
            )
        ),
        "validation_trace_sha256": (
            _sha256_file(
                output_dir
                / "validation_trace.json"
            )
        ),
        "convergence_checks_sha256": (
            _sha256_file(
                output_dir
                / "convergence_checks.json"
            )
        ),
    }

    (
        output_dir
        / "training_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "=== PHYSICAL DQN TRAINING COMPLETE ==="
    )

    for key in (
        "topology_id",
        "source_commit",
        "agent_seed",
        "episodes_completed",
        "converged",
        "stop_reason",
        "final_optimization_steps",
        "final_replay_size",
        "protocol_sha256",
        "dataset_manifest_sha256",
        "physical_case_manifest_sha256",
        "checkpoint_file_sha256",
    ):
        print(
            f"{key}:",
            summary[
                key
            ],
        )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--topology",
        required=True,
        choices=(
            "manhattan",
            "superblock",
            "hex",
            "radial_concentric",
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            DEFAULT_OUTPUT_ROOT
        ),
    )

    parser.add_argument(
        "--allow-dirty-tree",
        action="store_true",
        help=(
            "Development/testing only. "
            "Do not use for frozen real training."
        ),
    )

    args = parser.parse_args()

    train_topology_until_convergence(
        topology_id=(
            args.topology
        ),
        output_root=(
            args.output_root
        ),
        require_clean_git_tree=(
            not args.allow_dirty_tree
        ),
    )


if __name__ == "__main__":
    main()
