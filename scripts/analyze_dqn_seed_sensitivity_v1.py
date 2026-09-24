from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


PLAN_PATH = Path(
    "configs/rl/dqn_seed_sensitivity_v1.yaml"
)

EXPECTED_PREREG_COMMIT = (
    "41cae5cfbc66ed88d04de1f8d85aff1b699e3c96"
)

EXPECTED_EXECUTION_COMMIT = (
    "b3dc3be2526f617115ed9eef1d4ea0395b5607da"
)

EXPECTED_BASE_PROTOCOL_SHA = (
    "878446fe12985e7f7fa9cc41c39e40efaa69bc8825c9ac102be24b205aa0413a"
)

EXPECTED_DATASET_SHA = (
    "18bbf059305886970d3bcfc363a2140a4fe32d837901bd3ccd67af8d24384133"
)

EXPECTED_CASE_SHA = (
    "9b56981be163ed8651e40fdc9e3384c9e641e8d6b9719637382b491c66a4d5c7"
)

OUTPUT_DIR = Path(
    "analysis/rl/dqn_seed_sensitivity_v1"
)

TOPOLOGIES = (
    "manhattan",
    "superblock",
    "hex",
    "radial_concentric",
)

TRACE_FILES = (
    ("training_trace.json", "training_trace_sha256"),
    ("validation_trace.json", "validation_trace_sha256"),
    ("convergence_checks.json", "convergence_checks_sha256"),
)

WINDOW = 64
CURRICULUM_PERIOD = 32
VALIDATION_INTERVAL = 32
REWARD_THRESHOLD = 0.0025
LOSS_THRESHOLD = 0.005
ADMINISTRATIVE_LIMIT = 1280


def sha256_file(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def normalized_slope(
    values,
    x=None,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if x is None:
        x = np.arange(
            len(values),
            dtype=np.float64,
        )
    else:
        x = np.asarray(
            x,
            dtype=np.float64,
        )

    if len(values) < 2:
        return 0.0

    raw = float(
        np.polyfit(
            x,
            values,
            1,
        )[0]
    )

    scale = max(
        float(
            np.mean(
                np.abs(values)
            )
        ),
        1.0e-12,
    )

    return raw / scale


def raw_and_normalized_slope(
    values,
    x=None,
    normalization_scale=None,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if x is None:
        x = np.arange(
            len(values),
            dtype=np.float64,
        )
    else:
        x = np.asarray(
            x,
            dtype=np.float64,
        )

    raw = float(
        np.polyfit(
            x,
            values,
            1,
        )[0]
    )

    if normalization_scale is None:
        normalization_scale = max(
            float(
                np.mean(
                    np.abs(values)
                )
            ),
            1.0e-12,
        )

    return (
        raw,
        raw / normalization_scale,
        normalization_scale,
    )


def km_summary(
    records,
    tau,
):
    observations = sorted(
        (
            float(row["episodes_completed"]),
            bool(row["converged"]),
        )
        for row in records
    )

    # Kaplan-Meier survival step function.
    survival = 1.0
    km_median = None

    event_times = sorted(
        {
            t
            for t, event
            in observations
            if event
        }
    )

    for t in event_times:
        at_risk = sum(
            time >= t
            for time, _
            in observations
        )

        events = sum(
            time == t and event
            for time, event
            in observations
        )

        if at_risk > 0:
            survival *= (
                1.0
                - events / at_risk
            )

        if (
            km_median is None
            and survival <= 0.5
        ):
            km_median = t

    # Restricted mean survival time to tau.
    #
    # Integral of KM survival curve from 0 to tau.
    survival = 1.0
    previous_time = 0.0
    rmst = 0.0

    all_times = sorted(
        {
            min(
                float(t),
                float(tau),
            )
            for t, _
            in observations
            if t <= tau
        }
    )

    for t in all_times:
        rmst += (
            survival
            * (
                t
                - previous_time
            )
        )

        at_risk = sum(
            time >= t
            for time, _
            in observations
        )

        events = sum(
            time == t
            and event
            and t <= tau
            for time, event
            in observations
        )

        if at_risk > 0:
            survival *= (
                1.0
                - events / at_risk
            )

        previous_time = t

    if previous_time < tau:
        rmst += (
            survival
            * (
                tau
                - previous_time
            )
        )

    return {
        "km_median_episodes":
            km_median,

        "rmst_to_1280_episodes":
            float(rmst),
    }


def phase_decomposition(
    training_trace,
):
    window = training_trace[
        -WINDOW:
    ]

    if len(window) != WINDOW:
        raise RuntimeError(
            "Training trace shorter than "
            f"{WINDOW} episodes."
        )

    rewards = np.asarray(
        [
            float(row["total_reward"])
            for row in window
        ],
        dtype=np.float64,
    )

    episode_indices = np.asarray(
        [
            int(row["episode_index"])
            for row in window
        ],
        dtype=np.int64,
    )

    phases = (
        episode_indices
        % CURRICULUM_PERIOD
    )

    phase_mean = {}

    for phase in range(
        CURRICULUM_PERIOD
    ):
        mask = (
            phases
            == phase
        )

        if int(
            np.sum(mask)
        ) != 2:
            raise RuntimeError(
                "Expected exactly two observations "
                f"for curriculum phase {phase}; "
                f"found {int(np.sum(mask))}."
            )

        phase_mean[phase] = float(
            np.mean(
                rewards[
                    mask
                ]
            )
        )

    phase_component = np.asarray(
        [
            phase_mean[
                int(phase)
            ]
            for phase in phases
        ],
        dtype=np.float64,
    )

    residual = (
        rewards
        - phase_component
    )

    scale = max(
        float(
            np.mean(
                np.abs(
                    rewards
                )
            )
        ),
        1.0e-12,
    )

    raw_slope, raw_norm, _ = (
        raw_and_normalized_slope(
            rewards,
            normalization_scale=scale,
        )
    )

    phase_slope, phase_norm, _ = (
        raw_and_normalized_slope(
            phase_component,
            normalization_scale=scale,
        )
    )

    residual_slope, residual_norm, _ = (
        raw_and_normalized_slope(
            residual,
            normalization_scale=scale,
        )
    )

    reconstruction_error = abs(
        raw_norm
        - (
            phase_norm
            + residual_norm
        )
    )

    return {
        "reward_window_mean_abs":
            scale,

        "reward_raw_slope":
            raw_slope,

        "normalized_reward_slope_recomputed":
            raw_norm,

        "phase_raw_slope":
            phase_slope,

        "phase_component_normalized_slope":
            phase_norm,

        "residual_raw_slope":
            residual_slope,

        "phase_demeaned_residual_normalized_slope":
            residual_norm,

        "phase_plus_residual_reconstruction_error":
            reconstruction_error,

        "phase_fraction_of_raw_abs":
            (
                abs(phase_norm)
                / max(
                    abs(raw_norm),
                    1.0e-12,
                )
            ),
    }


def cycle_mean_trend(
    training_trace,
):
    cycles = defaultdict(
        list
    )

    for row in training_trace:
        episode = int(
            row["episode_index"]
        )

        cycle = (
            episode
            // CURRICULUM_PERIOD
        )

        cycles[
            cycle
        ].append(
            float(
                row["total_reward"]
            )
        )

    complete = []

    for cycle in sorted(
        cycles
    ):
        values = cycles[
            cycle
        ]

        if len(values) == CURRICULUM_PERIOD:
            complete.append(
                (
                    cycle,
                    float(
                        np.mean(values)
                    ),
                )
            )

    if len(complete) < 8:
        raise RuntimeError(
            "Expected at least eight complete "
            "curriculum cycles."
        )

    final8 = complete[
        -8:
    ]

    values = np.asarray(
        [
            mean
            for _, mean
            in final8
        ],
        dtype=np.float64,
    )

    # Actual episode spacing between cycle means.
    x = np.asarray(
        [
            (
                cycle
                * CURRICULUM_PERIOD
                + (
                    CURRICULUM_PERIOD
                    - 1
                )
            )
            for cycle, _
            in final8
        ],
        dtype=np.float64,
    )

    trend = normalized_slope(
        values,
        x=x,
    )

    return {
        "last8_cycle_mean_reward":
            [
                float(v)
                for v in values
            ],

        "last8_cycle_end_episodes_zero_based":
            [
                int(v)
                for v in x
            ],

        "cycle_mean_reward_trend_per_episode":
            float(
                trend
            ),
    }


def validation_trends(
    validation_trace,
):
    grouped = defaultdict(
        list
    )

    for row in validation_trace:
        grouped[
            int(
                row[
                    "checkpoint_episode"
                ]
            )
        ].append(
            row
        )

    checkpoints = sorted(
        grouped
    )

    aggregates = []

    for checkpoint in checkpoints:
        rows = grouped[
            checkpoint
        ]

        if len(rows) != 8:
            raise RuntimeError(
                f"Checkpoint {checkpoint} has "
                f"{len(rows)} validation rows; "
                "expected 8."
            )

        identities = {
            (
                int(
                    row[
                        "validation_slot_id"
                    ]
                ),
                str(
                    row[
                        "challenge_id"
                    ]
                ),
            )
            for row in rows
        }

        if len(identities) != 8:
            raise RuntimeError(
                "Validation slot/challenge identity "
                f"duplication at checkpoint {checkpoint}."
            )

        if not all(
            bool(
                row[
                    "natural_completion"
                ]
            )
            for row in rows
        ):
            raise RuntimeError(
                "Non-natural validation completion "
                f"at checkpoint {checkpoint}."
            )

        aggregates.append(
            {
                "checkpoint":
                    checkpoint,

                "mean_reward":
                    float(
                        np.mean(
                            [
                                float(
                                    row[
                                        "total_reward"
                                    ]
                                )
                                for row in rows
                            ]
                        )
                    ),

                "mean_distance":
                    float(
                        np.mean(
                            [
                                float(
                                    row[
                                        "fleet_distance_km"
                                    ]
                                )
                                for row in rows
                            ]
                        )
                    ),
            }
        )

    if len(aggregates) < 8:
        raise RuntimeError(
            "Expected at least eight validation checkpoints."
        )

    final8 = aggregates[
        -8:
    ]

    x = np.asarray(
        [
            row[
                "checkpoint"
            ]
            for row in final8
        ],
        dtype=np.float64,
    )

    reward = np.asarray(
        [
            row[
                "mean_reward"
            ]
            for row in final8
        ],
        dtype=np.float64,
    )

    distance = np.asarray(
        [
            row[
                "mean_distance"
            ]
            for row in final8
        ],
        dtype=np.float64,
    )

    return {
        "validation_last8_checkpoints":
            [
                int(v)
                for v in x
            ],

        "validation_last8_mean_reward":
            [
                float(v)
                for v in reward
            ],

        "validation_last8_mean_distance_km":
            [
                float(v)
                for v in distance
            ],

        "greedy_validation_reward_trend_per_episode":
            float(
                normalized_slope(
                    reward,
                    x=x,
                )
            ),

        "greedy_validation_distance_trend_per_episode":
            float(
                normalized_slope(
                    distance,
                    x=x,
                )
            ),
    }


def analyze_run(
    run_dir,
    *,
    replicate_number,
    master_seed,
    topology,
):
    summary_path = (
        run_dir
        / "training_summary.json"
    )

    if not summary_path.exists():
        raise RuntimeError(
            f"Missing run: {run_dir}"
        )

    summary = json.loads(
        summary_path.read_text(
            encoding="utf-8"
        )
    )

    failures = []

    if (
        summary[
            "topology_id"
        ]
        != topology
    ):
        failures.append(
            "topology_id"
        )

    if (
        summary[
            "source_commit"
        ]
        != EXPECTED_EXECUTION_COMMIT
    ):
        failures.append(
            "source_commit"
        )

    if (
        summary[
            "dataset_manifest_sha256"
        ]
        != EXPECTED_DATASET_SHA
    ):
        failures.append(
            "dataset_manifest_sha256"
        )

    if (
        summary[
            "physical_case_manifest_sha256"
        ]
        != EXPECTED_CASE_SHA
    ):
        failures.append(
            "physical_case_manifest_sha256"
        )

    checkpoint_path = (
        run_dir
        / "model_checkpoint.pt"
    )

    if not checkpoint_path.exists():
        failures.append(
            "model_checkpoint.pt_missing"
        )
    else:
        if (
            sha256_file(
                checkpoint_path
            )
            != summary[
                "checkpoint_file_sha256"
            ]
        ):
            failures.append(
                "checkpoint_sha"
            )

    for filename, field in TRACE_FILES:
        path = (
            run_dir
            / filename
        )

        if not path.exists():
            failures.append(
                filename
                + "_missing"
            )
            continue

        actual = sha256_file(
            path
        )

        if (
            actual
            != summary[field]
        ):
            failures.append(
                filename
                + "_sha"
            )

    if failures:
        raise RuntimeError(
            f"{run_dir}: integrity failures: "
            + ", ".join(
                failures
            )
        )

    training_trace = json.loads(
        (
            run_dir
            / "training_trace.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    validation_trace = json.loads(
        (
            run_dir
            / "validation_trace.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    checks = json.loads(
        (
            run_dir
            / "convergence_checks.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    episodes_completed = int(
        summary[
            "episodes_completed"
        ]
    )

    if len(
        training_trace
    ) != episodes_completed:
        raise RuntimeError(
            f"{run_dir}: training trace length "
            "does not equal episodes_completed."
        )

    expected_indices = list(
        range(
            episodes_completed
        )
    )

    actual_indices = [
        int(
            row[
                "episode_index"
            ]
        )
        for row in training_trace
    ]

    if (
        actual_indices
        != expected_indices
    ):
        raise RuntimeError(
            f"{run_dir}: non-contiguous episode indices."
        )

    if not all(
        bool(
            row[
                "natural_completion"
            ]
        )
        for row in training_trace
    ):
        raise RuntimeError(
            f"{run_dir}: training natural-completion failure."
        )

    if not checks:
        raise RuntimeError(
            f"{run_dir}: no convergence checks."
        )

    final_check = checks[
        -1
    ]

    if (
        int(
            final_check[
                "checkpoint_episode"
            ]
        )
        != episodes_completed
    ):
        raise RuntimeError(
            f"{run_dir}: final convergence checkpoint "
            "does not match episodes_completed."
        )

    converged = bool(
        summary[
            "converged"
        ]
    )

    if converged:
        if (
            summary[
                "stop_reason"
            ]
            != "convergence_criteria_satisfied"
        ):
            raise RuntimeError(
                f"{run_dir}: converged stop reason mismatch."
            )

        if int(
            final_check[
                "consecutive_stable_checks"
            ]
        ) < 3:
            raise RuntimeError(
                f"{run_dir}: convergence without patience 3."
            )

    else:
        if (
            summary[
                "stop_reason"
            ]
            != "maximum_episodes_reached"
        ):
            raise RuntimeError(
                f"{run_dir}: non-converged stop reason mismatch."
            )

    phase = phase_decomposition(
        training_trace
    )

    # Independent recomputation of final 64-episode loss slope.
    loss_window = training_trace[
        -WINDOW:
    ]

    loss_values = np.asarray(
        [
            float(
                row[
                    "mean_loss"
                ]
            )
            for row in loss_window
        ],
        dtype=np.float64,
    )

    loss_norm = normalized_slope(
        loss_values
    )

    reward_error = abs(
        phase[
            "normalized_reward_slope_recomputed"
        ]
        - float(
            final_check[
                "normalized_reward_slope"
            ]
        )
    )

    loss_error = abs(
        loss_norm
        - float(
            final_check[
                "normalized_loss_slope"
            ]
        )
    )

    if reward_error > 1.0e-12:
        raise RuntimeError(
            f"{run_dir}: reward slope recomputation mismatch "
            f"{reward_error}"
        )

    if loss_error > 1.0e-12:
        raise RuntimeError(
            f"{run_dir}: loss slope recomputation mismatch "
            f"{loss_error}"
        )

    cycles = cycle_mean_trend(
        training_trace
    )

    validation = validation_trends(
        validation_trace
    )

    return {
        "replicate":
            int(
                replicate_number
            ),

        "master_seed":
            int(
                master_seed
            ),

        "topology":
            topology,

        "agent_seed":
            int(
                summary[
                    "agent_seed"
                ]
            ),

        "converged":
            converged,

        "event_observed":
            converged,

        "right_censored":
            not converged,

        "episodes_completed":
            episodes_completed,

        "stop_reason":
            summary[
                "stop_reason"
            ],

        "final_optimization_steps":
            int(
                summary[
                    "final_optimization_steps"
                ]
            ),

        "final_replay_size":
            int(
                summary[
                    "final_replay_size"
                ]
            ),

        "protocol_sha256":
            summary[
                "protocol_sha256"
            ],

        "checkpoint_file_sha256":
            summary[
                "checkpoint_file_sha256"
            ],

        "final_recorded_reward_slope":
            float(
                final_check[
                    "normalized_reward_slope"
                ]
            ),

        "final_recorded_loss_slope":
            float(
                final_check[
                    "normalized_loss_slope"
                ]
            ),

        "final_consecutive_stable_checks":
            int(
                final_check[
                    "consecutive_stable_checks"
                ]
            ),

        "final_reward_stable":
            bool(
                final_check[
                    "reward_stable"
                ]
            ),

        "final_loss_stable":
            bool(
                final_check[
                    "loss_stable"
                ]
            ),

        "reward_slope_recompute_abs_error":
            reward_error,

        "loss_slope_recompute_abs_error":
            loss_error,

        **phase,
        **cycles,
        **validation,
    }


def write_csv(
    path,
    rows,
    fields,
):
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    field:
                        row.get(
                            field
                        )
                    for field in fields
                }
            )


def main():
    plan = yaml.safe_load(
        PLAN_PATH.read_text(
            encoding="utf-8"
        )
    )

    if (
        plan[
            "base_convergence_protocol"
        ][
            "sha256"
        ]
        != EXPECTED_BASE_PROTOCOL_SHA
    ):
        raise RuntimeError(
            "Pre-registration base protocol SHA mismatch."
        )

    if tuple(
        plan[
            "topologies"
        ]
    ) != TOPOLOGIES:
        raise RuntimeError(
            "Pre-registered topology set mismatch."
        )

    master_seeds = [
        int(seed)
        for seed in plan[
            "replicates"
        ][
            "master_seeds"
        ]
    ]

    if len(
        master_seeds
    ) != 5:
        raise RuntimeError(
            "Expected five pre-registered seeds."
        )

    root = Path(
        plan[
            "output_root"
        ]
    )

    rows = []

    print(
        "=" * 88
    )
    print(
        "FINAL DQN SEED-SENSITIVITY ANALYSIS"
    )
    print(
        "=" * 88
    )

    for replicate_number, master_seed in enumerate(
        master_seeds,
        start=1,
    ):
        replicate_root = (
            root
            / (
                f"replicate_"
                f"{replicate_number:02d}"
                f"_seed_{master_seed}"
            )
        )

        for topology in TOPOLOGIES:
            run_dir = (
                replicate_root
                / topology
            )

            row = analyze_run(
                run_dir,
                replicate_number=(
                    replicate_number
                ),
                master_seed=(
                    master_seed
                ),
                topology=(
                    topology
                ),
            )

            rows.append(
                row
            )

            print(
                f"replicate={replicate_number} "
                f"topology={topology:<18} "
                f"episodes={row['episodes_completed']:<4} "
                f"converged={str(row['converged']):<5} "
                f"raw={row['final_recorded_reward_slope']:+.9f} "
                f"phase={row['phase_component_normalized_slope']:+.9f} "
                f"residual="
                f"{row['phase_demeaned_residual_normalized_slope']:+.9f}"
            )

    if len(
        rows
    ) != 20:
        raise RuntimeError(
            f"Expected 20 runs; found {len(rows)}."
        )

    topology_summary = []

    for topology in TOPOLOGIES:
        subset = [
            row
            for row in rows
            if (
                row[
                    "topology"
                ]
                == topology
            )
        ]

        event_count = sum(
            bool(
                row[
                    "converged"
                ]
            )
            for row in subset
        )

        censored_count = (
            len(subset)
            - event_count
        )

        observed_event_times = [
            int(
                row[
                    "episodes_completed"
                ]
            )
            for row in subset
            if row[
                "converged"
            ]
        ]

        km = km_summary(
            subset,
            tau=(
                ADMINISTRATIVE_LIMIT
            ),
        )

        topology_summary.append(
            {
                "topology":
                    topology,

                "runs":
                    len(
                        subset
                    ),

                "converged_runs":
                    int(
                        event_count
                    ),

                "right_censored_runs":
                    int(
                        censored_count
                    ),

                "convergence_fraction":
                    float(
                        event_count
                        / len(
                            subset
                        )
                    ),

                "event_times_episodes":
                    observed_event_times,

                "median_observed_event_time_episodes":
                    (
                        float(
                            np.median(
                                observed_event_times
                            )
                        )
                        if observed_event_times
                        else None
                    ),

                "km_median_episodes":
                    km[
                        "km_median_episodes"
                    ],

                "rmst_to_1280_episodes":
                    km[
                        "rmst_to_1280_episodes"
                    ],

                "median_abs_final_reward_slope":
                    float(
                        np.median(
                            [
                                abs(
                                    row[
                                        "final_recorded_reward_slope"
                                    ]
                                )
                                for row in subset
                            ]
                        )
                    ),

                "median_abs_phase_component":
                    float(
                        np.median(
                            [
                                abs(
                                    row[
                                        "phase_component_normalized_slope"
                                    ]
                                )
                                for row in subset
                            ]
                        )
                    ),

                "median_abs_phase_demeaned_residual":
                    float(
                        np.median(
                            [
                                abs(
                                    row[
                                        "phase_demeaned_residual_normalized_slope"
                                    ]
                                )
                                for row in subset
                            ]
                        )
                    ),

                "median_abs_cycle_mean_trend":
                    float(
                        np.median(
                            [
                                abs(
                                    row[
                                        "cycle_mean_reward_trend_per_episode"
                                    ]
                                )
                                for row in subset
                            ]
                        )
                    ),

                "median_abs_validation_reward_trend":
                    float(
                        np.median(
                            [
                                abs(
                                    row[
                                        "greedy_validation_reward_trend_per_episode"
                                    ]
                                )
                                for row in subset
                            ]
                        )
                    ),

                "median_abs_validation_distance_trend":
                    float(
                        np.median(
                            [
                                abs(
                                    row[
                                        "greedy_validation_distance_trend_per_episode"
                                    ]
                                )
                                for row in subset
                            ]
                        )
                    ),
            }
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_fields = (
        "replicate",
        "master_seed",
        "topology",
        "agent_seed",
        "converged",
        "right_censored",
        "episodes_completed",
        "stop_reason",
        "final_optimization_steps",
        "final_replay_size",
        "final_recorded_reward_slope",
        "final_recorded_loss_slope",
        "phase_component_normalized_slope",
        "phase_demeaned_residual_normalized_slope",
        "phase_fraction_of_raw_abs",
        "cycle_mean_reward_trend_per_episode",
        "greedy_validation_reward_trend_per_episode",
        "greedy_validation_distance_trend_per_episode",
        "reward_slope_recompute_abs_error",
        "loss_slope_recompute_abs_error",
        "protocol_sha256",
        "checkpoint_file_sha256",
    )

    write_csv(
        OUTPUT_DIR
        / "run_level_summary.csv",
        rows,
        run_fields,
    )

    topology_fields = (
        "topology",
        "runs",
        "converged_runs",
        "right_censored_runs",
        "convergence_fraction",
        "median_observed_event_time_episodes",
        "km_median_episodes",
        "rmst_to_1280_episodes",
        "median_abs_final_reward_slope",
        "median_abs_phase_component",
        "median_abs_phase_demeaned_residual",
        "median_abs_cycle_mean_trend",
        "median_abs_validation_reward_trend",
        "median_abs_validation_distance_trend",
    )

    write_csv(
        OUTPUT_DIR
        / "topology_summary.csv",
        topology_summary,
        topology_fields,
    )

    evidence = {
        "schema_version":
            "dqn-seed-sensitivity-analysis-v1",

        "pre_registration_commit":
            EXPECTED_PREREG_COMMIT,

        "execution_source_commit":
            EXPECTED_EXECUTION_COMMIT,

        "base_protocol_sha256":
            EXPECTED_BASE_PROTOCOL_SHA,

        "dataset_manifest_sha256":
            EXPECTED_DATASET_SHA,

        "physical_case_manifest_sha256":
            EXPECTED_CASE_SHA,

        "analysis_definitions": {
            "right_censoring":
                (
                    "Runs reaching 1280 episodes without satisfying "
                    "the predeclared stopping rule are treated as "
                    "right-censored at 1280, not as convergence at 1280."
                ),

            "phase_component":
                (
                    "Within the final 64 training episodes, rewards "
                    "are grouped by fixed curriculum phase "
                    "(episode_index mod 32). The two observations "
                    "for each phase are averaged, repeated in their "
                    "original order, and the OLS slope of this "
                    "reconstructed phase component is normalized by "
                    "the mean absolute raw reward of the same window."
                ),

            "phase_demeaned_residual":
                (
                    "Raw reward minus the reconstructed fixed-phase "
                    "component, with its OLS slope normalized by the "
                    "same raw-reward scale."
                ),

            "cycle_mean_trend":
                (
                    "Normalized OLS slope per episode of the mean "
                    "reward of the final eight complete 32-episode "
                    "curriculum cycles."
                ),

            "greedy_validation_trends":
                (
                    "Normalized OLS slope per episode over the final "
                    "eight validation checkpoints after averaging "
                    "the eight fixed validation conditions at each "
                    "checkpoint."
                ),

            "survival_summary":
                (
                    "Kaplan-Meier median time to stopping criterion "
                    "and restricted mean time to criterion through "
                    "the predeclared administrative limit of 1280 "
                    "episodes. No hypothesis test is performed."
                ),
        },

        "runs":
            rows,

        "topology_summary":
            topology_summary,
    }

    (
        OUTPUT_DIR
        / "analysis_evidence.json"
    ).write_text(
        json.dumps(
            evidence,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = {
        "schema_version":
            "dqn-seed-sensitivity-analysis-manifest-v1",

        "files": {},
    }

    for filename in (
        "run_level_summary.csv",
        "topology_summary.csv",
        "analysis_evidence.json",
    ):
        path = (
            OUTPUT_DIR
            / filename
        )

        manifest[
            "files"
        ][
            filename
        ] = {
            "sha256":
                sha256_file(
                    path
                ),

            "bytes":
                path.stat().st_size,
        }

    (
        OUTPUT_DIR
        / "analysis_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 88)
    print("TOPOLOGY SUMMARY")
    print("=" * 88)

    for row in topology_summary:
        print()
        print(
            row[
                "topology"
            ]
        )

        print(
            "converged:",
            f"{row['converged_runs']}/{row['runs']}",
        )

        print(
            "right_censored:",
            row[
                "right_censored_runs"
            ],
        )

        print(
            "KM_median_episodes:",
            row[
                "km_median_episodes"
            ],
        )

        print(
            "RMST_to_1280:",
            f"{row['rmst_to_1280_episodes']:.3f}",
        )

        print(
            "median_abs_final_reward_slope:",
            f"{row['median_abs_final_reward_slope']:.9f}",
        )

        print(
            "median_abs_phase_component:",
            f"{row['median_abs_phase_component']:.9f}",
        )

        print(
            "median_abs_phase_demeaned_residual:",
            f"{row['median_abs_phase_demeaned_residual']:.9f}",
        )

        print(
            "median_abs_cycle_mean_trend:",
            f"{row['median_abs_cycle_mean_trend']:.9f}",
        )

        print(
            "median_abs_validation_reward_trend:",
            f"{row['median_abs_validation_reward_trend']:.9f}",
        )

        print(
            "median_abs_validation_distance_trend:",
            f"{row['median_abs_validation_distance_trend']:.9f}",
        )

    max_reward_error = max(
        row[
            "reward_slope_recompute_abs_error"
        ]
        for row in rows
    )

    max_loss_error = max(
        row[
            "loss_slope_recompute_abs_error"
        ]
        for row in rows
    )

    max_reconstruction_error = max(
        row[
            "phase_plus_residual_reconstruction_error"
        ]
        for row in rows
    )

    print()
    print("=" * 88)
    print("FINAL ANALYSIS AUDIT")
    print("=" * 88)

    print(
        "runs_analyzed:",
        len(
            rows
        ),
    )

    print(
        "integrity_failures: 0"
    )

    print(
        "max_reward_slope_recompute_error:",
        f"{max_reward_error:.3e}",
    )

    print(
        "max_loss_slope_recompute_error:",
        f"{max_loss_error:.3e}",
    )

    print(
        "max_phase_reconstruction_error:",
        f"{max_reconstruction_error:.3e}",
    )

    print(
        "RIGHT_CENSORING_HANDLED: PASS"
    )

    print(
        "FINAL_SEED_SENSITIVITY_ANALYSIS: PASS"
    )

    print()
    print(
        "output_directory:",
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()
