from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path

import yaml

from smart_waste.rl.physical_convergence_protocol import (
    load_physical_convergence_protocol,
)
from smart_waste.rl.physical_convergence_training import (
    _derive_agent_seed,
    train_topology_until_convergence,
)


PREREG_COMMIT = (
    "41cae5cfbc66ed88d04de1f8d85aff1b699e3c96"
)

PLAN_PATH = Path(
    "configs/rl/dqn_seed_sensitivity_v1.yaml"
)

EXPECTED_BASE_PROTOCOL_SHA = (
    "878446fe12985e7f7fa9cc41c39e40efaa69bc8825c9ac102be24b205aa0413a"
)

REQUIRED_ARTIFACTS = (
    "model_checkpoint.pt",
    "training_summary.json",
    "training_trace.json",
    "validation_trace.json",
    "convergence_checks.json",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def git_head() -> str:
    return subprocess.check_output(
        [
            "git",
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()


def require_clean_tree() -> None:
    output = subprocess.check_output(
        [
            "git",
            "status",
            "--porcelain",
        ],
        text=True,
    )

    if output.strip():
        raise RuntimeError(
            "Git working tree is not clean:\n"
            + output
        )


def require_prereg_ancestor() -> None:
    result = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            PREREG_COMMIT,
            "HEAD",
        ],
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Pre-registration commit is not "
            "an ancestor of current HEAD"
        )


def verify_completed_run(
    *,
    run_dir: Path,
    topology: str,
    expected_protocol_sha: str,
    expected_agent_seed: int,
) -> dict | None:

    summary_path = (
        run_dir
        / "training_summary.json"
    )

    if not summary_path.exists():
        return None

    missing = [
        name
        for name in REQUIRED_ARTIFACTS
        if not (
            run_dir
            / name
        ).exists()
    ]

    if missing:
        raise RuntimeError(
            f"Incomplete existing run at {run_dir}: "
            f"missing={missing}"
        )

    summary = json.loads(
        summary_path.read_text(
            encoding="utf-8"
        )
    )

    if (
        summary["topology_id"]
        != topology
    ):
        raise RuntimeError(
            f"{run_dir}: topology mismatch"
        )

    if (
        summary["protocol_sha256"]
        != expected_protocol_sha
    ):
        raise RuntimeError(
            f"{run_dir}: protocol SHA mismatch"
        )

    if (
        int(
            summary["agent_seed"]
        )
        != int(
            expected_agent_seed
        )
    ):
        raise RuntimeError(
            f"{run_dir}: agent seed mismatch"
        )

    if (
        sha256_file(
            run_dir
            / "model_checkpoint.pt"
        )
        != summary[
            "checkpoint_file_sha256"
        ]
    ):
        raise RuntimeError(
            f"{run_dir}: checkpoint SHA mismatch"
        )

    checks = (
        (
            "training_trace.json",
            "training_trace_sha256",
        ),
        (
            "validation_trace.json",
            "validation_trace_sha256",
        ),
        (
            "convergence_checks.json",
            "convergence_checks_sha256",
        ),
    )

    for filename, field in checks:

        actual = sha256_file(
            run_dir
            / filename
        )

        expected = summary[
            field
        ]

        if actual != expected:
            raise RuntimeError(
                f"{run_dir}: "
                f"{filename} SHA mismatch"
            )

    return summary


def write_execution_state(
    *,
    output_root: Path,
    rows: list[dict],
) -> None:

    payload = {
        "schema_version":
            "dqn-seed-sensitivity-execution-state-v1",

        "pre_registration_commit":
            PREREG_COMMIT,

        "execution_source_commit":
            git_head(),

        "runs":
            rows,
    }

    (
        output_root
        / "execution_state.json"
    ).write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    parser.add_argument(
        "--replicate",
        type=int,
        default=None,
        help=(
            "Run only one replicate number "
            "(1-based)."
        ),
    )

    parser.add_argument(
        "--topology",
        choices=(
            "manhattan",
            "superblock",
            "hex",
            "radial_concentric",
        ),
        default=None,
        help="Run only one topology.",
    )

    args = parser.parse_args()


    require_clean_tree()
    require_prereg_ancestor()


    plan = yaml.safe_load(
        PLAN_PATH.read_text(
            encoding="utf-8"
        )
    )

    base_protocol_path = Path(
        plan[
            "base_convergence_protocol"
        ][
            "path"
        ]
    )

    base_protocol = (
        load_physical_convergence_protocol(
            base_protocol_path
        )
    )

    if (
        base_protocol.payload_sha256
        != EXPECTED_BASE_PROTOCOL_SHA
    ):
        raise RuntimeError(
            "Base Protocol v3 SHA mismatch"
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
            "Pre-registration base SHA mismatch"
        )


    topologies = tuple(
        plan[
            "topologies"
        ]
    )

    master_seeds = tuple(
        int(x)
        for x in plan[
            "replicates"
        ][
            "master_seeds"
        ]
    )

    expected_count = int(
        plan[
            "replicates"
        ][
            "count_per_topology"
        ]
    )

    if len(
        master_seeds
    ) != expected_count:
        raise RuntimeError(
            "Unexpected number of master seeds"
        )


    output_root = Path(
        plan[
            "output_root"
        ]
    )


    # --------------------------------------------------------
    # Selection without changing the pre-registered matrix.
    # CLI filters execution only.
    # --------------------------------------------------------

    replicate_indices = list(
        range(
            len(
                master_seeds
            )
        )
    )

    if args.replicate is not None:

        if not (
            1
            <= args.replicate
            <= len(
                master_seeds
            )
        ):
            raise ValueError(
                "replicate must be 1.."
                f"{len(master_seeds)}"
            )

        replicate_indices = [
            args.replicate - 1
        ]


    selected_topologies = (
        topologies
        if args.topology is None
        else (
            args.topology,
        )
    )


    matrix = []


    # --------------------------------------------------------
    # Build derived protocols.
    #
    # Only master_agent_seed changes.
    # All convergence/model/data settings remain exactly v3.
    # --------------------------------------------------------

    for replicate_index in replicate_indices:

        replicate_number = (
            replicate_index
            + 1
        )

        master_seed = (
            master_seeds[
                replicate_index
            ]
        )

        replicate_root = (
            output_root
            / (
                f"replicate_"
                f"{replicate_number:02d}"
                f"_seed_{master_seed}"
            )
        )

        derived_protocol_path = (
            replicate_root
            / "derived_protocol_v3.yaml"
        )


        payload = yaml.safe_load(
            base_protocol_path.read_text(
                encoding="utf-8"
            )
        )

        original_payload = copy.deepcopy(
            payload
        )

        payload[
            "seeds"
        ][
            "master_agent_seed"
        ] = int(
            master_seed
        )


        # Prove no field other than master seed changed.
        original_seed = (
            original_payload[
                "seeds"
            ][
                "master_agent_seed"
            ]
        )

        original_payload[
            "seeds"
        ][
            "master_agent_seed"
        ] = int(
            master_seed
        )

        if (
            payload
            != original_payload
        ):
            raise RuntimeError(
                "Derived protocol changed fields "
                "other than master_agent_seed"
            )

        original_payload[
            "seeds"
        ][
            "master_agent_seed"
        ] = original_seed


        if not args.dry_run:

            replicate_root.mkdir(
                parents=True,
                exist_ok=True,
            )

            derived_protocol_path.write_text(
                yaml.safe_dump(
                    payload,
                    sort_keys=False,
                    allow_unicode=True,
                ),
                encoding="utf-8",
            )

        else:

            # Dry-run needs an actual loader-compatible file
            # without modifying the repository.
            temp_root = Path(
                "/tmp/"
                "smart_waste_seed_sensitivity"
            )

            temp_root.mkdir(
                parents=True,
                exist_ok=True,
            )

            derived_protocol_path = (
                temp_root
                / (
                    f"replicate_"
                    f"{replicate_number:02d}"
                    ".yaml"
                )
            )

            derived_protocol_path.write_text(
                yaml.safe_dump(
                    payload,
                    sort_keys=False,
                    allow_unicode=True,
                ),
                encoding="utf-8",
            )


        derived_protocol = (
            load_physical_convergence_protocol(
                derived_protocol_path
            )
        )


        # Explicit invariants against base Protocol v3.
        invariant_fields = (
            "minimum_episodes",
            "maximum_episodes",
            "curriculum_period_episodes",
            "validation_interval_episodes",
            "training_window_episodes",
            "normalized_reward_slope_threshold",
            "normalized_loss_slope_threshold",
            "max_validation_fuel_relative_change",
            "max_validation_reward_relative_change",
            "patience_consecutive_checks",
            "require_training_natural_completion",
            "require_validation_natural_completion",
            "validation_metric_stability_required",
            "dataset_manifest_sha256",
        )

        for field in invariant_fields:

            if (
                getattr(
                    derived_protocol,
                    field,
                )
                != getattr(
                    base_protocol,
                    field,
                )
            ):
                raise RuntimeError(
                    "Protocol invariant changed: "
                    f"{field}"
                )


        if (
            derived_protocol.master_agent_seed
            != master_seed
        ):
            raise RuntimeError(
                "Derived master seed mismatch"
            )


        for topology in selected_topologies:

            expected_agent_seed = (
                _derive_agent_seed(
                    master_seed=(
                        master_seed
                    ),
                    topology_id=(
                        topology
                    ),
                )
            )

            matrix.append(
                {
                    "replicate":
                        replicate_number,

                    "master_seed":
                        master_seed,

                    "topology":
                        topology,

                    "agent_seed":
                        expected_agent_seed,

                    "protocol_sha256":
                        derived_protocol
                        .payload_sha256,

                    "protocol_path":
                        derived_protocol_path,

                    "replicate_root":
                        replicate_root,
                }
            )


    print(
        "=" * 86
    )
    print(
        "DQN SEED SENSITIVITY V1 EXECUTION MATRIX"
    )
    print(
        "=" * 86
    )

    print(
        "pre_registration_commit:",
        PREREG_COMMIT,
    )

    print(
        "execution_source_commit:",
        git_head(),
    )

    print(
        "base_protocol_sha256:",
        base_protocol.payload_sha256,
    )

    print(
        "selected_run_count:",
        len(
            matrix
        ),
    )

    print()


    for row in matrix:

        print(
            f"replicate={row['replicate']} "
            f"master_seed={row['master_seed']} "
            f"topology={row['topology']} "
            f"agent_seed={row['agent_seed']} "
            f"protocol_sha="
            f"{row['protocol_sha256']}"
        )


    if args.dry_run:

        print()
        print(
            "DRY_RUN: PASS"
        )

        return


    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    execution_rows = []


    # --------------------------------------------------------
    # Execute / resume.
    # --------------------------------------------------------

    for run_index, row in enumerate(
        matrix,
        start=1,
    ):

        replicate_root = row[
            "replicate_root"
        ]

        topology = row[
            "topology"
        ]

        run_dir = (
            replicate_root
            / topology
        )


        existing = verify_completed_run(
            run_dir=run_dir,
            topology=topology,
            expected_protocol_sha=(
                row[
                    "protocol_sha256"
                ]
            ),
            expected_agent_seed=(
                row[
                    "agent_seed"
                ]
            ),
        )


        print()
        print(
            "=" * 86
        )

        print(
            f"RUN {run_index}/{len(matrix)}"
        )

        print(
            f"replicate={row['replicate']} "
            f"master_seed={row['master_seed']} "
            f"topology={topology}"
        )

        print(
            "=" * 86
        )


        if existing is not None:

            print(
                "STATUS: COMPLETE — SKIPPING"
            )

            summary = existing

        else:

            print(
                "STATUS: STARTING"
            )

            summary = (
                train_topology_until_convergence(
                    topology_id=topology,
                    output_root=(
                        replicate_root
                    ),
                    protocol_path=(
                        row[
                            "protocol_path"
                        ]
                    ),
                    require_clean_git_tree=True,
                )
            )


            verified = (
                verify_completed_run(
                    run_dir=run_dir,
                    topology=topology,
                    expected_protocol_sha=(
                        row[
                            "protocol_sha256"
                        ]
                    ),
                    expected_agent_seed=(
                        row[
                            "agent_seed"
                        ]
                    ),
                )
            )

            if verified is None:
                raise RuntimeError(
                    "Run completed but verification "
                    "failed to find summary"
                )

            summary = verified


        execution_rows.append(
            {
                "replicate":
                    row[
                        "replicate"
                    ],

                "master_seed":
                    row[
                        "master_seed"
                    ],

                "topology":
                    topology,

                "agent_seed":
                    row[
                        "agent_seed"
                    ],

                "protocol_sha256":
                    row[
                        "protocol_sha256"
                    ],

                "converged":
                    bool(
                        summary[
                            "converged"
                        ]
                    ),

                "episodes_completed":
                    int(
                        summary[
                            "episodes_completed"
                        ]
                    ),

                "stop_reason":
                    summary[
                        "stop_reason"
                    ],

                "checkpoint_file_sha256":
                    summary[
                        "checkpoint_file_sha256"
                    ],
            }
        )


        write_execution_state(
            output_root=output_root,
            rows=execution_rows,
        )


        print(
            "STATUS: VERIFIED"
        )

        print(
            "converged:",
            summary[
                "converged"
            ],
        )

        print(
            "episodes_completed:",
            summary[
                "episodes_completed"
            ],
        )

        print(
            "stop_reason:",
            summary[
                "stop_reason"
            ],
        )


    print()
    print(
        "=" * 86
    )
    print(
        "DQN SEED SENSITIVITY EXECUTION COMPLETE"
    )
    print(
        "=" * 86
    )

    print(
        "completed_or_verified_runs:",
        len(
            execution_rows
        ),
    )


if __name__ == "__main__":
    main()
