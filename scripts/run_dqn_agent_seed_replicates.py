from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml


PLAN_PATH = Path(
    "configs/dqn_agent_seed_replicates.yaml"
)

DEFAULT_TEMPLATE = Path(
    "configs/dqn_convergence.yaml"
)

CONVERGENCE_RUNNER = Path(
    "scripts/run_dqn_convergence.py"
)

RAW_ROOT = Path(
    "results/raw/dqn_agent_seed_replicates"
)

PROCESSED_ROOT = Path(
    "results/processed/dqn_agent_seed_replicates"
)

MODEL_ROOT = Path(
    "results/models/dqn_agent_seed_replicates"
)

SUMMARY_PATH = (
    PROCESSED_ROOT
    / "replicate_summary.csv"
)


def load_yaml(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def save_yaml(
    data: dict,
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            data,
            file,
            sort_keys=False,
            allow_unicode=True,
        )


def sha256(path: Path) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as file:

        for block in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def seed_paths(
    seed: int,
) -> dict[str, Path]:

    name = f"seed_{seed}"

    raw_dir = RAW_ROOT / name
    processed_dir = (
        PROCESSED_ROOT / name
    )
    model_dir = MODEL_ROOT / name

    return {
        "raw_dir":
            raw_dir,

        "processed_dir":
            processed_dir,

        "model_dir":
            model_dir,

        "episodes":
            raw_dir
            / "training_episodes.csv",

        "evaluations":
            raw_dir
            / "greedy_evaluations.csv",

        "checks":
            processed_dir
            / "convergence_checks.csv",

        "protocol_snapshot":
            processed_dir
            / "protocol_snapshot.yaml",

        "run_config":
            processed_dir
            / "run_config.yaml",

        "completion":
            processed_dir
            / "completion.json",

        "model":
            model_dir
            / "final.pt",

        "checkpoints":
            model_dir
            / "checkpoints",
    }


def build_seed_config(
    template: dict,
    *,
    seed: int,
    city_master_seed: int,
) -> dict:

    cfg = copy.deepcopy(
        template
    )

    protocol = cfg[
        "protocol"
    ]

    protocol[
        "name"
    ] = (
        "dqn_confirmatory_"
        f"agent_seed_{seed}"
    )

    protocol[
        "agent_seed"
    ] = int(seed)

    protocol[
        "city_master_seed"
    ] = int(
        city_master_seed
    )

    paths = seed_paths(seed)

    cfg["outputs"] = {
        "episode_metrics_csv":
            str(paths["episodes"]),

        "greedy_evaluations_csv":
            str(paths["evaluations"]),

        "convergence_checks_csv":
            str(paths["checks"]),

        "model_checkpoint":
            str(paths["model"]),

        "checkpoint_directory":
            str(paths["checkpoints"]),

        "protocol_snapshot":
            str(
                paths[
                    "protocol_snapshot"
                ]
            ),
    }

    return cfg


def final_artifacts_exist(
    paths: dict[str, Path],
) -> bool:

    required = [
        paths["episodes"],
        paths["evaluations"],
        paths["checks"],
        paths["model"],
        paths["protocol_snapshot"],
    ]

    return all(
        path.is_file()
        for path in required
    )


def summarize_completed_run(
    *,
    seed: int,
    paths: dict[str, Path],
) -> dict:

    if not final_artifacts_exist(
        paths
    ):
        raise RuntimeError(
            f"Seed {seed}: final artifacts "
            "are incomplete"
        )

    episodes = pd.read_csv(
        paths["episodes"]
    )

    evaluations = pd.read_csv(
        paths["evaluations"]
    )

    checks = pd.read_csv(
        paths["checks"]
    )

    if len(episodes) == 0:
        raise RuntimeError(
            f"Seed {seed}: empty episode file"
        )

    if len(evaluations) == 0:
        raise RuntimeError(
            f"Seed {seed}: empty evaluation file"
        )

    checkpoint = torch.load(
        paths["model"],
        map_location="cpu",
        weights_only=False,
    )

    final_eval = (
        evaluations.iloc[-1]
    )

    converged = bool(
        checkpoint.get(
            "converged",
            False,
        )
    )

    stop_reason = str(
        checkpoint.get(
            "stop_reason",
            "unknown",
        )
    )

    episodes_completed = int(
        checkpoint.get(
            "episodes_completed",
            len(episodes),
        )
    )

    if (
        episodes_completed
        != len(episodes)
    ):
        raise RuntimeError(
            f"Seed {seed}: checkpoint says "
            f"{episodes_completed} episodes "
            f"but CSV has {len(episodes)}"
        )

    if not bool(
        episodes[
            "terminated"
        ].all()
    ):
        raise RuntimeError(
            f"Seed {seed}: at least one "
            "training episode did not "
            "terminate naturally"
        )

    if bool(
        episodes[
            "truncated"
        ].any()
    ):
        raise RuntimeError(
            f"Seed {seed}: at least one "
            "training episode truncated"
        )

    summary = {
        "agent_seed":
            int(seed),

        "completed":
            True,

        "converged":
            converged,

        "stop_reason":
            stop_reason,

        "episodes_completed":
            episodes_completed,

        "greedy_evaluations":
            int(
                len(evaluations)
            ),

        "convergence_checks":
            int(
                len(checks)
            ),

        "final_reward":
            float(
                final_eval[
                    "total_reward"
                ]
            ),

        "final_distance_km":
            float(
                final_eval[
                    "distance_km"
                ]
            ),

        "final_fuel_litres":
            float(
                final_eval[
                    "fuel_litres"
                ]
            ),

        "final_simulated_time_hours":
            float(
                final_eval[
                    "simulated_time_hours"
                ]
            ),

        "final_terminated":
            bool(
                final_eval[
                    "terminated"
                ]
            ),

        "final_truncated":
            bool(
                final_eval[
                    "truncated"
                ]
            ),

        "episodes_sha256":
            sha256(
                paths["episodes"]
            ),

        "evaluations_sha256":
            sha256(
                paths["evaluations"]
            ),

        "checks_sha256":
            sha256(
                paths["checks"]
            ),

        "model_sha256":
            sha256(
                paths["model"]
            ),

        "protocol_sha256":
            sha256(
                paths[
                    "protocol_snapshot"
                ]
            ),
    }

    paths[
        "completion"
    ].parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with paths[
        "completion"
    ].open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            sort_keys=True,
        )

    return summary


def load_completion(
    paths: dict[str, Path],
) -> dict | None:

    completion = paths[
        "completion"
    ]

    if not completion.is_file():
        return None

    if not final_artifacts_exist(
        paths
    ):
        return None

    with completion.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def clean_incomplete_seed(
    paths: dict[str, Path],
) -> None:

    for key in (
        "raw_dir",
        "processed_dir",
        "model_dir",
    ):

        directory = paths[key]

        if directory.exists():
            shutil.rmtree(
                directory
            )


def collect_completed(
    seeds: list[int],
) -> list[dict]:

    records: list[dict] = []

    for seed in seeds:

        paths = seed_paths(seed)

        record = load_completion(
            paths
        )

        if record is None:

            if final_artifacts_exist(
                paths
            ):
                record = (
                    summarize_completed_run(
                        seed=seed,
                        paths=paths,
                    )
                )

        if record is not None:
            records.append(
                record
            )

    return records


def write_summary(
    seeds: list[int],
) -> None:

    records = collect_completed(
        seeds
    )

    PROCESSED_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if records:

        df = pd.DataFrame(
            records
        ).sort_values(
            "agent_seed"
        )

        df.to_csv(
            SUMMARY_PATH,
            index=False,
        )

    else:

        pd.DataFrame(
            columns=[
                "agent_seed",
                "completed",
                "converged",
                "stop_reason",
                "episodes_completed",
                "final_reward",
                "final_distance_km",
                "final_fuel_litres",
            ]
        ).to_csv(
            SUMMARY_PATH,
            index=False,
        )


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Show planned replicates "
            "without training."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Run only one planned "
            "agent seed."
        ),
    )

    args = parser.parse_args()

    plan = load_yaml(
        PLAN_PATH
    )

    experiment = plan[
        "experiment"
    ]

    seeds = [
        int(value)
        for value
        in experiment[
            "agent_seeds"
        ]
    ]

    expected_count = int(
        experiment[
            "planned_replicates"
        ]
    )

    if len(seeds) != expected_count:
        raise RuntimeError(
            "planned_replicates does "
            "not match seed count"
        )

    if len(set(seeds)) != len(seeds):
        raise RuntimeError(
            "duplicate agent seeds"
        )

    development_seed = int(
        experiment[
            "development_seed_excluded_"
            "from_confirmatory_statistics"
        ]
    )

    if development_seed in seeds:
        raise RuntimeError(
            "development seed must not "
            "appear in confirmatory seeds"
        )

    city_master_seed = int(
        experiment[
            "fixed_city_master_seed"
        ]
    )

    template_path = Path(
        experiment.get(
            "template_config",
            DEFAULT_TEMPLATE,
        )
    )

    template = load_yaml(
        template_path
    )

    selected_seeds = seeds

    if args.seed is not None:

        if args.seed not in seeds:
            raise SystemExit(
                f"Seed {args.seed} is not "
                "in the planned seed list"
            )

        selected_seeds = [
            args.seed
        ]

    print("=" * 96)
    print(
        "CONFIRMATORY DQN AGENT-SEED REPLICATION"
    )
    print("=" * 96)

    print(
        "Development seed excluded :",
        development_seed,
    )

    print(
        "Fixed city/workload seed  :",
        city_master_seed,
    )

    print(
        "Planned replicates        :",
        len(seeds),
    )

    print(
        "Selected this invocation  :",
        len(selected_seeds),
    )

    print(
        "Agent seeds               :",
        selected_seeds,
    )

    print(
        "Resume granularity        : "
        "completed replicate"
    )

    print("-" * 96)

    if args.dry_run:

        for seed in selected_seeds:

            paths = seed_paths(
                seed
            )

            if load_completion(
                paths
            ) is not None:

                status = "COMPLETED"

            elif final_artifacts_exist(
                paths
            ):

                status = (
                    "FINAL ARTIFACTS PRESENT"
                )

            elif (
                paths["raw_dir"].exists()
                or paths[
                    "processed_dir"
                ].exists()
                or paths[
                    "model_dir"
                ].exists()
            ):

                status = (
                    "INCOMPLETE - WILL RESTART"
                )

            else:

                status = "PENDING"

            print(
                f"seed={seed}: {status}"
            )

        return

    for index, seed in enumerate(
        selected_seeds,
        start=1,
    ):

        paths = seed_paths(seed)

        completed = load_completion(
            paths
        )

        if completed is not None:

            print(
                f"[{index}/{len(selected_seeds)}] "
                f"seed={seed}: already completed; "
                "skipping"
            )

            continue

        if final_artifacts_exist(
            paths
        ):

            summarize_completed_run(
                seed=seed,
                paths=paths,
            )

            write_summary(seeds)

            print(
                f"[{index}/{len(selected_seeds)}] "
                f"seed={seed}: recovered completed "
                "run; skipping retraining"
            )

            continue

        print()
        print("=" * 96)

        print(
            f"[{index}/{len(selected_seeds)}] "
            f"START seed={seed}"
        )

        print("=" * 96)

        # Exact replay-buffer/RNG state is not
        # currently serialized. Therefore an
        # interrupted replicate is restarted
        # from episode 0, while all previously
        # completed replicates are preserved.
        clean_incomplete_seed(
            paths
        )

        cfg = build_seed_config(
            template,
            seed=seed,
            city_master_seed=(
                city_master_seed
            ),
        )

        save_yaml(
            cfg,
            paths["run_config"],
        )

        command = [
            sys.executable,
            str(
                CONVERGENCE_RUNNER
            ),
            "--config",
            str(
                paths["run_config"]
            ),
        ]

        print(
            "Command:",
            " ".join(command),
        )

        subprocess.run(
            command,
            check=True,
        )

        summary = (
            summarize_completed_run(
                seed=seed,
                paths=paths,
            )
        )

        write_summary(
            seeds
        )

        print(
            f"seed={seed} COMPLETE: "
            f"episodes="
            f"{summary['episodes_completed']}  "
            f"converged="
            f"{summary['converged']}  "
            f"fuel="
            f"{summary['final_fuel_litres']:.6f} L"
        )

    write_summary(
        seeds
    )

    completed_records = (
        collect_completed(
            seeds
        )
    )

    print()
    print("=" * 96)
    print(
        "REPLICATION STATUS"
    )
    print("=" * 96)

    print(
        "Completed :",
        len(completed_records),
        "/",
        len(seeds),
    )

    print(
        "Summary   :",
        SUMMARY_PATH,
    )

    if len(
        completed_records
    ) == len(seeds):

        converged_count = sum(
            bool(
                record[
                    "converged"
                ]
            )
            for record
            in completed_records
        )

        print(
            "Converged :",
            converged_count,
            "/",
            len(seeds),
        )

        print(
            "All planned confirmatory "
            "replicates completed."
        )

    print("=" * 96)


if __name__ == "__main__":
    main()
