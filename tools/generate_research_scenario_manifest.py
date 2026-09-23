from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

from smart_waste.experiments.research_scenario_matrix import (
    DEFAULT_WORKLOAD_MANIFEST,
    FROZEN_TOPOLOGY_ORDER,
    RESEARCH_SCENARIO_SCHEMA,
    build_research_scenario_matrix,
)


OUTPUT_DIR = Path(
    "data/processed/scenarios"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "research_physical_scenarios_manifest.json"
)


def sha256_file(
    path: Path,
) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def canonical_json_bytes(
    value: object,
) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode(
        "utf-8"
    )


def git_output(
    *args: str,
) -> str:
    return subprocess.check_output(
        [
            "git",
            *args,
        ],
        text=True,
    ).strip()


def main() -> None:
    status = git_output(
        "status",
        "--porcelain",
    )

    if status:
        raise RuntimeError(
            "Generate the frozen scenario manifest "
            "only from a clean Git working tree.\n"
            f"{status}"
        )

    git_commit = git_output(
        "rev-parse",
        "HEAD",
    )

    cases = (
        build_research_scenario_matrix()
    )

    records = []

    for case in cases:
        assembly = (
            case.assembly
        )

        snapshot = (
            assembly.snapshot
        )

        records.append(
            {
                "case_id": (
                    case.case_id
                ),
                "topology_id": (
                    case.topology_id
                ),
                "topology_index": (
                    case.topology_index
                ),
                "replicate_id": (
                    case.replicate_id
                ),
                "workload_id": (
                    case.workload.workload_id
                ),
                "workload_sha256": (
                    case.workload.sha256
                ),
                "scenario_id": (
                    assembly.scenario_id
                ),
                "scenario_sha256": (
                    assembly.scenario_sha256
                ),
                "road_node_count": (
                    len(
                        snapshot.road_nodes
                    )
                ),
                "road_edge_count": (
                    len(
                        snapshot.road_edges
                    )
                ),
                "bin_count": (
                    len(
                        snapshot.bins
                    )
                ),
                "truck_count": (
                    len(
                        snapshot.trucks
                    )
                ),
                "depot_road_node": (
                    snapshot.depot.road_node
                ),
                "depot_unloading_bays": (
                    snapshot.depot.unloading_bays
                ),
            }
        )

    workload_manifest_sha = (
        sha256_file(
            DEFAULT_WORKLOAD_MANIFEST
        )
    )

    manifest = {
        "schema_version": (
            RESEARCH_SCENARIO_SCHEMA
        ),
        "classification": (
            "FROZEN_PHYSICAL_INPUT_CANDIDATE"
        ),
        "generator_git_commit": (
            git_commit
        ),
        "workload_manifest_file": (
            str(
                DEFAULT_WORKLOAD_MANIFEST
            )
        ),
        "workload_manifest_sha256": (
            workload_manifest_sha
        ),
        "topology_order": list(
            FROZEN_TOPOLOGY_ORDER
        ),
        "topology_count": 4,
        "replicate_count": 10,
        "scenario_count": 40,
        "ordering": (
            "replicate-major then frozen topology order"
        ),
        "scenarios": (
            records
        ),
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_bytes(
        canonical_json_bytes(
            manifest
        )
    )

    output_sha = sha256_file(
        OUTPUT_PATH
    )

    print(
        "=== RESEARCH PHYSICAL SCENARIO MANIFEST ==="
    )
    print(
        "git_commit:",
        git_commit,
    )
    print(
        "workload_manifest_sha256:",
        workload_manifest_sha,
    )
    print(
        "scenario_count:",
        len(
            records
        ),
    )
    print(
        "unique_scenario_shas:",
        len(
            {
                record[
                    "scenario_sha256"
                ]
                for record in records
            }
        ),
    )

    for replicate_id in range(
        10
    ):
        replicate_records = [
            record
            for record in records
            if record[
                "replicate_id"
            ] == replicate_id
        ]

        workload_shas = {
            record[
                "workload_sha256"
            ]
            for record in replicate_records
        }

        print(
            f"replicate={replicate_id:02d}",
            f"cases={len(replicate_records)}",
            f"workload_shas={len(workload_shas)}",
        )

    print(
        "manifest:",
        OUTPUT_PATH,
    )
    print(
        "manifest_sha256:",
        output_sha,
    )
    print(
        "GENERATION: PASS"
    )


if __name__ == "__main__":
    main()
