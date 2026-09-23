from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from smart_waste.experiments.empirical_workload import (
    EmpiricalWorkloadSpec,
    generate_empirical_workload,
)


CONFIG_PATH = Path(
    "configs/workload/"
    "empirical_wyndham.yaml"
)

OUTPUT_DIR = Path(
    "data/processed/workload/"
    "research_workloads"
)

MASTER_MANIFEST_PATH = (
    OUTPUT_DIR
    / "research_workloads_manifest.json"
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


def main() -> None:
    config = yaml.safe_load(
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )

    research = config[
        "research_design"
    ]

    source = config[
        "source"
    ]

    num_bins = int(
        research[
            "num_bins"
        ]
    )

    master_seed = int(
        research[
            "master_workload_seed"
        ]
    )

    replicate_ids = tuple(
        int(value)
        for value in research[
            "replicate_ids"
        ]
    )

    if replicate_ids != tuple(
        range(
            10
        )
    ):
        raise RuntimeError(
            "Frozen research design requires "
            "replicate IDs 0..9"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Remove only prior generated replicate JSON files
    # governed by this exact generator.
    for old_file in OUTPUT_DIR.glob(
        "workload_rep_*.json"
    ):
        old_file.unlink()

    workload_records = []

    print(
        "=== GENERATING RESEARCH WORKLOADS ==="
    )

    for replicate_id in replicate_ids:
        spec = EmpiricalWorkloadSpec(
            num_bins=num_bins,
            master_seed=master_seed,
            replicate_id=replicate_id,
            profile_library_path=(
                source[
                    "profile_library"
                ]
            ),
            profile_library_sha256=(
                source[
                    "profile_library_sha256"
                ]
            ),
        )

        workload = (
            generate_empirical_workload(
                spec
            )
        )

        initial = np.asarray(
            workload.initial_fill_percent,
            dtype=float,
        )

        rates = np.asarray(
            workload.fill_rate_percent_per_hour,
            dtype=float,
        )

        output_record = {
            "schema_version": (
                workload.schema_version
            ),
            "replicate_id": (
                replicate_id
            ),
            "num_bins": (
                num_bins
            ),
            "master_seed": (
                master_seed
            ),
            "sampling_seed": (
                workload.sampling_seed
            ),
            "source_profile_sha256": (
                workload.source_profile_sha256
            ),
            "workload_id": (
                workload.workload_id
            ),
            "workload_sha256": (
                workload.sha256
            ),
            "profile_ids": list(
                workload.profile_ids
            ),
            "initial_fill_percent": list(
                workload.initial_fill_percent
            ),
            "fill_rate_percent_per_hour": list(
                workload.fill_rate_percent_per_hour
            ),
            "summary": {
                "initial_fill_mean": (
                    float(
                        initial.mean()
                    )
                ),
                "initial_fill_median": (
                    float(
                        np.median(
                            initial
                        )
                    )
                ),
                "initial_ge_80_count": (
                    int(
                        (
                            initial
                            >= 80.0
                        ).sum()
                    )
                ),
                "initial_ge_80_fraction": (
                    float(
                        (
                            initial
                            >= 80.0
                        ).mean()
                    )
                ),
                "zero_growth_count": (
                    int(
                        (
                            rates
                            == 0.0
                        ).sum()
                    )
                ),
                "positive_growth_count": (
                    int(
                        (
                            rates
                            > 0.0
                        ).sum()
                    )
                ),
                "fill_rate_mean": (
                    float(
                        rates.mean()
                    )
                ),
                "fill_rate_median": (
                    float(
                        np.median(
                            rates
                        )
                    )
                ),
                "fill_rate_max": (
                    float(
                        rates.max()
                    )
                ),
            },
        }

        output_path = (
            OUTPUT_DIR
            / (
                "workload_rep_"
                f"{replicate_id:02d}.json"
            )
        )

        output_path.write_bytes(
            canonical_json_bytes(
                output_record
            )
        )

        file_sha256 = (
            sha256_file(
                output_path
            )
        )

        workload_records.append(
            {
                "replicate_id": (
                    replicate_id
                ),
                "file": (
                    str(
                        output_path
                    )
                ),
                "file_sha256": (
                    file_sha256
                ),
                "workload_id": (
                    workload.workload_id
                ),
                "workload_sha256": (
                    workload.sha256
                ),
                "sampling_seed": (
                    workload.sampling_seed
                ),
                "summary": (
                    output_record[
                        "summary"
                    ]
                ),
            }
        )

        print(
            f"replicate={replicate_id:02d}",
            f"id={workload.workload_id}",
            f"sha={workload.sha256}",
            f"eligible80={output_record['summary']['initial_ge_80_count']}",
            f"zero_rate={output_record['summary']['zero_growth_count']}",
        )

    workload_hashes = {
        record[
            "workload_sha256"
        ]
        for record in workload_records
    }

    if len(
        workload_hashes
    ) != len(
        replicate_ids
    ):
        raise RuntimeError(
            "Research workloads are not all distinct"
        )

    master_manifest = {
        "schema_version": (
            "research-workload-manifest-v1"
        ),
        "classification": (
            "FROZEN_INPUT_CANDIDATE"
        ),
        "config_file": (
            str(
                CONFIG_PATH
            )
        ),
        "config_sha256": (
            sha256_file(
                CONFIG_PATH
            )
        ),
        "profile_library": (
            source[
                "profile_library"
            ]
        ),
        "profile_library_sha256": (
            source[
                "profile_library_sha256"
            ]
        ),
        "raw_dataset_sha256": (
            source[
                "raw_dataset_sha256"
            ]
        ),
        "num_bins": (
            num_bins
        ),
        "master_workload_seed": (
            master_seed
        ),
        "replicate_count": (
            len(
                replicate_ids
            )
        ),
        "replicate_ids": list(
            replicate_ids
        ),
        "topology_fairness_contract": [
            "manhattan",
            "superblock",
            "hex",
            "radial_concentric",
        ],
        "workloads": (
            workload_records
        ),
    }

    MASTER_MANIFEST_PATH.write_bytes(
        canonical_json_bytes(
            master_manifest
        )
    )

    print()
    print(
        "master_manifest:",
        MASTER_MANIFEST_PATH,
    )

    print(
        "master_manifest_sha256:",
        sha256_file(
            MASTER_MANIFEST_PATH
        ),
    )

    print(
        "distinct_workload_shas:",
        len(
            workload_hashes
        ),
    )

    print(
        "GENERATION: PASS"
    )


if __name__ == "__main__":
    main()
