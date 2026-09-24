from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Final

import pandas as pd

from smart_waste.experiments.empirical_workload import (
    DEFAULT_PROFILE_LIBRARY,
    EXPECTED_PROFILE_LIBRARY_SHA256,
)
from smart_waste.experiments.hazard_overlay import (
    CHALLENGE_ORDER,
    CHALLENGE_TYPES,
    HAZARD_SEVERITY,
)
from smart_waste.experiments.research_scenario_matrix import (
    DEFAULT_WORKLOAD_MANIFEST,
)


DQN_DATASET_SCHEMA: Final = (
    "dqn-independent-training-dataset-v1"
)

DQN_DATASET_MANIFEST_SCHEMA: Final = (
    "dqn-independent-training-dataset-manifest-v1"
)

MASTER_DQN_DATASET_SEED: Final = 20260924

NUM_BINS: Final = 1000

TRAINING_WORKLOAD_COUNT: Final = 8
VALIDATION_WORKLOAD_COUNT: Final = 2

DEFAULT_OUTPUT_DIR = Path(
    "data/processed/rl/"
    "dqn_training_dataset"
)

DEFAULT_MANIFEST_PATH = (
    DEFAULT_OUTPUT_DIR
    / "dqn_training_dataset_manifest.json"
)


class DQNTrainingDatasetError(
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


def _sha256_payload(
    value: object,
) -> str:
    return hashlib.sha256(
        _canonical_json_bytes(
            value
        )
    ).hexdigest()


def _sha256_file(
    path: Path,
) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def _load_json(
    path: Path,
) -> dict:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def _load_profile_library(
) -> pd.DataFrame:
    path = Path(
        DEFAULT_PROFILE_LIBRARY
    )

    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    actual_sha = (
        _sha256_file(
            path
        )
    )

    if (
        actual_sha
        != EXPECTED_PROFILE_LIBRARY_SHA256
    ):
        raise DQNTrainingDatasetError(
            "empirical profile-library SHA mismatch"
        )

    frame = pd.read_csv(
        path
    )

    required = {
        "profile_id",
        "initial_fill_percent",
        "fill_rate_percent_per_hour",
    }

    missing = (
        required
        - set(
            frame.columns
        )
    )

    if missing:
        raise DQNTrainingDatasetError(
            f"profile library missing columns: "
            f"{sorted(missing)}"
        )

    return frame


def _confirmatory_profile_ids(
) -> tuple[int, ...]:
    manifest = _load_json(
        DEFAULT_WORKLOAD_MANIFEST
    )

    observed = set()

    workloads = manifest.get(
        "workloads",
        (),
    )

    if len(
        workloads
    ) != 10:
        raise DQNTrainingDatasetError(
            "expected exactly ten frozen "
            "confirmatory workloads"
        )

    for record in workloads:
        workload_path = Path(
            record[
                "file"
            ]
        )

        payload = _load_json(
            workload_path
        )

        observed.update(
            int(value)
            for value in payload[
                "profile_ids"
            ]
        )

    return tuple(
        sorted(
            observed
        )
    )


def _profile_rank(
    profile_id: int,
) -> bytes:
    payload = (
        f"{DQN_DATASET_SCHEMA}|"
        f"master={MASTER_DQN_DATASET_SEED}|"
        f"profile={profile_id}"
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        payload
    ).digest()


def _hazard_rank(
    *,
    split: str,
    slot_id: int,
    bin_id: int,
) -> bytes:
    payload = (
        f"{DQN_DATASET_SCHEMA}|"
        f"hazard|"
        f"master={MASTER_DQN_DATASET_SEED}|"
        f"split={split}|"
        f"slot={slot_id}|"
        f"bin={bin_id}"
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        payload
    ).digest()


def _hazard_bins(
    *,
    split: str,
    slot_id: int,
) -> tuple[int, int, int]:
    ranked = sorted(
        range(
            NUM_BINS
        ),
        key=lambda bin_id: (
            _hazard_rank(
                split=split,
                slot_id=slot_id,
                bin_id=bin_id,
            ),
            bin_id,
        ),
    )

    predictive = ranked[0]
    emergency = ranked[1]
    moisture = ranked[2]

    return (
        predictive,
        emergency,
        moisture,
    )


def _hazard_assignments(
    *,
    split: str,
    slot_id: int,
    challenge_id: str,
) -> tuple[
    dict[str, object],
    ...,
]:
    (
        predictive_bin,
        emergency_bin,
        moisture_bin,
    ) = _hazard_bins(
        split=split,
        slot_id=slot_id,
    )

    predictive = {
        "bin_id": predictive_bin,
        "hazard_type": (
            "predicted_combustion"
        ),
        "routing_severity": (
            HAZARD_SEVERITY[
                "predicted_combustion"
            ]
        ),
    }

    emergency = {
        "bin_id": emergency_bin,
        "hazard_type": (
            "emergency_fire"
        ),
        "routing_severity": (
            HAZARD_SEVERITY[
                "emergency_fire"
            ]
        ),
    }

    moisture = {
        "bin_id": moisture_bin,
        "hazard_type": (
            "excess_moisture"
        ),
        "routing_severity": (
            HAZARD_SEVERITY[
                "excess_moisture"
            ]
        ),
    }

    if challenge_id == "H0":
        return ()

    if challenge_id == "H1":
        return (
            predictive,
        )

    if challenge_id == "H2":
        return (
            emergency,
        )

    if challenge_id == "H3":
        return (
            emergency,
            predictive,
            moisture,
        )

    raise DQNTrainingDatasetError(
        f"unknown challenge_id={challenge_id}"
    )


@dataclass(frozen=True)
class IndependentDQNWorkload:
    split: str
    slot_id: int

    workload_id: str
    workload_sha256: str

    profile_ids: tuple[
        int,
        ...,
    ]

    initial_fill_percent: tuple[
        float,
        ...,
    ]

    fill_rate_percent_per_hour: tuple[
        float,
        ...,
    ]

    hazard_assignments: dict[
        str,
        tuple[
            dict[str, object],
            ...,
        ],
    ]

    def hazard_severity_by_bin(
        self,
        challenge_id: str,
    ) -> dict[int, float]:
        if (
            challenge_id
            not in self.hazard_assignments
        ):
            raise DQNTrainingDatasetError(
                "unknown challenge"
            )

        return {
            int(
                assignment[
                    "bin_id"
                ]
            ): float(
                assignment[
                    "routing_severity"
                ]
            )
            for assignment
            in self.hazard_assignments[
                challenge_id
            ]
        }


def _workload_record(
    *,
    frame: pd.DataFrame,
    split: str,
    slot_id: int,
    profile_ids: tuple[int, ...],
) -> dict[str, object]:
    identity = {
        "schema_version": (
            DQN_DATASET_SCHEMA
        ),
        "master_seed": (
            MASTER_DQN_DATASET_SEED
        ),
        "split": split,
        "slot_id": slot_id,
        "num_bins": NUM_BINS,
        "profile_library_sha256": (
            EXPECTED_PROFILE_LIBRARY_SHA256
        ),
        "profile_ids": list(
            profile_ids
        ),
    }

    workload_sha = (
        _sha256_payload(
            identity
        )
    )

    hazards = {}

    for challenge_id in (
        CHALLENGE_ORDER
    ):
        hazards[
            challenge_id
        ] = list(
            _hazard_assignments(
                split=split,
                slot_id=slot_id,
                challenge_id=challenge_id,
            )
        )

    return {
        **identity,
        "workload_id": (
            f"dqn-{split}-"
            f"{slot_id:02d}-"
            f"{workload_sha[:12]}"
        ),
        "workload_sha256": (
            workload_sha
        ),
        "hazard_challenges": (
            hazards
        ),
    }


def build_dqn_training_dataset_manifest(
) -> dict[str, object]:
    frame = _load_profile_library()

    all_profile_ids = tuple(
        int(value)
        for value in frame[
            "profile_id"
        ]
    )

    confirmatory_ids = (
        _confirmatory_profile_ids()
    )

    confirmatory_set = set(
        confirmatory_ids
    )

    eligible = [
        profile_id
        for profile_id
        in all_profile_ids
        if (
            profile_id
            not in confirmatory_set
        )
    ]

    eligible.sort(
        key=lambda profile_id: (
            _profile_rank(
                profile_id
            ),
            profile_id,
        )
    )

    required = (
        (
            TRAINING_WORKLOAD_COUNT
            + VALIDATION_WORKLOAD_COUNT
        )
        * NUM_BINS
    )

    if len(
        eligible
    ) < required:
        raise DQNTrainingDatasetError(
            "not enough empirical profiles remain "
            "after confirmatory exclusion"
        )

    selected = tuple(
        eligible[
            :required
        ]
    )

    if len(
        set(
            selected
        )
    ) != required:
        raise DQNTrainingDatasetError(
            "independent dataset contains duplicates"
        )

    records = []

    cursor = 0

    for slot_id in range(
        TRAINING_WORKLOAD_COUNT
    ):
        ids = tuple(
            selected[
                cursor:
                cursor + NUM_BINS
            ]
        )

        cursor += NUM_BINS

        records.append(
            _workload_record(
                frame=frame,
                split="training",
                slot_id=slot_id,
                profile_ids=ids,
            )
        )

    for slot_id in range(
        VALIDATION_WORKLOAD_COUNT
    ):
        ids = tuple(
            selected[
                cursor:
                cursor + NUM_BINS
            ]
        )

        cursor += NUM_BINS

        records.append(
            _workload_record(
                frame=frame,
                split="validation",
                slot_id=slot_id,
                profile_ids=ids,
            )
        )

    confirmatory_identity_sha = (
        _sha256_payload(
            list(
                confirmatory_ids
            )
        )
    )

    dataset_identity = {
        "schema_version": (
            DQN_DATASET_MANIFEST_SCHEMA
        ),
        "dataset_schema_version": (
            DQN_DATASET_SCHEMA
        ),
        "master_seed": (
            MASTER_DQN_DATASET_SEED
        ),
        "profile_library": str(
            DEFAULT_PROFILE_LIBRARY
        ),
        "profile_library_sha256": (
            EXPECTED_PROFILE_LIBRARY_SHA256
        ),
        "num_bins": NUM_BINS,
        "training_workload_count": (
            TRAINING_WORKLOAD_COUNT
        ),
        "validation_workload_count": (
            VALIDATION_WORKLOAD_COUNT
        ),
        "confirmatory_profile_union_count": (
            len(
                confirmatory_ids
            )
        ),
        "confirmatory_profile_union_sha256": (
            confirmatory_identity_sha
        ),
        "selection_contract": (
            "exclude_all_confirmatory_profile_ids_"
            "then_deterministic_hash_rank_"
            "then_disjoint_partition"
        ),
        "hazard_contract": (
            "deterministic_training_only_controlled_"
            "H0_H1_H2_H3_challenges_"
            "independent_of_confirmatory_hazard_overlay"
        ),
        "challenge_order": list(
            CHALLENGE_ORDER
        ),
        "challenge_types": {
            challenge_id: (
                CHALLENGE_TYPES[
                    challenge_id
                ]
            )
            for challenge_id
            in CHALLENGE_ORDER
        },
        "workloads": records,
    }

    manifest_sha = (
        _sha256_payload(
            dataset_identity
        )
    )

    return {
        **dataset_identity,
        "manifest_sha256": (
            manifest_sha
        ),
    }


def workload_from_record(
    record: dict[str, object],
) -> IndependentDQNWorkload:
    frame = _load_profile_library()

    indexed = frame.set_index(
        "profile_id"
    )

    profile_ids = tuple(
        int(value)
        for value in record[
            "profile_ids"
        ]
    )

    selected = indexed.loc[
        list(
            profile_ids
        )
    ]

    initial_fill = tuple(
        float(value)
        for value in selected[
            "initial_fill_percent"
        ]
    )

    fill_rate = tuple(
        float(value)
        for value in selected[
            "fill_rate_percent_per_hour"
        ]
    )

    hazard_assignments = {
        challenge_id: tuple(
            dict(
                assignment
            )
            for assignment
            in record[
                "hazard_challenges"
            ][
                challenge_id
            ]
        )
        for challenge_id
        in CHALLENGE_ORDER
    }

    return IndependentDQNWorkload(
        split=str(
            record[
                "split"
            ]
        ),
        slot_id=int(
            record[
                "slot_id"
            ]
        ),
        workload_id=str(
            record[
                "workload_id"
            ]
        ),
        workload_sha256=str(
            record[
                "workload_sha256"
            ]
        ),
        profile_ids=profile_ids,
        initial_fill_percent=(
            initial_fill
        ),
        fill_rate_percent_per_hour=(
            fill_rate
        ),
        hazard_assignments=(
            hazard_assignments
        ),
    )


def write_dqn_training_dataset_manifest(
    path: Path = DEFAULT_MANIFEST_PATH,
) -> dict[str, object]:
    manifest = (
        build_dqn_training_dataset_manifest()
    )

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return manifest


def main() -> None:
    manifest = (
        write_dqn_training_dataset_manifest()
    )

    print(
        "schema_version:",
        manifest[
            "schema_version"
        ],
    )

    print(
        "training_workloads:",
        manifest[
            "training_workload_count"
        ],
    )

    print(
        "validation_workloads:",
        manifest[
            "validation_workload_count"
        ],
    )

    print(
        "confirmatory_profile_union_count:",
        manifest[
            "confirmatory_profile_union_count"
        ],
    )

    print(
        "manifest_sha256:",
        manifest[
            "manifest_sha256"
        ],
    )


if __name__ == "__main__":
    main()
