from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Final

from smart_waste.experiments.dqn_training_dataset import (
    DEFAULT_MANIFEST_PATH,
    IndependentDQNWorkload,
    workload_from_record,
)
from smart_waste.experiments.research_scenario_matrix import (
    FROZEN_TOPOLOGY_ORDER,
    build_research_scenario_matrix,
)
from smart_waste.experiments.scenario_snapshot import (
    PhysicalScenarioSnapshot,
    build_simulation_state,
    capture_physical_scenario,
)


PHYSICAL_TRAINING_CASE_SCHEMA: Final = (
    "dqn-independent-physical-case-v1"
)

PHYSICAL_TRAINING_CASE_MANIFEST_SCHEMA: Final = (
    "dqn-independent-physical-case-manifest-v1"
)

EXPECTED_DATASET_MANIFEST_SHA256: Final = (
    "18bbf059305886970d3bcfc363a2140a4fe32d837901bd3ccd67af8d24384133"
)

DEFAULT_OUTPUT_PATH = Path(
    "data/processed/rl/"
    "dqn_training_physical_cases_manifest.json"
)


class DQNPhysicalTrainingCaseError(
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


def _load_dataset_manifest() -> dict:
    path = Path(
        DEFAULT_MANIFEST_PATH
    )

    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    manifest = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    embedded_sha = manifest.get(
        "manifest_sha256"
    )

    if (
        embedded_sha
        != EXPECTED_DATASET_MANIFEST_SHA256
    ):
        raise DQNPhysicalTrainingCaseError(
            "unexpected frozen DQN dataset manifest SHA"
        )

    core = dict(
        manifest
    )

    core.pop(
        "manifest_sha256",
        None,
    )

    actual_sha = _sha256_payload(
        core
    )

    if (
        actual_sha
        != EXPECTED_DATASET_MANIFEST_SHA256
    ):
        raise DQNPhysicalTrainingCaseError(
            "DQN dataset manifest payload SHA mismatch"
        )

    return manifest


@dataclass(frozen=True)
class DQNPhysicalTrainingCase:
    topology_id: str

    split: str
    slot_id: int

    workload: IndependentDQNWorkload

    reference_physical_scenario_id: str
    reference_physical_scenario_sha256: str

    snapshot: PhysicalScenarioSnapshot

    @property
    def case_id(self) -> str:
        return (
            f"dqn-{self.split}-"
            f"{self.topology_id}-"
            f"{self.slot_id:02d}-"
            f"{self.snapshot.sha256[:12]}"
        )

    @property
    def case_sha256(self) -> str:
        payload = {
            "schema_version": (
                PHYSICAL_TRAINING_CASE_SCHEMA
            ),
            "topology_id": (
                self.topology_id
            ),
            "split": (
                self.split
            ),
            "slot_id": (
                self.slot_id
            ),
            "workload_id": (
                self.workload.workload_id
            ),
            "workload_sha256": (
                self.workload.workload_sha256
            ),
            "reference_physical_scenario_id": (
                self.reference_physical_scenario_id
            ),
            "reference_physical_scenario_sha256": (
                self.reference_physical_scenario_sha256
            ),
            "snapshot_sha256": (
                self.snapshot.sha256
            ),
        }

        return _sha256_payload(
            payload
        )

    def hazard_severity_by_bin(
        self,
        challenge_id: str,
    ) -> dict[int, float]:
        return (
            self.workload
            .hazard_severity_by_bin(
                challenge_id
            )
        )


def _reference_cases_by_topology():
    research_cases = (
        build_research_scenario_matrix()
    )

    result = {}

    for topology_id in (
        FROZEN_TOPOLOGY_ORDER
    ):
        matches = [
            case
            for case in research_cases
            if (
                case.topology_id
                == topology_id
                and case.replicate_id
                == 0
            )
        ]

        if len(
            matches
        ) != 1:
            raise DQNPhysicalTrainingCaseError(
                "expected exactly one replicate-0 "
                f"reference case for {topology_id}"
            )

        result[
            topology_id
        ] = matches[
            0
        ]

    return result


def _apply_independent_workload(
    *,
    reference_snapshot: PhysicalScenarioSnapshot,
    workload: IndependentDQNWorkload,
) -> PhysicalScenarioSnapshot:
    state = build_simulation_state(
        reference_snapshot
    )

    bin_ids = tuple(
        sorted(
            state.bins
        )
    )

    if (
        bin_ids
        != tuple(
            range(
                1000
            )
        )
    ):
        raise DQNPhysicalTrainingCaseError(
            "physical DQN requires bin IDs 0..999"
        )

    if (
        len(
            workload.initial_fill_percent
        )
        != 1000
        or len(
            workload.fill_rate_percent_per_hour
        )
        != 1000
    ):
        raise DQNPhysicalTrainingCaseError(
            "independent workload must contain "
            "exactly 1000 bins"
        )

    if (
        state.current_time_hours
        != 0.0
    ):
        raise DQNPhysicalTrainingCaseError(
            "reference physical scenario is not pristine"
        )

    for bin_id in bin_ids:
        bin_ = state.bins[
            bin_id
        ]

        fill = float(
            workload.initial_fill_percent[
                bin_id
            ]
        )

        rate = float(
            workload.fill_rate_percent_per_hour[
                bin_id
            ]
        )

        if not (
            0.0
            <= fill
            <= 100.0
        ):
            raise DQNPhysicalTrainingCaseError(
                "independent fill outside [0, 100]"
            )

        if rate < 0.0:
            raise DQNPhysicalTrainingCaseError(
                "negative independent fill rate"
            )

        bin_.fill_percent = fill
        bin_.fill_rate_percent_per_hour = (
            rate
        )

        if hasattr(
            bin_,
            "waste_age_hours",
        ):
            bin_.waste_age_hours = 0.0

    state.validate_physical_invariants()

    snapshot = (
        capture_physical_scenario(
            state
        )
    )

    return snapshot


def build_dqn_physical_training_cases(
) -> tuple[
    DQNPhysicalTrainingCase,
    ...,
]:
    manifest = (
        _load_dataset_manifest()
    )

    references = (
        _reference_cases_by_topology()
    )

    workload_records = (
        manifest[
            "workloads"
        ]
    )

    cases = []

    for topology_id in (
        FROZEN_TOPOLOGY_ORDER
    ):
        reference = references[
            topology_id
        ]

        for record in workload_records:
            workload = (
                workload_from_record(
                    record
                )
            )

            snapshot = (
                _apply_independent_workload(
                    reference_snapshot=(
                        reference
                        .assembly
                        .snapshot
                    ),
                    workload=workload,
                )
            )

            cases.append(
                DQNPhysicalTrainingCase(
                    topology_id=(
                        topology_id
                    ),
                    split=(
                        workload.split
                    ),
                    slot_id=(
                        workload.slot_id
                    ),
                    workload=workload,
                    reference_physical_scenario_id=(
                        reference
                        .assembly
                        .scenario_id
                    ),
                    reference_physical_scenario_sha256=(
                        reference
                        .assembly
                        .scenario_sha256
                    ),
                    snapshot=snapshot,
                )
            )

    return tuple(
        cases
    )


def cases_for_topology(
    topology_id: str,
    *,
    split: str | None = None,
) -> tuple[
    DQNPhysicalTrainingCase,
    ...,
]:
    if (
        topology_id
        not in FROZEN_TOPOLOGY_ORDER
    ):
        raise ValueError(
            f"unknown topology_id={topology_id}"
        )

    cases = [
        case
        for case
        in build_dqn_physical_training_cases()
        if (
            case.topology_id
            == topology_id
        )
    ]

    if split is not None:
        if split not in {
            "training",
            "validation",
        }:
            raise ValueError(
                "split must be training or validation"
            )

        cases = [
            case
            for case in cases
            if (
                case.split
                == split
            )
        ]

    return tuple(
        cases
    )


def build_physical_case_manifest(
) -> dict[str, object]:
    cases = (
        build_dqn_physical_training_cases()
    )

    records = []

    for case in cases:
        records.append(
            {
                "case_id": (
                    case.case_id
                ),
                "case_sha256": (
                    case.case_sha256
                ),
                "topology_id": (
                    case.topology_id
                ),
                "split": (
                    case.split
                ),
                "slot_id": (
                    case.slot_id
                ),
                "workload_id": (
                    case.workload.workload_id
                ),
                "workload_sha256": (
                    case.workload.workload_sha256
                ),
                "reference_physical_scenario_id": (
                    case
                    .reference_physical_scenario_id
                ),
                "reference_physical_scenario_sha256": (
                    case
                    .reference_physical_scenario_sha256
                ),
                "snapshot_id": (
                    case.snapshot.scenario_id
                ),
                "snapshot_sha256": (
                    case.snapshot.sha256
                ),
            }
        )

    core = {
        "schema_version": (
            PHYSICAL_TRAINING_CASE_MANIFEST_SCHEMA
        ),
        "dataset_manifest_sha256": (
            EXPECTED_DATASET_MANIFEST_SHA256
        ),
        "topology_order": list(
            FROZEN_TOPOLOGY_ORDER
        ),
        "case_count": len(
            records
        ),
        "training_cases_per_topology": 8,
        "validation_cases_per_topology": 2,
        "cases": records,
    }

    manifest_sha = (
        _sha256_payload(
            core
        )
    )

    return {
        **core,
        "manifest_sha256": (
            manifest_sha
        ),
    }


def write_physical_case_manifest(
    path: Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, object]:
    manifest = (
        build_physical_case_manifest()
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
        write_physical_case_manifest()
    )

    print(
        "schema_version:",
        manifest[
            "schema_version"
        ],
    )

    print(
        "case_count:",
        manifest[
            "case_count"
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
