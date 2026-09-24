from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Final

from smart_waste.experiments.research_scenario_matrix import (
    FROZEN_TOPOLOGY_ORDER,
    build_research_scenario_matrix,
    load_frozen_research_workload,
)


HAZARD_OVERLAY_SCHEMA: Final = (
    "research-hazard-overlay-v1"
)

HAZARD_MANIFEST_SCHEMA: Final = (
    "research-hazard-overlay-manifest-v1"
)

COMPOSITE_SCENARIO_SCHEMA: Final = (
    "physical-hazard-composite-v1"
)

HAZARD_OVERLAY_VERSION: Final = "1.0.0"

# Reproducibility seed only.
#
# It has no physical interpretation and does not define a hazard
# prevalence or spontaneous-event probability.
MASTER_HAZARD_SEED: Final = 20260924

NUM_BINS: Final = 1000

CHALLENGE_ORDER: Final = (
    "H0",
    "H1",
    "H2",
    "H3",
)

CHALLENGE_TYPES: Final = {
    "H0": "nominal",
    "H1": "predictive_combustion",
    "H2": "emergency_fire",
    "H3": "mixed_hazard",
}

HAZARD_SEVERITY: Final = {
    "excess_moisture": (
        1.0
        / 3.0
    ),
    "predicted_combustion": (
        2.0
        / 3.0
    ),
    "emergency_fire": 1.0,
}

DEFAULT_OUTPUT_DIR = Path(
    "data/processed/hazard/"
    "research_overlays"
)

DEFAULT_MANIFEST_PATH = (
    DEFAULT_OUTPUT_DIR
    / "research_hazard_overlays_manifest.json"
)


class HazardOverlayError(RuntimeError):
    """Raised when the frozen hazard-overlay contract is invalid."""


def _canonical_json_bytes(
    payload: object,
) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        ensure_ascii=False,
        allow_nan=False,
    ).encode(
        "utf-8"
    )


def _payload_sha256(
    payload: object,
) -> str:
    return hashlib.sha256(
        _canonical_json_bytes(
            payload
        )
    ).hexdigest()


def _file_sha256(
    path: Path,
) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def _derive_event_seed(
    *,
    replicate_id: int,
    challenge_id: str,
) -> int:
    """
    Stable identity/event seed.

    No stochastic hazard occurrence is generated from this seed in
    the primary controlled overlay. It is frozen for provenance and
    future event-schedule extensions.
    """

    payload = (
        f"{HAZARD_OVERLAY_SCHEMA}|"
        f"{MASTER_HAZARD_SEED}|"
        f"{replicate_id}|"
        f"{challenge_id}"
    ).encode(
        "utf-8"
    )

    digest = hashlib.sha256(
        payload
    ).digest()

    # Positive signed-64-bit-compatible integer.
    return (
        int.from_bytes(
            digest[:8],
            byteorder="big",
            signed=False,
        )
        & 0x7FFFFFFFFFFFFFFF
    )


def _ranked_hazard_slots(
    *,
    replicate_id: int,
) -> tuple[int, int, int]:
    """
    Select three deterministic distinct bin IDs.

    Selection depends ONLY on:
    - frozen master hazard seed,
    - workload replicate ID,
    - bin ID.

    It deliberately does NOT inspect:
    - topology,
    - road distance,
    - fill level,
    - DQN/HDR/TSR performance,
    - any downstream metric.

    Therefore the same hazard bins are reused across all four
    topologies for one workload replicate.
    """

    scored = []

    for bin_id in range(
        NUM_BINS
    ):
        payload = (
            f"{HAZARD_OVERLAY_SCHEMA}|"
            f"hazard-slot|"
            f"{MASTER_HAZARD_SEED}|"
            f"{replicate_id}|"
            f"{bin_id}"
        ).encode(
            "utf-8"
        )

        score = hashlib.sha256(
            payload
        ).digest()

        scored.append(
            (
                score,
                bin_id,
            )
        )

    scored.sort(
        key=lambda item: (
            item[0],
            item[1],
        )
    )

    selected = tuple(
        bin_id
        for _, bin_id in scored[
            :3
        ]
    )

    if len(
        set(
            selected
        )
    ) != 3:
        raise HazardOverlayError(
            "hazard slot selection did not "
            "produce three distinct bins"
        )

    return selected


@dataclass(frozen=True)
class HazardAssignment:
    """
    Controlled routing-hazard assignment.

    This is NOT a prevalence model and does not claim that hazards
    occur with any particular real-world probability.
    """

    bin_id: int
    hazard_type: str
    routing_severity: float

    def __post_init__(self) -> None:
        if not (
            0
            <= self.bin_id
            < NUM_BINS
        ):
            raise HazardOverlayError(
                "hazard assignment bin_id "
                "outside 0..999"
            )

        if (
            self.hazard_type
            not in HAZARD_SEVERITY
        ):
            raise HazardOverlayError(
                "unknown hazard_type: "
                f"{self.hazard_type}"
            )

        expected = HAZARD_SEVERITY[
            self.hazard_type
        ]

        if abs(
            float(
                self.routing_severity
            )
            - expected
        ) > 1e-12:
            raise HazardOverlayError(
                "routing severity does not match "
                "the frozen DQN hazard encoding"
            )

    def to_payload(
        self,
    ) -> dict[str, object]:
        return {
            "bin_id": int(
                self.bin_id
            ),
            "hazard_type": (
                self.hazard_type
            ),
            "routing_severity": float(
                self.routing_severity
            ),
        }


@dataclass(frozen=True)
class ResearchHazardOverlay:
    schema_version: str

    overlay_version: str

    master_hazard_seed: int

    workload_replicate_id: int

    workload_id: str

    workload_sha256: str

    challenge_id: str

    challenge_type: str

    event_seed: int

    assignments: tuple[
        HazardAssignment,
        ...,
    ]

    def __post_init__(self) -> None:
        if (
            self.schema_version
            != HAZARD_OVERLAY_SCHEMA
        ):
            raise HazardOverlayError(
                "unexpected hazard overlay schema"
            )

        if (
            self.overlay_version
            != HAZARD_OVERLAY_VERSION
        ):
            raise HazardOverlayError(
                "unexpected hazard overlay version"
            )

        if (
            self.challenge_id
            not in CHALLENGE_ORDER
        ):
            raise HazardOverlayError(
                "unknown challenge_id"
            )

        if (
            self.challenge_type
            != CHALLENGE_TYPES[
                self.challenge_id
            ]
        ):
            raise HazardOverlayError(
                "challenge type does not match "
                "challenge ID"
            )

        if not (
            0
            <= self.workload_replicate_id
            < 10
        ):
            raise HazardOverlayError(
                "workload replicate must be 0..9"
            )

        if not self.workload_id:
            raise HazardOverlayError(
                "workload_id cannot be empty"
            )

        if len(
            self.workload_sha256
        ) != 64:
            raise HazardOverlayError(
                "invalid workload SHA-256"
            )

        bin_ids = tuple(
            assignment.bin_id
            for assignment in self.assignments
        )

        if len(
            set(
                bin_ids
            )
        ) != len(
            bin_ids
        ):
            raise HazardOverlayError(
                "one overlay cannot assign multiple "
                "hazards to the same bin"
            )

        expected_count = {
            "H0": 0,
            "H1": 1,
            "H2": 1,
            "H3": 3,
        }[
            self.challenge_id
        ]

        if (
            len(
                self.assignments
            )
            != expected_count
        ):
            raise HazardOverlayError(
                "unexpected assignment count "
                f"for {self.challenge_id}"
            )

    def canonical_payload(
        self,
    ) -> dict[str, object]:
        return {
            "schema_version": (
                self.schema_version
            ),
            "overlay_version": (
                self.overlay_version
            ),
            "master_hazard_seed": int(
                self.master_hazard_seed
            ),
            "workload_replicate_id": int(
                self.workload_replicate_id
            ),
            "workload_id": (
                self.workload_id
            ),
            "workload_sha256": (
                self.workload_sha256
            ),
            "challenge_id": (
                self.challenge_id
            ),
            "challenge_type": (
                self.challenge_type
            ),
            "event_seed": int(
                self.event_seed
            ),
            "selection_contract": (
                "deterministic_hash_ranked_bins_"
                "independent_of_topology_fill_and_performance"
            ),
            "interpretation": (
                "controlled_routing_hazard_challenge_"
                "not_real_world_prevalence_model"
            ),
            "initial_full_elapsed_policy": (
                "zero_at_t0_primary_benchmark"
            ),
            "assignments": [
                assignment.to_payload()
                for assignment
                in self.assignments
            ],
        }

    @property
    def sha256(
        self,
    ) -> str:
        return _payload_sha256(
            self.canonical_payload()
        )

    @property
    def hazard_scenario_id(
        self,
    ) -> str:
        return (
            f"hazard-"
            f"rep{self.workload_replicate_id:02d}-"
            f"{self.challenge_id.lower()}-"
            f"{self.sha256[:12]}"
        )

    @property
    def routing_severity_by_bin(
        self,
    ) -> dict[int, float]:
        return {
            assignment.bin_id: float(
                assignment.routing_severity
            )
            for assignment
            in self.assignments
        }

    def to_json_dict(
        self,
    ) -> dict[str, object]:
        payload = dict(
            self.canonical_payload()
        )

        payload[
            "hazard_scenario_id"
        ] = self.hazard_scenario_id

        payload[
            "overlay_sha256"
        ] = self.sha256

        return payload


def _build_assignments(
    *,
    replicate_id: int,
    challenge_id: str,
) -> tuple[
    HazardAssignment,
    ...,
]:
    predictive_bin, emergency_bin, moisture_bin = (
        _ranked_hazard_slots(
            replicate_id=replicate_id
        )
    )

    predictive = HazardAssignment(
        bin_id=predictive_bin,
        hazard_type=(
            "predicted_combustion"
        ),
        routing_severity=(
            HAZARD_SEVERITY[
                "predicted_combustion"
            ]
        ),
    )

    emergency = HazardAssignment(
        bin_id=emergency_bin,
        hazard_type=(
            "emergency_fire"
        ),
        routing_severity=(
            HAZARD_SEVERITY[
                "emergency_fire"
            ]
        ),
    )

    moisture = HazardAssignment(
        bin_id=moisture_bin,
        hazard_type=(
            "excess_moisture"
        ),
        routing_severity=(
            HAZARD_SEVERITY[
                "excess_moisture"
            ]
        ),
    )

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
        # Stable ordering by operational priority.
        return (
            emergency,
            predictive,
            moisture,
        )

    raise HazardOverlayError(
        f"unknown challenge_id={challenge_id}"
    )


def build_research_hazard_overlay(
    *,
    replicate_id: int,
    challenge_id: str,
) -> ResearchHazardOverlay:
    if (
        challenge_id
        not in CHALLENGE_ORDER
    ):
        raise HazardOverlayError(
            f"unknown challenge_id={challenge_id}"
        )

    workload = (
        load_frozen_research_workload(
            replicate_id
        )
    )

    return ResearchHazardOverlay(
        schema_version=(
            HAZARD_OVERLAY_SCHEMA
        ),
        overlay_version=(
            HAZARD_OVERLAY_VERSION
        ),
        master_hazard_seed=(
            MASTER_HAZARD_SEED
        ),
        workload_replicate_id=(
            replicate_id
        ),
        workload_id=(
            workload.workload_id
        ),
        workload_sha256=(
            workload.sha256
        ),
        challenge_id=(
            challenge_id
        ),
        challenge_type=(
            CHALLENGE_TYPES[
                challenge_id
            ]
        ),
        event_seed=(
            _derive_event_seed(
                replicate_id=replicate_id,
                challenge_id=challenge_id,
            )
        ),
        assignments=(
            _build_assignments(
                replicate_id=replicate_id,
                challenge_id=challenge_id,
            )
        ),
    )


def build_research_hazard_overlays(
) -> tuple[
    ResearchHazardOverlay,
    ...,
]:
    return tuple(
        build_research_hazard_overlay(
            replicate_id=replicate_id,
            challenge_id=challenge_id,
        )
        for replicate_id in range(
            10
        )
        for challenge_id
        in CHALLENGE_ORDER
    )


def build_composite_identity(
    *,
    physical_scenario_id: str,
    physical_scenario_sha256: str,
    overlay: ResearchHazardOverlay,
) -> tuple[str, str]:
    payload = {
        "schema_version": (
            COMPOSITE_SCENARIO_SCHEMA
        ),
        "physical_scenario_id": (
            physical_scenario_id
        ),
        "physical_scenario_sha256": (
            physical_scenario_sha256
        ),
        "hazard_scenario_id": (
            overlay.hazard_scenario_id
        ),
        "hazard_overlay_sha256": (
            overlay.sha256
        ),
    }

    digest = _payload_sha256(
        payload
    )

    composite_id = (
        f"composite-{digest[:16]}"
    )

    return (
        composite_id,
        digest,
    )


def _overlay_filename(
    overlay: ResearchHazardOverlay,
) -> str:
    return (
        f"rep-{overlay.workload_replicate_id:02d}"
        f"__{overlay.challenge_id.lower()}.json"
    )


def write_research_hazard_overlays(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    manifest_path: (
        Path
        | None
    ) = None,
) -> dict[str, object]:
    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if manifest_path is None:
        manifest_path = (
            output_dir
            / "research_hazard_overlays_manifest.json"
        )
    else:
        manifest_path = Path(
            manifest_path
        )

    overlays = (
        build_research_hazard_overlays()
    )

    overlay_records = []

    overlay_by_key = {}

    for overlay in overlays:
        key = (
            overlay.workload_replicate_id,
            overlay.challenge_id,
        )

        overlay_by_key[
            key
        ] = overlay

        file_path = (
            output_dir
            / _overlay_filename(
                overlay
            )
        )

        json_payload = (
            overlay.to_json_dict()
        )

        file_path.write_text(
            json.dumps(
                json_payload,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

        overlay_records.append(
            {
                "workload_replicate_id": (
                    overlay
                    .workload_replicate_id
                ),
                "challenge_id": (
                    overlay.challenge_id
                ),
                "challenge_type": (
                    overlay.challenge_type
                ),
                "workload_id": (
                    overlay.workload_id
                ),
                "workload_sha256": (
                    overlay.workload_sha256
                ),
                "hazard_scenario_id": (
                    overlay.hazard_scenario_id
                ),
                "overlay_sha256": (
                    overlay.sha256
                ),
                "event_seed": (
                    overlay.event_seed
                ),
                "file": (
                    file_path.as_posix()
                ),
                "file_sha256": (
                    _file_sha256(
                        file_path
                    )
                ),
            }
        )

    cases = (
        build_research_scenario_matrix()
    )

    composite_records = []

    for case in cases:
        for challenge_id in (
            CHALLENGE_ORDER
        ):
            overlay = (
                overlay_by_key[
                    (
                        case.replicate_id,
                        challenge_id,
                    )
                ]
            )

            if (
                overlay.workload_id
                != case.assembly.workload_id
            ):
                raise HazardOverlayError(
                    "hazard overlay workload ID "
                    "does not match physical scenario"
                )

            if (
                overlay.workload_sha256
                != case.assembly.workload_sha256
            ):
                raise HazardOverlayError(
                    "hazard overlay workload SHA "
                    "does not match physical scenario"
                )

            (
                composite_id,
                composite_sha,
            ) = build_composite_identity(
                physical_scenario_id=(
                    case.assembly.scenario_id
                ),
                physical_scenario_sha256=(
                    case.assembly.scenario_sha256
                ),
                overlay=overlay,
            )

            composite_records.append(
                {
                    "replicate_id": (
                        case.replicate_id
                    ),
                    "topology_id": (
                        case.topology_id
                    ),
                    "challenge_id": (
                        challenge_id
                    ),
                    "physical_scenario_id": (
                        case.assembly.scenario_id
                    ),
                    "physical_scenario_sha256": (
                        case.assembly.scenario_sha256
                    ),
                    "hazard_scenario_id": (
                        overlay.hazard_scenario_id
                    ),
                    "hazard_overlay_sha256": (
                        overlay.sha256
                    ),
                    "composite_scenario_id": (
                        composite_id
                    ),
                    "composite_sha256": (
                        composite_sha
                    ),
                }
            )

    manifest_core = {
        "schema_version": (
            HAZARD_MANIFEST_SCHEMA
        ),
        "overlay_schema_version": (
            HAZARD_OVERLAY_SCHEMA
        ),
        "overlay_version": (
            HAZARD_OVERLAY_VERSION
        ),
        "master_hazard_seed": (
            MASTER_HAZARD_SEED
        ),
        "master_hazard_seed_semantics": (
            "reproducibility_only_"
            "no_physical_probability_meaning"
        ),
        "num_bins": NUM_BINS,
        "replicate_ids": list(
            range(
                10
            )
        ),
        "topology_fairness_contract": list(
            FROZEN_TOPOLOGY_ORDER
        ),
        "challenge_order": list(
            CHALLENGE_ORDER
        ),
        "challenge_types": {
            key: CHALLENGE_TYPES[
                key
            ]
            for key in (
                CHALLENGE_ORDER
            )
        },
        "challenge_interpretation": (
            "controlled_deterministic_challenges_"
            "not_hazard_prevalence_estimates"
        ),
        "initial_full_elapsed_policy": (
            "zero_at_t0_primary_benchmark"
        ),
        "overlay_count": len(
            overlay_records
        ),
        "composite_pairing_count": len(
            composite_records
        ),
        "overlays": (
            overlay_records
        ),
        "composite_pairings": (
            composite_records
        ),
    }

    manifest_sha = (
        _payload_sha256(
            manifest_core
        )
    )

    manifest = dict(
        manifest_core
    )

    manifest[
        "manifest_sha256"
    ] = manifest_sha

    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path.write_text(
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
    parser = argparse.ArgumentParser(
        description=(
            "Generate frozen deterministic "
            "research hazard overlays."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    args = parser.parse_args()

    manifest = (
        write_research_hazard_overlays(
            output_dir=args.output_dir
        )
    )

    print(
        "schema_version:",
        manifest[
            "schema_version"
        ],
    )

    print(
        "overlay_count:",
        manifest[
            "overlay_count"
        ],
    )

    print(
        "composite_pairing_count:",
        manifest[
            "composite_pairing_count"
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
