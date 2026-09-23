from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable

from smart_waste.experiments.empirical_workload import (
    EmpiricalPhysicalWorkloadRealization,
    EmpiricalWorkloadSpec,
    generate_empirical_workload,
)
from smart_waste.experiments.physical_scenario_factory import (
    PhysicalScenarioAssembly,
    assemble_physical_scenario,
    load_primary_physical_scenario_spec,
)
from smart_waste.movement.topologies.hexagonal_layout import (
    build_primary_hex_layout,
)
from smart_waste.movement.topologies.manhattan_layout import (
    build_primary_manhattan_layout,
)
from smart_waste.movement.topologies.radial_concentric_layout import (
    build_primary_radial_concentric_layout,
)
from smart_waste.movement.topologies.superblock_layout import (
    build_primary_superblock_layout,
)


RESEARCH_SCENARIO_SCHEMA = (
    "research-physical-scenario-matrix-v1"
)

DEFAULT_WORKLOAD_MANIFEST = Path(
    "data/processed/workload/"
    "research_workloads/"
    "research_workloads_manifest.json"
)

FROZEN_TOPOLOGY_ORDER = (
    "manhattan",
    "superblock",
    "hex",
    "radial_concentric",
)

FROZEN_TOPOLOGY_INDEX = {
    "manhattan": 1,
    "superblock": 2,
    "hex": 3,
    "radial_concentric": 4,
}

EXPECTED_GRAPH_SCALE = {
    "manhattan": (
        1225,
        2380,
    ),
    "superblock": (
        2665,
        3960,
    ),
    "hex": (
        3023,
        4402,
    ),
    "radial_concentric": (
        1201,
        2400,
    ),
}


@dataclass(frozen=True)
class ResearchScenarioCase:
    topology_id: str
    topology_index: int

    replicate_id: int

    workload: (
        EmpiricalPhysicalWorkloadRealization
    )

    assembly: PhysicalScenarioAssembly

    @property
    def case_id(self) -> str:
        return (
            f"rep-{self.replicate_id:02d}"
            f"__{self.topology_id}"
        )


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


def load_frozen_workload_manifest(
    path: Path = DEFAULT_WORKLOAD_MANIFEST,
) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Research workload manifest not found: {path}"
        )

    manifest = _load_json(
        path
    )

    if (
        manifest.get(
            "schema_version"
        )
        != "research-workload-manifest-v1"
    ):
        raise RuntimeError(
            "Unexpected research workload "
            "manifest schema"
        )

    if (
        tuple(
            manifest.get(
                "replicate_ids",
                (),
            )
        )
        != tuple(
            range(
                10
            )
        )
    ):
        raise RuntimeError(
            "Frozen research design requires "
            "replicate IDs 0..9"
        )

    if (
        tuple(
            manifest.get(
                "topology_fairness_contract",
                (),
            )
        )
        != FROZEN_TOPOLOGY_ORDER
    ):
        raise RuntimeError(
            "Frozen topology fairness contract "
            "does not match the four-topology design"
        )

    if (
        int(
            manifest.get(
                "num_bins",
                -1,
            )
        )
        != 1000
    ):
        raise RuntimeError(
            "Frozen research design requires 1000 bins"
        )

    return manifest


def _record_by_replicate(
    manifest: dict,
) -> dict[int, dict]:
    records = {
        int(
            record[
                "replicate_id"
            ]
        ): record
        for record in manifest[
            "workloads"
        ]
    }

    if tuple(
        sorted(
            records
        )
    ) != tuple(
        range(
            10
        )
    ):
        raise RuntimeError(
            "Research workload records must "
            "cover replicate IDs 0..9 exactly"
        )

    return records


def load_frozen_research_workload(
    replicate_id: int,
    *,
    manifest_path: Path = (
        DEFAULT_WORKLOAD_MANIFEST
    ),
) -> EmpiricalPhysicalWorkloadRealization:
    manifest = (
        load_frozen_workload_manifest(
            manifest_path
        )
    )

    records = (
        _record_by_replicate(
            manifest
        )
    )

    if replicate_id not in records:
        raise ValueError(
            f"Unknown replicate_id={replicate_id}"
        )

    record = records[
        replicate_id
    ]

    workload_path = Path(
        record[
            "file"
        ]
    )

    if not workload_path.is_file():
        raise FileNotFoundError(
            f"Frozen workload file missing: "
            f"{workload_path}"
        )

    actual_file_sha = (
        _sha256_file(
            workload_path
        )
    )

    if (
        actual_file_sha
        != record[
            "file_sha256"
        ]
    ):
        raise RuntimeError(
            "Frozen workload file SHA mismatch "
            f"for replicate {replicate_id}"
        )

    frozen_payload = _load_json(
        workload_path
    )

    spec = EmpiricalWorkloadSpec(
        num_bins=int(
            manifest[
                "num_bins"
            ]
        ),
        master_seed=int(
            manifest[
                "master_workload_seed"
            ]
        ),
        replicate_id=(
            replicate_id
        ),
        profile_library_path=(
            manifest[
                "profile_library"
            ]
        ),
        profile_library_sha256=(
            manifest[
                "profile_library_sha256"
            ]
        ),
    )

    regenerated = (
        generate_empirical_workload(
            spec
        )
    )

    if (
        regenerated.workload_id
        != record[
            "workload_id"
        ]
    ):
        raise RuntimeError(
            "Regenerated workload ID mismatch"
        )

    if (
        regenerated.sha256
        != record[
            "workload_sha256"
        ]
    ):
        raise RuntimeError(
            "Regenerated workload SHA mismatch"
        )

    if (
        frozen_payload[
            "workload_id"
        ]
        != regenerated.workload_id
    ):
        raise RuntimeError(
            "Frozen workload payload ID mismatch"
        )

    if (
        frozen_payload[
            "workload_sha256"
        ]
        != regenerated.sha256
    ):
        raise RuntimeError(
            "Frozen workload payload SHA mismatch"
        )

    if tuple(
        int(value)
        for value in frozen_payload[
            "profile_ids"
        ]
    ) != regenerated.profile_ids:
        raise RuntimeError(
            "Frozen profile mapping mismatch"
        )

    if tuple(
        float(value)
        for value in frozen_payload[
            "initial_fill_percent"
        ]
    ) != regenerated.initial_fill_percent:
        raise RuntimeError(
            "Frozen initial-fill vector mismatch"
        )

    if tuple(
        float(value)
        for value in frozen_payload[
            "fill_rate_percent_per_hour"
        ]
    ) != regenerated.fill_rate_percent_per_hour:
        raise RuntimeError(
            "Frozen fill-rate vector mismatch"
        )

    return regenerated


def _topology_builders() -> dict[
    str,
    Callable,
]:
    return {
        "manhattan": (
            build_primary_manhattan_layout
        ),
        "superblock": (
            build_primary_superblock_layout
        ),
        "hex": (
            build_primary_hex_layout
        ),
        "radial_concentric": (
            build_primary_radial_concentric_layout
        ),
    }


def build_research_scenario_matrix(
    *,
    workload_manifest_path: Path = (
        DEFAULT_WORKLOAD_MANIFEST
    ),
) -> tuple[
    ResearchScenarioCase,
    ...,
]:
    workload_manifest = (
        load_frozen_workload_manifest(
            workload_manifest_path
        )
    )

    physical_spec = (
        load_primary_physical_scenario_spec()
    )

    builders = (
        _topology_builders()
    )

    layouts = {
        topology_id: (
            builders[
                topology_id
            ]()
        )
        for topology_id
        in FROZEN_TOPOLOGY_ORDER
    }

    cases: list[
        ResearchScenarioCase
    ] = []

    for replicate_id in workload_manifest[
        "replicate_ids"
    ]:
        replicate_id = int(
            replicate_id
        )

        workload = (
            load_frozen_research_workload(
                replicate_id,
                manifest_path=(
                    workload_manifest_path
                ),
            )
        )

        for topology_id in (
            FROZEN_TOPOLOGY_ORDER
        ):
            layout = layouts[
                topology_id
            ]

            assembly = (
                assemble_physical_scenario(
                    road_graph=(
                        layout.topology.road_graph
                    ),
                    depot_node=(
                        layout.depot_node
                    ),
                    assignments=(
                        layout.bin_placement.assignments
                    ),
                    workload=workload,
                    spec=physical_spec,
                )
            )

            snapshot = (
                assembly.snapshot
            )

            expected_nodes, expected_edges = (
                EXPECTED_GRAPH_SCALE[
                    topology_id
                ]
            )

            if len(
                snapshot.road_nodes
            ) != expected_nodes:
                raise RuntimeError(
                    f"{topology_id}: unexpected "
                    "road-node count"
                )

            if len(
                snapshot.road_edges
            ) != expected_edges:
                raise RuntimeError(
                    f"{topology_id}: unexpected "
                    "road-edge count"
                )

            if len(
                snapshot.bins
            ) != 1000:
                raise RuntimeError(
                    f"{topology_id}: expected "
                    "1000 bins"
                )

            if len(
                snapshot.trucks
            ) != 10:
                raise RuntimeError(
                    f"{topology_id}: expected "
                    "10 trucks"
                )

            if (
                assembly.workload_sha256
                != workload.sha256
            ):
                raise RuntimeError(
                    "Scenario/workload SHA linkage "
                    "mismatch"
                )

            if (
                assembly.workload_replicate_id
                != replicate_id
            ):
                raise RuntimeError(
                    "Scenario/workload replicate "
                    "linkage mismatch"
                )

            cases.append(
                ResearchScenarioCase(
                    topology_id=(
                        topology_id
                    ),
                    topology_index=(
                        FROZEN_TOPOLOGY_INDEX[
                            topology_id
                        ]
                    ),
                    replicate_id=(
                        replicate_id
                    ),
                    workload=(
                        workload
                    ),
                    assembly=(
                        assembly
                    ),
                )
            )

    if len(
        cases
    ) != 40:
        raise RuntimeError(
            "Frozen research matrix must contain "
            "exactly 40 physical scenarios"
        )

    scenario_hashes = {
        case.assembly.scenario_sha256
        for case in cases
    }

    if len(
        scenario_hashes
    ) != 40:
        raise RuntimeError(
            "All 40 physical scenario SHA values "
            "must be distinct"
        )

    for replicate_id in range(
        10
    ):
        replicate_cases = tuple(
            case
            for case in cases
            if case.replicate_id
            == replicate_id
        )

        if tuple(
            case.topology_id
            for case in replicate_cases
        ) != FROZEN_TOPOLOGY_ORDER:
            raise RuntimeError(
                "Topology ordering invariant violated"
            )

        if len(
            {
                case.workload.sha256
                for case in replicate_cases
            }
        ) != 1:
            raise RuntimeError(
                "Cross-topology workload fairness "
                "invariant violated"
            )

    return tuple(
        cases
    )
