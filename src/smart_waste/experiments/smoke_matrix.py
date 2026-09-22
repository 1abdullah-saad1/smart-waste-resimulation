from __future__ import annotations

import json

from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from smart_waste.experiments.manifest import (
    ConfigBundleIdentity,
    GitIdentity,
    build_config_bundle_identity,
    discover_git_identity,
)
from smart_waste.experiments.matrix import (
    FDIMatrixExecutionResult,
    FDIMatrixSpec,
    execute_fdi_matrix,
)
from smart_waste.experiments.scenario_snapshot import (
    PhysicalScenarioSnapshot,
    capture_physical_scenario,
)
from smart_waste.models.bin import WasteBin
from smart_waste.models.depot import Depot
from smart_waste.models.truck import Truck
from smart_waste.movement.road_graph import RoadGraph
from smart_waste.simulation.state import SimulationState


DEVELOPMENT_CLASSIFICATION = (
    "DEVELOPMENT_NON_CONFIRMATORY"
)

SMOKE_MASTER_SEED = 20261001

SMOKE_ATTACK_RATES = (
    0.00,
    0.15,
    0.30,
)

SMOKE_ATTACK_TYPES = (
    "external_unauthenticated",
    "authenticated_compromise",
)

SMOKE_REPLICATE_IDS = (
    0,
    1,
)

EXPECTED_SMOKE_PAIR_COUNT = (
    len(SMOKE_ATTACK_RATES)
    * len(SMOKE_ATTACK_TYPES)
    * len(SMOKE_REPLICATE_IDS)
)


class DevelopmentSmokeMatrixError(RuntimeError):
    """Raised when development smoke evidence is incomplete."""


@dataclass(frozen=True)
class DevelopmentSmokeResult:
    classification: str

    scenario_id: str
    scenario_sha256: str

    git_commit_sha: str
    config_sha256: str

    matrix: FDIMatrixExecutionResult

    marker_path: str
    summary_path: str


def build_development_smoke_snapshot(
) -> PhysicalScenarioSnapshot:
    """
    Build a tiny deterministic physical scenario used only to test
    the experiment/provenance pipeline.

    This scenario is NOT a benchmark topology and its outputs must
    never be interpreted as confirmatory research results.
    """

    graph = nx.Graph()

    graph.graph[
        "evidence_classification"
    ] = DEVELOPMENT_CLASSIFICATION

    graph.graph[
        "purpose"
    ] = "experiment-pipeline-smoke-test"

    graph.add_node(
        "depot",
        x=0.0,
        y=0.0,
        kind="depot",
    )

    fills = (
        20.0,
        35.0,
        50.0,
        65.0,
        79.0,
        80.0,
        85.0,
        90.0,
        95.0,
        100.0,
    )

    bins: dict[
        int,
        WasteBin,
    ] = {}

    previous_node = "depot"

    for index, fill in enumerate(
        fills
    ):
        bin_id = (
            100
            + index * 10
        )

        node = (
            f"smoke-bin-{bin_id}"
        )

        graph.add_node(
            node,
            x=float(
                index + 1
            ),
            y=0.0,
            kind="bin",
        )

        graph.add_edge(
            previous_node,
            node,
            length_km=0.5,
            road_class="smoke-chain",
        )

        previous_node = node

        bins[
            bin_id
        ] = WasteBin(
            bin_id=bin_id,
            road_node=node,
            fill_percent=fill,
            fill_rate_percent_per_hour=0.0,
            full_mass_kg=440.0,
        )

    trucks = {
        truck_id: Truck(
            truck_id=truck_id,
            current_node="depot",
            capacity_tonnes=10.0,
            speed_km_per_hour=30.0,
            fuel_capacity_litres=200.0,
            fuel_efficiency_km_per_litre=2.5,
        )
        for truck_id in (
            0,
            1,
        )
    }

    state = SimulationState(
        road_graph=RoadGraph(
            graph
        ),
        bins=bins,
        trucks=trucks,
        depot=Depot(
            road_node="depot",
            unloading_bays=1,
            unload_time_minutes=11.0,
            refuel_rate_litres_per_minute=60.0,
        ),
    )

    return capture_physical_scenario(
        state
    )


def development_smoke_spec(
) -> FDIMatrixSpec:
    return FDIMatrixSpec(
        master_seed=SMOKE_MASTER_SEED,
        attack_rates=(
            SMOKE_ATTACK_RATES
        ),
        attack_types=(
            SMOKE_ATTACK_TYPES
        ),
        replicate_ids=(
            SMOKE_REPLICATE_IDS
        ),
        selection_mode=(
            "paired_nested"
        ),
        forged_fill_percent=100.0,
        threshold_percent=80.0,
        service_time_seconds=36.0,
        max_events=10_000,
    )


def _write_idempotent_bytes(
    path: Path,
    content: bytes,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists():
        existing = (
            path.read_bytes()
        )

        if existing == content:
            return

        raise DevelopmentSmokeMatrixError(
            "refusing to overwrite conflicting development "
            f"smoke evidence: {path}"
        )

    temporary = path.with_suffix(
        path.suffix
        + ".tmp"
    )

    temporary.write_bytes(
        content
    )

    temporary.replace(
        path
    )


def _write_development_marker(
    directory: Path,
) -> Path:
    path = (
        directory
        / "DEVELOPMENT_NON_CONFIRMATORY.txt"
    )

    content = (
        "DEVELOPMENT / NON-CONFIRMATORY\n"
        "\n"
        "These files validate the experiment execution and "
        "provenance pipeline only.\n"
        "They are not confirmatory research results and must not "
        "be used in manuscript statistical analysis.\n"
    ).encode(
        "utf-8"
    )

    _write_idempotent_bytes(
        path,
        content,
    )

    return path


def _write_smoke_summary(
    *,
    directory: Path,
    snapshot: PhysicalScenarioSnapshot,
    git_identity: GitIdentity,
    config_bundle: ConfigBundleIdentity,
    matrix: FDIMatrixExecutionResult,
) -> Path:
    path = (
        directory
        / "DEVELOPMENT_SMOKE_SUMMARY.json"
    )

    document = {
        "schema_version": (
            "development-smoke-summary-v1"
        ),
        "evidence_classification": (
            DEVELOPMENT_CLASSIFICATION
        ),
        "confirmatory": False,
        "purpose": (
            "experiment_execution_and_provenance_validation"
        ),
        "physical_scenario": {
            "scenario_id": (
                snapshot.scenario_id
            ),
            "sha256": (
                snapshot.sha256
            ),
        },
        "code": {
            "git_commit_sha": (
                git_identity.commit_sha
            ),
            "git_dirty": (
                git_identity.is_dirty
            ),
        },
        "configuration": {
            "sha256": (
                config_bundle.sha256
            ),
        },
        "matrix": {
            "master_seed": (
                SMOKE_MASTER_SEED
            ),
            "attack_rates": list(
                SMOKE_ATTACK_RATES
            ),
            "attack_types": list(
                SMOKE_ATTACK_TYPES
            ),
            "replicate_ids": list(
                SMOKE_REPLICATE_IDS
            ),
            "expected_pair_count": (
                matrix.expected_pair_count
            ),
            "completed_pair_count": (
                matrix.completed_pair_count
            ),
            "pair_ids": list(
                matrix.pair_ids
            ),
            "records": [
                {
                    "pair_id": (
                        record.pair_id
                    ),
                    "manifest_sha256": (
                        record.manifest_sha256
                    ),
                    "manifest_file": (
                        Path(
                            record.manifest_path
                        ).name
                    ),
                    "raw_result_file": (
                        Path(
                            record.raw_result_path
                        ).name
                    ),
                }
                for record in matrix.records
            ],
        },
    }

    content = (
        json.dumps(
            document,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode(
        "utf-8"
    )

    _write_idempotent_bytes(
        path,
        content,
    )

    return path


def run_development_smoke_matrix(
    *,
    project_root: str | Path,
    manifest_directory: str | Path | None = None,
    raw_directory: str | Path | None = None,
    git_identity: GitIdentity | None = None,
    config_bundle: ConfigBundleIdentity | None = None,
    require_clean_git: bool = True,
) -> DevelopmentSmokeResult:
    """
    Execute the deterministic development-only FDI smoke matrix.

    Production use resolves real Git/config provenance.
    Optional identity injection exists only to make isolated unit
    tests independent from the repository working-tree state.
    """

    root = Path(
        project_root
    ).resolve()

    snapshot = (
        build_development_smoke_snapshot()
    )

    resolved_git = (
        discover_git_identity(
            root
        )
        if git_identity is None
        else git_identity
    )

    resolved_config = (
        build_config_bundle_identity(
            root
        )
        if config_bundle is None
        else config_bundle
    )

    manifest_dir = (
        root
        / "results"
        / "manifests"
        / "development-smoke"
        if manifest_directory is None
        else Path(
            manifest_directory
        )
    )

    raw_dir = (
        root
        / "results"
        / "raw"
        / "development-smoke"
        if raw_directory is None
        else Path(
            raw_directory
        )
    )

    matrix = execute_fdi_matrix(
        (
            snapshot,
        ),
        spec=development_smoke_spec(),
        git_identity=resolved_git,
        config_bundle=(
            resolved_config
        ),
        manifest_directory=(
            manifest_dir
        ),
        raw_directory=raw_dir,
        require_clean_git=(
            require_clean_git
        ),
    )

    if (
        matrix.expected_pair_count
        != EXPECTED_SMOKE_PAIR_COUNT
    ):
        raise DevelopmentSmokeMatrixError(
            "smoke matrix expected-pair count drifted: "
            f"expected={EXPECTED_SMOKE_PAIR_COUNT}, "
            f"actual={matrix.expected_pair_count}"
        )

    if not matrix.is_complete:
        raise DevelopmentSmokeMatrixError(
            "development smoke matrix is incomplete"
        )

    if (
        len(
            set(
                matrix.pair_ids
            )
        )
        != EXPECTED_SMOKE_PAIR_COUNT
    ):
        raise DevelopmentSmokeMatrixError(
            "development smoke matrix pair IDs are not unique"
        )

    marker_path = (
        _write_development_marker(
            manifest_dir
        )
    )

    summary_path = (
        _write_smoke_summary(
            directory=manifest_dir,
            snapshot=snapshot,
            git_identity=resolved_git,
            config_bundle=(
                resolved_config
            ),
            matrix=matrix,
        )
    )

    return DevelopmentSmokeResult(
        classification=(
            DEVELOPMENT_CLASSIFICATION
        ),
        scenario_id=(
            snapshot.scenario_id
        ),
        scenario_sha256=(
            snapshot.sha256
        ),
        git_commit_sha=(
            resolved_git.commit_sha
        ),
        config_sha256=(
            resolved_config.sha256
        ),
        matrix=matrix,
        marker_path=str(
            marker_path
        ),
        summary_path=str(
            summary_path
        ),
    )
