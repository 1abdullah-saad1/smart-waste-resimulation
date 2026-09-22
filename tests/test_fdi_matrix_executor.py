import json

import networkx as nx
import pytest

from smart_waste.experiments.manifest import (
    ConfigBundleIdentity,
    ConfigFileIdentity,
    ExperimentManifestError,
    GitIdentity,
)
from smart_waste.experiments.matrix import (
    FDIMatrixError,
    FDIMatrixSpec,
    execute_fdi_matrix,
    write_paired_fdi_raw_result,
)
from smart_waste.experiments.scenario_snapshot import (
    capture_physical_scenario,
)
from smart_waste.models.bin import WasteBin
from smart_waste.models.depot import Depot
from smart_waste.models.truck import Truck
from smart_waste.movement.road_graph import RoadGraph
from smart_waste.simulation.state import SimulationState


MASTER_SEED = 20261001


def make_snapshot(
    *,
    bin_id: int = 101,
    fill_percent: float = 20.0,
):
    graph = nx.Graph()

    graph.add_node(
        "depot",
        x=0.0,
        y=0.0,
    )

    node = f"bin-{bin_id}"

    graph.add_node(
        node,
        x=1.0,
        y=0.0,
    )

    graph.add_edge(
        "depot",
        node,
        length_km=1.0,
    )

    state = SimulationState(
        road_graph=RoadGraph(
            graph
        ),
        bins={
            bin_id: WasteBin(
                bin_id=bin_id,
                road_node=node,
                fill_percent=(
                    fill_percent
                ),
                fill_rate_percent_per_hour=0.0,
                full_mass_kg=440.0,
            )
        },
        trucks={
            0: Truck(
                truck_id=0,
                current_node="depot",
                capacity_tonnes=10.0,
                speed_km_per_hour=30.0,
                fuel_capacity_litres=200.0,
                fuel_efficiency_km_per_litre=2.5,
            )
        },
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


def fake_git():
    return GitIdentity(
        commit_sha="a" * 40,
        is_dirty=False,
    )


def fake_config():
    return ConfigBundleIdentity(
        files=(
            ConfigFileIdentity(
                path="configs/test.yaml",
                sha256="b" * 64,
                size_bytes=12,
            ),
        ),
        sha256="c" * 64,
    )


def test_matrix_spec_canonicalizes_order_and_duplicates() -> None:
    spec = FDIMatrixSpec(
        master_seed=MASTER_SEED,
        attack_rates=(
            0.30,
            0.0,
            0.15,
            0.15,
        ),
        attack_types=(
            "external_unauthenticated",
            "external_unauthenticated",
        ),
        replicate_ids=(
            2,
            0,
            1,
            1,
        ),
    )

    assert spec.attack_rates == (
        0.0,
        0.15,
        0.30,
    )

    assert spec.replicate_ids == (
        0,
        1,
        2,
    )

    assert spec.attack_types == (
        "external_unauthenticated",
    )


def test_small_matrix_writes_all_expected_pairs(
    tmp_path,
) -> None:
    spec = FDIMatrixSpec(
        master_seed=MASTER_SEED,
        attack_rates=(
            0.0,
            1.0,
        ),
        attack_types=(
            "external_unauthenticated",
        ),
        replicate_ids=(
            0,
            1,
        ),
    )

    result = execute_fdi_matrix(
        (
            make_snapshot(),
        ),
        spec=spec,
        git_identity=fake_git(),
        config_bundle=fake_config(),
        manifest_directory=(
            tmp_path
            / "manifests"
        ),
        raw_directory=(
            tmp_path
            / "raw"
        ),
    )

    assert result.expected_pair_count == 4
    assert result.completed_pair_count == 4
    assert result.is_complete

    assert len(
        set(
            result.pair_ids
        )
    ) == 4

    assert len(
        list(
            (
                tmp_path
                / "manifests"
            ).glob(
                "*__manifest.json"
            )
        )
    ) == 4

    assert len(
        list(
            (
                tmp_path
                / "raw"
            ).glob(
                "*__results.json"
            )
        )
    ) == 4


def test_matrix_outputs_reference_matching_manifest_hashes(
    tmp_path,
) -> None:
    result = execute_fdi_matrix(
        (
            make_snapshot(),
        ),
        spec=FDIMatrixSpec(
            master_seed=MASTER_SEED,
            attack_rates=(
                1.0,
            ),
            attack_types=(
                "external_unauthenticated",
            ),
            replicate_ids=(
                0,
            ),
        ),
        git_identity=fake_git(),
        config_bundle=fake_config(),
        manifest_directory=(
            tmp_path
            / "manifests"
        ),
        raw_directory=(
            tmp_path
            / "raw"
        ),
    )

    record = result.records[
        0
    ]

    raw = json.loads(
        open(
            record.raw_result_path,
            encoding="utf-8",
        ).read()
    )

    manifest = json.loads(
        open(
            record.manifest_path,
            encoding="utf-8",
        ).read()
    )

    assert (
        raw[
            "manifest_sha256"
        ]
        == manifest[
            "manifest_sha256"
        ]
        == record.manifest_sha256
    )

    assert (
        raw[
            "pair_id"
        ]
        == manifest[
            "pair_id"
        ]
        == record.pair_id
    )


def test_external_full_attack_differs_between_security_paths(
    tmp_path,
) -> None:
    result = execute_fdi_matrix(
        (
            make_snapshot(
                fill_percent=20.0
            ),
        ),
        spec=FDIMatrixSpec(
            master_seed=MASTER_SEED,
            attack_rates=(
                1.0,
            ),
            attack_types=(
                "external_unauthenticated",
            ),
            replicate_ids=(
                0,
            ),
        ),
        git_identity=fake_git(),
        config_bundle=fake_config(),
        manifest_directory=(
            tmp_path
            / "manifests"
        ),
        raw_directory=(
            tmp_path
            / "raw"
        ),
    )

    raw = json.loads(
        open(
            result.records[
                0
            ].raw_result_path,
            encoding="utf-8",
        ).read()
    )

    assert raw[
        "unprotected"
    ][
        "serviced_bin_ids"
    ] == [
        101
    ]

    assert raw[
        "poa_verified"
    ][
        "serviced_bin_ids"
    ] == []


def test_zero_attack_paths_are_identical(
    tmp_path,
) -> None:
    result = execute_fdi_matrix(
        (
            make_snapshot(
                fill_percent=90.0
            ),
        ),
        spec=FDIMatrixSpec(
            master_seed=MASTER_SEED,
            attack_rates=(
                0.0,
            ),
            attack_types=(
                "external_unauthenticated",
            ),
            replicate_ids=(
                0,
            ),
        ),
        git_identity=fake_git(),
        config_bundle=fake_config(),
        manifest_directory=(
            tmp_path
            / "manifests"
        ),
        raw_directory=(
            tmp_path
            / "raw"
        ),
    )

    raw = json.loads(
        open(
            result.records[
                0
            ].raw_result_path,
            encoding="utf-8",
        ).read()
    )

    assert (
        raw[
            "unprotected"
        ][
            "completed_services"
        ]
        == raw[
            "poa_verified"
        ][
            "completed_services"
        ]
    )

    assert (
        raw[
            "unprotected"
        ][
            "metrics"
        ][
            "total_distance_km"
        ]
        == pytest.approx(
            raw[
                "poa_verified"
            ][
                "metrics"
            ][
                "total_distance_km"
            ]
        )
    )


def test_authenticated_compromise_paths_are_identical(
    tmp_path,
) -> None:
    result = execute_fdi_matrix(
        (
            make_snapshot(
                fill_percent=20.0
            ),
        ),
        spec=FDIMatrixSpec(
            master_seed=MASTER_SEED,
            attack_rates=(
                1.0,
            ),
            attack_types=(
                "authenticated_compromise",
            ),
            replicate_ids=(
                0,
            ),
        ),
        git_identity=fake_git(),
        config_bundle=fake_config(),
        manifest_directory=(
            tmp_path
            / "manifests"
        ),
        raw_directory=(
            tmp_path
            / "raw"
        ),
    )

    raw = json.loads(
        open(
            result.records[
                0
            ].raw_result_path,
            encoding="utf-8",
        ).read()
    )

    assert (
        raw[
            "unprotected"
        ][
            "serviced_bin_ids"
        ]
        == raw[
            "poa_verified"
        ][
            "serviced_bin_ids"
        ]
        == [
            101
        ]
    )


def test_matrix_rejects_duplicate_physical_scenarios(
    tmp_path,
) -> None:
    snapshot = make_snapshot()

    with pytest.raises(
        FDIMatrixError,
        match="duplicate physical scenarios",
    ):
        execute_fdi_matrix(
            (
                snapshot,
                snapshot,
            ),
            spec=FDIMatrixSpec(
                master_seed=MASTER_SEED,
                attack_rates=(
                    0.0,
                ),
                attack_types=(
                    "external_unauthenticated",
                ),
                replicate_ids=(
                    0,
                ),
            ),
            git_identity=fake_git(),
            config_bundle=fake_config(),
            manifest_directory=(
                tmp_path
                / "manifests"
            ),
            raw_directory=(
                tmp_path
                / "raw"
            ),
        )


def test_matrix_requires_clean_git_when_requested(
    tmp_path,
) -> None:
    with pytest.raises(
        ExperimentManifestError,
        match="clean Git",
    ):
        execute_fdi_matrix(
            (
                make_snapshot(),
            ),
            spec=FDIMatrixSpec(
                master_seed=MASTER_SEED,
                attack_rates=(
                    0.0,
                ),
                attack_types=(
                    "external_unauthenticated",
                ),
                replicate_ids=(
                    0,
                ),
            ),
            git_identity=GitIdentity(
                commit_sha="a" * 40,
                is_dirty=True,
            ),
            config_bundle=fake_config(),
            manifest_directory=(
                tmp_path
                / "manifests"
            ),
            raw_directory=(
                tmp_path
                / "raw"
            ),
        )


def test_matrix_outputs_are_idempotent(
    tmp_path,
) -> None:
    kwargs = dict(
        snapshots=(
            make_snapshot(),
        ),
        spec=FDIMatrixSpec(
            master_seed=MASTER_SEED,
            attack_rates=(
                1.0,
            ),
            attack_types=(
                "external_unauthenticated",
            ),
            replicate_ids=(
                0,
            ),
        ),
        git_identity=fake_git(),
        config_bundle=fake_config(),
        manifest_directory=(
            tmp_path
            / "manifests"
        ),
        raw_directory=(
            tmp_path
            / "raw"
        ),
    )

    first = execute_fdi_matrix(
        **kwargs
    )

    second = execute_fdi_matrix(
        **kwargs
    )

    assert (
        first.pair_ids
        == second.pair_ids
    )

    assert (
        first.records[
            0
        ].manifest_sha256
        == second.records[
            0
        ].manifest_sha256
    )


def test_raw_writer_refuses_conflicting_existing_output(
    tmp_path,
) -> None:
    result = execute_fdi_matrix(
        (
            make_snapshot(),
        ),
        spec=FDIMatrixSpec(
            master_seed=MASTER_SEED,
            attack_rates=(
                1.0,
            ),
            attack_types=(
                "external_unauthenticated",
            ),
            replicate_ids=(
                0,
            ),
        ),
        git_identity=fake_git(),
        config_bundle=fake_config(),
        manifest_directory=(
            tmp_path
            / "manifests"
        ),
        raw_directory=(
            tmp_path
            / "raw"
        ),
    )

    raw_path = result.records[
        0
    ].raw_result_path

    with open(
        raw_path,
        "w",
        encoding="utf-8",
    ) as stream:
        stream.write(
            '{"corrupt": true}\n'
        )

    # Re-running must refuse to silently replace corrupted/raw
    # conflicting evidence.
    with pytest.raises(
        FDIMatrixError,
        match="refusing to overwrite",
    ):
        execute_fdi_matrix(
            (
                make_snapshot(),
            ),
            spec=FDIMatrixSpec(
                master_seed=MASTER_SEED,
                attack_rates=(
                    1.0,
                ),
                attack_types=(
                    "external_unauthenticated",
                ),
                replicate_ids=(
                    0,
                ),
            ),
            git_identity=fake_git(),
            config_bundle=fake_config(),
            manifest_directory=(
                tmp_path
                / "manifests"
            ),
            raw_directory=(
                tmp_path
                / "raw"
            ),
        )
