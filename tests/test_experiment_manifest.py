import hashlib
import json
from pathlib import Path

import networkx as nx
import pytest

from smart_waste.attacks.fdi import (
    generate_fdi_attack_from_truth,
)
from smart_waste.experiments.manifest import (
    ConfigBundleIdentity,
    ConfigFileIdentity,
    ExperimentManifestError,
    GitIdentity,
    build_config_bundle_identity,
    build_fdi_experiment_manifest,
    default_fdi_config_paths,
    write_fdi_experiment_manifest,
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


def make_snapshot():
    graph = nx.Graph()

    graph.add_node(
        "depot",
        x=0.0,
        y=0.0,
    )

    graph.add_node(
        "bin-101",
        x=1.0,
        y=0.0,
    )

    graph.add_node(
        "bin-407",
        x=2.0,
        y=0.0,
    )

    graph.add_edge(
        "depot",
        "bin-101",
        length_km=1.0,
    )

    graph.add_edge(
        "bin-101",
        "bin-407",
        length_km=1.0,
    )

    state = SimulationState(
        road_graph=RoadGraph(
            graph
        ),
        bins={
            101: WasteBin(
                bin_id=101,
                road_node="bin-101",
                fill_percent=20.0,
                fill_rate_percent_per_hour=0.0,
                full_mass_kg=440.0,
            ),
            407: WasteBin(
                bin_id=407,
                road_node="bin-407",
                fill_percent=95.0,
                fill_rate_percent_per_hour=0.0,
                full_mass_kg=440.0,
            ),
        },
        trucks={
            0: Truck(
                truck_id=0,
                current_node="depot",
                capacity_tonnes=10.0,
                speed_km_per_hour=30.0,
                fuel_capacity_litres=200.0,
                fuel_efficiency_km_per_litre=2.5,
            ),
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


def make_scenario(
    snapshot,
    *,
    attack_rate=0.50,
):
    truth = {
        row.bin_id: row.fill_percent
        for row in snapshot.bins
    }

    return generate_fdi_attack_from_truth(
        truth,
        master_seed=MASTER_SEED,
        attack_rate=attack_rate,
        attack_type=(
            "external_unauthenticated"
        ),
        replicate_id=7,
    )


def fake_config_bundle():
    files = (
        ConfigFileIdentity(
            path="configs/a.yaml",
            sha256="a" * 64,
            size_bytes=10,
        ),
        ConfigFileIdentity(
            path="configs/b.yaml",
            sha256="b" * 64,
            size_bytes=20,
        ),
    )

    return ConfigBundleIdentity(
        files=files,
        sha256="c" * 64,
    )


def fake_git(
    *,
    dirty=False,
):
    return GitIdentity(
        commit_sha="d" * 40,
        is_dirty=dirty,
    )


def test_default_fdi_config_bundle_exists_in_project() -> None:
    project_root = Path(
        __file__
    ).resolve().parents[1]

    bundle = (
        build_config_bundle_identity(
            project_root
        )
    )

    assert tuple(
        row.path
        for row in bundle.files
    ) == tuple(
        sorted(
            default_fdi_config_paths()
        )
    )

    assert len(
        bundle.sha256
    ) == 64

    assert all(
        len(row.sha256) == 64
        for row in bundle.files
    )


def test_config_bundle_hash_changes_with_exact_file_bytes(
    tmp_path,
) -> None:
    first = (
        tmp_path
        / "a.yaml"
    )

    second = (
        tmp_path
        / "b.yaml"
    )

    first.write_text(
        "value: 1\n",
        encoding="utf-8",
    )

    second.write_text(
        "value: 2\n",
        encoding="utf-8",
    )

    before = (
        build_config_bundle_identity(
            tmp_path,
            relative_paths=(
                "a.yaml",
                "b.yaml",
            ),
        )
    )

    first.write_text(
        "value: 1\n# changed bytes\n",
        encoding="utf-8",
    )

    after = (
        build_config_bundle_identity(
            tmp_path,
            relative_paths=(
                "a.yaml",
                "b.yaml",
            ),
        )
    )

    assert before.sha256 != after.sha256

    assert (
        before.files[0].sha256
        != after.files[0].sha256
    )


def test_config_bundle_is_independent_of_requested_path_order(
    tmp_path,
) -> None:
    (
        tmp_path
        / "a.yaml"
    ).write_text(
        "a: 1\n",
        encoding="utf-8",
    )

    (
        tmp_path
        / "b.yaml"
    ).write_text(
        "b: 2\n",
        encoding="utf-8",
    )

    first = build_config_bundle_identity(
        tmp_path,
        relative_paths=(
            "a.yaml",
            "b.yaml",
        ),
    )

    second = build_config_bundle_identity(
        tmp_path,
        relative_paths=(
            "b.yaml",
            "a.yaml",
        ),
    )

    assert first == second


def test_manifest_has_deterministic_pair_and_run_ids() -> None:
    snapshot = make_snapshot()

    scenario = make_scenario(
        snapshot
    )

    manifest = (
        build_fdi_experiment_manifest(
            snapshot=snapshot,
            scenario=scenario,
            git_identity=fake_git(),
            config_bundle=(
                fake_config_bundle()
            ),
            master_seed=MASTER_SEED,
        )
    )

    assert (
        manifest.pair_id
        == (
            "fdi"
            f"__{snapshot.scenario_id}"
            "__rep-0007"
            "__rate-0p5"
            "__external-unauthenticated"
        )
    )

    assert (
        manifest.run_id(
            "unprotected"
        )
        == (
            manifest.pair_id
            + "__unprotected"
        )
    )

    assert (
        manifest.run_id(
            "poa_verified"
        )
        == (
            manifest.pair_id
            + "__poa-verified"
        )
    )


def test_manifest_hash_is_deterministic() -> None:
    snapshot = make_snapshot()
    scenario = make_scenario(
        snapshot
    )

    kwargs = dict(
        snapshot=snapshot,
        scenario=scenario,
        git_identity=fake_git(),
        config_bundle=(
            fake_config_bundle()
        ),
        master_seed=MASTER_SEED,
    )

    first = (
        build_fdi_experiment_manifest(
            **kwargs
        )
    )

    second = (
        build_fdi_experiment_manifest(
            **kwargs
        )
    )

    assert (
        first.manifest_sha256
        == second.manifest_sha256
    )

    assert (
        first.to_dict()
        == second.to_dict()
    )


def test_manifest_hash_changes_when_git_commit_changes() -> None:
    snapshot = make_snapshot()
    scenario = make_scenario(
        snapshot
    )

    first = (
        build_fdi_experiment_manifest(
            snapshot=snapshot,
            scenario=scenario,
            git_identity=GitIdentity(
                commit_sha="a" * 40,
                is_dirty=False,
            ),
            config_bundle=(
                fake_config_bundle()
            ),
            master_seed=MASTER_SEED,
        )
    )

    second = (
        build_fdi_experiment_manifest(
            snapshot=snapshot,
            scenario=scenario,
            git_identity=GitIdentity(
                commit_sha="b" * 40,
                is_dirty=False,
            ),
            config_bundle=(
                fake_config_bundle()
            ),
            master_seed=MASTER_SEED,
        )
    )

    assert (
        first.manifest_sha256
        != second.manifest_sha256
    )


def test_manifest_rejects_dirty_git_by_default() -> None:
    snapshot = make_snapshot()

    with pytest.raises(
        ExperimentManifestError,
        match="clean Git",
    ):
        build_fdi_experiment_manifest(
            snapshot=snapshot,
            scenario=make_scenario(
                snapshot
            ),
            git_identity=fake_git(
                dirty=True
            ),
            config_bundle=(
                fake_config_bundle()
            ),
            master_seed=MASTER_SEED,
        )


def test_manifest_can_record_dirty_git_for_development_only() -> None:
    snapshot = make_snapshot()

    manifest = (
        build_fdi_experiment_manifest(
            snapshot=snapshot,
            scenario=make_scenario(
                snapshot
            ),
            git_identity=fake_git(
                dirty=True
            ),
            config_bundle=(
                fake_config_bundle()
            ),
            master_seed=MASTER_SEED,
            require_clean_git=False,
        )
    )

    assert manifest.git_dirty


def test_manifest_writer_is_idempotent_and_refuses_conflict(
    tmp_path,
) -> None:
    snapshot = make_snapshot()

    manifest = (
        build_fdi_experiment_manifest(
            snapshot=snapshot,
            scenario=make_scenario(
                snapshot
            ),
            git_identity=fake_git(),
            config_bundle=(
                fake_config_bundle()
            ),
            master_seed=MASTER_SEED,
        )
    )

    first_path = (
        write_fdi_experiment_manifest(
            manifest,
            tmp_path,
        )
    )

    second_path = (
        write_fdi_experiment_manifest(
            manifest,
            tmp_path,
        )
    )

    assert first_path == second_path

    document = json.loads(
        first_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        document[
            "manifest_sha256"
        ]
        == manifest.manifest_sha256
    )

    first_path.write_text(
        '{"corrupt": true}\n',
        encoding="utf-8",
    )

    with pytest.raises(
        ExperimentManifestError,
        match="refusing to overwrite",
    ):
        write_fdi_experiment_manifest(
            manifest,
            tmp_path,
        )


def test_written_manifest_contains_exact_config_hashes(
    tmp_path,
) -> None:
    snapshot = make_snapshot()

    manifest = (
        build_fdi_experiment_manifest(
            snapshot=snapshot,
            scenario=make_scenario(
                snapshot
            ),
            git_identity=fake_git(),
            config_bundle=(
                fake_config_bundle()
            ),
            master_seed=MASTER_SEED,
        )
    )

    path = write_fdi_experiment_manifest(
        manifest,
        tmp_path,
    )

    document = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        document[
            "configuration"
        ][
            "sha256"
        ]
        == "c" * 64
    )

    assert [
        row[
            "sha256"
        ]
        for row in document[
            "configuration"
        ][
            "files"
        ]
    ] == [
        "a" * 64,
        "b" * 64,
    ]
