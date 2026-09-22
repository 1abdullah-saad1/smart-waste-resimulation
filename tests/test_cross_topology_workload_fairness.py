import pytest

from smart_waste.experiments.physical_scenario_factory import (
    assemble_physical_scenario,
    load_primary_physical_scenario_spec,
)
from smart_waste.experiments.workload import (
    PhysicalWorkloadSpec,
    generate_physical_workload,
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


@pytest.fixture(scope="module")
def shared_workload():
    # Structural fairness fixture only.
    # These are not frozen research workload parameters.
    return generate_physical_workload(
        PhysicalWorkloadSpec(
            num_bins=1000,
            master_seed=20261001,
            replicate_id=0,
            minimum_initial_fill_percent=0.0,
            maximum_initial_fill_percent=100.0,
            minimum_fill_rate_percent_per_hour=0.1,
            maximum_fill_rate_percent_per_hour=1.0,
        )
    )


@pytest.fixture(scope="module")
def physical_spec():
    return load_primary_physical_scenario_spec()


def _assemble(layout, workload, physical_spec):
    return assemble_physical_scenario(
        road_graph=layout.topology.road_graph,
        depot_node=layout.depot_node,
        assignments=layout.bin_placement.assignments,
        workload=workload,
        spec=physical_spec,
    )


@pytest.fixture(scope="module")
def assemblies(shared_workload, physical_spec):
    layouts = {
        "manhattan": build_primary_manhattan_layout(),
        "superblock": build_primary_superblock_layout(),
        "hex": build_primary_hex_layout(),
        "radial_concentric": (
            build_primary_radial_concentric_layout()
        ),
    }

    return {
        name: _assemble(
            layout,
            shared_workload,
            physical_spec,
        )
        for name, layout in layouts.items()
    }


def test_exactly_four_topologies_are_frozen(
    assemblies,
) -> None:
    assert tuple(
        assemblies.keys()
    ) == (
        "manhattan",
        "superblock",
        "hex",
        "radial_concentric",
    )


def test_all_topologies_share_exact_workload(
    assemblies,
    shared_workload,
) -> None:
    assert {
        assembly.workload_sha256
        for assembly in assemblies.values()
    } == {
        shared_workload.sha256
    }

    assert {
        assembly.workload_id
        for assembly in assemblies.values()
    } == {
        shared_workload.workload_id
    }


def test_per_bin_fill_and_rate_are_identical(
    assemblies,
    shared_workload,
) -> None:
    for assembly in assemblies.values():
        bins = assembly.snapshot.bins

        assert len(bins) == 1000

        for bin_id, row in enumerate(bins):
            assert row.bin_id == bin_id

            assert (
                row.fill_percent
                == shared_workload.initial_fill_percent[
                    bin_id
                ]
            )

            assert (
                row.fill_rate_percent_per_hour
                == shared_workload.fill_rate_percent_per_hour[
                    bin_id
                ]
            )


def test_physical_fleet_scale_is_identical(
    assemblies,
) -> None:
    for assembly in assemblies.values():
        snapshot = assembly.snapshot

        assert len(snapshot.bins) == 1000
        assert len(snapshot.trucks) == 10

        assert (
            snapshot.depot.unloading_bays
            == 1
        )

        assert all(
            truck.capacity_tonnes
            == pytest.approx(10.0)
            for truck in snapshot.trucks
        )

        assert all(
            truck.speed_km_per_hour
            == pytest.approx(30.0)
            for truck in snapshot.trucks
        )

        assert all(
            truck.fuel_capacity_litres
            == pytest.approx(200.0)
            for truck in snapshot.trucks
        )

        assert all(
            truck.fuel_efficiency_km_per_litre
            == pytest.approx(2.5)
            for truck in snapshot.trucks
        )


def test_all_scenario_identities_are_distinct(
    assemblies,
) -> None:
    assert len(
        {
            assembly.scenario_sha256
            for assembly in assemblies.values()
        }
    ) == 4

    assert len(
        {
            assembly.scenario_id
            for assembly in assemblies.values()
        }
    ) == 4
