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
from smart_waste.movement.topologies.superblock_layout import (
    build_primary_superblock_layout,
)


@pytest.fixture(scope="module")
def shared_workload():
    """
    Structural fairness fixture only.

    These distribution bounds are test parameters, not frozen
    research benchmark parameters.
    """

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


def _assemble(
    layout,
    *,
    workload,
    physical_spec,
):
    return assemble_physical_scenario(
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


@pytest.fixture(scope="module")
def manhattan_assembly(
    shared_workload,
    physical_spec,
):
    return _assemble(
        build_primary_manhattan_layout(),
        workload=shared_workload,
        physical_spec=physical_spec,
    )


@pytest.fixture(scope="module")
def superblock_assembly(
    shared_workload,
    physical_spec,
):
    return _assemble(
        build_primary_superblock_layout(),
        workload=shared_workload,
        physical_spec=physical_spec,
    )


@pytest.fixture(scope="module")
def hex_assembly(
    shared_workload,
    physical_spec,
):
    return _assemble(
        build_primary_hex_layout(),
        workload=shared_workload,
        physical_spec=physical_spec,
    )


def _assemblies(
    manhattan_assembly,
    superblock_assembly,
    hex_assembly,
):
    return (
        manhattan_assembly,
        superblock_assembly,
        hex_assembly,
    )


def test_cross_topology_workload_identity_is_identical(
    shared_workload,
    manhattan_assembly,
    superblock_assembly,
    hex_assembly,
) -> None:
    assemblies = _assemblies(
        manhattan_assembly,
        superblock_assembly,
        hex_assembly,
    )

    assert {
        assembly.workload_sha256
        for assembly in assemblies
    } == {
        shared_workload.sha256
    }

    assert {
        assembly.workload_id
        for assembly in assemblies
    } == {
        shared_workload.workload_id
    }

    assert {
        assembly.workload_replicate_id
        for assembly in assemblies
    } == {
        0
    }


def test_cross_topology_per_bin_workload_is_identical(
    shared_workload,
    manhattan_assembly,
    superblock_assembly,
    hex_assembly,
) -> None:
    assemblies = _assemblies(
        manhattan_assembly,
        superblock_assembly,
        hex_assembly,
    )

    snapshots_by_topology = tuple(
        assembly.snapshot
        for assembly in assemblies
    )

    for snapshot in snapshots_by_topology:
        assert tuple(
            row.bin_id
            for row in snapshot.bins
        ) == tuple(
            range(
                1000
            )
        )

    for bin_id in range(
        1000
    ):
        rows = tuple(
            snapshot.bins[
                bin_id
            ]
            for snapshot
            in snapshots_by_topology
        )

        expected_fill = (
            shared_workload.initial_fill_percent[
                bin_id
            ]
        )

        expected_rate = (
            shared_workload.fill_rate_percent_per_hour[
                bin_id
            ]
        )

        assert {
            row.fill_percent
            for row in rows
        } == {
            expected_fill
        }

        assert {
            row.fill_rate_percent_per_hour
            for row in rows
        } == {
            expected_rate
        }

        assert len(
            {
                row.full_mass_kg
                for row in rows
            }
        ) == 1


def test_cross_topology_geometry_is_distinct(
    manhattan_assembly,
    superblock_assembly,
    hex_assembly,
) -> None:
    manhattan = (
        manhattan_assembly.snapshot
    )

    superblock = (
        superblock_assembly.snapshot
    )

    hexagonal = (
        hex_assembly.snapshot
    )

    assert (
        len(
            manhattan.road_nodes
        ),
        len(
            manhattan.road_edges
        ),
    ) == (
        1225,
        2380,
    )

    assert (
        len(
            superblock.road_nodes
        ),
        len(
            superblock.road_edges
        ),
    ) == (
        2665,
        3960,
    )

    assert (
        len(
            hexagonal.road_nodes
        ),
        len(
            hexagonal.road_edges
        ),
    ) == (
        3023,
        4402,
    )

    road_assignments = {
        tuple(
            row.road_node
            for row in snapshot.bins
        )
        for snapshot in (
            manhattan,
            superblock,
            hexagonal,
        )
    }

    assert len(
        road_assignments
    ) == 3


def test_cross_topology_scenario_identities_are_distinct(
    manhattan_assembly,
    superblock_assembly,
    hex_assembly,
) -> None:
    assemblies = _assemblies(
        manhattan_assembly,
        superblock_assembly,
        hex_assembly,
    )

    assert len(
        {
            assembly.scenario_sha256
            for assembly in assemblies
        }
    ) == 3

    assert len(
        {
            assembly.scenario_id
            for assembly in assemblies
        }
    ) == 3
