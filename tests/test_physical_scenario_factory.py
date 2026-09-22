from pathlib import Path

import pytest

from smart_waste.experiments.physical_scenario_factory import (
    PhysicalScenarioFactoryError,
    assemble_physical_scenario,
    build_pristine_simulation_state,
    load_primary_physical_scenario_spec,
)
from smart_waste.experiments.scenario_snapshot import (
    build_simulation_state,
    capture_physical_scenario,
)
from smart_waste.experiments.workload import (
    PhysicalWorkloadSpec,
    generate_physical_workload,
)
from smart_waste.models.truck import TruckStatus
from smart_waste.movement.bin_placement import (
    BinRoadAssignment,
)
from smart_waste.movement.topologies.manhattan_layout import (
    build_primary_manhattan_layout,
)


@pytest.fixture(scope="module")
def layout():
    return build_primary_manhattan_layout()


@pytest.fixture(scope="module")
def workload():
    # Structural test fixture only.
    # These bounds are NOT benchmark parameters.
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
    return load_primary_physical_scenario_spec(
        Path("configs")
    )


@pytest.fixture(scope="module")
def assembly(
    layout,
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


def test_primary_config_adapter_loads_exact_physics(
    physical_spec,
) -> None:
    assert (
        physical_spec.bin_full_mass_kg
        == pytest.approx(440.0)
    )

    assert physical_spec.truck_count == 10

    assert (
        physical_spec.truck_capacity_tonnes
        == pytest.approx(10.0)
    )

    assert (
        physical_spec.truck_speed_km_per_hour
        == pytest.approx(30.0)
    )

    assert (
        physical_spec.fuel_tank_capacity_litres
        == pytest.approx(200.0)
    )

    assert (
        physical_spec.fuel_efficiency_km_per_litre
        == pytest.approx(2.5)
    )

    assert (
        physical_spec.depot_unloading_bays
        == 1
    )

    assert (
        physical_spec.depot_unload_time_minutes
        == pytest.approx(11.0)
    )

    assert (
        physical_spec.depot_refuel_rate_litres_per_minute
        == pytest.approx(60.0)
    )


def test_pristine_state_has_exact_primary_scale(
    layout,
    workload,
    physical_spec,
) -> None:
    state = build_pristine_simulation_state(
        road_graph=(
            layout.topology.road_graph
        ),
        depot_node=layout.depot_node,
        assignments=(
            layout.bin_placement.assignments
        ),
        workload=workload,
        spec=physical_spec,
    )

    assert state.current_time_hours == 0.0

    assert len(state.bins) == 1000
    assert len(state.trucks) == 10

    assert (
        state.depot.road_node
        == 612
    )


def test_bins_match_assignment_and_workload_exactly(
    layout,
    workload,
    physical_spec,
) -> None:
    state = build_pristine_simulation_state(
        road_graph=(
            layout.topology.road_graph
        ),
        depot_node=layout.depot_node,
        assignments=(
            layout.bin_placement.assignments
        ),
        workload=workload,
        spec=physical_spec,
    )

    for assignment in (
        layout.bin_placement.assignments
    ):
        bin_ = state.bins[
            assignment.bin_id
        ]

        assert (
            bin_.road_node
            == assignment.road_node
        )

        assert (
            bin_.fill_percent
            == workload.initial_fill_percent[
                assignment.bin_id
            ]
        )

        assert (
            bin_.fill_rate_percent_per_hour
            == workload.fill_rate_percent_per_hour[
                assignment.bin_id
            ]
        )

        assert (
            bin_.full_mass_kg
            == pytest.approx(440.0)
        )

        assert bin_.waste_age_hours == 0.0
        assert bin_.collected_count == 0


def test_trucks_are_pristine_at_central_depot(
    layout,
    workload,
    physical_spec,
) -> None:
    state = build_pristine_simulation_state(
        road_graph=(
            layout.topology.road_graph
        ),
        depot_node=layout.depot_node,
        assignments=(
            layout.bin_placement.assignments
        ),
        workload=workload,
        spec=physical_spec,
    )

    for truck_id, truck in state.trucks.items():
        assert truck.truck_id == truck_id

        assert (
            truck.current_node
            == layout.depot_node
        )

        assert truck.status == TruckStatus.IDLE
        assert truck.current_load_tonnes == 0.0

        assert (
            truck.fuel_remaining_litres
            == pytest.approx(200.0)
        )

        assert (
            truck.route_nodes
            == [layout.depot_node]
        )

        assert truck.cumulative_distance_km == 0.0

        assert (
            truck.cumulative_fuel_used_litres
            == 0.0
        )

        assert (
            truck.cumulative_refuelled_litres
            == 0.0
        )


def test_depot_matches_primary_physics(
    assembly,
) -> None:
    depot = assembly.snapshot.depot

    assert depot.road_node == 612
    assert depot.unloading_bays == 1

    assert (
        depot.unload_time_minutes
        == pytest.approx(11.0)
    )

    assert (
        depot.refuel_rate_litres_per_minute
        == pytest.approx(60.0)
    )


def test_snapshot_round_trip_preserves_hash(
    assembly,
) -> None:
    rebuilt = build_simulation_state(
        assembly.snapshot
    )

    recaptured = capture_physical_scenario(
        rebuilt
    )

    assert (
        recaptured.sha256
        == assembly.snapshot.sha256
    )

    assert (
        recaptured.scenario_id
        == assembly.snapshot.scenario_id
    )


def test_assembly_is_deterministic(
    layout,
    workload,
    physical_spec,
) -> None:
    first = assemble_physical_scenario(
        road_graph=(
            layout.topology.road_graph
        ),
        depot_node=layout.depot_node,
        assignments=(
            layout.bin_placement.assignments
        ),
        workload=workload,
        spec=physical_spec,
    )

    second = assemble_physical_scenario(
        road_graph=(
            layout.topology.road_graph
        ),
        depot_node=layout.depot_node,
        assignments=(
            layout.bin_placement.assignments
        ),
        workload=workload,
        spec=physical_spec,
    )

    assert (
        first.scenario_sha256
        == second.scenario_sha256
    )

    assert first.snapshot == second.snapshot


def test_workload_identity_is_preserved_verbatim(
    assembly,
    workload,
) -> None:
    assert (
        assembly.workload_id
        == workload.workload_id
    )

    assert (
        assembly.workload_sha256
        == workload.sha256
    )

    assert (
        assembly.workload_replicate_id
        == workload.spec.replicate_id
    )


def test_scenario_hash_changes_when_road_physics_changes(
    layout,
    workload,
    physical_spec,
) -> None:
    state = build_pristine_simulation_state(
        road_graph=(
            layout.topology.road_graph
        ),
        depot_node=layout.depot_node,
        assignments=(
            layout.bin_placement.assignments
        ),
        workload=workload,
        spec=physical_spec,
    )

    baseline = capture_physical_scenario(
        state
    )

    source, target = next(
        iter(
            state.road_graph.graph.edges
        )
    )

    state.road_graph.graph.edges[
        source,
        target,
    ][
        "length_km"
    ] += 0.001

    changed = capture_physical_scenario(
        state
    )

    assert baseline.sha256 != changed.sha256

    # The workload itself did not change.
    assert (
        workload.sha256
        == workload.sha256
    )


def test_state_graph_does_not_alias_layout_graph(
    layout,
    workload,
    physical_spec,
) -> None:
    state = build_pristine_simulation_state(
        road_graph=(
            layout.topology.road_graph
        ),
        depot_node=layout.depot_node,
        assignments=(
            layout.bin_placement.assignments
        ),
        workload=workload,
        spec=physical_spec,
    )

    assert (
        state.road_graph.graph
        is not layout.topology.road_graph.graph
    )

    source, target = next(
        iter(
            state.road_graph.graph.edges
        )
    )

    original = (
        layout.topology.road_graph.graph.edges[
            source,
            target,
        ][
            "length_km"
        ]
    )

    state.road_graph.graph.edges[
        source,
        target,
    ][
        "length_km"
    ] += 1.0

    assert (
        layout.topology.road_graph.graph.edges[
            source,
            target,
        ][
            "length_km"
        ]
        == original
    )


def test_rejects_workload_count_mismatch(
    layout,
    physical_spec,
) -> None:
    short_workload = generate_physical_workload(
        PhysicalWorkloadSpec(
            num_bins=999,
            master_seed=20261001,
            replicate_id=0,
            minimum_initial_fill_percent=0.0,
            maximum_initial_fill_percent=100.0,
            minimum_fill_rate_percent_per_hour=0.1,
            maximum_fill_rate_percent_per_hour=1.0,
        )
    )

    with pytest.raises(
        PhysicalScenarioFactoryError,
        match="count does not match workload",
    ):
        build_pristine_simulation_state(
            road_graph=(
                layout.topology.road_graph
            ),
            depot_node=layout.depot_node,
            assignments=(
                layout.bin_placement.assignments
            ),
            workload=short_workload,
            spec=physical_spec,
        )


def test_rejects_noncanonical_bin_ids(
    layout,
    workload,
    physical_spec,
) -> None:
    assignments = list(
        layout.bin_placement.assignments
    )

    assignments[
        0
    ] = BinRoadAssignment(
        bin_id=1001,
        road_node=assignments[0].road_node,
    )

    with pytest.raises(
        PhysicalScenarioFactoryError,
        match="canonical contiguous",
    ):
        build_pristine_simulation_state(
            road_graph=(
                layout.topology.road_graph
            ),
            depot_node=layout.depot_node,
            assignments=assignments,
            workload=workload,
            spec=physical_spec,
        )


def test_rejects_depot_as_bin_node(
    layout,
    workload,
    physical_spec,
) -> None:
    assignments = list(
        layout.bin_placement.assignments
    )

    assignments[
        0
    ] = BinRoadAssignment(
        bin_id=0,
        road_node=layout.depot_node,
    )

    with pytest.raises(
        PhysicalScenarioFactoryError,
        match="cannot also host a bin",
    ):
        build_pristine_simulation_state(
            road_graph=(
                layout.topology.road_graph
            ),
            depot_node=layout.depot_node,
            assignments=assignments,
            workload=workload,
            spec=physical_spec,
        )


def test_snapshot_excludes_run_level_service_time(
    assembly,
) -> None:
    assert not hasattr(
        assembly.snapshot,
        "bin_service_time_seconds",
    )
