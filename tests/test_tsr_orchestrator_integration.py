import networkx as nx
import pytest

from smart_waste.collection.tsr import (
    TSRCollectionPolicy,
)
from smart_waste.models.bin import WasteBin
from smart_waste.models.depot import Depot
from smart_waste.models.truck import (
    Truck,
    TruckStatus,
)
from smart_waste.movement.road_graph import RoadGraph
from smart_waste.simulation.engine import (
    SimulationEngine,
)
from smart_waste.simulation.orchestrator import (
    SimulationOrchestrator,
)
from smart_waste.simulation.state import (
    SimulationState,
)


SERVICE_SECONDS = 36.0


def make_state(
    *,
    initial_load_tonnes: float = 0.0,
) -> SimulationState:
    graph = nx.Graph()

    graph.add_node(
        "depot",
        x=0.0,
        y=0.0,
    )

    for bin_id in range(4):
        node = f"bin-{bin_id}"

        graph.add_node(
            node,
            x=float(
                bin_id + 1
            ),
            y=0.0,
        )

        graph.add_edge(
            "depot",
            node,
            length_km=float(
                bin_id + 1
            ),
        )

    for bin_id in range(3):
        graph.add_edge(
            f"bin-{bin_id}",
            f"bin-{bin_id + 1}",
            length_km=1.0,
        )

    road = RoadGraph(
        graph
    )

    bins = {
        bin_id: WasteBin(
            bin_id=bin_id,
            road_node=f"bin-{bin_id}",
            fill_percent=50.0,
            fill_rate_percent_per_hour=0.0,
            full_mass_kg=440.0,
        )
        for bin_id in range(4)
    }

    trucks = {
        truck_id: Truck(
            truck_id=truck_id,
            current_node="depot",
            capacity_tonnes=10.0,
            speed_km_per_hour=30.0,
            fuel_capacity_litres=200.0,
            fuel_efficiency_km_per_litre=2.5,
            current_load_tonnes=(
                initial_load_tonnes
            ),
        )
        for truck_id in range(2)
    }

    depot = Depot(
        road_node="depot",
        unloading_bays=1,
        unload_time_minutes=11.0,
        refuel_rate_litres_per_minute=60.0,
    )

    return SimulationState(
        road_graph=road,
        bins=bins,
        trucks=trucks,
        depot=depot,
    )


def make_orchestrator(
    state: SimulationState,
) -> tuple[
    SimulationOrchestrator,
    TSRCollectionPolicy,
]:
    policy = TSRCollectionPolicy(
        road_graph=state.road_graph,
        truck_ids=(
            0,
            1,
        ),
        candidate_count=2,
        target_relative_range=0.10,
    )

    engine = SimulationEngine(
        state=state
    )

    orchestrator = SimulationOrchestrator(
        state=state,
        engine=engine,
        policy=policy,
        service_time_seconds=SERVICE_SECONDS,
    )

    return (
        orchestrator,
        policy,
    )


def test_tsr_end_to_end_collects_every_bin_once() -> None:
    state = make_state()

    orchestrator, policy = (
        make_orchestrator(
            state
        )
    )

    orchestrator.start()

    processed = (
        orchestrator.engine.run(
            max_events=1000
        )
    )

    assert processed > 0

    assert sorted(
        bin_id
        for _, bin_id in (
            orchestrator.completed_services
        )
    ) == [
        0,
        1,
        2,
        3,
    ]

    assert len(
        orchestrator.completed_services
    ) == 4

    assert policy.completed_bin_count == 4


def test_tsr_end_to_end_finishes_all_trucks_at_depot() -> None:
    state = make_state()

    orchestrator, _ = (
        make_orchestrator(
            state
        )
    )

    orchestrator.start()

    orchestrator.engine.run(
        max_events=1000
    )

    for truck in state.trucks.values():
        assert (
            truck.status
            == TruckStatus.FINISHED
        )

        assert truck.current_node == "depot"

        assert (
            truck.current_load_tonnes
            == pytest.approx(0.0)
        )

        assert (
            truck.fuel_remaining_litres
            == pytest.approx(200.0)
        )


def test_tsr_static_service_order_matches_planned_routes() -> None:
    state = make_state()

    orchestrator, policy = (
        make_orchestrator(
            state
        )
    )

    orchestrator.start()

    orchestrator.engine.run(
        max_events=1000
    )

    actual_by_truck = {
        truck_id: []
        for truck_id in policy.truck_ids
    }

    for (
        truck_id,
        bin_id,
    ) in orchestrator.completed_services:
        actual_by_truck[
            truck_id
        ].append(
            bin_id
        )

    for truck_id in policy.truck_ids:
        assert tuple(
            actual_by_truck[
                truck_id
            ]
        ) == policy.plan.route_for_truck(
            truck_id
        )


def test_tsr_routes_have_disjoint_bin_assignments() -> None:
    state = make_state()

    orchestrator, policy = (
        make_orchestrator(
            state
        )
    )

    orchestrator.start()

    route_0 = set(
        policy.plan.route_for_truck(
            0
        )
    )

    route_1 = set(
        policy.plan.route_for_truck(
            1
        )
    )

    assert route_0.isdisjoint(
        route_1
    )

    assert (
        route_0
        | route_1
    ) == {
        0,
        1,
        2,
        3,
    }


def test_tsr_releases_all_reservations_after_completion() -> None:
    state = make_state()

    orchestrator, _ = (
        make_orchestrator(
            state
        )
    )

    orchestrator.start()

    orchestrator.engine.run(
        max_events=1000
    )

    assert len(
        orchestrator.reservation_book
    ) == 0


def test_tsr_does_not_skip_or_reorder_after_capacity_return() -> None:
    state = make_state(
        initial_load_tonnes=9.9
    )

    orchestrator, policy = (
        make_orchestrator(
            state
        )
    )

    planned_first = {}

    # start() initializes policy before dispatch.
    results = orchestrator.start()

    for truck_id in policy.truck_ids:
        planned_first[
            truck_id
        ] = policy.plan.route_for_truck(
            truck_id
        )[0]

    # Initial load prevents immediate collection, so both trucks
    # must physically service depot before resuming the same fixed
    # TSR route.
    assert any(
        result.depot_return_reason
        is not None
        for result in results
    )

    orchestrator.engine.run(
        max_events=1000
    )

    actual_first = {}

    for (
        truck_id,
        bin_id,
    ) in orchestrator.completed_services:
        actual_first.setdefault(
            truck_id,
            bin_id,
        )

    assert actual_first == planned_first

    for truck_id in policy.truck_ids:
        assert tuple(
            bin_id
            for (
                completed_truck_id,
                bin_id,
            ) in orchestrator.completed_services
            if completed_truck_id
            == truck_id
        ) == policy.plan.route_for_truck(
            truck_id
        )


def test_global_completion_wakes_previous_no_candidate_truck() -> None:
    state = make_state()

    orchestrator, policy = (
        make_orchestrator(
            state
        )
    )

    orchestrator.start()

    orchestrator.engine.run(
        max_events=1000
    )

    # In multi-truck static TSR, one truck can finish its own route
    # before the other one. At that point the global policy is not
    # complete, so that truck legitimately receives NO_CANDIDATE.
    #
    # When the last remaining route completes, every idle truck
    # must be awakened and passed through the normal terminal depot
    # lifecycle rather than remaining stranded in IDLE state.
    no_candidate_trucks = {
        result.truck_id
        for result in orchestrator.dispatch_history
        if result.action.value
        == "no_candidate"
    }

    assert no_candidate_trucks

    for truck_id in no_candidate_trucks:
        truck = state.trucks[
            truck_id
        ]

        assert truck.status == TruckStatus.FINISHED
        assert truck.current_node == "depot"
        assert (
            truck.current_load_tonnes
            == pytest.approx(0.0)
        )
        assert (
            truck.fuel_remaining_litres
            == pytest.approx(
                truck.fuel_capacity_litres
            )
        )

    # All TSR work must still have been completed exactly once.
    assert policy.completed_bin_count == 4

    assert sorted(
        bin_id
        for _, bin_id
        in orchestrator.completed_services
    ) == [
        0,
        1,
        2,
        3,
    ]

    assert len(
        orchestrator.reservation_book
    ) == 0
