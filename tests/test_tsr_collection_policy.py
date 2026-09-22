import networkx as nx
import pytest

from smart_waste.collection.base import (
    PolicyView,
)
from smart_waste.collection.tsr import (
    TSRCollectionPolicy,
    TSRPolicyError,
)
from smart_waste.models.bin import WasteBin
from smart_waste.models.depot import Depot
from smart_waste.models.truck import (
    Truck,
)
from smart_waste.movement.road_graph import RoadGraph
from smart_waste.simulation.policy_view import (
    build_policy_view,
)
from smart_waste.simulation.reservations import (
    BinReservationBook,
)
from smart_waste.simulation.state import (
    SimulationState,
)


def make_state(
    *,
    truck_count: int = 2,
) -> SimulationState:
    graph = nx.Graph()

    graph.add_node(
        "depot",
        x=0.0,
        y=0.0,
    )

    # Six bins in longitudinal order.
    for bin_id in range(6):
        graph.add_node(
            f"bin-{bin_id}",
            x=float(
                bin_id + 1
            ),
            y=0.0,
        )

        graph.add_edge(
            "depot",
            f"bin-{bin_id}",
            length_km=float(
                bin_id + 1
            ),
        )

    # Add local links so road distance is not restricted to
    # independent star legs.
    for bin_id in range(5):
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
        for bin_id in range(6)
    }

    trucks = {
        truck_id: Truck(
            truck_id=truck_id,
            current_node="depot",
            capacity_tonnes=10.0,
            speed_km_per_hour=30.0,
            fuel_capacity_litres=200.0,
            fuel_efficiency_km_per_litre=2.5,
        )
        for truck_id in range(
            truck_count
        )
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


def make_policy(
    state: SimulationState,
) -> TSRCollectionPolicy:
    return TSRCollectionPolicy(
        road_graph=state.road_graph,
        truck_ids=tuple(
            state.trucks
        ),
        candidate_count=2,
        target_relative_range=0.10,
    )


def test_policy_initializes_static_routes() -> None:
    state = make_state()
    policy = make_policy(
        state
    )

    policy.initialize(
        build_policy_view(
            state
        )
    )

    assert policy.initialized

    all_bins = sorted(
        bin_id
        for _, route in (
            policy.plan.routes_by_truck
        )
        for bin_id in route
    )

    assert all_bins == list(
        range(6)
    )


def test_policy_cannot_initialize_twice() -> None:
    state = make_state()
    policy = make_policy(
        state
    )

    view = build_policy_view(
        state
    )

    policy.initialize(
        view
    )

    with pytest.raises(
        TSRPolicyError,
        match="already initialized",
    ):
        policy.initialize(
            view
        )


def test_each_bin_is_owned_by_exactly_one_static_route() -> None:
    state = make_state()
    policy = make_policy(
        state
    )

    policy.initialize(
        build_policy_view(
            state
        )
    )

    planned = [
        bin_id
        for _, route in (
            policy.plan.routes_by_truck
        )
        for bin_id in route
    ]

    assert len(planned) == 6
    assert len(set(planned)) == 6


def test_select_returns_first_static_route_bin() -> None:
    state = make_state()
    policy = make_policy(
        state
    )

    view = build_policy_view(
        state
    )

    policy.initialize(
        view
    )

    route = policy.plan.route_for_truck(
        0
    )

    selected = policy.select_next_bin(
        view=view,
        truck_id=0,
    )

    assert selected == route[0]


def test_route_does_not_advance_without_service_completion() -> None:
    state = make_state()
    policy = make_policy(
        state
    )

    view = build_policy_view(
        state
    )

    policy.initialize(
        view
    )

    first = policy.select_next_bin(
        view=view,
        truck_id=0,
    )

    second = policy.select_next_bin(
        view=view,
        truck_id=0,
    )

    assert second == first


def test_service_completion_advances_exactly_one_position() -> None:
    state = make_state()
    policy = make_policy(
        state
    )

    view = build_policy_view(
        state
    )

    policy.initialize(
        view
    )

    route = policy.plan.route_for_truck(
        0
    )

    first = policy.select_next_bin(
        view=view,
        truck_id=0,
    )

    assert first == route[0]

    policy.on_service_complete(
        view=view,
        truck_id=0,
        bin_id=first,
    )

    assert policy.completed_bin_count == 1

    second = policy.select_next_bin(
        view=view,
        truck_id=0,
    )

    assert second == route[1]


def test_out_of_order_service_completion_is_rejected() -> None:
    state = make_state()
    policy = make_policy(
        state
    )

    view = build_policy_view(
        state
    )

    policy.initialize(
        view
    )

    route = policy.plan.route_for_truck(
        0
    )

    with pytest.raises(
        TSRPolicyError,
        match="expected static TSR bin",
    ):
        policy.on_service_complete(
            view=view,
            truck_id=0,
            bin_id=route[1],
        )


def test_reserved_static_bin_is_rejected() -> None:
    state = make_state()
    policy = make_policy(
        state
    )

    initial_view = build_policy_view(
        state
    )

    policy.initialize(
        initial_view
    )

    next_bin = (
        policy.plan.route_for_truck(
            0
        )[0]
    )

    book = BinReservationBook()

    book.reserve(
        bin_id=next_bin,
        truck_id=99,
    )

    reserved_view = build_policy_view(
        state,
        reservations=book.snapshot(),
    )

    with pytest.raises(
        TSRPolicyError,
        match="unexpectedly reserved",
    ):
        policy.select_next_bin(
            view=reserved_view,
            truck_id=0,
        )


def test_policy_complete_only_after_all_routes_advance() -> None:
    state = make_state()
    policy = make_policy(
        state
    )

    view = build_policy_view(
        state
    )

    policy.initialize(
        view
    )

    assert not policy.is_complete(
        view
    )

    for truck_id in policy.truck_ids:
        route = policy.plan.route_for_truck(
            truck_id
        )

        for bin_id in route:
            assert (
                policy.select_next_bin(
                    view=view,
                    truck_id=truck_id,
                )
                == bin_id
            )

            policy.on_service_complete(
                view=view,
                truck_id=truck_id,
                bin_id=bin_id,
            )

    assert policy.is_complete(
        view
    )


def test_finished_truck_route_returns_none() -> None:
    state = make_state()
    policy = make_policy(
        state
    )

    view = build_policy_view(
        state
    )

    policy.initialize(
        view
    )

    route = policy.plan.route_for_truck(
        0
    )

    for bin_id in route:
        policy.on_service_complete(
            view=view,
            truck_id=0,
            bin_id=bin_id,
        )

    assert (
        policy.select_next_bin(
            view=view,
            truck_id=0,
        )
        is None
    )


def test_primary_partition_flag_rejects_small_fixture() -> None:
    state = make_state()

    policy = TSRCollectionPolicy(
        road_graph=state.road_graph,
        truck_ids=tuple(
            state.trucks
        ),
        require_primary_partition=True,
    )

    with pytest.raises(
        Exception,
        match="10 zones",
    ):
        policy.initialize(
            build_policy_view(
                state
            )
        )
