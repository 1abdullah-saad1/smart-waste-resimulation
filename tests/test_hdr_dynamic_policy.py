import networkx as nx
import pytest

from smart_waste.collection.base import (
    BinPolicyView,
    PolicyView,
    TruckPolicyView,
)
from smart_waste.collection.hdr import (
    HDRCollectionPolicy,
    HDRPolicyError,
    build_hdr_distance_oracle,
)
from smart_waste.models.truck import (
    TruckStatus,
)
from smart_waste.movement.road_graph import (
    RoadGraph,
)


def make_view_and_graph(
    *,
    truck_node: str = "depot",
    reservations: dict[int, int] | None = None,
    reported: dict[int, float] | None = None,
) -> tuple[
    PolicyView,
    RoadGraph,
]:
    graph = nx.Graph()

    graph.add_node(
        "depot",
        x=0.0,
        y=0.0,
    )

    graph.add_node(
        "a",
        x=1.0,
        y=0.0,
    )

    graph.add_node(
        "b",
        x=2.0,
        y=0.0,
    )

    graph.add_node(
        "c",
        x=100.0,
        y=0.0,
    )

    # Physical road distances intentionally differ from coordinate
    # appearance. HDR must use length_km, never Euclidean geometry.
    graph.add_edge(
        "depot",
        "a",
        length_km=8.0,
    )

    graph.add_edge(
        "depot",
        "b",
        length_km=3.0,
    )

    graph.add_edge(
        "depot",
        "c",
        length_km=3.0,
    )

    graph.add_edge(
        "a",
        "b",
        length_km=1.0,
    )

    graph.add_edge(
        "b",
        "c",
        length_km=9.0,
    )

    road = RoadGraph(
        graph
    )

    reported_map = {
        0: 90.0,
        1: 90.0,
        2: 90.0,
    }

    if reported is not None:
        reported_map.update(
            reported
        )

    reservation_map = (
        {}
        if reservations is None
        else reservations
    )

    bins = (
        BinPolicyView(
            bin_id=0,
            road_node="a",
            fill_percent=90.0,
            fill_rate_percent_per_hour=0.0,
            waste_mass_tonnes=0.396,
            reported_fill_percent=(
                reported_map[0]
            ),
            reserved_by_truck_id=(
                reservation_map.get(0)
            ),
        ),
        BinPolicyView(
            bin_id=1,
            road_node="b",
            fill_percent=90.0,
            fill_rate_percent_per_hour=0.0,
            waste_mass_tonnes=0.396,
            reported_fill_percent=(
                reported_map[1]
            ),
            reserved_by_truck_id=(
                reservation_map.get(1)
            ),
        ),
        BinPolicyView(
            bin_id=2,
            road_node="c",
            fill_percent=90.0,
            fill_rate_percent_per_hour=0.0,
            waste_mass_tonnes=0.396,
            reported_fill_percent=(
                reported_map[2]
            ),
            reserved_by_truck_id=(
                reservation_map.get(2)
            ),
        ),
    )

    trucks = (
        TruckPolicyView(
            truck_id=0,
            current_node=truck_node,
            remaining_capacity_tonnes=10.0,
            fuel_remaining_litres=200.0,
            status=TruckStatus.IDLE,
        ),
        TruckPolicyView(
            truck_id=1,
            current_node="depot",
            remaining_capacity_tonnes=10.0,
            fuel_remaining_litres=200.0,
            status=TruckStatus.IDLE,
        ),
    )

    return (
        PolicyView(
            current_time_hours=0.0,
            depot_node="depot",
            bins=bins,
            trucks=trucks,
        ),
        road,
    )


def make_policy(
    view: PolicyView,
    road: RoadGraph,
) -> HDRCollectionPolicy:
    policy = HDRCollectionPolicy(
        road_graph=road,
        threshold_percent=80.0,
    )

    policy.initialize(
        view
    )

    return policy


def test_initialize_freezes_eligible_pending_set() -> None:
    view, road = make_view_and_graph(
        reported={
            0: 79.0,
            1: 80.0,
            2: 100.0,
        }
    )

    policy = make_policy(
        view,
        road,
    )

    assert (
        policy.eligibility_snapshot.eligible_bin_ids
        == (
            1,
            2,
        )
    )

    assert policy.pending_bin_ids == (
        1,
        2,
    )


def test_selection_uses_shortest_road_distance() -> None:
    view, road = make_view_and_graph()

    policy = make_policy(
        view,
        road,
    )

    selected = policy.select_next_bin(
        view=view,
        truck_id=0,
    )

    # a looks geometrically nearest, but road shortest paths are:
    #
    # depot -> a = depot -> b -> a = 4 km
    # depot -> b = 3 km
    # depot -> c = 3 km
    #
    # b and c tie at 3 km; smaller bin_id 1 wins.
    assert selected == 1


def test_equal_road_distance_tie_breaks_by_bin_id() -> None:
    view, road = make_view_and_graph()

    policy = make_policy(
        view,
        road,
    )

    assert (
        policy.select_next_bin(
            view=view,
            truck_id=0,
        )
        == 1
    )


def test_reserved_candidate_is_excluded() -> None:
    view, road = make_view_and_graph(
        reservations={
            1: 99,
        }
    )

    policy = make_policy(
        view,
        road,
    )

    # bin 1 is nearest but reserved.
    # depot -> bin 2 = 3 km, so bin 2 is selected.
    assert (
        policy.select_next_bin(
            view=view,
            truck_id=0,
        )
        == 2
    )


def test_all_pending_reserved_returns_none_without_completion() -> None:
    view, road = make_view_and_graph(
        reservations={
            0: 10,
            1: 11,
            2: 12,
        }
    )

    policy = make_policy(
        view,
        road,
    )

    assert (
        policy.select_next_bin(
            view=view,
            truck_id=0,
        )
        is None
    )

    assert not policy.is_complete(
        view
    )

    assert policy.pending_bin_ids == (
        0,
        1,
        2,
    )


def test_successful_service_removes_only_that_pending_bin() -> None:
    view, road = make_view_and_graph()

    policy = make_policy(
        view,
        road,
    )

    policy.on_service_complete(
        view=view,
        truck_id=0,
        bin_id=1,
    )

    assert policy.completed_bin_ids == (
        1,
    )

    assert policy.pending_bin_ids == (
        0,
        2,
    )


def test_below_threshold_bins_do_not_block_completion() -> None:
    view, road = make_view_and_graph(
        reported={
            0: 20.0,
            1: 80.0,
            2: 20.0,
        }
    )

    policy = make_policy(
        view,
        road,
    )

    assert policy.pending_bin_ids == (
        1,
    )

    policy.on_service_complete(
        view=view,
        truck_id=0,
        bin_id=1,
    )

    assert policy.is_complete(
        view
    )


def test_dynamic_reselection_uses_new_truck_location() -> None:
    initial_view, road = make_view_and_graph()

    policy = make_policy(
        initial_view,
        road,
    )

    # From depot, bin 1 wins.
    assert (
        policy.select_next_bin(
            view=initial_view,
            truck_id=0,
        )
        == 1
    )

    policy.on_service_complete(
        view=initial_view,
        truck_id=0,
        bin_id=1,
    )

    # Simulate the next policy snapshot after physical service:
    # truck 0 is now at road node b.
    next_view, _ = make_view_and_graph(
        truck_node="b",
    )

    # bin 0 is now 1 km away from b, while bin 2 is 6 km via
    # depot (b -> depot -> c = 3 + 3).
    assert (
        policy.select_next_bin(
            view=next_view,
            truck_id=0,
        )
        == 0
    )


def test_later_telemetry_does_not_change_frozen_membership() -> None:
    initial_view, road = make_view_and_graph(
        reported={
            0: 90.0,
            1: 20.0,
            2: 20.0,
        }
    )

    policy = make_policy(
        initial_view,
        road,
    )

    assert policy.pending_bin_ids == (
        0,
    )

    later_view, _ = make_view_and_graph(
        reported={
            0: 0.0,
            1: 100.0,
            2: 100.0,
        }
    )

    # Snapshot membership remains frozen. New reports do not add
    # bins 1 or 2 during the same benchmark run.
    assert policy.pending_bin_ids == (
        0,
    )

    assert (
        policy.select_next_bin(
            view=later_view,
            truck_id=0,
        )
        == 0
    )


def test_nonpending_service_completion_is_rejected() -> None:
    view, road = make_view_and_graph(
        reported={
            0: 20.0,
            1: 90.0,
            2: 20.0,
        }
    )

    policy = make_policy(
        view,
        road,
    )

    with pytest.raises(
        HDRPolicyError,
        match="not pending",
    ):
        policy.on_service_complete(
            view=view,
            truck_id=0,
            bin_id=0,
        )


def test_distance_oracle_reuses_same_source_search() -> None:
    view, road = make_view_and_graph()

    oracle = build_hdr_distance_oracle(
        view=view,
        road_graph=road,
    )

    assert oracle.source_search_count == 0

    assert oracle.distance_km(
        "depot",
        "a",
    ) == pytest.approx(
        4.0
    )

    assert oracle.source_search_count == 1

    assert oracle.distance_km(
        "depot",
        "c",
    ) == pytest.approx(
        3.0
    )

    assert oracle.source_search_count == 1


def test_policy_cannot_initialize_twice() -> None:
    view, road = make_view_and_graph()

    policy = make_policy(
        view,
        road,
    )

    with pytest.raises(
        HDRPolicyError,
        match="already initialized",
    ):
        policy.initialize(
            view
        )
