import networkx as nx
import pytest

from smart_waste.collection.base import (
    BinPolicyView,
    PolicyView,
)
from smart_waste.collection.tsr import (
    TSRZone,
    balance_adjacent_zones,
    build_tsr_distance_oracle,
    build_zone_distance_matrix,
    plan_all_zone_routes,
)
from smart_waste.movement.road_graph import RoadGraph


def make_case():
    graph = nx.Graph()

    graph.add_node(
        "depot",
        x=0.0,
        y=0.0,
    )

    for bin_id in range(6):
        graph.add_node(
            f"bin-{bin_id}",
            x=float(bin_id + 1),
            y=0.0,
        )

    # Connected chain.
    previous = "depot"

    for bin_id in range(6):
        node = f"bin-{bin_id}"

        graph.add_edge(
            previous,
            node,
            length_km=1.0,
        )

        previous = node

    road = RoadGraph(
        graph
    )

    view = PolicyView(
        current_time_hours=0.0,
        depot_node="depot",
        bins=tuple(
            BinPolicyView(
                bin_id=bin_id,
                road_node=f"bin-{bin_id}",
                fill_percent=50.0,
                fill_rate_percent_per_hour=0.0,
                waste_mass_tonnes=0.22,
            )
            for bin_id in range(6)
        ),
        trucks=(),
    )

    zones = (
        TSRZone(
            truck_id=0,
            bin_ids=(
                0,
                1,
                2,
            ),
        ),
        TSRZone(
            truck_id=1,
            bin_ids=(
                3,
                4,
                5,
            ),
        ),
    )

    return (
        view,
        road,
        zones,
    )


def test_oracle_matches_shortest_road_distance() -> None:
    view, road, _ = make_case()

    oracle = build_tsr_distance_oracle(
        view=view,
        road_graph=road,
    )

    assert oracle.distance_km(
        "depot",
        "bin-5",
    ) == pytest.approx(
        6.0
    )

    assert oracle.distance_km(
        "bin-1",
        "bin-4",
    ) == pytest.approx(
        3.0
    )


def test_same_source_runs_dijkstra_only_once() -> None:
    view, road, _ = make_case()

    oracle = build_tsr_distance_oracle(
        view=view,
        road_graph=road,
    )

    assert oracle.source_search_count == 0

    oracle.distance_km(
        "depot",
        "bin-0",
    )

    assert oracle.source_search_count == 1

    oracle.distance_km(
        "depot",
        "bin-5",
    )

    assert oracle.source_search_count == 1


def test_rebuilding_same_zone_reuses_cached_sources() -> None:
    view, road, zones = make_case()

    oracle = build_tsr_distance_oracle(
        view=view,
        road_graph=road,
    )

    first = build_zone_distance_matrix(
        zone=zones[0],
        view=view,
        road_graph=road,
        distance_oracle=oracle,
    )

    searches_after_first = (
        oracle.source_search_count
    )

    second = build_zone_distance_matrix(
        zone=zones[0],
        view=view,
        road_graph=road,
        distance_oracle=oracle,
    )

    assert second == first

    assert (
        oracle.source_search_count
        == searches_after_first
    )


def test_all_zone_planning_is_bounded_by_unique_service_sources() -> None:
    view, road, zones = make_case()

    oracle = build_tsr_distance_oracle(
        view=view,
        road_graph=road,
    )

    plan_all_zone_routes(
        zones=zones,
        view=view,
        road_graph=road,
        distance_oracle=oracle,
    )

    # depot + six unique bin road nodes.
    assert (
        oracle.source_search_count
        <= 7
    )

    assert (
        oracle.cached_source_count
        == oracle.source_search_count
    )


def test_repeated_all_zone_planning_adds_no_dijkstra_searches() -> None:
    view, road, zones = make_case()

    oracle = build_tsr_distance_oracle(
        view=view,
        road_graph=road,
    )

    first = plan_all_zone_routes(
        zones=zones,
        view=view,
        road_graph=road,
        distance_oracle=oracle,
    )

    first_count = (
        oracle.source_search_count
    )

    second = plan_all_zone_routes(
        zones=zones,
        view=view,
        road_graph=road,
        distance_oracle=oracle,
    )

    assert first == second

    assert (
        oracle.source_search_count
        == first_count
    )


def test_balancing_reuses_shared_oracle() -> None:
    view, road, zones = make_case()

    oracle = build_tsr_distance_oracle(
        view=view,
        road_graph=road,
    )

    result = balance_adjacent_zones(
        zones=zones,
        view=view,
        road_graph=road,
        candidate_count=3,
        target_relative_range=0.10,
        distance_oracle=oracle,
    )

    assert result.final_routes

    # Regardless of how many candidate exchanges are evaluated,
    # each unique registered source can trigger Dijkstra at most
    # once.
    assert (
        oracle.source_search_count
        <= len(
            oracle.service_nodes
        )
    )


def test_cached_and_uncached_planning_are_identical() -> None:
    view, road, zones = make_case()

    uncached = plan_all_zone_routes(
        zones=zones,
        view=view,
        road_graph=road,
    )

    oracle = build_tsr_distance_oracle(
        view=view,
        road_graph=road,
    )

    cached = plan_all_zone_routes(
        zones=zones,
        view=view,
        road_graph=road,
        distance_oracle=oracle,
    )

    assert cached == uncached


def test_multiple_bins_on_same_road_node_share_source() -> None:
    graph = nx.Graph()

    graph.add_node(
        "depot",
        x=0.0,
        y=0.0,
    )

    graph.add_node(
        "shared",
        x=1.0,
        y=0.0,
    )

    graph.add_edge(
        "depot",
        "shared",
        length_km=2.0,
    )

    road = RoadGraph(
        graph
    )

    view = PolicyView(
        current_time_hours=0.0,
        depot_node="depot",
        bins=(
            BinPolicyView(
                bin_id=0,
                road_node="shared",
                fill_percent=50.0,
                fill_rate_percent_per_hour=0.0,
                waste_mass_tonnes=0.22,
            ),
            BinPolicyView(
                bin_id=1,
                road_node="shared",
                fill_percent=60.0,
                fill_rate_percent_per_hour=0.0,
                waste_mass_tonnes=0.264,
            ),
        ),
        trucks=(),
    )

    oracle = build_tsr_distance_oracle(
        view=view,
        road_graph=road,
    )

    assert oracle.service_nodes == (
        "depot",
        "shared",
    )

    assert oracle.distance_km(
        "shared",
        "shared",
    ) == 0.0

    assert oracle.distance_km(
        "depot",
        "shared",
    ) == pytest.approx(
        2.0
    )

    assert oracle.source_search_count == 1
