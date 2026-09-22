import networkx as nx
import pytest

from smart_waste.collection.base import (
    BinPolicyView,
    PolicyView,
)
from smart_waste.collection.tsr import (
    TSRBalanceObjective,
    TSRBoundary,
    TSRPlanningError,
    TSRRoute,
    TSRZone,
    build_boundary_candidates,
    build_longitudinal_boundaries,
    closest_boundary_bins,
    tsr_balance_objective,
)
from smart_waste.movement.road_graph import RoadGraph


def make_view_and_graph(
    coordinates: dict[
        int,
        tuple[float, float],
    ],
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

    previous = "depot"

    for bin_id, (
        x,
        y,
    ) in sorted(
        coordinates.items()
    ):
        node = f"bin-{bin_id}"

        graph.add_node(
            node,
            x=x,
            y=y,
        )

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
            for bin_id in sorted(
                coordinates
            )
        ),
        trucks=(),
    )

    return (
        view,
        road,
    )


def route(
    truck_id: int,
    distance: float,
) -> TSRRoute:
    return TSRRoute(
        truck_id=truck_id,
        initial_nn_bin_ids=(
            truck_id,
        ),
        optimized_bin_ids=(
            truck_id,
        ),
        initial_nn_distance_km=distance,
        optimized_distance_km=distance,
    )


def test_balance_objective_calculation() -> None:
    objective = tsr_balance_objective(
        (
            route(0, 80.0),
            route(1, 100.0),
            route(2, 120.0),
        )
    )

    assert objective.distance_range_km == pytest.approx(
        40.0
    )

    assert objective.total_distance_km == pytest.approx(
        300.0
    )

    assert objective.mean_distance_km == pytest.approx(
        100.0
    )

    assert objective.relative_range == pytest.approx(
        0.4
    )


def test_balance_objective_is_lexicographic() -> None:
    better_range = TSRBalanceObjective(
        distance_range_km=10.0,
        total_distance_km=500.0,
        mean_distance_km=100.0,
        relative_range=0.1,
    )

    worse_range = TSRBalanceObjective(
        distance_range_km=11.0,
        total_distance_km=100.0,
        mean_distance_km=20.0,
        relative_range=0.55,
    )

    assert better_range < worse_range

    same_range_lower_total = TSRBalanceObjective(
        distance_range_km=10.0,
        total_distance_km=400.0,
        mean_distance_km=80.0,
        relative_range=0.125,
    )

    assert (
        same_range_lower_total
        < better_range
    )


def test_ten_percent_balance_target() -> None:
    objective = TSRBalanceObjective(
        distance_range_km=10.0,
        total_distance_km=1000.0,
        mean_distance_km=100.0,
        relative_range=0.10,
    )

    assert objective.target_met(
        0.10
    )

    not_balanced = TSRBalanceObjective(
        distance_range_km=10.1,
        total_distance_km=1000.0,
        mean_distance_km=100.0,
        relative_range=0.101,
    )

    assert not not_balanced.target_met(
        0.10
    )


def test_initial_boundary_is_midpoint_between_adjacent_zones() -> None:
    coordinates = {
        0: (0.0, 0.0),
        1: (1.0, 0.0),
        2: (3.0, 0.0),
        3: (4.0, 0.0),
    }

    view, road = make_view_and_graph(
        coordinates
    )

    boundaries = build_longitudinal_boundaries(
        zones=(
            TSRZone(
                truck_id=0,
                bin_ids=(
                    0,
                    1,
                ),
            ),
            TSRZone(
                truck_id=1,
                bin_ids=(
                    2,
                    3,
                ),
            ),
        ),
        view=view,
        road_graph=road,
    )

    assert len(boundaries) == 1

    assert boundaries[0].left_truck_id == 0
    assert boundaries[0].right_truck_id == 1

    assert boundaries[0].x_coordinate == pytest.approx(
        2.0
    )


def test_closest_boundary_bins_are_deterministic() -> None:
    coordinates = {
        0: (0.0, 0.0),
        1: (1.0, 4.0),
        2: (1.0, 2.0),
        3: (1.0, 2.0),
        4: (1.5, 9.0),
    }

    view, road = make_view_and_graph(
        coordinates
    )

    zone = TSRZone(
        truck_id=0,
        bin_ids=(
            0,
            1,
            2,
            3,
            4,
        ),
    )

    boundary = TSRBoundary(
        left_truck_id=0,
        right_truck_id=1,
        x_coordinate=2.0,
    )

    candidates = closest_boundary_bins(
        zone=zone,
        boundary=boundary,
        view=view,
        road_graph=road,
        candidate_count=4,
    )

    # bin 4 is closest by x.
    # bins 2 and 3 tie in x and y -> bin_id breaks tie.
    # bin 1 follows because its y is larger.
    assert candidates == (
        4,
        2,
        3,
        1,
    )


def test_candidate_count_is_capped_by_zone_size() -> None:
    coordinates = {
        0: (0.0, 0.0),
        1: (1.0, 0.0),
    }

    view, road = make_view_and_graph(
        coordinates
    )

    zone = TSRZone(
        truck_id=0,
        bin_ids=(
            0,
            1,
        ),
    )

    boundary = TSRBoundary(
        left_truck_id=0,
        right_truck_id=1,
        x_coordinate=2.0,
    )

    assert len(
        closest_boundary_bins(
            zone=zone,
            boundary=boundary,
            view=view,
            road_graph=road,
            candidate_count=10,
        )
    ) == 2


def test_boundary_candidates_are_built_for_each_adjacent_pair() -> None:
    coordinates = {
        0: (0.0, 0.0),
        1: (1.0, 0.0),
        2: (2.0, 0.0),
        3: (3.0, 0.0),
        4: (4.0, 0.0),
        5: (5.0, 0.0),
    }

    view, road = make_view_and_graph(
        coordinates
    )

    zones = (
        TSRZone(
            truck_id=0,
            bin_ids=(
                0,
                1,
            ),
        ),
        TSRZone(
            truck_id=1,
            bin_ids=(
                2,
                3,
            ),
        ),
        TSRZone(
            truck_id=2,
            bin_ids=(
                4,
                5,
            ),
        ),
    )

    boundaries = build_longitudinal_boundaries(
        zones=zones,
        view=view,
        road_graph=road,
    )

    candidates = build_boundary_candidates(
        zones=zones,
        boundaries=boundaries,
        view=view,
        road_graph=road,
        candidate_count=1,
    )

    assert len(boundaries) == 2
    assert len(candidates) == 2

    assert candidates[0].left_bin_ids == (
        1,
    )
    assert candidates[0].right_bin_ids == (
        2,
    )

    assert candidates[1].left_bin_ids == (
        3,
    )
    assert candidates[1].right_bin_ids == (
        4,
    )


def test_overlapping_zones_cannot_define_boundaries() -> None:
    coordinates = {
        0: (0.0, 0.0),
        1: (1.0, 0.0),
        2: (2.0, 0.0),
    }

    view, road = make_view_and_graph(
        coordinates
    )

    with pytest.raises(
        TSRPlanningError,
        match="overlap",
    ):
        build_longitudinal_boundaries(
            zones=(
                TSRZone(
                    truck_id=0,
                    bin_ids=(
                        0,
                        1,
                    ),
                ),
                TSRZone(
                    truck_id=1,
                    bin_ids=(
                        1,
                        2,
                    ),
                ),
            ),
            view=view,
            road_graph=road,
        )
