import networkx as nx
import pytest

from smart_waste.collection.base import (
    BinPolicyView,
    PolicyView,
)
from smart_waste.collection.tsr import (
    TSRPlanningError,
    TSRZone,
    build_longitudinal_zones,
    build_zone_distance_matrix,
    deterministic_two_opt,
    nearest_neighbor_route,
    plan_all_zone_routes,
    plan_zone_route,
    validate_primary_tsr_partition,
)
from smart_waste.movement.road_graph import RoadGraph


def make_view_and_graph(
    coordinates: dict[
        int,
        tuple[float, float],
    ],
    edges: list[
        tuple[int | str, int | str, float]
    ],
    *,
    bin_order: list[int] | None = None,
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

    for bin_id, (
        x,
        y,
    ) in coordinates.items():
        graph.add_node(
            f"bin-{bin_id}",
            x=x,
            y=y,
        )

    for first, second, distance in edges:
        first_node = (
            f"bin-{first}"
            if isinstance(first, int)
            else first
        )

        second_node = (
            f"bin-{second}"
            if isinstance(second, int)
            else second
        )

        graph.add_edge(
            first_node,
            second_node,
            length_km=distance,
        )

    road_graph = RoadGraph(
        graph
    )

    order = (
        list(coordinates)
        if bin_order is None
        else bin_order
    )

    bins = tuple(
        BinPolicyView(
            bin_id=bin_id,
            road_node=f"bin-{bin_id}",
            fill_percent=50.0,
            fill_rate_percent_per_hour=0.0,
            waste_mass_tonnes=0.22,
        )
        for bin_id in order
    )

    view = PolicyView(
        current_time_hours=0.0,
        depot_node="depot",
        bins=bins,
        trucks=(),
    )

    return (
        view,
        road_graph,
    )


def make_complete_weighted_graph(
    coordinates: dict[
        int,
        tuple[float, float],
    ],
    distances: dict[
        tuple[int | str, int | str],
        float,
    ],
) -> tuple[
    PolicyView,
    RoadGraph,
]:
    edges = [
        (
            first,
            second,
            distance,
        )
        for (
            first,
            second
        ), distance in distances.items()
    ]

    return make_view_and_graph(
        coordinates,
        edges,
    )


def test_longitudinal_partition_uses_x_y_bin_id_order() -> None:
    coordinates = {
        4: (2.0, 0.0),
        3: (1.0, 2.0),
        2: (1.0, 1.0),
        1: (1.0, 1.0),
    }

    edges = [
        ("depot", 1, 1.0),
        (1, 2, 1.0),
        (2, 3, 1.0),
        (3, 4, 1.0),
    ]

    view, road = make_view_and_graph(
        coordinates,
        edges,
        bin_order=[
            4,
            2,
            3,
            1,
        ],
    )

    zones = build_longitudinal_zones(
        view=view,
        road_graph=road,
        truck_ids=[
            1,
            0,
        ],
    )

    assert [
        zone.truck_id
        for zone in zones
    ] == [
        0,
        1,
    ]

    assert zones[0].bin_ids == (
        1,
        2,
    )

    assert zones[1].bin_ids == (
        3,
        4,
    )


def test_primary_partition_is_exactly_10_by_100() -> None:
    graph = nx.path_graph(
        1001
    )

    mapping = {
        0: "depot",
        **{
            index: f"bin-{index - 1}"
            for index in range(
                1,
                1001,
            )
        },
    }

    graph = nx.relabel_nodes(
        graph,
        mapping,
    )

    graph.nodes["depot"][
        "x"
    ] = 0.0
    graph.nodes["depot"][
        "y"
    ] = 0.0

    for bin_id in range(1000):
        node = f"bin-{bin_id}"

        graph.nodes[node][
            "x"
        ] = float(bin_id)

        graph.nodes[node][
            "y"
        ] = 0.0

    for first, second in graph.edges:
        graph[first][second][
            "length_km"
        ] = 0.5

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
                fill_percent=0.0,
                fill_rate_percent_per_hour=0.0,
                waste_mass_tonnes=0.0,
            )
            for bin_id in reversed(
                range(1000)
            )
        ),
        trucks=(),
    )

    zones = build_longitudinal_zones(
        view=view,
        road_graph=road,
        truck_ids=range(10),
    )

    validate_primary_tsr_partition(
        zones
    )

    assert len(zones) == 10

    assert all(
        len(zone.bin_ids) == 100
        for zone in zones
    )

    assert zones[0].bin_ids == tuple(
        range(
            0,
            100,
        )
    )

    assert zones[9].bin_ids == tuple(
        range(
            900,
            1000,
        )
    )


def test_partition_rejects_unequal_zone_sizes() -> None:
    coordinates = {
        0: (0.0, 0.0),
        1: (1.0, 0.0),
        2: (2.0, 0.0),
    }

    edges = [
        ("depot", 0, 1.0),
        (0, 1, 1.0),
        (1, 2, 1.0),
    ]

    view, road = make_view_and_graph(
        coordinates,
        edges,
    )

    with pytest.raises(
        TSRPlanningError,
        match="divisible",
    ):
        build_longitudinal_zones(
            view=view,
            road_graph=road,
            truck_ids=[
                0,
                1,
            ],
        )


def test_partition_requires_node_coordinates() -> None:
    graph = nx.Graph()

    graph.add_node(
        "depot",
        x=0.0,
        y=0.0,
    )

    graph.add_node(
        "bin-0"
    )

    graph.add_edge(
        "depot",
        "bin-0",
        length_km=1.0,
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
                road_node="bin-0",
                fill_percent=0.0,
                fill_rate_percent_per_hour=0.0,
                waste_mass_tonnes=0.0,
            ),
        ),
        trucks=(),
    )

    with pytest.raises(
        TSRPlanningError,
        match="missing x/y",
    ):
        build_longitudinal_zones(
            view=view,
            road_graph=road,
            truck_ids=[
                0,
            ],
        )


def test_distance_matrix_uses_shortest_road_distance() -> None:
    coordinates = {
        0: (1.0, 0.0),
        1: (2.0, 0.0),
    }

    edges = [
        ("depot", 0, 7.0),
        ("depot", 1, 20.0),
        (0, 1, 4.0),
    ]

    view, road = make_view_and_graph(
        coordinates,
        edges,
    )

    zone = TSRZone(
        truck_id=0,
        bin_ids=(
            0,
            1,
        ),
    )

    matrix = build_zone_distance_matrix(
        zone=zone,
        view=view,
        road_graph=road,
    )

    assert (
        matrix.depot_to_bin_km(0)
        == pytest.approx(7.0)
    )

    # Shortest depot -> bin1 is depot -> bin0 -> bin1:
    # 7 + 4 = 11 km, not the direct 20 km edge.
    assert (
        matrix.depot_to_bin_km(1)
        == pytest.approx(11.0)
    )

    assert (
        matrix.between_bins_km(
            0,
            1,
        )
        == pytest.approx(4.0)
    )


def test_nearest_neighbor_tie_breaks_by_bin_id() -> None:
    coordinates = {
        7: (1.0, 0.0),
        3: (2.0, 0.0),
        5: (3.0, 0.0),
    }

    distances = {
        ("depot", 7): 5.0,
        ("depot", 3): 5.0,
        ("depot", 5): 9.0,
        (3, 7): 2.0,
        (3, 5): 2.0,
        (7, 5): 2.0,
    }

    view, road = make_complete_weighted_graph(
        coordinates,
        distances,
    )

    zone = TSRZone(
        truck_id=0,
        bin_ids=(
            7,
            3,
            5,
        ),
    )

    matrix = build_zone_distance_matrix(
        zone=zone,
        view=view,
        road_graph=road,
    )

    route = nearest_neighbor_route(
        matrix
    )

    # Depot tie 3 vs 7 -> smaller bin_id 3.
    # Then 5 vs 7 are tied -> smaller bin_id 5.
    assert route == (
        3,
        5,
        7,
    )


def test_closed_route_distance_includes_return_to_depot() -> None:
    coordinates = {
        0: (1.0, 0.0),
        1: (2.0, 0.0),
    }

    distances = {
        ("depot", 0): 2.0,
        ("depot", 1): 3.0,
        (0, 1): 4.0,
    }

    view, road = make_complete_weighted_graph(
        coordinates,
        distances,
    )

    matrix = build_zone_distance_matrix(
        zone=TSRZone(
            truck_id=0,
            bin_ids=(
                0,
                1,
            ),
        ),
        view=view,
        road_graph=road,
    )

    assert matrix.route_distance_km(
        (
            0,
            1,
        )
    ) == pytest.approx(
        2.0
        + 4.0
        + 3.0
    )


def test_two_opt_improves_crossed_route() -> None:
    coordinates = {
        0: (0.0, 1.0),
        1: (1.0, 0.0),
        2: (0.0, -1.0),
        3: (-1.0, 0.0),
    }

    # Complete graph with metric-like weights chosen so route
    # 0 -> 2 -> 1 -> 3 contains expensive crossings.
    distances = {
        ("depot", 0): 1.0,
        ("depot", 1): 1.0,
        ("depot", 2): 1.0,
        ("depot", 3): 1.0,
        (0, 1): 1.0,
        (1, 2): 1.0,
        (2, 3): 1.0,
        (3, 0): 1.0,
        (0, 2): 3.0,
        (1, 3): 3.0,
    }

    view, road = make_complete_weighted_graph(
        coordinates,
        distances,
    )

    matrix = build_zone_distance_matrix(
        zone=TSRZone(
            truck_id=0,
            bin_ids=(
                0,
                1,
                2,
                3,
            ),
        ),
        view=view,
        road_graph=road,
    )

    initial = (
        0,
        2,
        1,
        3,
    )

    optimized = deterministic_two_opt(
        matrix,
        initial,
    )

    assert (
        matrix.route_distance_km(
            optimized
        )
        < matrix.route_distance_km(
            initial
        )
    )

    assert set(optimized) == {
        0,
        1,
        2,
        3,
    }


def test_two_opt_is_deterministic() -> None:
    coordinates = {
        0: (0.0, 1.0),
        1: (1.0, 0.0),
        2: (0.0, -1.0),
        3: (-1.0, 0.0),
    }

    distances = {
        ("depot", 0): 1.0,
        ("depot", 1): 1.0,
        ("depot", 2): 1.0,
        ("depot", 3): 1.0,
        (0, 1): 1.0,
        (1, 2): 1.0,
        (2, 3): 1.0,
        (3, 0): 1.0,
        (0, 2): 3.0,
        (1, 3): 3.0,
    }

    view, road = make_complete_weighted_graph(
        coordinates,
        distances,
    )

    matrix = build_zone_distance_matrix(
        zone=TSRZone(
            truck_id=0,
            bin_ids=(
                0,
                1,
                2,
                3,
            ),
        ),
        view=view,
        road_graph=road,
    )

    initial = (
        0,
        2,
        1,
        3,
    )

    first = deterministic_two_opt(
        matrix,
        initial,
    )

    second = deterministic_two_opt(
        matrix,
        initial,
    )

    assert first == second


def test_zone_plan_never_worsens_nearest_neighbor() -> None:
    coordinates = {
        0: (1.0, 0.0),
        1: (2.0, 0.0),
        2: (3.0, 0.0),
        3: (4.0, 0.0),
    }

    distances = {
        ("depot", 0): 1.0,
        ("depot", 1): 2.0,
        ("depot", 2): 3.0,
        ("depot", 3): 4.0,
        (0, 1): 1.0,
        (0, 2): 2.0,
        (0, 3): 3.0,
        (1, 2): 1.0,
        (1, 3): 2.0,
        (2, 3): 1.0,
    }

    view, road = make_complete_weighted_graph(
        coordinates,
        distances,
    )

    route = plan_zone_route(
        zone=TSRZone(
            truck_id=0,
            bin_ids=(
                0,
                1,
                2,
                3,
            ),
        ),
        view=view,
        road_graph=road,
    )

    assert (
        route.optimized_distance_km
        <= route.initial_nn_distance_km
        + 1.0e-12
    )

    assert set(
        route.optimized_bin_ids
    ) == {
        0,
        1,
        2,
        3,
    }


def test_all_zone_routes_preserve_static_assignments() -> None:
    coordinates = {
        0: (1.0, 0.0),
        1: (2.0, 0.0),
        2: (3.0, 0.0),
        3: (4.0, 0.0),
    }

    distances = {
        ("depot", 0): 1.0,
        ("depot", 1): 2.0,
        ("depot", 2): 3.0,
        ("depot", 3): 4.0,
        (0, 1): 1.0,
        (0, 2): 2.0,
        (0, 3): 3.0,
        (1, 2): 1.0,
        (1, 3): 2.0,
        (2, 3): 1.0,
    }

    view, road = make_complete_weighted_graph(
        coordinates,
        distances,
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
    )

    routes = plan_all_zone_routes(
        zones=zones,
        view=view,
        road_graph=road,
    )

    assert [
        route.truck_id
        for route in routes
    ] == [
        0,
        1,
    ]

    assert set(
        routes[0].optimized_bin_ids
    ) == {
        0,
        1,
    }

    assert set(
        routes[1].optimized_bin_ids
    ) == {
        2,
        3,
    }


def test_overlapping_zones_are_rejected() -> None:
    coordinates = {
        0: (1.0, 0.0),
        1: (2.0, 0.0),
        2: (3.0, 0.0),
    }

    distances = {
        ("depot", 0): 1.0,
        ("depot", 1): 2.0,
        ("depot", 2): 3.0,
        (0, 1): 1.0,
        (0, 2): 2.0,
        (1, 2): 1.0,
    }

    view, road = make_complete_weighted_graph(
        coordinates,
        distances,
    )

    with pytest.raises(
        TSRPlanningError,
        match="overlap",
    ):
        plan_all_zone_routes(
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
