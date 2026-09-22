import networkx as nx
import pytest

from smart_waste.movement.road_graph import (
    RoadGraph,
    RoadGraphError,
)
from smart_waste.movement.routing_distance import (
    NoRoadPathError,
    road_distance_km,
    shortest_road_path,
)
from smart_waste.movement.travel_time import (
    travel_time_hours,
    travel_time_minutes,
)


def make_graph() -> RoadGraph:
    graph = nx.Graph()

    graph.add_edge(
        "depot",
        "a",
        length_km=2.0,
    )
    graph.add_edge(
        "a",
        "b",
        length_km=3.0,
    )
    graph.add_edge(
        "depot",
        "b",
        length_km=10.0,
    )

    return RoadGraph(graph)


def test_shortest_road_distance_uses_edge_lengths() -> None:
    road = make_graph()

    assert road_distance_km(
        road,
        "depot",
        "b",
    ) == pytest.approx(5.0)


def test_shortest_road_path() -> None:
    road = make_graph()

    assert shortest_road_path(
        road,
        "depot",
        "b",
    ) == (
        "depot",
        "a",
        "b",
    )


def test_total_road_length() -> None:
    road = make_graph()

    assert road.total_road_length_km == pytest.approx(
        15.0
    )


def test_road_density() -> None:
    road = make_graph()

    assert road.road_density_km_per_km2(
        area_km2=5.0
    ) == pytest.approx(3.0)


def test_density_validator_accepts_valid_network() -> None:
    graph = nx.Graph()

    graph.add_edge(
        0,
        1,
        length_km=500.0,
    )

    road = RoadGraph(graph)

    road.validate_density(
        area_km2=50.0,
        target_km_per_km2=10.0,
        tolerance_fraction=0.05,
    )


def test_density_validator_rejects_invalid_network() -> None:
    road = make_graph()

    with pytest.raises(RoadGraphError):
        road.validate_density(
            area_km2=50.0,
            target_km_per_km2=10.0,
            tolerance_fraction=0.05,
        )


def test_zero_length_edge_is_rejected() -> None:
    graph = nx.Graph()

    graph.add_edge(
        0,
        1,
        length_km=0.0,
    )

    with pytest.raises(RoadGraphError):
        RoadGraph(graph)


def test_missing_edge_length_is_rejected() -> None:
    graph = nx.Graph()
    graph.add_edge(0, 1)

    with pytest.raises(RoadGraphError):
        RoadGraph(graph)


def test_disconnected_graph_detected() -> None:
    graph = nx.Graph()

    graph.add_edge(
        0,
        1,
        length_km=1.0,
    )
    graph.add_edge(
        2,
        3,
        length_km=1.0,
    )

    road = RoadGraph(graph)

    assert not road.is_connected

    with pytest.raises(RoadGraphError):
        road.validate_connected()


def test_service_nodes_must_exist() -> None:
    road = make_graph()

    with pytest.raises(RoadGraphError):
        road.validate_service_nodes(
            ("depot", "missing")
        )


def test_no_path_error() -> None:
    graph = nx.Graph()

    graph.add_edge(
        0,
        1,
        length_km=1.0,
    )
    graph.add_edge(
        2,
        3,
        length_km=1.0,
    )

    road = RoadGraph(graph)

    with pytest.raises(NoRoadPathError):
        road_distance_km(
            road,
            0,
            3,
        )


def test_travel_time_at_primary_speed() -> None:
    assert travel_time_hours(
        distance_km=30.0,
        speed_km_per_hour=30.0,
    ) == pytest.approx(1.0)


def test_travel_time_minutes() -> None:
    assert travel_time_minutes(
        distance_km=15.0,
        speed_km_per_hour=30.0,
    ) == pytest.approx(30.0)


def test_road_distance_can_exceed_straight_line_distance() -> None:
    graph = nx.Graph()

    graph.add_node(
        "a",
        x_km=0.0,
        y_km=0.0,
    )
    graph.add_node(
        "b",
        x_km=1.0,
        y_km=0.0,
    )
    graph.add_node(
        "c",
        x_km=1.0,
        y_km=1.0,
    )

    graph.add_edge(
        "a",
        "c",
        length_km=2.0,
    )
    graph.add_edge(
        "c",
        "b",
        length_km=2.0,
    )

    road = RoadGraph(graph)

    assert road_distance_km(
        road,
        "a",
        "b",
    ) == pytest.approx(4.0)
