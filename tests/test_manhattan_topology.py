from math import sqrt

import networkx as nx
import pytest

from smart_waste.movement.road_graph import (
    RoadGraph,
    RoadGraphError,
)
from smart_waste.movement.topologies.manhattan import (
    MANHATTAN_TOPOLOGY_ID,
    ManhattanTopologySpec,
    generate_manhattan_topology,
)
from smart_waste.movement.topologies.validation import (
    RoadTopologyValidationError,
    validate_benchmark_road_topology,
)


def test_primary_manhattan_has_exact_structure() -> None:
    topology = generate_manhattan_topology()

    assert (
        topology.validation.topology_id
        == MANHATTAN_TOPOLOGY_ID
    )

    assert (
        topology.validation.node_count
        == 1225
    )

    assert (
        topology.validation.edge_count
        == 2380
    )

    assert topology.road_graph.is_connected


def test_primary_manhattan_physical_geometry() -> None:
    topology = generate_manhattan_topology()

    spec = topology.spec

    assert spec.side_km == pytest.approx(
        sqrt(
            50.0
        )
    )

    assert spec.spacing_km == pytest.approx(
        sqrt(
            50.0
        )
        / 34.0
    )

    assert (
        topology.validation.min_x_km
        == pytest.approx(
            0.0
        )
    )

    assert (
        topology.validation.min_y_km
        == pytest.approx(
            0.0
        )
    )

    assert (
        topology.validation.max_x_km
        == pytest.approx(
            sqrt(
                50.0
            )
        )
    )

    assert (
        topology.validation.max_y_km
        == pytest.approx(
            sqrt(
                50.0
            )
        )
    )


def test_primary_manhattan_road_length_and_density() -> None:
    topology = generate_manhattan_topology()

    expected_length = (
        70.0
        * sqrt(
            50.0
        )
    )

    assert (
        topology.validation.total_road_length_km
        == pytest.approx(
            expected_length
        )
    )

    assert (
        topology.validation.total_road_length_km
        == pytest.approx(
            494.9747468305833
        )
    )

    assert (
        topology.validation.road_density_km_per_km2
        == pytest.approx(
            9.899494936611665
        )
    )

    assert (
        9.5
        <= topology.validation.road_density_km_per_km2
        <= 10.5
    )


def test_manhattan_corner_to_corner_road_distance() -> None:
    topology = generate_manhattan_topology()

    graph = (
        topology.road_graph.graph
    )

    first = 0
    last = 1224

    distance = nx.shortest_path_length(
        graph,
        first,
        last,
        weight="length_km",
    )

    assert distance == pytest.approx(
        2.0
        * sqrt(
            50.0
        )
    )


def test_manhattan_generation_is_deterministic() -> None:
    first = generate_manhattan_topology()
    second = generate_manhattan_topology()

    assert (
        sorted(
            first.road_graph.graph.nodes(
                data=True
            )
        )
        == sorted(
            second.road_graph.graph.nodes(
                data=True
            )
        )
    )

    first_edges = sorted(
        (
            min(source, target),
            max(source, target),
            tuple(
                sorted(
                    data.items()
                )
            ),
        )
        for source, target, data
        in first.road_graph.graph.edges(
            data=True
        )
    )

    second_edges = sorted(
        (
            min(source, target),
            max(source, target),
            tuple(
                sorted(
                    data.items()
                )
            ),
        )
        for source, target, data
        in second.road_graph.graph.edges(
            data=True
        )
    )

    assert first_edges == second_edges


def test_manhattan_metadata_contains_no_policy_identity() -> None:
    topology = generate_manhattan_topology()

    metadata = (
        topology.road_graph.graph.graph
    )

    assert (
        metadata[
            "topology_id"
        ]
        == "manhattan"
    )

    forbidden = {
        "hdr",
        "tsr",
        "fdi",
        "poa",
        "fuel_performance",
        "route_performance",
    }

    assert not (
        forbidden
        & set(
            metadata
        )
    )


def test_topology_validator_rejects_missing_coordinates() -> None:
    graph = nx.Graph()

    graph.add_edge(
        0,
        1,
        length_km=500.0,
    )

    road = RoadGraph(
        graph
    )

    with pytest.raises(
        RoadTopologyValidationError,
        match="missing x/y",
    ):
        validate_benchmark_road_topology(
            road,
            topology_id="invalid",
            area_km2=50.0,
            target_density_km_per_km2=10.0,
            density_tolerance_fraction=0.05,
        )


def test_manhattan_spec_rejects_invalid_intersection_count() -> None:
    with pytest.raises(
        ValueError,
        match="at least 2",
    ):
        ManhattanTopologySpec(
            intersections_per_axis=1
        )


def test_road_graph_rejects_nonfinite_edge_length() -> None:
    for invalid in (
        float("nan"),
        float("inf"),
        float("-inf"),
    ):
        graph = nx.Graph()

        graph.add_edge(
            0,
            1,
            length_km=invalid,
        )

        with pytest.raises(
            RoadGraphError,
            match="finite physical length",
        ):
            RoadGraph(
                graph
            )
