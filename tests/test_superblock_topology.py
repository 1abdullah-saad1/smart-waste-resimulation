from math import sqrt

import networkx as nx
import pytest

from smart_waste.movement.topologies.superblock import (
    SUPERBLOCK_TOPOLOGY_ID,
    SuperblockTopologySpec,
    generate_superblock_topology,
)


def _coordinates(topology):
    return {
        (
            data["x"],
            data["y"],
        )
        for _, data
        in topology.road_graph.graph.nodes(
            data=True
        )
    }


def test_primary_superblock_exact_structure() -> None:
    topology = (
        generate_superblock_topology()
    )

    assert (
        topology.validation.topology_id
        == SUPERBLOCK_TOPOLOGY_ID
    )

    assert (
        topology.spec.macroblocks_per_axis
        == 18
    )

    assert (
        topology.validation.node_count
        == 2665
    )

    assert (
        topology.validation.edge_count
        == 3960
    )

    assert topology.road_graph.is_connected


def test_primary_superblock_road_length_and_density() -> None:
    topology = (
        generate_superblock_topology()
    )

    expected_length = (
        74.0
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
            523.2590180780452
        )
    )

    assert (
        topology.validation.road_density_km_per_km2
        == pytest.approx(
            10.465180361560904
        )
    )

    assert (
        9.5
        <= topology.validation.road_density_km_per_km2
        <= 10.5
    )


def test_superblock_has_exact_candidate_count() -> None:
    topology = (
        generate_superblock_topology()
    )

    candidate_nodes = tuple(
        node_id
        for node_id, attributes
        in topology.road_graph.graph.nodes(
            data=True
        )
        if attributes[
            "superblock_bin_candidate"
        ]
    )

    assert len(
        candidate_nodes
    ) == 1045

    assert (
        len(
            candidate_nodes
        )
        >= 1001
    )


def test_superblock_candidates_are_exactly_corners_and_midpoints() -> None:
    topology = (
        generate_superblock_topology()
    )

    boundaries = (
        topology.spec.boundaries_km
    )

    n = (
        topology.spec.macroblocks_per_axis
    )

    expected = set()

    for y in boundaries:
        for x in boundaries:
            expected.add(
                (
                    x,
                    y,
                )
            )

    for y in boundaries:
        for column in range(
            n
        ):
            x0 = boundaries[
                column
            ]
            x1 = boundaries[
                column + 1
            ]

            expected.add(
                (
                    (
                        x0 + x1
                    )
                    / 2.0,
                    y,
                )
            )

    for x in boundaries:
        for row in range(
            n
        ):
            y0 = boundaries[
                row
            ]
            y1 = boundaries[
                row + 1
            ]

            expected.add(
                (
                    x,
                    (
                        y0 + y1
                    )
                    / 2.0,
                )
            )

    actual = {
        (
            attributes[
                "x"
            ],
            attributes[
                "y"
            ],
        )
        for _, attributes
        in topology.road_graph.graph.nodes(
            data=True
        )
        if attributes[
            "superblock_bin_candidate"
        ]
    }

    assert actual == expected
    assert len(actual) == 1045


def test_superblock_graph_is_planar() -> None:
    topology = (
        generate_superblock_topology()
    )

    is_planar, _ = nx.check_planarity(
        topology.road_graph.graph
    )

    assert is_planar


def test_every_superblock_edge_matches_physical_coordinates() -> None:
    topology = (
        generate_superblock_topology()
    )

    graph = (
        topology.road_graph.graph
    )

    for source, target, attributes in graph.edges(
        data=True
    ):
        source_data = graph.nodes[
            source
        ]

        target_data = graph.nodes[
            target
        ]

        dx = abs(
            source_data["x"]
            - target_data["x"]
        )

        dy = abs(
            source_data["y"]
            - target_data["y"]
        )

        assert (
            dx == pytest.approx(0.0)
            or dy == pytest.approx(0.0)
        )

        assert not (
            dx == pytest.approx(0.0)
            and dy == pytest.approx(0.0)
        )

        expected_length = (
            dx + dy
        )

        assert (
            attributes[
                "length_km"
            ]
            == pytest.approx(
                expected_length
            )
        )


def test_checkerboard_internal_thirds_are_present() -> None:
    topology = (
        generate_superblock_topology()
    )

    coordinates = (
        _coordinates(
            topology
        )
    )

    # Cell (0, 0), even:
    # N-S at 1/3, E-W at 2/3.
    even_cross = (
        0.4 / 3.0,
        0.4 * 2.0 / 3.0,
    )

    # Cell (0, 1), odd:
    # N-S at 2/3, E-W at 1/3.
    odd_cross = (
        0.4
        + 0.4 * 2.0 / 3.0,
        0.4 / 3.0,
    )

    assert even_cross in coordinates
    assert odd_cross in coordinates


def test_superblock_generation_is_deterministic() -> None:
    first = (
        generate_superblock_topology()
    )

    second = (
        generate_superblock_topology()
    )

    assert (
        first.road_graph.graph.graph
        == second.road_graph.graph.graph
    )

    assert (
        list(
            first.road_graph.graph.nodes(
                data=True
            )
        )
        == list(
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
                    attributes.items()
                )
            ),
        )
        for source, target, attributes
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
                    attributes.items()
                )
            ),
        )
        for source, target, attributes
        in second.road_graph.graph.edges(
            data=True
        )
    )

    assert first_edges == second_edges


def test_superblock_metadata_has_no_policy_identity() -> None:
    topology = (
        generate_superblock_topology()
    )

    metadata = (
        topology.road_graph.graph.graph
    )

    assert (
        metadata[
            "topology_id"
        ]
        == "superblock"
    )

    assert (
        metadata[
            "topology_index"
        ]
        == 2
    )

    forbidden = {
        "hdr",
        "tsr",
        "fdi",
        "poa",
        "reward",
        "route_performance",
        "fuel_performance",
    }

    assert not (
        forbidden
        & {
            str(key).lower()
            for key in metadata
        }
    )


def test_superblock_spec_expected_counts() -> None:
    spec = (
        SuperblockTopologySpec()
    )

    assert (
        spec.macroblocks_per_axis
        == 18
    )

    assert (
        spec.expected_node_count
        == 2665
    )

    assert (
        spec.expected_edge_count
        == 3960
    )

    assert (
        spec.expected_candidate_count
        == 1045
    )
