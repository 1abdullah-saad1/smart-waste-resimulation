from math import sqrt

import networkx as nx
import pytest

from smart_waste.movement.topologies.hexagonal import (
    HEX_TOPOLOGY_ID,
    HexTopologySpec,
    generate_hex_topology,
)


@pytest.fixture(scope="module")
def topology():
    return generate_hex_topology()


def test_primary_hex_structure(
    topology,
) -> None:
    assert (
        topology.validation.topology_id
        == HEX_TOPOLOGY_ID
    )

    assert (
        topology.validation.node_count
        == 3023
    )

    assert (
        topology.validation.edge_count
        == 4402
    )

    assert topology.road_graph.is_connected


def test_primary_hex_road_length_and_density(
    topology,
) -> None:
    assert (
        topology.validation.total_road_length_km
        == pytest.approx(
            501.7380444075,
            abs=1e-8,
        )
    )

    assert (
        topology.validation.road_density_km_per_km2
        == pytest.approx(
            10.03476088815,
            abs=1e-9,
        )
    )

    assert (
        9.5
        <= topology.validation.road_density_km_per_km2
        <= 10.5
    )


def test_hex_side_is_frozen_primary_value(
    topology,
) -> None:
    assert (
        topology.spec.hex_side_km
        == pytest.approx(
            0.11547
        )
    )

    assert (
        topology.spec.hex_side_km
        * 1000.0
        == pytest.approx(
            115.47
        )
    )


def test_hex_theoretical_infinite_density_is_near_target(
    topology,
) -> None:
    assert (
        topology.spec
        .theoretical_infinite_density_km_per_km2
        == pytest.approx(
            10.0000046625,
            rel=1e-6,
        )
    )


def test_hex_has_sufficient_shared_vertex_candidates(
    topology,
) -> None:
    graph = (
        topology.road_graph.graph
    )

    candidates = tuple(
        node_id
        for node_id, attributes
        in graph.nodes(
            data=True
        )
        if attributes[
            "hex_bin_candidate"
        ]
    )

    assert len(
        candidates
    ) == 2870

    assert len(
        candidates
    ) >= 1001

    assert all(
        graph.degree[
            node_id
        ] == 3
        for node_id in candidates
    )


def test_hex_boundary_nodes_are_not_bin_candidates(
    topology,
) -> None:
    graph = (
        topology.road_graph.graph
    )

    side = sqrt(
        50.0
    )

    boundary_nodes = [
        node_id
        for node_id, attributes
        in graph.nodes(
            data=True
        )
        if (
            attributes[
                "x"
            ]
            == pytest.approx(
                0.0
            )
            or attributes[
                "x"
            ]
            == pytest.approx(
                side
            )
            or attributes[
                "y"
            ]
            == pytest.approx(
                0.0
            )
            or attributes[
                "y"
            ]
            == pytest.approx(
                side
            )
        )
    ]

    assert boundary_nodes

    assert all(
        not graph.nodes[
            node_id
        ][
            "hex_bin_candidate"
        ]
        for node_id in boundary_nodes
    )


def test_every_hex_edge_has_positive_physical_length(
    topology,
) -> None:
    graph = (
        topology.road_graph.graph
    )

    for source, target, attributes in graph.edges(
        data=True
    ):
        source_data = (
            graph.nodes[
                source
            ]
        )

        target_data = (
            graph.nodes[
                target
            ]
        )

        dx = (
            source_data["x"]
            - target_data["x"]
        )

        dy = (
            source_data["y"]
            - target_data["y"]
        )

        coordinate_length = (
            dx * dx
            + dy * dy
        ) ** 0.5

        assert (
            attributes[
                "length_km"
            ]
            == pytest.approx(
                coordinate_length
            )
        )

        assert (
            attributes[
                "length_km"
            ]
            > 0.0
        )


def test_hex_graph_is_planar(
    topology,
) -> None:
    planar, _ = (
        nx.check_planarity(
            topology.road_graph.graph
        )
    )

    assert planar


def test_hex_generation_is_deterministic() -> None:
    first = (
        generate_hex_topology()
    )

    second = (
        generate_hex_topology()
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


def test_hex_metadata_contains_no_policy_identity(
    topology,
) -> None:
    metadata = (
        topology.road_graph.graph.graph
    )

    assert (
        metadata[
            "topology_id"
        ]
        == "hex"
    )

    assert (
        metadata[
            "topology_index"
        ]
        == 3
    )

    assert (
        metadata[
            "hex_orientation"
        ]
        == "flat_top"
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


def test_invalid_hex_side_is_rejected() -> None:
    with pytest.raises(
        ValueError,
    ):
        HexTopologySpec(
            hex_side_km=0.0
        )
