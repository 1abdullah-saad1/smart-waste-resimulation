import networkx as nx
import pytest

from smart_waste.movement.topologies.hexagonal_layout import (
    PRIMARY_HEX_BIN_COUNT,
    build_primary_hex_layout,
)


@pytest.fixture(scope="module")
def primary_layout():
    return build_primary_hex_layout()


def test_hex_has_exact_candidate_pool(
    primary_layout,
) -> None:
    assert len(
        primary_layout.candidate_nodes
    ) == 2870

    graph = (
        primary_layout
        .topology
        .road_graph
        .graph
    )

    assert all(
        graph.nodes[
            node_id
        ][
            "hex_bin_candidate"
        ]
        for node_id
        in primary_layout.candidate_nodes
    )


def test_hex_places_exactly_1000_unique_bins(
    primary_layout,
) -> None:
    placement = (
        primary_layout.bin_placement
    )

    assert (
        placement.bin_count
        == PRIMARY_HEX_BIN_COUNT
        == 1000
    )

    assert len(
        placement.road_nodes
    ) == 1000

    assert len(
        set(
            placement.road_nodes
        )
    ) == 1000

    assert (
        primary_layout.depot_node
        not in placement.road_nodes
    )


def test_every_hex_bin_is_shared_vertex_candidate(
    primary_layout,
) -> None:
    graph = (
        primary_layout
        .topology
        .road_graph
        .graph
    )

    candidate_set = set(
        primary_layout.candidate_nodes
    )

    for node_id in (
        primary_layout
        .bin_placement
        .road_nodes
    ):
        assert node_id in candidate_set

        assert (
            graph.nodes[
                node_id
            ][
                "hex_bin_candidate"
            ]
        )

        assert (
            graph.degree[
                node_id
            ]
            == 3
        )


def test_hex_bin_ids_are_canonical_and_contiguous(
    primary_layout,
) -> None:
    assignments = (
        primary_layout
        .bin_placement
        .assignments
    )

    assert tuple(
        row.bin_id
        for row in assignments
    ) == tuple(
        range(
            1000
        )
    )

    road_nodes = tuple(
        row.road_node
        for row in assignments
    )

    assert road_nodes == tuple(
        sorted(
            road_nodes
        )
    )


def test_hex_first_selection_is_farthest_candidate(
    primary_layout,
) -> None:
    graph = (
        primary_layout
        .topology
        .road_graph
        .graph
    )

    depot = (
        primary_layout.depot_node
    )

    distances = (
        nx.single_source_dijkstra_path_length(
            graph,
            depot,
            weight="length_km",
        )
    )

    candidates = tuple(
        node_id
        for node_id
        in primary_layout.candidate_nodes
        if node_id != depot
    )

    maximum_distance = max(
        distances[
            node_id
        ]
        for node_id in candidates
    )

    tied = tuple(
        node_id
        for node_id in candidates
        if distances[
            node_id
        ] == pytest.approx(
            maximum_distance
        )
    )

    assert (
        primary_layout
        .bin_placement
        .selection_order[
            0
        ]
        == min(
            tied
        )
    )


def test_hex_layout_is_deterministic(
    primary_layout,
) -> None:
    second = (
        build_primary_hex_layout()
    )

    assert (
        second.depot_node
        == primary_layout.depot_node
    )

    assert (
        second.candidate_nodes
        == primary_layout.candidate_nodes
    )

    assert (
        second.bin_placement
        == primary_layout.bin_placement
    )


def test_hex_depot_is_valid_road_node(
    primary_layout,
) -> None:
    graph = (
        primary_layout
        .topology
        .road_graph
        .graph
    )

    assert (
        primary_layout.depot_node
        in graph
    )

    assert (
        primary_layout.depot_node
        not in primary_layout
        .bin_placement
        .road_nodes
    )


def test_hex_selection_order_contains_exactly_1000_unique_nodes(
    primary_layout,
) -> None:
    order = (
        primary_layout
        .bin_placement
        .selection_order
    )

    assert len(
        order
    ) == 1000

    assert len(
        set(
            order
        )
    ) == 1000
