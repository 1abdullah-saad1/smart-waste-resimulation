import networkx as nx
import pytest

from smart_waste.movement.bin_placement import (
    BinPlacementError,
    select_network_maxmin_bin_nodes,
)
from smart_waste.movement.road_graph import (
    RoadGraph,
)
from smart_waste.movement.topologies.superblock_layout import (
    PRIMARY_SUPERBLOCK_BIN_COUNT,
    build_primary_superblock_layout,
)


@pytest.fixture(scope="module")
def primary_layout():
    return build_primary_superblock_layout()


def test_superblock_has_exact_candidate_pool(
    primary_layout,
) -> None:
    assert len(
        primary_layout.candidate_nodes
    ) == 1045

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
            "superblock_bin_candidate"
        ]
        for node_id in (
            primary_layout.candidate_nodes
        )
    )


def test_superblock_places_exactly_1000_unique_bins(
    primary_layout,
) -> None:
    placement = (
        primary_layout.bin_placement
    )

    assert (
        placement.bin_count
        == PRIMARY_SUPERBLOCK_BIN_COUNT
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


def test_every_superblock_bin_uses_candidate_node(
    primary_layout,
) -> None:
    candidate_set = set(
        primary_layout.candidate_nodes
    )

    assert all(
        node_id in candidate_set
        for node_id in (
            primary_layout
            .bin_placement
            .road_nodes
        )
    )


def test_superblock_bin_ids_are_canonical_and_contiguous(
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


def test_superblock_first_selection_is_farthest_candidate(
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

    available_candidates = tuple(
        node_id
        for node_id
        in primary_layout.candidate_nodes
        if node_id != depot
    )

    maximum_distance = max(
        distances[
            node_id
        ]
        for node_id
        in available_candidates
    )

    expected = min(
        node_id
        for node_id
        in available_candidates
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
        == expected
    )


def test_superblock_layout_is_deterministic(
    primary_layout,
) -> None:
    second = (
        build_primary_superblock_layout()
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


def test_candidate_restricted_maxmin_never_uses_other_nodes() -> None:
    graph = nx.Graph()

    graph.add_node(
        "depot",
        x=0.0,
        y=0.0,
    )

    graph.add_node(
        "candidate-a",
        x=1.0,
        y=0.0,
    )

    graph.add_node(
        "candidate-b",
        x=2.0,
        y=0.0,
    )

    graph.add_node(
        "forbidden",
        x=100.0,
        y=0.0,
    )

    graph.add_edge(
        "depot",
        "candidate-a",
        length_km=1.0,
    )

    graph.add_edge(
        "candidate-a",
        "candidate-b",
        length_km=1.0,
    )

    graph.add_edge(
        "candidate-b",
        "forbidden",
        length_km=100.0,
    )

    selected = (
        select_network_maxmin_bin_nodes(
            RoadGraph(
                graph
            ),
            depot_node="depot",
            bin_count=2,
            candidate_nodes=[
                "candidate-a",
                "candidate-b",
            ],
        )
    )

    assert set(
        selected
    ) == {
        "candidate-a",
        "candidate-b",
    }

    assert "forbidden" not in selected


def test_candidate_restricted_maxmin_rejects_duplicate_pool() -> None:
    graph = nx.Graph()

    graph.add_node(
        0,
        x=0.0,
        y=0.0,
    )

    graph.add_node(
        1,
        x=1.0,
        y=0.0,
    )

    graph.add_edge(
        0,
        1,
        length_km=1.0,
    )

    with pytest.raises(
        BinPlacementError,
        match="must be unique",
    ):
        select_network_maxmin_bin_nodes(
            RoadGraph(
                graph
            ),
            depot_node=0,
            bin_count=1,
            candidate_nodes=[
                1,
                1,
            ],
        )
