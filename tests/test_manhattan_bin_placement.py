import networkx as nx
import pytest

from smart_waste.movement.bin_placement import (
    BinPlacementError,
    build_network_maxmin_bin_placement,
    select_central_road_node,
    select_network_maxmin_bin_nodes,
)
from smart_waste.movement.road_graph import (
    RoadGraph,
)
from smart_waste.movement.topologies.manhattan import (
    generate_manhattan_topology,
)
from smart_waste.movement.topologies.manhattan_layout import (
    PRIMARY_MANHATTAN_BIN_COUNT,
    build_primary_manhattan_layout,
)


@pytest.fixture(scope="module")
def primary_layout():
    return build_primary_manhattan_layout()


def test_primary_manhattan_depot_is_exact_central_node(
    primary_layout,
) -> None:
    assert (
        primary_layout.depot_node
        == 612
    )

    graph = (
        primary_layout
        .topology
        .road_graph
        .graph
    )

    center = (
        primary_layout
        .topology
        .spec
        .side_km
        / 2.0
    )

    assert (
        graph.nodes[
            612
        ][
            "x"
        ]
        == pytest.approx(
            center
        )
    )

    assert (
        graph.nodes[
            612
        ][
            "y"
        ]
        == pytest.approx(
            center
        )
    )


def test_primary_manhattan_has_exactly_1000_unique_bin_nodes(
    primary_layout,
) -> None:
    placement = (
        primary_layout.bin_placement
    )

    assert (
        placement.bin_count
        == PRIMARY_MANHATTAN_BIN_COUNT
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


def test_primary_bin_ids_are_canonical_and_contiguous(
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

    nodes = tuple(
        row.road_node
        for row in assignments
    )

    assert nodes == tuple(
        sorted(
            nodes
        )
    )


def test_primary_selected_nodes_all_exist_on_road_graph(
    primary_layout,
) -> None:
    graph = (
        primary_layout
        .topology
        .road_graph
        .graph
    )

    assert all(
        node in graph
        for node in (
            primary_layout
            .bin_placement
            .road_nodes
        )
    )


def test_primary_maxmin_starts_from_canonical_farthest_corner(
    primary_layout,
) -> None:
    # All four corners are equally far from central node 612.
    # Canonical lowest node ID wins the exact tie.
    assert (
        primary_layout
        .bin_placement
        .selection_order[
            0
        ]
        == 0
    )


def test_primary_selection_order_is_unique(
    primary_layout,
) -> None:
    order = (
        primary_layout
        .bin_placement
        .selection_order
    )

    assert len(order) == 1000

    assert len(
        set(
            order
        )
    ) == 1000


def test_maxmin_uses_road_distance_not_coordinate_distance() -> None:
    graph = nx.Graph()

    # Geometrically A appears far from depot, but road distance
    # makes B farther. The selection must follow road distance.
    graph.add_node(
        "depot",
        x=0.0,
        y=0.0,
    )

    graph.add_node(
        "A",
        x=100.0,
        y=0.0,
    )

    graph.add_node(
        "B",
        x=1.0,
        y=0.0,
    )

    graph.add_edge(
        "depot",
        "A",
        length_km=1.0,
    )

    graph.add_edge(
        "depot",
        "B",
        length_km=10.0,
    )

    selected = (
        select_network_maxmin_bin_nodes(
            RoadGraph(
                graph
            ),
            depot_node="depot",
            bin_count=1,
        )
    )

    assert selected == (
        "B",
    )


def test_maxmin_ties_use_canonical_lowest_node_id() -> None:
    graph = nx.Graph()

    graph.add_node(
        10,
        x=0.0,
        y=0.0,
    )

    graph.add_node(
        2,
        x=-1.0,
        y=0.0,
    )

    graph.add_node(
        7,
        x=1.0,
        y=0.0,
    )

    graph.add_edge(
        10,
        2,
        length_km=1.0,
    )

    graph.add_edge(
        10,
        7,
        length_km=1.0,
    )

    selected = (
        select_network_maxmin_bin_nodes(
            RoadGraph(
                graph
            ),
            depot_node=10,
            bin_count=1,
        )
    )

    assert selected == (
        2,
    )


def test_central_node_selection_is_deterministic_on_tie() -> None:
    graph = nx.Graph()

    graph.add_node(
        9,
        x=-1.0,
        y=0.0,
    )

    graph.add_node(
        3,
        x=1.0,
        y=0.0,
    )

    graph.add_edge(
        9,
        3,
        length_km=2.0,
    )

    road = RoadGraph(
        graph
    )

    assert (
        select_central_road_node(
            road,
            center_x_km=0.0,
            center_y_km=0.0,
        )
        == 3
    )


def test_placement_rejects_more_bins_than_non_depot_nodes() -> None:
    graph = nx.Graph()

    graph.add_node(
        "depot",
        x=0.0,
        y=0.0,
    )

    graph.add_node(
        "only",
        x=1.0,
        y=0.0,
    )

    graph.add_edge(
        "depot",
        "only",
        length_km=1.0,
    )

    road = RoadGraph(
        graph
    )

    with pytest.raises(
        BinPlacementError,
        match="exceeds available",
    ):
        build_network_maxmin_bin_placement(
            road,
            depot_node="depot",
            bin_count=2,
        )


def test_primary_layout_is_deterministic() -> None:
    first = (
        build_primary_manhattan_layout()
    )

    second = (
        build_primary_manhattan_layout()
    )

    assert (
        first.depot_node
        == second.depot_node
        == 612
    )

    assert (
        first.bin_placement
        == second.bin_placement
    )
