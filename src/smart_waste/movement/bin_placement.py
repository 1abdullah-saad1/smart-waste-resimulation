from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import networkx as nx

from smart_waste.movement.road_graph import (
    RoadGraph,
)


NodeId = int | str


class BinPlacementError(ValueError):
    """Raised when physical bin placement is invalid."""


@dataclass(frozen=True)
class BinRoadAssignment:
    bin_id: int
    road_node: NodeId


@dataclass(frozen=True)
class NetworkMaxMinPlacement:
    """
    Deterministic road-network bin placement result.

    `selection_order` preserves the greedy max-min construction
    order for reproducibility/audit.

    `assignments` deliberately assign bin IDs only after the final
    selected node set is canonically sorted. Therefore arbitrary
    greedy selection order does not define later bin identity.
    """

    depot_node: NodeId

    selection_order: tuple[
        NodeId,
        ...,
    ]

    assignments: tuple[
        BinRoadAssignment,
        ...,
    ]

    @property
    def bin_count(self) -> int:
        return len(
            self.assignments
        )

    @property
    def road_nodes(self) -> tuple[
        NodeId,
        ...,
    ]:
        return tuple(
            assignment.road_node
            for assignment in self.assignments
        )


def _normalize_node_id(
    node_id: object,
) -> NodeId:
    if isinstance(
        node_id,
        bool,
    ):
        raise BinPlacementError(
            "boolean road-node IDs are not supported"
        )

    if isinstance(
        node_id,
        int,
    ):
        return node_id

    if isinstance(
        node_id,
        str,
    ):
        return node_id

    raise BinPlacementError(
        "road-node IDs must be integer or string"
    )


def canonical_node_sort_key(
    node_id: NodeId,
) -> tuple[
    int,
    int | str,
]:
    node_id = _normalize_node_id(
        node_id
    )

    if isinstance(
        node_id,
        int,
    ):
        return (
            0,
            node_id,
        )

    return (
        1,
        node_id,
    )


def _node_xy(
    road_graph: RoadGraph,
    node_id: NodeId,
) -> tuple[
    float,
    float,
]:
    if node_id not in road_graph.graph:
        raise BinPlacementError(
            f"road node does not exist: {node_id!r}"
        )

    data = road_graph.graph.nodes[
        node_id
    ]

    if (
        "x" not in data
        or "y" not in data
    ):
        raise BinPlacementError(
            f"road node {node_id!r} "
            "is missing x/y coordinates"
        )

    try:
        x = float(
            data["x"]
        )
        y = float(
            data["y"]
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise BinPlacementError(
            f"road node {node_id!r} "
            "has nonnumeric x/y coordinates"
        ) from exc

    if (
        not isfinite(x)
        or not isfinite(y)
    ):
        raise BinPlacementError(
            f"road node {node_id!r} "
            "has nonfinite x/y coordinates"
        )

    return (
        x,
        y,
    )


def select_central_road_node(
    road_graph: RoadGraph,
    *,
    center_x_km: float,
    center_y_km: float,
) -> NodeId:
    """
    Select the physical road node nearest the requested geometric
    center.

    Coordinate distance is used only for depot LOCATION, never for
    truck routing or bin max-min placement.

    Exact ties use canonical node ID order.
    """

    if (
        not isfinite(
            center_x_km
        )
        or not isfinite(
            center_y_km
        )
    ):
        raise BinPlacementError(
            "depot center coordinates must be finite"
        )

    best_node: NodeId | None = None
    best_squared_distance: float | None = None

    for raw_node in sorted(
        road_graph.graph.nodes,
        key=canonical_node_sort_key,
    ):
        node = _normalize_node_id(
            raw_node
        )

        x, y = _node_xy(
            road_graph,
            node,
        )

        squared_distance = (
            (
                x
                - center_x_km
            )
            ** 2
            + (
                y
                - center_y_km
            )
            ** 2
        )

        if (
            best_squared_distance is None
            or squared_distance
            < best_squared_distance
        ):
            best_node = node
            best_squared_distance = (
                squared_distance
            )

    if best_node is None:
        raise BinPlacementError(
            "road graph contains no nodes"
        )

    return best_node


def _distances_from(
    road_graph: RoadGraph,
    source: NodeId,
) -> dict[
    NodeId,
    float,
]:
    raw = (
        nx.single_source_dijkstra_path_length(
            road_graph.graph,
            source,
            weight=(
                road_graph.length_attribute
            ),
        )
    )

    distances: dict[
        NodeId,
        float,
    ] = {}

    for raw_node, raw_distance in raw.items():
        node = _normalize_node_id(
            raw_node
        )

        distance = float(
            raw_distance
        )

        if not isfinite(
            distance
        ):
            raise BinPlacementError(
                "shortest-road distance must be finite"
            )

        distances[
            node
        ] = distance

    return distances


def select_network_maxmin_bin_nodes(
    road_graph: RoadGraph,
    *,
    depot_node: NodeId,
    bin_count: int,
) -> tuple[
    NodeId,
    ...,
]:
    """
    Deterministic greedy network-distance max-min selection.

    Initial bin:
        farthest road node from depot.

    Each subsequent bin:
        node maximizing its minimum shortest-road distance to
        any previously selected bin.

    Ties:
        canonical lowest road-node ID.

    Returned tuple preserves greedy selection order.
    """

    if bin_count <= 0:
        raise BinPlacementError(
            "bin_count must be positive"
        )

    depot_node = _normalize_node_id(
        depot_node
    )

    if depot_node not in road_graph.graph:
        raise BinPlacementError(
            "depot node is absent from road graph"
        )

    road_graph.validate_connected()

    candidates = tuple(
        _normalize_node_id(
            node
        )
        for node in sorted(
            (
                node
                for node in road_graph.graph.nodes
                if node != depot_node
            ),
            key=canonical_node_sort_key,
        )
    )

    if bin_count > len(
        candidates
    ):
        raise BinPlacementError(
            "bin_count exceeds available non-depot road nodes"
        )

    depot_distances = (
        _distances_from(
            road_graph,
            depot_node,
        )
    )

    first_node = candidates[
        0
    ]

    first_distance = (
        depot_distances[
            first_node
        ]
    )

    for node in candidates[
        1:
    ]:
        distance = (
            depot_distances[
                node
            ]
        )

        if distance > first_distance:
            first_node = node
            first_distance = distance

    selected: list[
        NodeId
    ] = [
        first_node
    ]

    selected_set: set[
        NodeId
    ] = {
        first_node
    }

    minimum_selected_distance = (
        _distances_from(
            road_graph,
            first_node,
        )
    )

    while len(
        selected
    ) < bin_count:
        best_node: NodeId | None = None
        best_distance = -1.0

        for node in candidates:
            if node in selected_set:
                continue

            distance = (
                minimum_selected_distance[
                    node
                ]
            )

            if distance > best_distance:
                best_node = node
                best_distance = distance

        if best_node is None:
            raise BinPlacementError(
                "unable to select another bin node"
            )

        selected.append(
            best_node
        )

        selected_set.add(
            best_node
        )

        new_distances = (
            _distances_from(
                road_graph,
                best_node,
            )
        )

        for node in candidates:
            if node in selected_set:
                continue

            distance = (
                new_distances[
                    node
                ]
            )

            if (
                distance
                < minimum_selected_distance[
                    node
                ]
            ):
                minimum_selected_distance[
                    node
                ] = distance

    return tuple(
        selected
    )


def build_network_maxmin_bin_placement(
    road_graph: RoadGraph,
    *,
    depot_node: NodeId,
    bin_count: int,
) -> NetworkMaxMinPlacement:
    """
    Build auditable max-min placement and stable bin identities.
    """

    selection_order = (
        select_network_maxmin_bin_nodes(
            road_graph,
            depot_node=depot_node,
            bin_count=bin_count,
        )
    )

    canonical_nodes = tuple(
        sorted(
            selection_order,
            key=canonical_node_sort_key,
        )
    )

    assignments = tuple(
        BinRoadAssignment(
            bin_id=bin_id,
            road_node=node,
        )
        for bin_id, node in enumerate(
            canonical_nodes
        )
    )

    return NetworkMaxMinPlacement(
        depot_node=depot_node,
        selection_order=(
            selection_order
        ),
        assignments=assignments,
    )
