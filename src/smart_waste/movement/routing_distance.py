from __future__ import annotations

import networkx as nx

from smart_waste.movement.road_graph import RoadGraph


class NoRoadPathError(RuntimeError):
    """Raised when two service nodes have no road connection."""


def road_distance_km(
    road_graph: RoadGraph,
    source: int | str,
    target: int | str,
) -> float:
    """
    Return weighted shortest-path road distance.

    Euclidean distance is never used as physical truck travel.
    """

    try:
        return float(
            nx.shortest_path_length(
                road_graph.graph,
                source=source,
                target=target,
                weight=road_graph.length_attribute,
            )
        )
    except nx.NodeNotFound as exc:
        raise NoRoadPathError(str(exc)) from exc
    except nx.NetworkXNoPath as exc:
        raise NoRoadPathError(str(exc)) from exc


def shortest_road_path(
    road_graph: RoadGraph,
    source: int | str,
    target: int | str,
) -> tuple[int | str, ...]:
    """Return the physical shortest road path."""

    try:
        path = nx.shortest_path(
            road_graph.graph,
            source=source,
            target=target,
            weight=road_graph.length_attribute,
        )
    except nx.NodeNotFound as exc:
        raise NoRoadPathError(str(exc)) from exc
    except nx.NetworkXNoPath as exc:
        raise NoRoadPathError(str(exc)) from exc

    return tuple(path)
