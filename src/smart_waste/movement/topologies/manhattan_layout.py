from __future__ import annotations

from dataclasses import dataclass

from smart_waste.movement.bin_placement import (
    NetworkMaxMinPlacement,
    build_network_maxmin_bin_placement,
    select_central_road_node,
)
from smart_waste.movement.topologies.manhattan import (
    ManhattanTopology,
    generate_manhattan_topology,
)


PRIMARY_MANHATTAN_BIN_COUNT = 1000


@dataclass(frozen=True)
class ManhattanPhysicalLayout:
    """
    Topology plus topology-specific depot/bin placement.

    No workload, fill dynamics, truck state, routing policy or
    security condition is included here.
    """

    topology: ManhattanTopology
    depot_node: int
    bin_placement: NetworkMaxMinPlacement


def build_primary_manhattan_layout(
) -> ManhattanPhysicalLayout:
    topology = generate_manhattan_topology()

    center = (
        topology.spec.side_km
        / 2.0
    )

    depot_node = select_central_road_node(
        topology.road_graph,
        center_x_km=center,
        center_y_km=center,
    )

    if not isinstance(
        depot_node,
        int,
    ):
        raise RuntimeError(
            "primary Manhattan node IDs must be integers"
        )

    placement = build_network_maxmin_bin_placement(
        topology.road_graph,
        depot_node=depot_node,
        bin_count=PRIMARY_MANHATTAN_BIN_COUNT,
    )

    return ManhattanPhysicalLayout(
        topology=topology,
        depot_node=depot_node,
        bin_placement=placement,
    )
