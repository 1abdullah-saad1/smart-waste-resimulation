from __future__ import annotations

from dataclasses import dataclass

from smart_waste.movement.bin_placement import (
    NetworkMaxMinPlacement,
    build_network_maxmin_bin_placement,
    canonical_node_sort_key,
    select_central_road_node,
)
from smart_waste.movement.topologies.hexagonal import (
    HexTopology,
    generate_hex_topology,
)


PRIMARY_HEX_BIN_COUNT = 1000


@dataclass(frozen=True)
class HexPhysicalLayout:
    """
    Hexagonal road topology with a central depot and bins restricted
    to shared interior hex vertices.
    """

    topology: HexTopology

    depot_node: int

    candidate_nodes: tuple[int, ...]

    bin_placement: NetworkMaxMinPlacement


def build_primary_hex_layout(
) -> HexPhysicalLayout:
    topology = (
        generate_hex_topology()
    )

    center = (
        topology.spec.side_km
        / 2.0
    )

    depot_node = (
        select_central_road_node(
            topology.road_graph,
            center_x_km=center,
            center_y_km=center,
        )
    )

    if not isinstance(
        depot_node,
        int,
    ):
        raise RuntimeError(
            "primary Hex node IDs must be integers"
        )

    candidate_nodes = tuple(
        sorted(
            (
                node_id
                for node_id, attributes
                in topology.road_graph.graph.nodes(
                    data=True
                )
                if attributes[
                    "hex_bin_candidate"
                ]
            ),
            key=canonical_node_sort_key,
        )
    )

    if len(
        candidate_nodes
    ) < (
        PRIMARY_HEX_BIN_COUNT
        + 1
    ):
        raise RuntimeError(
            "Hex topology does not provide enough "
            "shared-vertex candidates"
        )

    placement = (
        build_network_maxmin_bin_placement(
            topology.road_graph,
            depot_node=depot_node,
            bin_count=(
                PRIMARY_HEX_BIN_COUNT
            ),
            candidate_nodes=list(
                candidate_nodes
            ),
        )
    )

    return HexPhysicalLayout(
        topology=topology,
        depot_node=depot_node,
        candidate_nodes=candidate_nodes,
        bin_placement=placement,
    )
