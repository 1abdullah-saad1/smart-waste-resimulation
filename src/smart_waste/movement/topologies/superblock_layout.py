from __future__ import annotations

from dataclasses import dataclass

from smart_waste.movement.bin_placement import (
    NetworkMaxMinPlacement,
    canonical_node_sort_key,
    build_network_maxmin_bin_placement,
    select_central_road_node,
)
from smart_waste.movement.topologies.superblock import (
    SuperblockTopology,
    generate_superblock_topology,
)


PRIMARY_SUPERBLOCK_BIN_COUNT = 1000


@dataclass(frozen=True)
class SuperblockPhysicalLayout:
    """
    Superblock topology with topology-specific depot and bin
    placement.

    Bin placement is restricted to the frozen Superblock candidate
    contract:
        macroblock corners + macroblock-side midpoints.
    """

    topology: SuperblockTopology

    depot_node: int

    candidate_nodes: tuple[int, ...]

    bin_placement: NetworkMaxMinPlacement


def build_primary_superblock_layout(
) -> SuperblockPhysicalLayout:
    topology = (
        generate_superblock_topology()
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
            "primary Superblock node IDs must be integers"
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
                    "superblock_bin_candidate"
                ]
            ),
            key=canonical_node_sort_key,
        )
    )

    if len(
        candidate_nodes
    ) != topology.spec.expected_candidate_count:
        raise RuntimeError(
            "Superblock candidate-count invariant violated"
        )

    placement = (
        build_network_maxmin_bin_placement(
            topology.road_graph,
            depot_node=depot_node,
            bin_count=(
                PRIMARY_SUPERBLOCK_BIN_COUNT
            ),
            candidate_nodes=list(
                candidate_nodes
            ),
        )
    )

    return SuperblockPhysicalLayout(
        topology=topology,
        depot_node=depot_node,
        candidate_nodes=candidate_nodes,
        bin_placement=placement,
    )
