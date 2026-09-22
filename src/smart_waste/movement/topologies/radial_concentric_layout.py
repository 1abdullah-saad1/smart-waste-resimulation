from __future__ import annotations

from dataclasses import dataclass
from math import floor

from smart_waste.movement.bin_placement import (
    BinRoadAssignment,
    canonical_node_sort_key,
    select_central_road_node,
)
from smart_waste.movement.topologies.radial_concentric import (
    RadialConcentricTopology,
    RadialConcentricTopologySpec,
    generate_radial_concentric_topology,
)


PRIMARY_RADIAL_BIN_COUNT = 1000


@dataclass(frozen=True)
class RadialWeightedPlacement:
    """
    Deterministic radial-density placement.

    Ring occupancy follows the frozen benchmark weight:

        w(r) = 2 - r / R

    The weighting affects bin placement only. It does not modify
    road distances, routing, workload, or simulation physics.
    """

    depot_node: int

    candidate_nodes: tuple[int, ...]

    ring_weights: tuple[float, ...]
    ring_quotas: tuple[int, ...]

    selection_order: tuple[int, ...]

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
    def road_nodes(self) -> tuple[int, ...]:
        return tuple(
            row.road_node
            for row in self.assignments
        )


@dataclass(frozen=True)
class RadialConcentricPhysicalLayout:
    topology: RadialConcentricTopology

    depot_node: int

    candidate_nodes: tuple[int, ...]

    bin_placement: RadialWeightedPlacement


def radial_ring_weight(
    *,
    ring_index: int,
    ring_count: int,
) -> float:
    """
    Evaluate w(r) = 2 - r/R for an equally spaced ring.

    Since r_i / R = i / ring_count:
        w_i = 2 - i / ring_count
    """

    if ring_count <= 0:
        raise ValueError(
            "ring_count must be positive"
        )

    if not (
        1
        <= ring_index
        <= ring_count
    ):
        raise ValueError(
            "ring_index must be within 1..ring_count"
        )

    return (
        2.0
        - (
            ring_index
            / ring_count
        )
    )


def allocate_weighted_ring_quotas(
    *,
    spec: RadialConcentricTopologySpec,
    bin_count: int,
) -> tuple[int, ...]:
    """
    Allocate exactly bin_count bins across radial rings.

    Method:
    - proportional allocation using w(r),
    - cap each ring at spoke_count,
    - iteratively remove saturated rings,
    - Hamilton/largest-remainder allocation for remaining seats,
    - deterministic ties by lower ring index.
    """

    if bin_count <= 0:
        raise ValueError(
            "bin_count must be positive"
        )

    total_capacity = (
        spec.ring_count
        * spec.spoke_count
    )

    if bin_count > total_capacity:
        raise ValueError(
            "bin_count exceeds radial candidate capacity"
        )

    weights = tuple(
        radial_ring_weight(
            ring_index=ring_index,
            ring_count=spec.ring_count,
        )
        for ring_index in range(
            1,
            spec.ring_count + 1,
        )
    )

    quotas = [
        0
        for _ in range(
            spec.ring_count
        )
    ]

    active = list(
        range(
            spec.ring_count
        )
    )

    remaining = (
        bin_count
    )

    while active:
        total_weight = sum(
            weights[index]
            for index in active
        )

        provisional = {
            index: (
                remaining
                * weights[index]
                / total_weight
            )
            for index in active
        }

        saturated = [
            index
            for index in active
            if provisional[
                index
            ] > spec.spoke_count
        ]

        if not saturated:
            break

        for index in saturated:
            quotas[
                index
            ] = (
                spec.spoke_count
            )

            remaining -= (
                spec.spoke_count
            )

        saturated_set = set(
            saturated
        )

        active = [
            index
            for index in active
            if index
            not in saturated_set
        ]

    if active:
        total_weight = sum(
            weights[index]
            for index in active
        )

        provisional = {
            index: (
                remaining
                * weights[index]
                / total_weight
            )
            for index in active
        }

        floor_allocations = {
            index: floor(
                provisional[
                    index
                ]
            )
            for index in active
        }

        for index in active:
            quotas[
                index
            ] = (
                floor_allocations[
                    index
                ]
            )

        seats_left = (
            bin_count
            - sum(
                quotas
            )
        )

        ranked = sorted(
            active,
            key=lambda index: (
                -(
                    provisional[index]
                    - floor_allocations[index]
                ),
                index,
            ),
        )

        for index in ranked[
            :seats_left
        ]:
            quotas[
                index
            ] += 1

    if sum(
        quotas
    ) != bin_count:
        raise RuntimeError(
            "radial quota allocation failed "
            "to preserve bin count"
        )

    if any(
        quota < 0
        or quota > spec.spoke_count
        for quota in quotas
    ):
        raise RuntimeError(
            "radial quota capacity invariant violated"
        )

    return tuple(
        quotas
    )


def _select_circular_spokes(
    *,
    spoke_count: int,
    quota: int,
) -> tuple[int, ...]:
    """
    Deterministic circular farthest-point selection.

    Start with spoke 0. Every subsequent spoke maximizes its minimum
    cyclic distance from already selected spokes. Ties use the
    smallest spoke index.
    """

    if spoke_count <= 0:
        raise ValueError(
            "spoke_count must be positive"
        )

    if not (
        0
        <= quota
        <= spoke_count
    ):
        raise ValueError(
            "quota must be within 0..spoke_count"
        )

    if quota == 0:
        return ()

    selected = [
        0
    ]

    selected_set = {
        0
    }

    while len(
        selected
    ) < quota:
        best_spoke = None
        best_distance = -1

        for spoke_index in range(
            spoke_count
        ):
            if spoke_index in selected_set:
                continue

            minimum_distance = min(
                min(
                    (
                        spoke_index
                        - chosen
                    )
                    % spoke_count,
                    (
                        chosen
                        - spoke_index
                    )
                    % spoke_count,
                )
                for chosen in selected
            )

            if (
                minimum_distance
                > best_distance
            ):
                best_distance = (
                    minimum_distance
                )

                best_spoke = (
                    spoke_index
                )

        if best_spoke is None:
            raise RuntimeError(
                "circular spoke selection failed"
            )

        selected.append(
            best_spoke
        )

        selected_set.add(
            best_spoke
        )

    return tuple(
        selected
    )


def build_primary_radial_concentric_layout(
) -> RadialConcentricPhysicalLayout:
    topology = (
        generate_radial_concentric_topology()
    )

    spec = (
        topology.spec
    )

    depot_node = (
        select_central_road_node(
            topology.road_graph,
            center_x_km=(
                spec.center_x_km
            ),
            center_y_km=(
                spec.center_y_km
            ),
        )
    )

    if not isinstance(
        depot_node,
        int,
    ):
        raise RuntimeError(
            "primary Radial-Concentric node IDs "
            "must be integers"
        )

    candidate_nodes = tuple(
        sorted(
            (
                node_id
                for node_id, attributes
                in topology.road_graph.graph.nodes(
                    data=True
                )
                if attributes.get(
                    "radial_bin_candidate",
                    False,
                )
            ),
            key=canonical_node_sort_key,
        )
    )

    if len(
        candidate_nodes
    ) != (
        spec.expected_candidate_count
    ):
        raise RuntimeError(
            "Radial-Concentric candidate-count "
            "invariant violated"
        )

    ring_weights = tuple(
        radial_ring_weight(
            ring_index=ring_index,
            ring_count=(
                spec.ring_count
            ),
        )
        for ring_index in range(
            1,
            spec.ring_count + 1,
        )
    )

    ring_quotas = (
        allocate_weighted_ring_quotas(
            spec=spec,
            bin_count=(
                PRIMARY_RADIAL_BIN_COUNT
            ),
        )
    )

    node_by_ring_spoke: dict[
        tuple[int, int],
        int,
    ] = {}

    for node_id in candidate_nodes:
        attributes = (
            topology.road_graph.graph.nodes[
                node_id
            ]
        )

        key = (
            int(
                attributes[
                    "ring_index"
                ]
            ),
            int(
                attributes[
                    "spoke_index"
                ]
            ),
        )

        if key in node_by_ring_spoke:
            raise RuntimeError(
                "duplicate radial ring/spoke node"
            )

        node_by_ring_spoke[
            key
        ] = node_id

    selected_nodes: list[
        int
    ] = []

    for ring_index, quota in enumerate(
        ring_quotas,
        start=1,
    ):
        selected_spokes = (
            _select_circular_spokes(
                spoke_count=(
                    spec.spoke_count
                ),
                quota=quota,
            )
        )

        for spoke_index in selected_spokes:
            selected_nodes.append(
                node_by_ring_spoke[
                    (
                        ring_index,
                        spoke_index,
                    )
                ]
            )

    if len(
        selected_nodes
    ) != PRIMARY_RADIAL_BIN_COUNT:
        raise RuntimeError(
            "Radial-Concentric selected-bin count "
            "invariant violated"
        )

    if len(
        set(
            selected_nodes
        )
    ) != len(
        selected_nodes
    ):
        raise RuntimeError(
            "Radial-Concentric selected nodes "
            "must be unique"
        )

    if depot_node in selected_nodes:
        raise RuntimeError(
            "Radial-Concentric depot cannot host a bin"
        )

    canonical_nodes = tuple(
        sorted(
            selected_nodes,
            key=canonical_node_sort_key,
        )
    )

    assignments = tuple(
        BinRoadAssignment(
            bin_id=bin_id,
            road_node=node_id,
        )
        for bin_id, node_id
        in enumerate(
            canonical_nodes
        )
    )

    placement = (
        RadialWeightedPlacement(
            depot_node=depot_node,
            candidate_nodes=(
                candidate_nodes
            ),
            ring_weights=(
                ring_weights
            ),
            ring_quotas=(
                ring_quotas
            ),
            selection_order=tuple(
                selected_nodes
            ),
            assignments=assignments,
        )
    )

    return (
        RadialConcentricPhysicalLayout(
            topology=topology,
            depot_node=depot_node,
            candidate_nodes=(
                candidate_nodes
            ),
            bin_placement=placement,
        )
    )
