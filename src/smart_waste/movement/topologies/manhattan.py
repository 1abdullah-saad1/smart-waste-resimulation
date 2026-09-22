from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt

import networkx as nx

from smart_waste.movement.road_graph import (
    RoadGraph,
)
from smart_waste.movement.topologies.validation import (
    RoadTopologyValidationReport,
    validate_benchmark_road_topology,
)


MANHATTAN_TOPOLOGY_ID = "manhattan"
MANHATTAN_TOPOLOGY_INDEX = 1

DEFAULT_AREA_KM2 = 50.0
DEFAULT_INTERSECTIONS_PER_AXIS = 35

DEFAULT_TARGET_DENSITY_KM_PER_KM2 = 10.0
DEFAULT_DENSITY_TOLERANCE_FRACTION = 0.05


@dataclass(frozen=True)
class ManhattanTopologySpec:
    """
    Deterministic revised-benchmark Manhattan topology.

    `intersections_per_axis` denotes physical intersections/nodes,
    not road segments.
    """

    area_km2: float = DEFAULT_AREA_KM2

    intersections_per_axis: int = (
        DEFAULT_INTERSECTIONS_PER_AXIS
    )

    target_density_km_per_km2: float = (
        DEFAULT_TARGET_DENSITY_KM_PER_KM2
    )

    density_tolerance_fraction: float = (
        DEFAULT_DENSITY_TOLERANCE_FRACTION
    )

    def __post_init__(self) -> None:
        if (
            not isfinite(
                self.area_km2
            )
            or self.area_km2 <= 0.0
        ):
            raise ValueError(
                "area_km2 must be finite and positive"
            )

        if self.intersections_per_axis < 2:
            raise ValueError(
                "intersections_per_axis must be at least 2"
            )

        if (
            not isfinite(
                self.target_density_km_per_km2
            )
            or self.target_density_km_per_km2 <= 0.0
        ):
            raise ValueError(
                "target density must be finite and positive"
            )

        if not (
            0.0
            <= self.density_tolerance_fraction
            < 1.0
        ):
            raise ValueError(
                "density tolerance must be in [0, 1)"
            )

    @property
    def side_km(self) -> float:
        return sqrt(
            self.area_km2
        )

    @property
    def spacing_km(self) -> float:
        return (
            self.side_km
            / (
                self.intersections_per_axis
                - 1
            )
        )

    @property
    def expected_node_count(self) -> int:
        return (
            self.intersections_per_axis
            ** 2
        )

    @property
    def expected_edge_count(self) -> int:
        n = (
            self.intersections_per_axis
        )

        return (
            2
            * n
            * (
                n - 1
            )
        )

    @property
    def expected_total_road_length_km(
        self,
    ) -> float:
        return (
            self.expected_edge_count
            * self.spacing_km
        )


@dataclass(frozen=True)
class ManhattanTopology:
    spec: ManhattanTopologySpec
    road_graph: RoadGraph
    validation: RoadTopologyValidationReport


def _node_id(
    row: int,
    column: int,
    *,
    width: int,
) -> int:
    return (
        row
        * width
        + column
    )


def generate_manhattan_topology(
    spec: ManhattanTopologySpec | None = None,
) -> ManhattanTopology:
    """
    Construct the deterministic 35×35-intersection primary
    Manhattan benchmark.

    No routing-policy or performance information is accepted by
    this generator.
    """

    resolved = (
        ManhattanTopologySpec()
        if spec is None
        else spec
    )

    n = (
        resolved.intersections_per_axis
    )

    spacing = (
        resolved.spacing_km
    )

    graph = nx.Graph()

    graph.graph.update(
        {
            "generator_schema": (
                "road-topology-v1"
            ),
            "topology_id": (
                MANHATTAN_TOPOLOGY_ID
            ),
            "topology_index": (
                MANHATTAN_TOPOLOGY_INDEX
            ),
            "benchmark_role": (
                "revised_road_topology"
            ),
            "area_km2": (
                resolved.area_km2
            ),
            "intersections_per_axis": n,
        }
    )

    for row in range(
        n
    ):
        for column in range(
            n
        ):
            node_id = _node_id(
                row,
                column,
                width=n,
            )

            graph.add_node(
                node_id,
                x=float(
                    column
                    * spacing
                ),
                y=float(
                    row
                    * spacing
                ),
                row=row,
                column=column,
            )

    for row in range(
        n
    ):
        for column in range(
            n
        ):
            source = _node_id(
                row,
                column,
                width=n,
            )

            if column + 1 < n:
                target = _node_id(
                    row,
                    column + 1,
                    width=n,
                )

                graph.add_edge(
                    source,
                    target,
                    length_km=spacing,
                    orientation="horizontal",
                )

            if row + 1 < n:
                target = _node_id(
                    row + 1,
                    column,
                    width=n,
                )

                graph.add_edge(
                    source,
                    target,
                    length_km=spacing,
                    orientation="vertical",
                )

    road_graph = RoadGraph(
        graph=graph,
        length_attribute="length_km",
    )

    report = (
        validate_benchmark_road_topology(
            road_graph,
            topology_id=(
                MANHATTAN_TOPOLOGY_ID
            ),
            area_km2=(
                resolved.area_km2
            ),
            target_density_km_per_km2=(
                resolved.target_density_km_per_km2
            ),
            density_tolerance_fraction=(
                resolved.density_tolerance_fraction
            ),
        )
    )

    if (
        report.node_count
        != resolved.expected_node_count
    ):
        raise RuntimeError(
            "Manhattan node-count invariant violated"
        )

    if (
        report.edge_count
        != resolved.expected_edge_count
    ):
        raise RuntimeError(
            "Manhattan edge-count invariant violated"
        )

    return ManhattanTopology(
        spec=resolved,
        road_graph=road_graph,
        validation=report,
    )
