from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite, sqrt

import networkx as nx

from smart_waste.movement.road_graph import (
    RoadGraph,
)
from smart_waste.movement.topologies.validation import (
    RoadTopologyValidationReport,
    validate_benchmark_road_topology,
)


SUPERBLOCK_TOPOLOGY_ID = "superblock"
SUPERBLOCK_TOPOLOGY_INDEX = 2

DEFAULT_AREA_KM2 = 50.0
DEFAULT_MACROBLOCK_SIZE_KM = 0.4

DEFAULT_TARGET_DENSITY_KM_PER_KM2 = 10.0
DEFAULT_DENSITY_TOLERANCE_FRACTION = 0.05


Coordinate = tuple[float, float]


@dataclass(frozen=True)
class SuperblockTopologySpec:
    """
    Deterministic Barcelona-like revised-benchmark topology.

    The nominal 0.4 km macroblock dimension is applied from the
    origin. The final row/column absorbs the remaining city extent.
    """

    area_km2: float = DEFAULT_AREA_KM2

    nominal_macroblock_size_km: float = (
        DEFAULT_MACROBLOCK_SIZE_KM
    )

    target_density_km_per_km2: float = (
        DEFAULT_TARGET_DENSITY_KM_PER_KM2
    )

    density_tolerance_fraction: float = (
        DEFAULT_DENSITY_TOLERANCE_FRACTION
    )

    def __post_init__(self) -> None:
        if (
            not isfinite(self.area_km2)
            or self.area_km2 <= 0.0
        ):
            raise ValueError(
                "area_km2 must be finite and positive"
            )

        if (
            not isfinite(
                self.nominal_macroblock_size_km
            )
            or self.nominal_macroblock_size_km <= 0.0
        ):
            raise ValueError(
                "nominal_macroblock_size_km must be "
                "finite and positive"
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
    def macroblocks_per_axis(self) -> int:
        return int(
            ceil(
                self.side_km
                / self.nominal_macroblock_size_km
            )
        )

    @property
    def boundaries_km(self) -> tuple[float, ...]:
        n = self.macroblocks_per_axis

        return tuple(
            (
                float(
                    index
                    * self.nominal_macroblock_size_km
                )
                if index < n
                else self.side_km
            )
            for index in range(
                n + 1
            )
        )

    @property
    def expected_candidate_count(self) -> int:
        n = self.macroblocks_per_axis

        corners = (
            (n + 1)
            * (n + 1)
        )

        side_midpoints = (
            2
            * n
            * (n + 1)
        )

        return (
            corners
            + side_midpoints
        )

    @property
    def expected_node_count(self) -> int:
        n = self.macroblocks_per_axis

        corners = (
            (n + 1)
            * (n + 1)
        )

        side_midpoints = (
            2
            * n
            * (n + 1)
        )

        internal_boundary_endpoints = (
            4
            * n
            * n
        )

        internal_crossings = (
            n
            * n
        )

        return (
            corners
            + side_midpoints
            + internal_boundary_endpoints
            + internal_crossings
        )

    @property
    def expected_edge_count(self) -> int:
        n = self.macroblocks_per_axis

        boundary_edges = (
            8
            * n
            * n
            + 4
            * n
        )

        internal_edges = (
            4
            * n
            * n
        )

        return (
            boundary_edges
            + internal_edges
        )

    @property
    def expected_total_road_length_km(
        self,
    ) -> float:
        n = self.macroblocks_per_axis

        return (
            (4 * n + 2)
            * self.side_km
        )


@dataclass(frozen=True)
class SuperblockTopology:
    spec: SuperblockTopologySpec
    road_graph: RoadGraph
    validation: RoadTopologyValidationReport


def _internal_fractions(
    row: int,
    column: int,
) -> tuple[float, float]:
    """
    Return:
        (north_south_fraction, east_west_fraction)
    """

    if (
        (row + column)
        % 2
        == 0
    ):
        return (
            1.0 / 3.0,
            2.0 / 3.0,
        )

    return (
        2.0 / 3.0,
        1.0 / 3.0,
    )


def _ensure_coordinate_node(
    graph: nx.Graph,
    coordinate: Coordinate,
    *,
    bin_candidate: bool = False,
) -> None:
    if coordinate not in graph:
        graph.add_node(
            coordinate,
            superblock_bin_candidate=(
                bool(bin_candidate)
            ),
        )

        return

    if bin_candidate:
        graph.nodes[
            coordinate
        ][
            "superblock_bin_candidate"
        ] = True


def _add_axis_aligned_edge(
    graph: nx.Graph,
    source: Coordinate,
    target: Coordinate,
    *,
    street_class: str,
) -> None:
    _ensure_coordinate_node(
        graph,
        source,
    )

    _ensure_coordinate_node(
        graph,
        target,
    )

    x1, y1 = source
    x2, y2 = target

    if source == target:
        raise RuntimeError(
            "zero-length Superblock edge"
        )

    if x1 == x2:
        orientation = "vertical"
        length = abs(
            y2 - y1
        )
    elif y1 == y2:
        orientation = "horizontal"
        length = abs(
            x2 - x1
        )
    else:
        raise RuntimeError(
            "Superblock edge must be axis-aligned"
        )

    if (
        not isfinite(length)
        or length <= 0.0
    ):
        raise RuntimeError(
            "invalid Superblock edge length"
        )

    if graph.has_edge(
        source,
        target,
    ):
        raise RuntimeError(
            "duplicate Superblock physical edge"
        )

    graph.add_edge(
        source,
        target,
        length_km=float(length),
        orientation=orientation,
        street_class=street_class,
    )


def generate_superblock_topology(
    spec: SuperblockTopologySpec | None = None,
) -> SuperblockTopology:
    """
    Generate the deterministic Barcelona-like Superblock network.

    The generator consumes no workload, routing policy, security
    state, performance metric or random seed.
    """

    resolved = (
        SuperblockTopologySpec()
        if spec is None
        else spec
    )

    boundaries = (
        resolved.boundaries_km
    )

    n = (
        resolved.macroblocks_per_axis
    )

    coordinate_graph = nx.Graph()

    # ---------------------------------------------------------
    # Candidate points:
    # all macroblock corners + all macroblock-side midpoints.
    # ---------------------------------------------------------

    for y in boundaries:
        for x in boundaries:
            _ensure_coordinate_node(
                coordinate_graph,
                (
                    x,
                    y,
                ),
                bin_candidate=True,
            )

    # Horizontal macroblock-side midpoints.
    for line_index in range(
        n + 1
    ):
        y = boundaries[
            line_index
        ]

        for column in range(
            n
        ):
            x0 = boundaries[
                column
            ]
            x1 = boundaries[
                column + 1
            ]

            midpoint = (
                (
                    x0 + x1
                )
                / 2.0
            )

            _ensure_coordinate_node(
                coordinate_graph,
                (
                    midpoint,
                    y,
                ),
                bin_candidate=True,
            )

    # Vertical macroblock-side midpoints.
    for line_index in range(
        n + 1
    ):
        x = boundaries[
            line_index
        ]

        for row in range(
            n
        ):
            y0 = boundaries[
                row
            ]
            y1 = boundaries[
                row + 1
            ]

            midpoint = (
                (
                    y0 + y1
                )
                / 2.0
            )

            _ensure_coordinate_node(
                coordinate_graph,
                (
                    x,
                    midpoint,
                ),
                bin_candidate=True,
            )

    # ---------------------------------------------------------
    # Macroblock horizontal boundaries.
    #
    # Each segment is split at:
    # - its midpoint,
    # - N-S internal streets terminating from cells above/below.
    # ---------------------------------------------------------

    for line_index in range(
        n + 1
    ):
        y = boundaries[
            line_index
        ]

        for column in range(
            n
        ):
            x0 = boundaries[
                column
            ]
            x1 = boundaries[
                column + 1
            ]

            width = (
                x1 - x0
            )

            split_x = {
                x0,
                (
                    x0 + x1
                )
                / 2.0,
                x1,
            }

            if line_index > 0:
                fraction, _ = (
                    _internal_fractions(
                        line_index - 1,
                        column,
                    )
                )

                split_x.add(
                    x0
                    + fraction
                    * width
                )

            if line_index < n:
                fraction, _ = (
                    _internal_fractions(
                        line_index,
                        column,
                    )
                )

                split_x.add(
                    x0
                    + fraction
                    * width
                )

            ordered_x = sorted(
                split_x
            )

            for index in range(
                len(ordered_x) - 1
            ):
                _add_axis_aligned_edge(
                    coordinate_graph,
                    (
                        ordered_x[index],
                        y,
                    ),
                    (
                        ordered_x[index + 1],
                        y,
                    ),
                    street_class="macro_boundary",
                )

    # ---------------------------------------------------------
    # Macroblock vertical boundaries.
    #
    # Each segment is split at:
    # - its midpoint,
    # - E-W internal streets terminating from cells left/right.
    # ---------------------------------------------------------

    for line_index in range(
        n + 1
    ):
        x = boundaries[
            line_index
        ]

        for row in range(
            n
        ):
            y0 = boundaries[
                row
            ]
            y1 = boundaries[
                row + 1
            ]

            height = (
                y1 - y0
            )

            split_y = {
                y0,
                (
                    y0 + y1
                )
                / 2.0,
                y1,
            }

            if line_index > 0:
                _, fraction = (
                    _internal_fractions(
                        row,
                        line_index - 1,
                    )
                )

                split_y.add(
                    y0
                    + fraction
                    * height
                )

            if line_index < n:
                _, fraction = (
                    _internal_fractions(
                        row,
                        line_index,
                    )
                )

                split_y.add(
                    y0
                    + fraction
                    * height
                )

            ordered_y = sorted(
                split_y
            )

            for index in range(
                len(ordered_y) - 1
            ):
                _add_axis_aligned_edge(
                    coordinate_graph,
                    (
                        x,
                        ordered_y[index],
                    ),
                    (
                        x,
                        ordered_y[index + 1],
                    ),
                    street_class="macro_boundary",
                )

    # ---------------------------------------------------------
    # Internal checkerboard streets.
    # ---------------------------------------------------------

    for row in range(
        n
    ):
        y0 = boundaries[
            row
        ]
        y1 = boundaries[
            row + 1
        ]

        height = (
            y1 - y0
        )

        for column in range(
            n
        ):
            x0 = boundaries[
                column
            ]
            x1 = boundaries[
                column + 1
            ]

            width = (
                x1 - x0
            )

            (
                north_south_fraction,
                east_west_fraction,
            ) = _internal_fractions(
                row,
                column,
            )

            internal_x = (
                x0
                + north_south_fraction
                * width
            )

            internal_y = (
                y0
                + east_west_fraction
                * height
            )

            bottom = (
                internal_x,
                y0,
            )

            crossing = (
                internal_x,
                internal_y,
            )

            top = (
                internal_x,
                y1,
            )

            left = (
                x0,
                internal_y,
            )

            right = (
                x1,
                internal_y,
            )

            _add_axis_aligned_edge(
                coordinate_graph,
                bottom,
                crossing,
                street_class="internal",
            )

            _add_axis_aligned_edge(
                coordinate_graph,
                crossing,
                top,
                street_class="internal",
            )

            _add_axis_aligned_edge(
                coordinate_graph,
                left,
                crossing,
                street_class="internal",
            )

            _add_axis_aligned_edge(
                coordinate_graph,
                crossing,
                right,
                street_class="internal",
            )

    # ---------------------------------------------------------
    # Canonical integer road-node IDs.
    #
    # Sort row-major by physical y, then x.
    # ---------------------------------------------------------

    ordered_coordinates = tuple(
        sorted(
            coordinate_graph.nodes,
            key=lambda coordinate: (
                coordinate[1],
                coordinate[0],
            ),
        )
    )

    node_id_by_coordinate = {
        coordinate: node_id
        for node_id, coordinate
        in enumerate(
            ordered_coordinates
        )
    }

    graph = nx.Graph()

    graph.graph.update(
        {
            "generator_schema": (
                "road-topology-v1"
            ),
            "topology_id": (
                SUPERBLOCK_TOPOLOGY_ID
            ),
            "topology_index": (
                SUPERBLOCK_TOPOLOGY_INDEX
            ),
            "benchmark_role": (
                "revised_road_topology"
            ),
            "area_km2": (
                resolved.area_km2
            ),
            "nominal_macroblock_size_km": (
                resolved.nominal_macroblock_size_km
            ),
            "macroblocks_per_axis": n,
            "internal_layout": (
                "checkerboard_thirds"
            ),
        }
    )

    for coordinate in ordered_coordinates:
        node_id = (
            node_id_by_coordinate[
                coordinate
            ]
        )

        attributes = (
            coordinate_graph.nodes[
                coordinate
            ]
        )

        graph.add_node(
            node_id,
            x=float(
                coordinate[0]
            ),
            y=float(
                coordinate[1]
            ),
            superblock_bin_candidate=bool(
                attributes.get(
                    "superblock_bin_candidate",
                    False,
                )
            ),
        )

    for (
        source_coordinate,
        target_coordinate,
        attributes,
    ) in coordinate_graph.edges(
        data=True
    ):
        graph.add_edge(
            node_id_by_coordinate[
                source_coordinate
            ],
            node_id_by_coordinate[
                target_coordinate
            ],
            **dict(
                attributes
            ),
        )

    road_graph = RoadGraph(
        graph=graph,
        length_attribute="length_km",
    )

    report = (
        validate_benchmark_road_topology(
            road_graph,
            topology_id=(
                SUPERBLOCK_TOPOLOGY_ID
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
            "Superblock node-count invariant violated: "
            f"{report.node_count} != "
            f"{resolved.expected_node_count}"
        )

    if (
        report.edge_count
        != resolved.expected_edge_count
    ):
        raise RuntimeError(
            "Superblock edge-count invariant violated: "
            f"{report.edge_count} != "
            f"{resolved.expected_edge_count}"
        )

    candidate_count = sum(
        1
        for _, attributes
        in graph.nodes(
            data=True
        )
        if attributes[
            "superblock_bin_candidate"
        ]
    )

    if (
        candidate_count
        != resolved.expected_candidate_count
    ):
        raise RuntimeError(
            "Superblock bin-candidate invariant violated: "
            f"{candidate_count} != "
            f"{resolved.expected_candidate_count}"
        )

    return SuperblockTopology(
        spec=resolved,
        road_graph=road_graph,
        validation=report,
    )
