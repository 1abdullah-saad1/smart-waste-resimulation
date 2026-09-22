from __future__ import annotations

from dataclasses import dataclass
from math import (
    ceil,
    cos,
    dist,
    floor,
    isfinite,
    radians,
    sin,
    sqrt,
)

import networkx as nx

from smart_waste.movement.road_graph import (
    RoadGraph,
)
from smart_waste.movement.topologies.validation import (
    RoadTopologyValidationReport,
    validate_benchmark_road_topology,
)


HEX_TOPOLOGY_ID = "hex"
HEX_TOPOLOGY_INDEX = 3

DEFAULT_AREA_KM2 = 50.0
DEFAULT_HEX_SIDE_KM = 0.11547

DEFAULT_TARGET_DENSITY_KM_PER_KM2 = 10.0
DEFAULT_DENSITY_TOLERANCE_FRACTION = 0.05

_COORDINATE_DIGITS = 12

Coordinate = tuple[float, float]
EdgeKey = tuple[
    Coordinate,
    Coordinate,
]


@dataclass(frozen=True)
class HexTopologySpec:
    """
    Deterministic flat-top regular hexagonal road topology.

    The infinite tessellation is clipped to the square benchmark
    city. Boundary clipping changes only the finite-domain edge
    fragments; the interior regular hexagonal geometry is unchanged.
    """

    area_km2: float = DEFAULT_AREA_KM2
    hex_side_km: float = DEFAULT_HEX_SIDE_KM

    target_density_km_per_km2: float = (
        DEFAULT_TARGET_DENSITY_KM_PER_KM2
    )

    density_tolerance_fraction: float = (
        DEFAULT_DENSITY_TOLERANCE_FRACTION
    )

    def __post_init__(self) -> None:
        positive_values = {
            "area_km2": self.area_km2,
            "hex_side_km": self.hex_side_km,
            "target_density_km_per_km2": (
                self.target_density_km_per_km2
            ),
        }

        for name, value in positive_values.items():
            if (
                not isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(
                    f"{name} must be finite and positive"
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
    def hex_height_km(self) -> float:
        return (
            sqrt(3.0)
            * self.hex_side_km
        )

    @property
    def horizontal_center_spacing_km(self) -> float:
        return (
            1.5
            * self.hex_side_km
        )

    @property
    def theoretical_infinite_density_km_per_km2(
        self,
    ) -> float:
        return (
            2.0
            / (
                sqrt(3.0)
                * self.hex_side_km
            )
        )


@dataclass(frozen=True)
class HexTopology:
    spec: HexTopologySpec
    road_graph: RoadGraph
    validation: RoadTopologyValidationReport


def _canonical_coordinate(
    x: float,
    y: float,
) -> Coordinate:
    return (
        round(
            float(x),
            _COORDINATE_DIGITS,
        ),
        round(
            float(y),
            _COORDINATE_DIGITS,
        ),
    )


def _canonical_edge(
    first: Coordinate,
    second: Coordinate,
) -> EdgeKey:
    if first <= second:
        return (
            first,
            second,
        )

    return (
        second,
        first,
    )


def _clip_segment_to_square(
    first: Coordinate,
    second: Coordinate,
    *,
    side_km: float,
) -> tuple[
    Coordinate,
    Coordinate,
] | None:
    """
    Liang-Barsky clipping against:
        0 <= x <= side_km
        0 <= y <= side_km
    """

    x0, y0 = first
    x1, y1 = second

    dx = (
        x1 - x0
    )

    dy = (
        y1 - y0
    )

    p = (
        -dx,
        dx,
        -dy,
        dy,
    )

    q = (
        x0,
        side_km - x0,
        y0,
        side_km - y0,
    )

    lower = 0.0
    upper = 1.0

    for coefficient, bound in zip(
        p,
        q,
        strict=True,
    ):
        if abs(
            coefficient
        ) <= 1e-15:
            if bound < 0.0:
                return None

            continue

        ratio = (
            bound
            / coefficient
        )

        if coefficient < 0.0:
            if ratio > upper:
                return None

            lower = max(
                lower,
                ratio,
            )

        else:
            if ratio < lower:
                return None

            upper = min(
                upper,
                ratio,
            )

    clipped_first = (
        x0 + lower * dx,
        y0 + lower * dy,
    )

    clipped_second = (
        x0 + upper * dx,
        y0 + upper * dy,
    )

    first_canonical = (
        _canonical_coordinate(
            *clipped_first
        )
    )

    second_canonical = (
        _canonical_coordinate(
            *clipped_second
        )
    )

    if first_canonical == second_canonical:
        return None

    return (
        first_canonical,
        second_canonical,
    )


def _generate_unclipped_hex_edges(
    spec: HexTopologySpec,
) -> set[
    EdgeKey
]:
    """
    Generate enough cells around the benchmark square that every
    physical hex edge intersecting the square exists before clipping.
    """

    side = (
        spec.side_km
    )

    hex_side = (
        spec.hex_side_km
    )

    height = (
        spec.hex_height_km
    )

    horizontal_spacing = (
        spec.horizontal_center_spacing_km
    )

    margin = (
        2.0
        * hex_side
    )

    minimum_column = (
        floor(
            -margin
            / horizontal_spacing
        )
        - 2
    )

    maximum_column = (
        ceil(
            (
                side
                + margin
            )
            / horizontal_spacing
        )
        + 2
    )

    edges: set[
        EdgeKey
    ] = set()

    for column in range(
        minimum_column,
        maximum_column + 1,
    ):
        center_x = (
            horizontal_spacing
            * column
        )

        vertical_offset = (
            height / 2.0
            if column % 2
            else 0.0
        )

        minimum_row = (
            floor(
                (
                    -margin
                    - vertical_offset
                )
                / height
            )
            - 2
        )

        maximum_row = (
            ceil(
                (
                    side
                    + margin
                    - vertical_offset
                )
                / height
            )
            + 2
        )

        for row in range(
            minimum_row,
            maximum_row + 1,
        ):
            center_y = (
                height
                * row
                + vertical_offset
            )

            vertices = tuple(
                _canonical_coordinate(
                    center_x
                    + hex_side
                    * cos(
                        radians(
                            60.0
                            * vertex_index
                        )
                    ),
                    center_y
                    + hex_side
                    * sin(
                        radians(
                            60.0
                            * vertex_index
                        )
                    ),
                )
                for vertex_index in range(
                    6
                )
            )

            for vertex_index in range(
                6
            ):
                first = (
                    vertices[
                        vertex_index
                    ]
                )

                second = (
                    vertices[
                        (
                            vertex_index
                            + 1
                        )
                        % 6
                    ]
                )

                edges.add(
                    _canonical_edge(
                        first,
                        second,
                    )
                )

    return edges


def generate_hex_topology(
    spec: HexTopologySpec | None = None,
) -> HexTopology:
    """
    Generate the deterministic regular hexagonal benchmark network.

    No workload, routing policy, attack state, performance result or
    random generator contributes to the topology.
    """

    resolved = (
        HexTopologySpec()
        if spec is None
        else spec
    )

    raw_edges = (
        _generate_unclipped_hex_edges(
            resolved
        )
    )

    clipped_edges: dict[
        EdgeKey,
        float,
    ] = {}

    original_vertices: set[
        Coordinate
    ] = set()

    for first, second in raw_edges:
        clipped = (
            _clip_segment_to_square(
                first,
                second,
                side_km=(
                    resolved.side_km
                ),
            )
        )

        if clipped is None:
            continue

        clipped_first, clipped_second = (
            clipped
        )

        edge = (
            _canonical_edge(
                clipped_first,
                clipped_second,
            )
        )

        length = dist(
            clipped_first,
            clipped_second,
        )

        if (
            not isfinite(length)
            or length <= 0.0
        ):
            raise RuntimeError(
                "invalid clipped Hex edge length"
            )

        clipped_edges[
            edge
        ] = float(
            length
        )

        # An endpoint is a true shared hex vertex only when clipping
        # did not create it artificially at the city boundary.
        if clipped_first in (
            first,
            second,
        ):
            original_vertices.add(
                clipped_first
            )

        if clipped_second in (
            first,
            second,
        ):
            original_vertices.add(
                clipped_second
            )

    coordinate_graph = nx.Graph()

    for (
        first,
        second,
    ), length in clipped_edges.items():
        coordinate_graph.add_edge(
            first,
            second,
            length_km=length,
            street_class="hex_edge",
        )

    if not nx.is_connected(
        coordinate_graph
    ):
        raise RuntimeError(
            "Hex clipped road graph must be connected"
        )

    # A true shared interior honeycomb vertex has degree exactly 3.
    candidate_coordinates = {
        coordinate
        for coordinate
        in original_vertices
        if (
            coordinate in coordinate_graph
            and coordinate_graph.degree[
                coordinate
            ] == 3
        )
    }

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
                HEX_TOPOLOGY_ID
            ),
            "topology_index": (
                HEX_TOPOLOGY_INDEX
            ),
            "benchmark_role": (
                "revised_road_topology"
            ),
            "area_km2": (
                resolved.area_km2
            ),
            "hex_side_km": (
                resolved.hex_side_km
            ),
            "hex_orientation": (
                "flat_top"
            ),
            "boundary_policy": (
                "clip_to_square"
            ),
        }
    )

    for coordinate in ordered_coordinates:
        node_id = (
            node_id_by_coordinate[
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
            hex_bin_candidate=(
                coordinate
                in candidate_coordinates
            ),
        )

    for (
        first,
        second,
    ), length in clipped_edges.items():
        graph.add_edge(
            node_id_by_coordinate[
                first
            ],
            node_id_by_coordinate[
                second
            ],
            length_km=float(
                length
            ),
            street_class="hex_edge",
        )

    road_graph = RoadGraph(
        graph=graph,
        length_attribute="length_km",
    )

    validation = (
        validate_benchmark_road_topology(
            road_graph,
            topology_id=(
                HEX_TOPOLOGY_ID
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

    candidate_count = sum(
        1
        for _, attributes
        in graph.nodes(
            data=True
        )
        if attributes[
            "hex_bin_candidate"
        ]
    )

    if candidate_count < 1001:
        raise RuntimeError(
            "Hex topology must provide at least 1001 "
            "shared-vertex candidates"
        )

    return HexTopology(
        spec=resolved,
        road_graph=road_graph,
        validation=validation,
    )
