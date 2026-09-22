from __future__ import annotations

from dataclasses import dataclass
from math import (
    cos,
    isfinite,
    pi,
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


RADIAL_CONCENTRIC_TOPOLOGY_ID = (
    "radial_concentric"
)
RADIAL_CONCENTRIC_TOPOLOGY_INDEX = 4

DEFAULT_AREA_KM2 = 50.0
DEFAULT_RING_COUNT = 20
DEFAULT_SPOKE_COUNT = 60

DEFAULT_TARGET_DENSITY_KM_PER_KM2 = 10.0
DEFAULT_DENSITY_TOLERANCE_FRACTION = 0.05


@dataclass(frozen=True)
class RadialConcentricTopologySpec:
    """
    Deterministic disk-shaped radial-concentric benchmark topology.

    Twenty equally spaced circular rings are intersected by sixty
    equally spaced radial spokes.

    Circular ring edges use physical arc length rather than chord
    length.
    """

    area_km2: float = DEFAULT_AREA_KM2

    ring_count: int = DEFAULT_RING_COUNT
    spoke_count: int = DEFAULT_SPOKE_COUNT

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

        if self.ring_count <= 0:
            raise ValueError(
                "ring_count must be positive"
            )

        if self.spoke_count < 3:
            raise ValueError(
                "spoke_count must be at least 3"
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
    def radius_km(self) -> float:
        return sqrt(
            self.area_km2
            / pi
        )

    @property
    def center_x_km(self) -> float:
        return self.radius_km

    @property
    def center_y_km(self) -> float:
        return self.radius_km

    @property
    def radial_spacing_km(self) -> float:
        return (
            self.radius_km
            / self.ring_count
        )

    @property
    def angular_spacing_radians(self) -> float:
        return (
            2.0
            * pi
            / self.spoke_count
        )

    @property
    def expected_node_count(self) -> int:
        return (
            1
            + self.ring_count
            * self.spoke_count
        )

    @property
    def expected_edge_count(self) -> int:
        return (
            2
            * self.ring_count
            * self.spoke_count
        )

    @property
    def expected_candidate_count(self) -> int:
        return (
            self.ring_count
            * self.spoke_count
        )

    @property
    def expected_spoke_length_km(self) -> float:
        return (
            self.spoke_count
            * self.radius_km
        )

    @property
    def expected_ring_length_km(self) -> float:
        # Sum of ring radii:
        #
        # R / n * (1 + ... + n)
        # = R * (n + 1) / 2
        #
        # Circumference sum:
        # 2π * that value
        # = π R (n + 1)
        return (
            pi
            * self.radius_km
            * (
                self.ring_count
                + 1
            )
        )

    @property
    def expected_total_road_length_km(
        self,
    ) -> float:
        return (
            self.expected_spoke_length_km
            + self.expected_ring_length_km
        )


@dataclass(frozen=True)
class RadialConcentricTopology:
    spec: RadialConcentricTopologySpec
    road_graph: RoadGraph
    validation: RoadTopologyValidationReport


def _ring_node_id(
    *,
    ring_index: int,
    spoke_index: int,
    spoke_count: int,
) -> int:
    """
    Center is node 0.

    Ring nodes are then assigned deterministically:
        ring 1 -> 1 ... spoke_count
        ring 2 -> ...
    """

    return (
        1
        + (
            ring_index
            - 1
        )
        * spoke_count
        + spoke_index
    )


def generate_radial_concentric_topology(
    spec: RadialConcentricTopologySpec | None = None,
) -> RadialConcentricTopology:
    """
    Generate the deterministic radial-concentric road network.

    No workload, collection policy, attack state, performance
    result or random generator contributes to this topology.
    """

    resolved = (
        RadialConcentricTopologySpec()
        if spec is None
        else spec
    )

    graph = nx.Graph()

    graph.graph.update(
        {
            "generator_schema": (
                "road-topology-v1"
            ),
            "topology_id": (
                RADIAL_CONCENTRIC_TOPOLOGY_ID
            ),
            "topology_index": (
                RADIAL_CONCENTRIC_TOPOLOGY_INDEX
            ),
            "benchmark_role": (
                "revised_road_topology"
            ),
            "area_km2": (
                resolved.area_km2
            ),
            "city_shape": "disk",
            "radius_km": (
                resolved.radius_km
            ),
            "ring_count": (
                resolved.ring_count
            ),
            "spoke_count": (
                resolved.spoke_count
            ),
            "radial_spacing": (
                "equal_radius"
            ),
            "ring_edge_metric": (
                "circular_arc_length"
            ),
        }
    )

    center_node = 0

    graph.add_node(
        center_node,
        x=float(
            resolved.center_x_km
        ),
        y=float(
            resolved.center_y_km
        ),
        node_role="center",
        ring_index=0,
        radius_km=0.0,
        radial_bin_candidate=False,
    )

    # ---------------------------------------------------------
    # Nodes at every ring/spoke intersection.
    # ---------------------------------------------------------

    for ring_index in range(
        1,
        resolved.ring_count + 1,
    ):
        radius = (
            ring_index
            * resolved.radial_spacing_km
        )

        for spoke_index in range(
            resolved.spoke_count
        ):
            angle = (
                spoke_index
                * resolved.angular_spacing_radians
            )

            node_id = (
                _ring_node_id(
                    ring_index=ring_index,
                    spoke_index=spoke_index,
                    spoke_count=(
                        resolved.spoke_count
                    ),
                )
            )

            x = (
                resolved.center_x_km
                + radius
                * cos(
                    angle
                )
            )

            y = (
                resolved.center_y_km
                + radius
                * sin(
                    angle
                )
            )

            graph.add_node(
                node_id,
                x=float(x),
                y=float(y),
                node_role=(
                    "ring_spoke_intersection"
                ),
                ring_index=ring_index,
                spoke_index=spoke_index,
                radius_km=float(
                    radius
                ),
                angle_radians=float(
                    angle
                ),
                radial_bin_candidate=True,
            )

    # ---------------------------------------------------------
    # Radial spoke edges.
    # ---------------------------------------------------------

    for spoke_index in range(
        resolved.spoke_count
    ):
        previous_node = (
            center_node
        )

        for ring_index in range(
            1,
            resolved.ring_count + 1,
        ):
            current_node = (
                _ring_node_id(
                    ring_index=ring_index,
                    spoke_index=spoke_index,
                    spoke_count=(
                        resolved.spoke_count
                    ),
                )
            )

            graph.add_edge(
                previous_node,
                current_node,
                length_km=float(
                    resolved.radial_spacing_km
                ),
                street_class="spoke",
                spoke_index=spoke_index,
                radial_segment_index=(
                    ring_index
                ),
            )

            previous_node = (
                current_node
            )

    # ---------------------------------------------------------
    # Circular ring arc edges.
    # ---------------------------------------------------------

    for ring_index in range(
        1,
        resolved.ring_count + 1,
    ):
        radius = (
            ring_index
            * resolved.radial_spacing_km
        )

        arc_length = (
            radius
            * resolved.angular_spacing_radians
        )

        for spoke_index in range(
            resolved.spoke_count
        ):
            next_spoke = (
                (
                    spoke_index
                    + 1
                )
                % resolved.spoke_count
            )

            source = (
                _ring_node_id(
                    ring_index=ring_index,
                    spoke_index=spoke_index,
                    spoke_count=(
                        resolved.spoke_count
                    ),
                )
            )

            target = (
                _ring_node_id(
                    ring_index=ring_index,
                    spoke_index=next_spoke,
                    spoke_count=(
                        resolved.spoke_count
                    ),
                )
            )

            graph.add_edge(
                source,
                target,
                length_km=float(
                    arc_length
                ),
                street_class="ring",
                ring_index=ring_index,
                arc_index=spoke_index,
            )

    road_graph = RoadGraph(
        graph=graph,
        length_attribute="length_km",
    )

    validation = (
        validate_benchmark_road_topology(
            road_graph,
            topology_id=(
                RADIAL_CONCENTRIC_TOPOLOGY_ID
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
        validation.node_count
        != resolved.expected_node_count
    ):
        raise RuntimeError(
            "Radial-Concentric node-count invariant violated: "
            f"{validation.node_count} != "
            f"{resolved.expected_node_count}"
        )

    if (
        validation.edge_count
        != resolved.expected_edge_count
    ):
        raise RuntimeError(
            "Radial-Concentric edge-count invariant violated: "
            f"{validation.edge_count} != "
            f"{resolved.expected_edge_count}"
        )

    candidate_count = sum(
        1
        for _, attributes
        in graph.nodes(
            data=True
        )
        if attributes.get(
            "radial_bin_candidate",
            False,
        )
    )

    if (
        candidate_count
        != resolved.expected_candidate_count
    ):
        raise RuntimeError(
            "Radial-Concentric candidate-count invariant violated: "
            f"{candidate_count} != "
            f"{resolved.expected_candidate_count}"
        )

    return RadialConcentricTopology(
        spec=resolved,
        road_graph=road_graph,
        validation=validation,
    )
