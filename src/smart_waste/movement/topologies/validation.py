from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from smart_waste.movement.road_graph import (
    RoadGraph,
)


class RoadTopologyValidationError(ValueError):
    """Raised when a benchmark road topology is structurally invalid."""


@dataclass(frozen=True)
class RoadTopologyValidationReport:
    topology_id: str

    node_count: int
    edge_count: int

    area_km2: float

    total_road_length_km: float
    road_density_km_per_km2: float

    min_x_km: float
    max_x_km: float
    min_y_km: float
    max_y_km: float


def validate_benchmark_road_topology(
    road_graph: RoadGraph,
    *,
    topology_id: str,
    area_km2: float,
    target_density_km_per_km2: float,
    density_tolerance_fraction: float,
) -> RoadTopologyValidationReport:
    """
    Validate topology-only benchmark properties.

    This layer deliberately knows nothing about collection policy,
    security, fuel performance, route performance or experiment
    outcomes.
    """

    if not topology_id.strip():
        raise RoadTopologyValidationError(
            "topology_id cannot be empty"
        )

    if (
        not isfinite(area_km2)
        or area_km2 <= 0.0
    ):
        raise RoadTopologyValidationError(
            "area_km2 must be finite and positive"
        )

    road_graph.validate_connected()

    road_graph.validate_density(
        area_km2=area_km2,
        target_km_per_km2=(
            target_density_km_per_km2
        ),
        tolerance_fraction=(
            density_tolerance_fraction
        ),
    )

    x_values: list[float] = []
    y_values: list[float] = []

    for node_id, data in (
        road_graph.graph.nodes(
            data=True
        )
    ):
        if (
            "x" not in data
            or "y" not in data
        ):
            raise RoadTopologyValidationError(
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
            raise RoadTopologyValidationError(
                f"road node {node_id!r} "
                "has nonnumeric x/y coordinates"
            ) from exc

        if (
            not isfinite(x)
            or not isfinite(y)
        ):
            raise RoadTopologyValidationError(
                f"road node {node_id!r} "
                "has nonfinite x/y coordinates"
            )

        x_values.append(
            x
        )

        y_values.append(
            y
        )

    return RoadTopologyValidationReport(
        topology_id=topology_id,
        node_count=(
            road_graph.graph.number_of_nodes()
        ),
        edge_count=(
            road_graph.graph.number_of_edges()
        ),
        area_km2=float(
            area_km2
        ),
        total_road_length_km=(
            road_graph.total_road_length_km
        ),
        road_density_km_per_km2=(
            road_graph.road_density_km_per_km2(
                area_km2=area_km2
            )
        ),
        min_x_km=min(
            x_values
        ),
        max_x_km=max(
            x_values
        ),
        min_y_km=min(
            y_values
        ),
        max_y_km=max(
            y_values
        ),
    )
