from __future__ import annotations

from dataclasses import dataclass

import networkx as nx


class RoadGraphError(ValueError):
    """Raised when a road graph violates simulation invariants."""


@dataclass
class RoadGraph:
    """
    Validated physical road network.

    Truck movement is permitted only over graph edges whose
    physical length is stored in `length_km`.
    """

    graph: nx.Graph
    length_attribute: str = "length_km"

    def __post_init__(self) -> None:
        if self.graph.is_directed():
            raise RoadGraphError(
                "primary road graph must be undirected"
            )

        if self.graph.number_of_nodes() == 0:
            raise RoadGraphError(
                "road graph cannot be empty"
            )

        if self.graph.number_of_edges() == 0:
            raise RoadGraphError(
                "road graph must contain edges"
            )

        self._validate_edge_lengths()

    def _validate_edge_lengths(self) -> None:
        for source, target, data in self.graph.edges(data=True):
            if self.length_attribute not in data:
                raise RoadGraphError(
                    f"edge ({source!r}, {target!r}) is missing "
                    f"{self.length_attribute!r}"
                )

            length = float(
                data[self.length_attribute]
            )

            if length <= 0.0:
                raise RoadGraphError(
                    f"edge ({source!r}, {target!r}) "
                    "must have positive physical length"
                )

    @property
    def is_connected(self) -> bool:
        return nx.is_connected(self.graph)

    @property
    def total_road_length_km(self) -> float:
        return float(
            sum(
                float(data[self.length_attribute])
                for _, _, data in self.graph.edges(data=True)
            )
        )

    def road_density_km_per_km2(
        self,
        *,
        area_km2: float,
    ) -> float:
        if area_km2 <= 0.0:
            raise ValueError(
                "area_km2 must be positive"
            )

        return self.total_road_length_km / area_km2

    def validate_connected(self) -> None:
        if not self.is_connected:
            raise RoadGraphError(
                "road graph must be connected"
            )

    def validate_service_nodes(
        self,
        nodes: list[int | str] | tuple[int | str, ...],
    ) -> None:
        missing = [
            node
            for node in nodes
            if node not in self.graph
        ]

        if missing:
            raise RoadGraphError(
                f"service nodes missing from road graph: {missing}"
            )

    def validate_density(
        self,
        *,
        area_km2: float,
        target_km_per_km2: float,
        tolerance_fraction: float,
    ) -> None:
        if target_km_per_km2 <= 0.0:
            raise ValueError(
                "target_km_per_km2 must be positive"
            )

        if not 0.0 <= tolerance_fraction < 1.0:
            raise ValueError(
                "tolerance_fraction must be in [0, 1)"
            )

        density = self.road_density_km_per_km2(
            area_km2=area_km2,
        )

        lower = (
            target_km_per_km2
            * (1.0 - tolerance_fraction)
        )
        upper = (
            target_km_per_km2
            * (1.0 + tolerance_fraction)
        )

        if not lower <= density <= upper:
            raise RoadGraphError(
                "road density outside benchmark acceptance range: "
                f"{density:.6f} not in "
                f"[{lower:.6f}, {upper:.6f}] km/km^2"
            )
