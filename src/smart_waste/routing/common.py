from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RouteMetrics:
    truck_id: int

    distance_km: float = 0.0
    fuel_litres: float = 0.0

    collected_mass_tonnes: float = 0.0

    bins_visited: int = 0
    depot_returns: int = 0

    route: list[int] = field(default_factory=list)


@dataclass
class FleetMetrics:
    truck_routes: list[RouteMetrics]

    @property
    def total_distance_km(self) -> float:
        return sum(
            route.distance_km
            for route in self.truck_routes
        )

    @property
    def total_fuel_litres(self) -> float:
        return sum(
            route.fuel_litres
            for route in self.truck_routes
        )

    @property
    def total_collected_mass_tonnes(self) -> float:
        return sum(
            route.collected_mass_tonnes
            for route in self.truck_routes
        )

    @property
    def total_bins_visited(self) -> int:
        return sum(
            route.bins_visited
            for route in self.truck_routes
        )

    @property
    def total_depot_returns(self) -> int:
        return sum(
            route.depot_returns
            for route in self.truck_routes
        )