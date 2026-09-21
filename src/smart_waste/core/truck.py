from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Truck:
    """
    Representation of one collection vehicle.
    """

    truck_id: int
    capacity_tonnes: float
    fuel_efficiency_km_per_litre: float
    x_km: float
    y_km: float

    current_load_tonnes: float = 0.0
    distance_travelled_km: float = 0.0

    def __post_init__(self) -> None:
        if self.truck_id < 0:
            raise ValueError("truck_id must be non-negative")

        if self.capacity_tonnes <= 0:
            raise ValueError(
                "capacity_tonnes must be greater than zero"
            )

        if self.fuel_efficiency_km_per_litre <= 0:
            raise ValueError(
                "fuel_efficiency_km_per_litre must be greater than zero"
            )

        if self.current_load_tonnes < 0:
            raise ValueError(
                "current_load_tonnes cannot be negative"
            )

        if self.current_load_tonnes > self.capacity_tonnes:
            raise ValueError(
                "current load cannot exceed truck capacity"
            )

    @property
    def remaining_capacity_tonnes(self) -> float:
        return self.capacity_tonnes - self.current_load_tonnes

    @property
    def fuel_consumed_litres(self) -> float:
        return (
            self.distance_travelled_km
            / self.fuel_efficiency_km_per_litre
        )

    @property
    def location(self) -> tuple[float, float]:
        return self.x_km, self.y_km

    def reset(
        self,
        x_km: float,
        y_km: float,
    ) -> None:
        self.x_km = x_km
        self.y_km = y_km
        self.current_load_tonnes = 0.0
        self.distance_travelled_km = 0.0