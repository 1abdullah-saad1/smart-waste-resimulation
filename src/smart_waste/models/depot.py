from __future__ import annotations

from dataclasses import dataclass

from smart_waste.models.fuel import (
    refuel_time_minutes,
)
from smart_waste.models.truck import Truck


@dataclass(frozen=True)
class Depot:
    """Physical service configuration of the central depot."""

    road_node: int | str
    unloading_bays: int
    unload_time_minutes: float
    refuel_rate_litres_per_minute: float

    def __post_init__(self) -> None:
        if self.unloading_bays <= 0:
            raise ValueError(
                "unloading_bays must be positive"
            )

        if self.unload_time_minutes < 0.0:
            raise ValueError(
                "unload_time_minutes cannot be negative"
            )

        if self.refuel_rate_litres_per_minute <= 0.0:
            raise ValueError(
                "refuel_rate_litres_per_minute must be positive"
            )

    def unload_duration_minutes(
        self,
        truck: Truck,
    ) -> float:
        if truck.current_load_tonnes <= 0.0:
            return 0.0

        return self.unload_time_minutes

    def refuel_duration_minutes(
        self,
        truck: Truck,
    ) -> float:
        return refuel_time_minutes(
            current_litres=truck.fuel_remaining_litres,
            tank_capacity_litres=(
                truck.fuel_capacity_litres
            ),
            rate_litres_per_minute=(
                self.refuel_rate_litres_per_minute
            ),
        )

    def service_duration_minutes(
        self,
        truck: Truck,
    ) -> float:
        """
        Sequential primary depot service:
        unload first, then refuel.
        Queue time is handled by the event-driven engine.
        """

        return (
            self.unload_duration_minutes(truck)
            + self.refuel_duration_minutes(truck)
        )
