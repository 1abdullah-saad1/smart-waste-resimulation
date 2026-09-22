from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from smart_waste.models.fuel import (
    fuel_required_litres,
)


class TruckStatus(str, Enum):
    IDLE = "idle"
    TRAVELLING = "travelling"
    SERVICING_BIN = "servicing_bin"
    WAITING_AT_DEPOT = "waiting_at_depot"
    UNLOADING = "unloading"
    REFUELLING = "refuelling"
    FINISHED = "finished"


@dataclass
class Truck:
    """Mutable physical state of one collection truck."""

    truck_id: int
    current_node: int | str
    capacity_tonnes: float
    speed_km_per_hour: float
    fuel_capacity_litres: float
    fuel_efficiency_km_per_litre: float

    current_load_tonnes: float = 0.0
    fuel_remaining_litres: float | None = None
    cumulative_distance_km: float = 0.0
    cumulative_fuel_used_litres: float = 0.0
    cumulative_refuelled_litres: float = 0.0

    depot_returns: int = 0
    capacity_returns: int = 0
    fuel_returns: int = 0
    combined_returns: int = 0

    status: TruckStatus = TruckStatus.IDLE
    next_event_time_hours: float = 0.0

    route_nodes: list[int | str] = field(
        default_factory=list
    )

    def __post_init__(self) -> None:
        if self.truck_id < 0:
            raise ValueError(
                "truck_id must be non-negative"
            )

        if self.capacity_tonnes <= 0.0:
            raise ValueError(
                "capacity_tonnes must be positive"
            )

        if self.speed_km_per_hour <= 0.0:
            raise ValueError(
                "speed_km_per_hour must be positive"
            )

        if self.fuel_capacity_litres <= 0.0:
            raise ValueError(
                "fuel_capacity_litres must be positive"
            )

        if self.fuel_efficiency_km_per_litre <= 0.0:
            raise ValueError(
                "fuel_efficiency_km_per_litre must be positive"
            )

        if self.current_load_tonnes < 0.0:
            raise ValueError(
                "current_load_tonnes cannot be negative"
            )

        if self.current_load_tonnes > self.capacity_tonnes:
            raise ValueError(
                "current load cannot exceed truck capacity"
            )

        if self.fuel_remaining_litres is None:
            self.fuel_remaining_litres = (
                self.fuel_capacity_litres
            )

        if not (
            0.0
            <= self.fuel_remaining_litres
            <= self.fuel_capacity_litres
        ):
            raise ValueError(
                "fuel_remaining_litres must be within tank capacity"
            )

        if self.next_event_time_hours < 0.0:
            raise ValueError(
                "next_event_time_hours cannot be negative"
            )

        if not self.route_nodes:
            self.route_nodes.append(self.current_node)

    @property
    def remaining_capacity_tonnes(self) -> float:
        return (
            self.capacity_tonnes
            - self.current_load_tonnes
        )

    def can_load(self, mass_tonnes: float) -> bool:
        if mass_tonnes < 0.0:
            raise ValueError(
                "mass_tonnes cannot be negative"
            )

        return (
            self.current_load_tonnes
            + mass_tonnes
            <= self.capacity_tonnes
        )

    def load_waste(self, mass_tonnes: float) -> None:
        if not self.can_load(mass_tonnes):
            raise ValueError(
                "waste mass exceeds remaining truck capacity"
            )

        self.current_load_tonnes += mass_tonnes

    def unload(self) -> float:
        unloaded = self.current_load_tonnes
        self.current_load_tonnes = 0.0
        return unloaded

    def travel(
        self,
        *,
        destination_node: int | str,
        distance_km: float,
    ) -> float:
        """
        Apply one physical road movement.

        Returns elapsed travel time in hours.
        """

        required = fuel_required_litres(
            distance_km,
            efficiency_km_per_litre=(
                self.fuel_efficiency_km_per_litre
            ),
        )

        if required > self.fuel_remaining_litres:
            raise ValueError(
                "insufficient fuel for requested movement"
            )

        self.fuel_remaining_litres -= required

        if self.fuel_remaining_litres < -1e-12:
            raise RuntimeError(
                "fuel safety invariant violated"
            )

        self.fuel_remaining_litres = max(
            0.0,
            self.fuel_remaining_litres,
        )

        self.cumulative_distance_km += distance_km
        self.cumulative_fuel_used_litres += required
        self.current_node = destination_node
        self.route_nodes.append(destination_node)

        return distance_km / self.speed_km_per_hour

    def refuel_full(self) -> float:
        """Refill the tank and return litres added."""

        added = (
            self.fuel_capacity_litres
            - self.fuel_remaining_litres
        )

        self.fuel_remaining_litres = (
            self.fuel_capacity_litres
        )

        self.cumulative_refuelled_litres += added

        return added
