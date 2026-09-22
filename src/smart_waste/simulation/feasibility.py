from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from smart_waste.models.fuel import (
    candidate_fuel_requirement_litres,
    fuel_required_litres,
)
from smart_waste.models.truck import TruckStatus
from smart_waste.movement.routing_distance import (
    road_distance_km,
)
from smart_waste.simulation.bin_service import (
    service_duration_hours,
)
from smart_waste.simulation.state import (
    SimulationState,
)


class FeasibilityError(RuntimeError):
    """Raised when feasibility cannot be evaluated safely."""


class FeasibilityFailure(str, Enum):
    NONE = "none"
    CAPACITY = "capacity"
    FUEL = "fuel"
    CAPACITY_AND_FUEL = "capacity_and_fuel"


@dataclass(frozen=True)
class CandidateFeasibility:
    truck_id: int
    bin_id: int

    capacity_feasible: bool
    fuel_feasible: bool

    projected_collection_mass_tonnes: float
    remaining_capacity_tonnes: float

    distance_to_candidate_km: float
    candidate_to_depot_km: float

    required_fuel_litres: float
    fuel_remaining_litres: float

    @property
    def feasible(self) -> bool:
        return (
            self.capacity_feasible
            and self.fuel_feasible
        )

    @property
    def failure(self) -> FeasibilityFailure:
        if self.feasible:
            return FeasibilityFailure.NONE

        if (
            not self.capacity_feasible
            and not self.fuel_feasible
        ):
            return (
                FeasibilityFailure.CAPACITY_AND_FUEL
            )

        if not self.capacity_feasible:
            return FeasibilityFailure.CAPACITY

        return FeasibilityFailure.FUEL


def evaluate_candidate_feasibility(
    *,
    state: SimulationState,
    truck_id: int,
    bin_id: int,
    service_time_seconds: float,
) -> CandidateFeasibility:
    """
    Evaluate physical feasibility before dispatching a truck.

    A candidate is feasible only if:

    1. the truck can carry the projected bin mass at service
       completion, and
    2. it has enough fuel to travel from its current position to
       the candidate and then from the candidate back to depot.

    Physical distances are shortest-path road distances.
    """

    if truck_id not in state.trucks:
        raise FeasibilityError(
            f"unknown truck_id: {truck_id}"
        )

    if bin_id not in state.bins:
        raise FeasibilityError(
            f"unknown bin_id: {bin_id}"
        )

    truck = state.trucks[truck_id]
    bin_ = state.bins[bin_id]

    if truck.status != TruckStatus.IDLE:
        raise FeasibilityError(
            f"truck {truck_id} must be idle "
            "before candidate evaluation"
        )

    service_hours = service_duration_hours(
        service_time_seconds
    )

    projected_mass = (
        bin_.projected_waste_mass_tonnes(
            service_hours
        )
    )

    capacity_feasible = truck.can_load(
        projected_mass
    )

    distance_to_candidate = road_distance_km(
        state.road_graph,
        truck.current_node,
        bin_.road_node,
    )

    candidate_to_depot = road_distance_km(
        state.road_graph,
        bin_.road_node,
        state.depot.road_node,
    )

    required_fuel = (
        candidate_fuel_requirement_litres(
            distance_to_candidate_km=(
                distance_to_candidate
            ),
            candidate_to_depot_km=(
                candidate_to_depot
            ),
            efficiency_km_per_litre=(
                truck.fuel_efficiency_km_per_litre
            ),
        )
    )

    fuel_feasible = (
        truck.fuel_remaining_litres
        >= required_fuel
    )

    return CandidateFeasibility(
        truck_id=truck_id,
        bin_id=bin_id,
        capacity_feasible=capacity_feasible,
        fuel_feasible=fuel_feasible,
        projected_collection_mass_tonnes=(
            projected_mass
        ),
        remaining_capacity_tonnes=(
            truck.remaining_capacity_tonnes
        ),
        distance_to_candidate_km=(
            distance_to_candidate
        ),
        candidate_to_depot_km=(
            candidate_to_depot
        ),
        required_fuel_litres=required_fuel,
        fuel_remaining_litres=(
            truck.fuel_remaining_litres
        ),
    )


def fuel_required_to_depot_litres(
    *,
    state: SimulationState,
    truck_id: int,
) -> float:
    """
    Fuel required for an immediate safe return to depot.
    """

    if truck_id not in state.trucks:
        raise FeasibilityError(
            f"unknown truck_id: {truck_id}"
        )

    truck = state.trucks[truck_id]

    distance_to_depot = road_distance_km(
        state.road_graph,
        truck.current_node,
        state.depot.road_node,
    )

    return fuel_required_litres(
        distance_to_depot,
        efficiency_km_per_litre=(
            truck.fuel_efficiency_km_per_litre
        ),
    )


def can_return_to_depot(
    *,
    state: SimulationState,
    truck_id: int,
) -> bool:
    truck = state.trucks[truck_id]

    return (
        truck.fuel_remaining_litres
        >= fuel_required_to_depot_litres(
            state=state,
            truck_id=truck_id,
        )
    )
