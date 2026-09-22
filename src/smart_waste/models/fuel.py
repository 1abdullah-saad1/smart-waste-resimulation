from __future__ import annotations


def fuel_required_litres(
    distance_km: float,
    *,
    efficiency_km_per_litre: float,
) -> float:
    """Fuel required for a road-network distance."""

    if distance_km < 0.0:
        raise ValueError("distance_km cannot be negative")

    if efficiency_km_per_litre <= 0.0:
        raise ValueError(
            "efficiency_km_per_litre must be positive"
        )

    return distance_km / efficiency_km_per_litre


def candidate_fuel_requirement_litres(
    *,
    distance_to_candidate_km: float,
    candidate_to_depot_km: float,
    efficiency_km_per_litre: float,
) -> float:
    """
    Dynamic safety requirement:

    enough fuel to reach the candidate and subsequently
    return from that candidate to the depot.
    """

    total_distance = (
        distance_to_candidate_km
        + candidate_to_depot_km
    )

    return fuel_required_litres(
        total_distance,
        efficiency_km_per_litre=(
            efficiency_km_per_litre
        ),
    )


def is_candidate_fuel_feasible(
    *,
    fuel_remaining_litres: float,
    distance_to_candidate_km: float,
    candidate_to_depot_km: float,
    efficiency_km_per_litre: float,
) -> bool:
    if fuel_remaining_litres < 0.0:
        raise ValueError(
            "fuel_remaining_litres cannot be negative"
        )

    required = candidate_fuel_requirement_litres(
        distance_to_candidate_km=(
            distance_to_candidate_km
        ),
        candidate_to_depot_km=(
            candidate_to_depot_km
        ),
        efficiency_km_per_litre=(
            efficiency_km_per_litre
        ),
    )

    return fuel_remaining_litres >= required


def refuel_time_minutes(
    *,
    current_litres: float,
    tank_capacity_litres: float,
    rate_litres_per_minute: float,
) -> float:
    if tank_capacity_litres <= 0.0:
        raise ValueError(
            "tank_capacity_litres must be positive"
        )

    if rate_litres_per_minute <= 0.0:
        raise ValueError(
            "rate_litres_per_minute must be positive"
        )

    if not 0.0 <= current_litres <= tank_capacity_litres:
        raise ValueError(
            "current_litres must be within tank capacity"
        )

    missing = tank_capacity_litres - current_litres

    return missing / rate_litres_per_minute
