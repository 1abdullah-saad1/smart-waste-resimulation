from __future__ import annotations

from smart_waste.core.city import City
from smart_waste.core.truck import Truck


def create_fleet(
    city: City,
    *,
    num_trucks: int,
    truck_capacity_tonnes: float,
    fuel_efficiency_km_per_litre: float,
) -> list[Truck]:
    """
    Create a homogeneous fleet located initially at the depot.
    """

    if num_trucks <= 0:
        raise ValueError("num_trucks must be greater than zero")

    return [
        Truck(
            truck_id=truck_id,
            capacity_tonnes=truck_capacity_tonnes,
            fuel_efficiency_km_per_litre=(
                fuel_efficiency_km_per_litre
            ),
            x_km=city.depot_x_km,
            y_km=city.depot_y_km,
        )
        for truck_id in range(num_trucks)
    ]