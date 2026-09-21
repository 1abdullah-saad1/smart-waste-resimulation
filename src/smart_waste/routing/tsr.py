from __future__ import annotations

from math import atan2

import numpy as np

from smart_waste.core.city import City
from smart_waste.core.distance import euclidean_distance
from smart_waste.routing.common import (
    FleetMetrics,
    RouteMetrics,
)


def assign_bins_to_static_sectors(
    city: City,
    num_trucks: int,
) -> list[list[int]]:
    """
    Deterministically divide bins into geographical angular
    sectors around the depot.

    The assignment uses geometry only and never uses fill level.
    """

    if num_trucks <= 0:
        raise ValueError(
            "num_trucks must be greater than zero"
        )

    depot_x, depot_y = city.depot_location

    angular_bins = []

    for bin_ in city.bins:
        angle = atan2(
            bin_.y_km - depot_y,
            bin_.x_km - depot_x,
        )

        angular_bins.append(
            (angle, bin_.bin_id)
        )

    angular_bins.sort(
        key=lambda item: (item[0], item[1])
    )

    assignments = [
        []
        for _ in range(num_trucks)
    ]

    # Split the angularly sorted bins into contiguous sectors.
    chunks = np.array_split(
        np.array(
            [bin_id for _, bin_id in angular_bins],
            dtype=int,
        ),
        num_trucks,
    )

    for truck_id, chunk in enumerate(chunks):
        assignments[truck_id] = [
            int(bin_id)
            for bin_id in chunk
        ]

    return assignments


def nearest_neighbor_order(
    city: City,
    bin_ids: list[int],
) -> list[int]:
    """
    Produce a deterministic nearest-neighbor route starting
    from the depot.

    Tie-breaking uses bin_id to ensure reproducibility.
    """

    remaining = set(bin_ids)

    current_location = city.depot_location
    ordered = []

    while remaining:
        next_bin_id = min(
            remaining,
            key=lambda bin_id: (
                euclidean_distance(
                    current_location,
                    city.bins[bin_id].location,
                ),
                bin_id,
            ),
        )

        ordered.append(next_bin_id)

        current_location = (
            city.bins[next_bin_id].location
        )

        remaining.remove(next_bin_id)

    return ordered


def simulate_tsr(
    city: City,
    *,
    num_trucks: int,
    truck_capacity_tonnes: float,
    fuel_efficiency_km_per_litre: float,
    bin_volume_m3: float,
    waste_density_kg_per_m3: float,
) -> FleetMetrics:
    """
    Simulate Traditional Static Routing (TSR).

    Every bin is visited.

    Routes are fixed using geographical sectors and
    deterministic nearest-neighbor ordering.

    When the remaining truck capacity cannot accommodate
    the next bin, the vehicle returns to the depot,
    unloads, and resumes its fixed route.
    """

    assignments = assign_bins_to_static_sectors(
        city,
        num_trucks,
    )

    truck_results: list[RouteMetrics] = []

    depot = city.depot_location

    for truck_id, assigned_bins in enumerate(assignments):

        ordered_bins = nearest_neighbor_order(
            city,
            assigned_bins,
        )

        metrics = RouteMetrics(
            truck_id=truck_id,
        )

        current_location = depot
        current_load_tonnes = 0.0

        for bin_id in ordered_bins:

            bin_ = city.bins[bin_id]

            demand_tonnes = bin_.waste_mass_tonnes(
                bin_volume_m3=bin_volume_m3,
                waste_density_kg_per_m3=(
                    waste_density_kg_per_m3
                ),
            )

            if demand_tonnes > truck_capacity_tonnes:
                raise ValueError(
                    f"Bin {bin_id} demand exceeds "
                    "truck capacity"
                )

            remaining_capacity = (
                truck_capacity_tonnes
                - current_load_tonnes
            )

            if demand_tonnes > remaining_capacity:

                distance_to_depot = euclidean_distance(
                    current_location,
                    depot,
                )

                metrics.distance_km += (
                    distance_to_depot
                )

                metrics.depot_returns += 1

                current_location = depot
                current_load_tonnes = 0.0

            distance_to_bin = euclidean_distance(
                current_location,
                bin_.location,
            )

            metrics.distance_km += distance_to_bin

            metrics.bins_visited += 1

            metrics.collected_mass_tonnes += (
                demand_tonnes
            )

            metrics.route.append(bin_id)

            current_load_tonnes += demand_tonnes

            current_location = bin_.location

        # Final return to depot after completing route.
        metrics.distance_km += euclidean_distance(
            current_location,
            depot,
        )

        metrics.fuel_litres = (
            metrics.distance_km
            / fuel_efficiency_km_per_litre
        )

        truck_results.append(metrics)

    return FleetMetrics(
        truck_routes=truck_results
    )