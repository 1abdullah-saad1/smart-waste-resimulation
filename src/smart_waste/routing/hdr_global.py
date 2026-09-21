from __future__ import annotations

from collections.abc import Mapping

from smart_waste.core.city import City
from smart_waste.core.distance import euclidean_distance
from smart_waste.routing.common import FleetMetrics, RouteMetrics
from smart_waste.routing.hdr import select_hdr_candidates


def simulate_global_greedy_hdr(
    city: City,
    *,
    num_trucks: int,
    truck_capacity_tonnes: float,
    fuel_efficiency_km_per_litre: float,
    bin_volume_m3: float,
    waste_density_kg_per_m3: float,
    threshold_percent: float = 80.0,
    reported_fill_percent: Mapping[int, float] | None = None,
) -> FleetMetrics:
    """
    Global fleet-level greedy nearest-neighbour HDR.

    All trucks start at the depot.
    At each step, choose the feasible (truck, bin) pair with the
    smallest travel distance.

    If no remaining candidate fits a truck's remaining capacity,
    that truck returns to the depot, unloads, and becomes available
    again.

    Physical collected mass always uses true fill.
    """

    candidate_ids = set(
        select_hdr_candidates(
            city,
            threshold_percent=threshold_percent,
            reported_fill_percent=reported_fill_percent,
        )
    )

    depot = city.depot_location

    metrics = [
        RouteMetrics(truck_id=i)
        for i in range(num_trucks)
    ]

    locations = [
        depot
        for _ in range(num_trucks)
    ]

    loads = [
        0.0
        for _ in range(num_trucks)
    ]

    while candidate_ids:

        best_choice = None

        for truck_id in range(num_trucks):

            remaining_capacity = (
                truck_capacity_tonnes
                - loads[truck_id]
            )

            for bin_id in candidate_ids:
                bin_ = city.bins[bin_id]

                demand = bin_.waste_mass_tonnes(
                    bin_volume_m3=bin_volume_m3,
                    waste_density_kg_per_m3=(
                        waste_density_kg_per_m3
                    ),
                )

                if demand > remaining_capacity:
                    continue

                distance = euclidean_distance(
                    locations[truck_id],
                    bin_.location,
                )

                choice = (
                    distance,
                    truck_id,
                    bin_id,
                    demand,
                )

                if (
                    best_choice is None
                    or choice < best_choice
                ):
                    best_choice = choice

        if best_choice is None:
            # No truck can accept another candidate.
            # Return all non-empty trucks to depot and unload.
            returned_any = False

            for truck_id in range(num_trucks):
                if loads[truck_id] > 0.0:
                    distance = euclidean_distance(
                        locations[truck_id],
                        depot,
                    )

                    metrics[truck_id].distance_km += distance
                    metrics[truck_id].depot_returns += 1

                    locations[truck_id] = depot
                    loads[truck_id] = 0.0
                    returned_any = True

            if not returned_any:
                raise RuntimeError(
                    "No candidate can be served even by an empty truck"
                )

            continue

        distance, truck_id, bin_id, demand = best_choice

        bin_ = city.bins[bin_id]

        metrics[truck_id].distance_km += distance
        metrics[truck_id].bins_visited += 1
        metrics[truck_id].collected_mass_tonnes += demand
        metrics[truck_id].route.append(bin_id)

        loads[truck_id] += demand
        locations[truck_id] = bin_.location

        candidate_ids.remove(bin_id)

    # Final depot return.
    for truck_id in range(num_trucks):
        if metrics[truck_id].bins_visited > 0:
            metrics[truck_id].distance_km += euclidean_distance(
                locations[truck_id],
                depot,
            )

        metrics[truck_id].fuel_litres = (
            metrics[truck_id].distance_km
            / fuel_efficiency_km_per_litre
        )

    return FleetMetrics(
        truck_routes=metrics
    )