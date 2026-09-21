from __future__ import annotations

from collections.abc import Mapping

from smart_waste.core.city import City
from smart_waste.core.distance import euclidean_distance
from smart_waste.routing.common import (
    FleetMetrics,
    RouteMetrics,
)
from smart_waste.routing.tsr import (
    assign_bins_to_static_sectors,
    nearest_neighbor_order,
)


def select_hdr_candidates(
    city: City,
    *,
    threshold_percent: float = 80.0,
    reported_fill_percent: Mapping[int, float] | None = None,
) -> tuple[int, ...]:
    """
    Select bins eligible for Heuristic Dynamic Routing.

    Selection is based on REPORTED fill level.

    If no reported-fill override is supplied, the true simulated
    fill level is used.

    This distinction is intentional because later FDI experiments
    will modify reported telemetry without modifying the physical
    waste mass in the bin.
    """

    if not 0.0 <= threshold_percent <= 100.0:
        raise ValueError(
            "threshold_percent must be between 0 and 100"
        )

    selected: list[int] = []

    for bin_ in city.bins:

        if reported_fill_percent is None:
            reported_fill = bin_.fill_percent
        else:
            reported_fill = reported_fill_percent.get(
                bin_.bin_id,
                bin_.fill_percent,
            )

        if not 0.0 <= reported_fill <= 100.0:
            raise ValueError(
                f"Invalid reported fill for bin {bin_.bin_id}: "
                f"{reported_fill}"
            )

        if reported_fill >= threshold_percent:
            selected.append(bin_.bin_id)

    return tuple(selected)


def simulate_hdr(
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
    Simulate the Heuristic Dynamic Routing baseline.

    1. Preserve the same geographical truck sectors used by TSR.
    2. Select only bins whose reported fill >= threshold.
    3. Visit eligible bins using deterministic nearest-neighbour.
    4. Respect physical truck capacity.
    5. Physical collected mass is always derived from TRUE fill,
       not reported telemetry.
    """

    candidate_ids = set(
        select_hdr_candidates(
            city,
            threshold_percent=threshold_percent,
            reported_fill_percent=reported_fill_percent,
        )
    )

    sector_assignments = assign_bins_to_static_sectors(
        city,
        num_trucks,
    )

    depot = city.depot_location

    truck_results: list[RouteMetrics] = []

    for truck_id, sector_bins in enumerate(
        sector_assignments
    ):
        eligible_bins = [
            bin_id
            for bin_id in sector_bins
            if bin_id in candidate_ids
        ]

        ordered_bins = nearest_neighbor_order(
            city,
            eligible_bins,
        )

        metrics = RouteMetrics(
            truck_id=truck_id,
        )

        current_location = depot
        current_load_tonnes = 0.0

        for bin_id in ordered_bins:
            bin_ = city.bins[bin_id]

            # Important:
            # collection mass comes from physical TRUE fill.
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

                metrics.distance_km += (
                    euclidean_distance(
                        current_location,
                        depot,
                    )
                )

                metrics.depot_returns += 1

                current_location = depot
                current_load_tonnes = 0.0

            metrics.distance_km += (
                euclidean_distance(
                    current_location,
                    bin_.location,
                )
            )

            metrics.bins_visited += 1
            metrics.collected_mass_tonnes += (
                demand_tonnes
            )
            metrics.route.append(bin_id)

            current_load_tonnes += demand_tonnes
            current_location = bin_.location

        # A truck with no eligible bins does not leave the depot.
        if ordered_bins:
            metrics.distance_km += (
                euclidean_distance(
                    current_location,
                    depot,
                )
            )

        metrics.fuel_litres = (
            metrics.distance_km
            / fuel_efficiency_km_per_litre
        )

        truck_results.append(metrics)

    return FleetMetrics(
        truck_routes=truck_results
    )