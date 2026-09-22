from __future__ import annotations

from collections.abc import Mapping

from smart_waste.collection.base import (
    BinPolicyView,
    PolicyView,
    TruckPolicyView,
)
from smart_waste.simulation.state import (
    SimulationState,
)


def build_policy_view(
    state: SimulationState,
    *,
    reservations: Mapping[int, int] | None = None,
) -> PolicyView:
    """
    Build a deterministic read-only policy snapshot.

    Reservation information is copied as scalar coordination
    metadata. Policies never receive mutable simulation objects.
    """

    reservation_map = (
        {}
        if reservations is None
        else dict(reservations)
    )

    bins = tuple(
        BinPolicyView(
            bin_id=bin_.bin_id,
            road_node=bin_.road_node,
            fill_percent=float(
                bin_.fill_percent
            ),
            fill_rate_percent_per_hour=float(
                bin_.fill_rate_percent_per_hour
            ),
            waste_mass_tonnes=float(
                bin_.waste_mass_tonnes
            ),
            reserved_by_truck_id=(
                reservation_map.get(
                    bin_.bin_id
                )
            ),
        )
        for bin_ in sorted(
            state.bins.values(),
            key=lambda item: item.bin_id,
        )
    )

    trucks = tuple(
        TruckPolicyView(
            truck_id=truck.truck_id,
            current_node=truck.current_node,
            remaining_capacity_tonnes=float(
                truck.remaining_capacity_tonnes
            ),
            fuel_remaining_litres=float(
                truck.fuel_remaining_litres
            ),
            status=truck.status,
        )
        for truck in sorted(
            state.trucks.values(),
            key=lambda item: item.truck_id,
        )
    )

    return PolicyView(
        current_time_hours=float(
            state.current_time_hours
        ),
        depot_node=state.depot.road_node,
        bins=bins,
        trucks=trucks,
    )
