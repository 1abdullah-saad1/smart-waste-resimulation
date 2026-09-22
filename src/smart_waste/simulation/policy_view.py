from __future__ import annotations

from collections.abc import Mapping
from math import isfinite

from smart_waste.collection.base import (
    BinPolicyView,
    PolicyView,
    TruckPolicyView,
)
from smart_waste.simulation.state import (
    SimulationState,
)


class PolicyViewError(RuntimeError):
    """Raised when a policy snapshot cannot be built safely."""


def _validate_reported_fill(
    *,
    bin_id: int,
    value: float,
) -> float:
    try:
        reported = float(
            value
        )
    except (TypeError, ValueError) as exc:
        raise PolicyViewError(
            f"reported fill for bin {bin_id} must be numeric"
        ) from exc

    if not isfinite(
        reported
    ):
        raise PolicyViewError(
            f"reported fill for bin {bin_id} must be finite"
        )

    if not (
        0.0
        <= reported
        <= 100.0
    ):
        raise PolicyViewError(
            f"reported fill for bin {bin_id} must be "
            "between 0 and 100"
        )

    return reported


def build_policy_view(
    state: SimulationState,
    *,
    reservations: Mapping[int, int] | None = None,
    reported_fill_percent: Mapping[int, float] | None = None,
) -> PolicyView:
    """
    Build a deterministic read-only policy snapshot.

    Physical truth and routing telemetry are deliberately separate:

        BinPolicyView.fill_percent
            true physical simulated fill

        BinPolicyView.reported_fill_percent
            telemetry presented to routing logic

    If no telemetry override exists for a bin, its reported value
    equals its current physical fill.

    This enables future FDI experiments to alter routing input
    without mutating physical waste state or collection mass.

    Reservation information is copied as scalar coordination
    metadata. Policies never receive mutable simulation objects.
    """

    reservation_map = (
        {}
        if reservations is None
        else dict(
            reservations
        )
    )

    telemetry_map = (
        {}
        if reported_fill_percent is None
        else dict(
            reported_fill_percent
        )
    )

    known_bin_ids = set(
        state.bins
    )

    unknown_telemetry_ids = (
        set(telemetry_map)
        - known_bin_ids
    )

    if unknown_telemetry_ids:
        raise PolicyViewError(
            "reported telemetry references unknown bin IDs: "
            f"{sorted(unknown_telemetry_ids)}"
        )

    unknown_reservation_ids = (
        set(reservation_map)
        - known_bin_ids
    )

    if unknown_reservation_ids:
        raise PolicyViewError(
            "reservations reference unknown bin IDs: "
            f"{sorted(unknown_reservation_ids)}"
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
            reported_fill_percent=(
                _validate_reported_fill(
                    bin_id=bin_.bin_id,
                    value=telemetry_map.get(
                        bin_.bin_id,
                        bin_.fill_percent,
                    ),
                )
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
