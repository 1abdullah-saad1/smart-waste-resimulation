from __future__ import annotations

from math import isclose

from smart_waste.models.events import (
    EventType,
    SimulationEvent,
)
from smart_waste.models.truck import TruckStatus
from smart_waste.simulation.engine import (
    SimulationEngine,
)
from smart_waste.simulation.event_queue import (
    EventQueue,
)
from smart_waste.simulation.state import (
    SimulationState,
)


class BinServiceError(RuntimeError):
    """Raised when physical bin service is inconsistent."""


def service_duration_hours(
    service_time_seconds: float,
) -> float:
    if service_time_seconds <= 0.0:
        raise ValueError(
            "service_time_seconds must be positive"
        )

    return service_time_seconds / 3600.0


def _projected_collection_mass_tonnes(
    *,
    current_fill_percent: float,
    fill_rate_percent_per_hour: float,
    elapsed_hours: float,
    full_mass_kg: float,
) -> float:
    """
    Project the mass that will be physically present when
    service completes.

    This prevents a truck from beginning service when the small
    amount of waste accumulated during the service interval would
    make its capacity infeasible.
    """

    projected_fill = min(
        100.0,
        current_fill_percent
        + fill_rate_percent_per_hour
        * elapsed_hours,
    )

    return (
        full_mass_kg
        * projected_fill
        / 100.0
        / 1000.0
    )


def schedule_bin_service(
    *,
    state: SimulationState,
    event_queue: EventQueue,
    truck_id: int,
    bin_id: int,
    service_time_seconds: float,
) -> SimulationEvent:
    """
    Start physical service of one bin.

    Collection is not applied immediately. The truck and bin are
    updated only when BIN_SERVICE_COMPLETE is processed.
    """

    if truck_id not in state.trucks:
        raise BinServiceError(
            f"unknown truck_id: {truck_id}"
        )

    if bin_id not in state.bins:
        raise BinServiceError(
            f"unknown bin_id: {bin_id}"
        )

    truck = state.trucks[truck_id]
    bin_ = state.bins[bin_id]

    if truck.status != TruckStatus.IDLE:
        raise BinServiceError(
            f"truck {truck_id} is not idle"
        )

    if truck.current_node != bin_.road_node:
        raise BinServiceError(
            f"truck {truck_id} is not located at bin {bin_id}"
        )

    duration_hours = service_duration_hours(
        service_time_seconds
    )

    projected_mass_tonnes = (
        _projected_collection_mass_tonnes(
            current_fill_percent=bin_.fill_percent,
            fill_rate_percent_per_hour=(
                bin_.fill_rate_percent_per_hour
            ),
            elapsed_hours=duration_hours,
            full_mass_kg=bin_.full_mass_kg,
        )
    )

    if not truck.can_load(projected_mass_tonnes):
        raise BinServiceError(
            f"truck {truck_id} lacks capacity for bin {bin_id}"
        )

    completion_time = (
        state.current_time_hours
        + duration_hours
    )

    truck.status = TruckStatus.SERVICING_BIN
    truck.next_event_time_hours = completion_time

    return event_queue.schedule(
        time_hours=completion_time,
        event_type=EventType.BIN_SERVICE_COMPLETE,
        entity_id=truck_id,
        payload={
            "bin_id": bin_id,
            "service_time_seconds": float(
                service_time_seconds
            ),
            "projected_mass_tonnes": (
                projected_mass_tonnes
            ),
        },
    )


def handle_bin_service_complete(
    state: SimulationState,
    event: SimulationEvent,
) -> None:
    """
    Complete physical bin collection after service duration.
    """

    if event.event_type != EventType.BIN_SERVICE_COMPLETE:
        raise BinServiceError(
            "bin service handler received wrong event type"
        )

    if not isinstance(event.entity_id, int):
        raise BinServiceError(
            "bin service event requires integer truck_id"
        )

    truck_id = event.entity_id

    if truck_id not in state.trucks:
        raise BinServiceError(
            f"unknown truck_id: {truck_id}"
        )

    bin_id = event.payload.get("bin_id")

    if not isinstance(bin_id, int):
        raise BinServiceError(
            "bin service event requires integer bin_id"
        )

    if bin_id not in state.bins:
        raise BinServiceError(
            f"unknown bin_id: {bin_id}"
        )

    truck = state.trucks[truck_id]
    bin_ = state.bins[bin_id]

    if truck.status != TruckStatus.SERVICING_BIN:
        raise BinServiceError(
            f"truck {truck_id} is not servicing a bin"
        )

    if truck.current_node != bin_.road_node:
        raise BinServiceError(
            "truck is no longer located at serviced bin"
        )

    if truck.next_event_time_hours is None:
        raise BinServiceError(
            "servicing truck has no completion time"
        )

    if not isclose(
        event.time_hours,
        truck.next_event_time_hours,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise BinServiceError(
            "service completion timestamp does not match "
            "truck scheduled event time"
        )

    collected_mass_tonnes = (
        bin_.waste_mass_tonnes
    )

    if not truck.can_load(
        collected_mass_tonnes
    ):
        raise BinServiceError(
            "truck capacity invariant would be violated"
        )

    bin_.collect()
    truck.load_waste(
        collected_mass_tonnes
    )

    truck.next_event_time_hours = None
    truck.status = TruckStatus.IDLE


def register_bin_service_handlers(
    engine: SimulationEngine,
) -> None:
    engine.register_handler(
        EventType.BIN_SERVICE_COMPLETE,
        handle_bin_service_complete,
    )
