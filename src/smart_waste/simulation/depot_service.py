from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from math import isclose

from smart_waste.models.events import (
    EventType,
    SimulationEvent,
)
from smart_waste.models.truck import TruckStatus
from smart_waste.simulation.engine import SimulationEngine
from smart_waste.simulation.event_queue import EventQueue
from smart_waste.simulation.movement import schedule_truck_travel
from smart_waste.simulation.state import SimulationState


class DepotServiceError(RuntimeError):
    """Raised when depot-service state becomes inconsistent."""


class DepotReturnReason(str, Enum):
    ROUTINE = "routine"
    CAPACITY = "capacity"
    FUEL = "fuel"
    COMBINED = "combined"


@dataclass
class DepotServiceState:
    """
    Runtime state of the deterministic FCFS depot service channel.

    Primary benchmark:
    one bay, FCFS queue, sequential unload then refuel.
    """

    waiting_truck_ids: deque[int] = field(
        default_factory=deque
    )

    starting_truck_ids: set[int] = field(
        default_factory=set
    )

    active_truck_ids: set[int] = field(
        default_factory=set
    )

    @property
    def queue_length(self) -> int:
        return len(self.waiting_truck_ids)


def schedule_depot_return(
    *,
    state: SimulationState,
    event_queue: EventQueue,
    truck_id: int,
    reason: DepotReturnReason,
) -> SimulationEvent:
    """Schedule physical road travel back to the central depot."""

    if truck_id not in state.trucks:
        raise DepotServiceError(
            f"unknown truck_id: {truck_id}"
        )

    return schedule_truck_travel(
        state=state,
        event_queue=event_queue,
        truck_id=truck_id,
        destination_node=state.depot.road_node,
        arrival_event_type=EventType.DEPOT_ARRIVAL,
        metadata={
            "depot_return_reason": reason.value,
        },
    )


def _record_completed_return(
    state: SimulationState,
    *,
    truck_id: int,
    reason: DepotReturnReason,
) -> None:
    truck = state.trucks[truck_id]

    truck.depot_returns += 1

    if reason == DepotReturnReason.CAPACITY:
        truck.capacity_returns += 1
    elif reason == DepotReturnReason.FUEL:
        truck.fuel_returns += 1
    elif reason == DepotReturnReason.COMBINED:
        truck.combined_returns += 1


def _dispatch_waiting_trucks(
    *,
    state: SimulationState,
    event_queue: EventQueue,
    depot_state: DepotServiceState,
) -> None:
    """
    Reserve free depot-service channels in deterministic FCFS order.
    """

    occupied = (
        len(depot_state.active_truck_ids)
        + len(depot_state.starting_truck_ids)
    )

    free_bays = (
        state.depot.unloading_bays
        - occupied
    )

    if free_bays <= 0:
        return

    for truck_id in depot_state.waiting_truck_ids:
        if free_bays <= 0:
            break

        if (
            truck_id in depot_state.starting_truck_ids
            or truck_id in depot_state.active_truck_ids
        ):
            continue

        depot_state.starting_truck_ids.add(
            truck_id
        )

        event_queue.schedule(
            time_hours=state.current_time_hours,
            event_type=EventType.DEPOT_SERVICE_START,
            entity_id=truck_id,
        )

        free_bays -= 1


def handle_depot_arrival(
    state: SimulationState,
    event: SimulationEvent,
    *,
    event_queue: EventQueue,
    depot_state: DepotServiceState,
) -> None:
    """
    Enter the FCFS depot queue after physical movement completes.

    The movement handler must run before this handler.
    """

    if event.event_type != EventType.DEPOT_ARRIVAL:
        raise DepotServiceError(
            "depot arrival handler received wrong event type"
        )

    if not isinstance(event.entity_id, int):
        raise DepotServiceError(
            "DEPOT_ARRIVAL requires integer truck_id"
        )

    truck_id = event.entity_id

    if truck_id not in state.trucks:
        raise DepotServiceError(
            f"unknown truck_id: {truck_id}"
        )

    truck = state.trucks[truck_id]

    if truck.current_node != state.depot.road_node:
        raise DepotServiceError(
            "truck has not physically arrived at depot"
        )

    if truck.status != TruckStatus.IDLE:
        raise DepotServiceError(
            "movement arrival must complete before depot handling"
        )

    if (
        truck_id in depot_state.waiting_truck_ids
        or truck_id in depot_state.starting_truck_ids
        or truck_id in depot_state.active_truck_ids
    ):
        raise DepotServiceError(
            f"truck {truck_id} is already in depot service"
        )

    raw_reason = event.payload.get(
        "depot_return_reason",
        DepotReturnReason.ROUTINE.value,
    )

    try:
        reason = DepotReturnReason(
            str(raw_reason)
        )
    except ValueError as exc:
        raise DepotServiceError(
            f"invalid depot return reason: {raw_reason!r}"
        ) from exc

    _record_completed_return(
        state,
        truck_id=truck_id,
        reason=reason,
    )

    truck.status = TruckStatus.WAITING_AT_DEPOT

    depot_state.waiting_truck_ids.append(
        truck_id
    )

    _dispatch_waiting_trucks(
        state=state,
        event_queue=event_queue,
        depot_state=depot_state,
    )


def handle_depot_service_start(
    state: SimulationState,
    event: SimulationEvent,
    *,
    event_queue: EventQueue,
    depot_state: DepotServiceState,
) -> None:
    if event.event_type != EventType.DEPOT_SERVICE_START:
        raise DepotServiceError(
            "depot start handler received wrong event type"
        )

    if not isinstance(event.entity_id, int):
        raise DepotServiceError(
            "DEPOT_SERVICE_START requires integer truck_id"
        )

    truck_id = event.entity_id

    if truck_id not in depot_state.starting_truck_ids:
        raise DepotServiceError(
            "truck does not own a reserved depot bay"
        )

    if not depot_state.waiting_truck_ids:
        raise DepotServiceError(
            "depot waiting queue is unexpectedly empty"
        )

    if depot_state.waiting_truck_ids[0] != truck_id:
        raise DepotServiceError(
            "FCFS queue ordering violated"
        )

    truck = state.trucks[truck_id]

    if truck.status != TruckStatus.WAITING_AT_DEPOT:
        raise DepotServiceError(
            "truck is not waiting for depot service"
        )

    depot_state.waiting_truck_ids.popleft()
    depot_state.starting_truck_ids.remove(
        truck_id
    )
    depot_state.active_truck_ids.add(
        truck_id
    )

    unload_minutes = (
        state.depot.unload_duration_minutes(
            truck
        )
    )

    if unload_minutes > 0.0:
        truck.status = TruckStatus.UNLOADING

        completion_time = (
            state.current_time_hours
            + unload_minutes / 60.0
        )

        truck.next_event_time_hours = (
            completion_time
        )

        event_queue.schedule(
            time_hours=completion_time,
            event_type=EventType.DEPOT_UNLOAD_COMPLETE,
            entity_id=truck_id,
        )

        return

    truck.status = TruckStatus.REFUELLING

    refuel_minutes = (
        state.depot.refuel_duration_minutes(
            truck
        )
    )

    completion_time = (
        state.current_time_hours
        + refuel_minutes / 60.0
    )

    truck.next_event_time_hours = completion_time

    event_queue.schedule(
        time_hours=completion_time,
        event_type=EventType.DEPOT_SERVICE_COMPLETE,
        entity_id=truck_id,
    )


def handle_depot_unload_complete(
    state: SimulationState,
    event: SimulationEvent,
    *,
    event_queue: EventQueue,
    depot_state: DepotServiceState,
) -> None:
    if event.event_type != EventType.DEPOT_UNLOAD_COMPLETE:
        raise DepotServiceError(
            "unload handler received wrong event type"
        )

    if not isinstance(event.entity_id, int):
        raise DepotServiceError(
            "DEPOT_UNLOAD_COMPLETE requires integer truck_id"
        )

    truck_id = event.entity_id

    if truck_id not in depot_state.active_truck_ids:
        raise DepotServiceError(
            "truck does not occupy a depot service channel"
        )

    truck = state.trucks[truck_id]

    if truck.status != TruckStatus.UNLOADING:
        raise DepotServiceError(
            "truck is not unloading"
        )

    if truck.next_event_time_hours is None or not isclose(
        event.time_hours,
        truck.next_event_time_hours,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise DepotServiceError(
            "unload timestamp does not match truck state"
        )

    truck.unload()
    truck.status = TruckStatus.REFUELLING

    refuel_minutes = (
        state.depot.refuel_duration_minutes(
            truck
        )
    )

    completion_time = (
        state.current_time_hours
        + refuel_minutes / 60.0
    )

    truck.next_event_time_hours = completion_time

    event_queue.schedule(
        time_hours=completion_time,
        event_type=EventType.DEPOT_SERVICE_COMPLETE,
        entity_id=truck_id,
    )


def handle_depot_service_complete(
    state: SimulationState,
    event: SimulationEvent,
    *,
    event_queue: EventQueue,
    depot_state: DepotServiceState,
) -> None:
    if event.event_type != EventType.DEPOT_SERVICE_COMPLETE:
        raise DepotServiceError(
            "depot completion handler received wrong event type"
        )

    if not isinstance(event.entity_id, int):
        raise DepotServiceError(
            "DEPOT_SERVICE_COMPLETE requires integer truck_id"
        )

    truck_id = event.entity_id

    if truck_id not in depot_state.active_truck_ids:
        raise DepotServiceError(
            "truck does not occupy a depot service channel"
        )

    truck = state.trucks[truck_id]

    if truck.status != TruckStatus.REFUELLING:
        raise DepotServiceError(
            "truck is not refuelling"
        )

    if truck.next_event_time_hours is None or not isclose(
        event.time_hours,
        truck.next_event_time_hours,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise DepotServiceError(
            "refuel timestamp does not match truck state"
        )

    truck.refuel_full()

    truck.next_event_time_hours = None
    truck.status = TruckStatus.IDLE

    depot_state.active_truck_ids.remove(
        truck_id
    )

    _dispatch_waiting_trucks(
        state=state,
        event_queue=event_queue,
        depot_state=depot_state,
    )


def register_depot_handlers(
    engine: SimulationEngine,
    *,
    depot_state: DepotServiceState,
) -> None:
    """
    Register depot handlers.

    Movement handlers must be registered first so DEPOT_ARRIVAL
    first completes road movement and only then enters the queue.
    """

    def arrival_handler(
        state: SimulationState,
        event: SimulationEvent,
    ) -> None:
        handle_depot_arrival(
            state,
            event,
            event_queue=engine.event_queue,
            depot_state=depot_state,
        )

    def start_handler(
        state: SimulationState,
        event: SimulationEvent,
    ) -> None:
        handle_depot_service_start(
            state,
            event,
            event_queue=engine.event_queue,
            depot_state=depot_state,
        )

    def unload_handler(
        state: SimulationState,
        event: SimulationEvent,
    ) -> None:
        handle_depot_unload_complete(
            state,
            event,
            event_queue=engine.event_queue,
            depot_state=depot_state,
        )

    def completion_handler(
        state: SimulationState,
        event: SimulationEvent,
    ) -> None:
        handle_depot_service_complete(
            state,
            event,
            event_queue=engine.event_queue,
            depot_state=depot_state,
        )

    engine.register_handler(
        EventType.DEPOT_ARRIVAL,
        arrival_handler,
    )

    engine.register_handler(
        EventType.DEPOT_SERVICE_START,
        start_handler,
    )

    engine.register_handler(
        EventType.DEPOT_UNLOAD_COMPLETE,
        unload_handler,
    )

    engine.register_handler(
        EventType.DEPOT_SERVICE_COMPLETE,
        completion_handler,
    )
