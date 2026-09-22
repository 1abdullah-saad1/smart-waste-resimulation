from __future__ import annotations

from math import isclose
from typing import Any

from smart_waste.models.events import (
    EventType,
    SimulationEvent,
)
from smart_waste.models.truck import TruckStatus
from smart_waste.movement.routing_distance import (
    road_distance_km,
    shortest_road_path,
)
from smart_waste.movement.travel_time import (
    travel_time_hours,
)
from smart_waste.simulation.engine import SimulationEngine
from smart_waste.simulation.event_queue import EventQueue
from smart_waste.simulation.state import SimulationState


class MovementError(RuntimeError):
    """Raised when scheduled physical movement is inconsistent."""


def schedule_truck_travel(
    *,
    state: SimulationState,
    event_queue: EventQueue,
    truck_id: int,
    destination_node: int | str,
    arrival_event_type: EventType = EventType.TRUCK_ARRIVAL,
    metadata: dict[str, Any] | None = None,
) -> SimulationEvent:
    """
    Begin physical road travel and schedule its arrival.

    The truck remains physically at the origin until the
    corresponding arrival event is processed.
    """

    if arrival_event_type not in {
        EventType.TRUCK_ARRIVAL,
        EventType.DEPOT_ARRIVAL,
    }:
        raise MovementError(
            "arrival_event_type must be TRUCK_ARRIVAL "
            "or DEPOT_ARRIVAL"
        )

    if truck_id not in state.trucks:
        raise MovementError(
            f"unknown truck_id: {truck_id}"
        )

    if destination_node not in state.road_graph.graph:
        raise MovementError(
            "destination node is not in road graph: "
            f"{destination_node!r}"
        )

    truck = state.trucks[truck_id]

    if truck.status != TruckStatus.IDLE:
        raise MovementError(
            f"truck {truck_id} is not idle"
        )

    origin_node = truck.current_node

    distance_km = road_distance_km(
        state.road_graph,
        origin_node,
        destination_node,
    )

    path = shortest_road_path(
        state.road_graph,
        origin_node,
        destination_node,
    )

    elapsed_hours = travel_time_hours(
        distance_km=distance_km,
        speed_km_per_hour=truck.speed_km_per_hour,
    )

    arrival_time_hours = (
        state.current_time_hours
        + elapsed_hours
    )

    truck.begin_travel(
        destination_node=destination_node,
        distance_km=distance_km,
    )

    truck.next_event_time_hours = (
        arrival_time_hours
    )

    payload: dict[str, Any] = {
        "origin_node": origin_node,
        "destination_node": destination_node,
        "distance_km": distance_km,
        "path": list(path),
    }

    if metadata is not None:
        reserved_keys = (
            set(payload)
            & set(metadata)
        )

        if reserved_keys:
            raise MovementError(
                "movement metadata cannot override reserved keys: "
                f"{sorted(reserved_keys)}"
            )

        payload.update(metadata)

    return event_queue.schedule(
        time_hours=arrival_time_hours,
        event_type=arrival_event_type,
        entity_id=truck_id,
        payload=payload,
    )


def handle_truck_arrival(
    state: SimulationState,
    event: SimulationEvent,
) -> None:
    """Finalize physical truck movement."""

    if event.event_type not in {
        EventType.TRUCK_ARRIVAL,
        EventType.DEPOT_ARRIVAL,
    }:
        raise MovementError(
            "arrival handler received wrong event type"
        )

    if not isinstance(event.entity_id, int):
        raise MovementError(
            "truck arrival event requires integer truck_id"
        )

    truck_id = event.entity_id

    if truck_id not in state.trucks:
        raise MovementError(
            f"unknown truck_id: {truck_id}"
        )

    truck = state.trucks[truck_id]

    if truck.status != TruckStatus.TRAVELLING:
        raise MovementError(
            f"truck {truck_id} is not travelling"
        )

    expected_destination = event.payload.get(
        "destination_node"
    )

    if expected_destination != truck.destination_node:
        raise MovementError(
            "arrival destination does not match "
            "truck pending destination"
        )

    expected_distance = event.payload.get(
        "distance_km"
    )

    if expected_distance is None:
        raise MovementError(
            "arrival event is missing distance_km"
        )

    if not isclose(
        float(expected_distance),
        truck.pending_distance_km,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise MovementError(
            "arrival distance does not match "
            "truck pending distance"
        )

    if truck.next_event_time_hours is None:
        raise MovementError(
            "travelling truck has no scheduled event time"
        )

    if not isclose(
        event.time_hours,
        truck.next_event_time_hours,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise MovementError(
            "arrival timestamp does not match "
            "truck scheduled event time"
        )

    truck.complete_travel()


def register_movement_handlers(
    engine: SimulationEngine,
) -> None:
    """
    Register physical arrival handling.

    DEPOT_ARRIVAL may subsequently have an additional depot
    handler. Registration order is therefore significant.
    """

    engine.register_handler(
        EventType.TRUCK_ARRIVAL,
        handle_truck_arrival,
    )

    engine.register_handler(
        EventType.DEPOT_ARRIVAL,
        handle_truck_arrival,
    )
