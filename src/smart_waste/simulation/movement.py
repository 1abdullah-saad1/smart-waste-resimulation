from __future__ import annotations

from math import isclose

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
from smart_waste.simulation.engine import (
    SimulationEngine,
)
from smart_waste.simulation.event_queue import (
    EventQueue,
)
from smart_waste.simulation.state import (
    SimulationState,
)


class MovementError(RuntimeError):
    """Raised when scheduled physical movement is inconsistent."""


def schedule_truck_travel(
    *,
    state: SimulationState,
    event_queue: EventQueue,
    truck_id: int,
    destination_node: int | str,
) -> SimulationEvent:
    """
    Begin a physical road trip and schedule its arrival.

    The truck remains at its origin node until the
    TRUCK_ARRIVAL event is processed.
    """

    if truck_id not in state.trucks:
        raise MovementError(
            f"unknown truck_id: {truck_id}"
        )

    if destination_node not in state.road_graph.graph:
        raise MovementError(
            f"destination node is not in road graph: "
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
        speed_km_per_hour=(
            truck.speed_km_per_hour
        ),
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

    return event_queue.schedule(
        time_hours=arrival_time_hours,
        event_type=EventType.TRUCK_ARRIVAL,
        entity_id=truck_id,
        payload={
            "origin_node": origin_node,
            "destination_node": destination_node,
            "distance_km": distance_km,
            "path": list(path),
        },
    )


def handle_truck_arrival(
    state: SimulationState,
    event: SimulationEvent,
) -> None:
    """
    Finalize physical movement at the scheduled arrival time.
    """

    if event.event_type != EventType.TRUCK_ARRIVAL:
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

    if (
        expected_destination
        != truck.destination_node
    ):
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
    """Register physical movement event handlers."""

    engine.register_handler(
        EventType.TRUCK_ARRIVAL,
        handle_truck_arrival,
    )
