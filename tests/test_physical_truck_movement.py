import networkx as nx
import pytest

from smart_waste.models.bin import WasteBin
from smart_waste.models.depot import Depot
from smart_waste.models.events import EventType
from smart_waste.models.truck import (
    Truck,
    TruckStatus,
)
from smart_waste.movement.road_graph import RoadGraph
from smart_waste.simulation.engine import SimulationEngine
from smart_waste.simulation.event_queue import EventQueue
from smart_waste.simulation.movement import (
    MovementError,
    handle_truck_arrival,
    register_movement_handlers,
    schedule_truck_travel,
)
from smart_waste.simulation.state import SimulationState


def make_state() -> SimulationState:
    graph = nx.Graph()

    graph.add_edge(
        "depot",
        "junction",
        length_km=2.0,
    )

    graph.add_edge(
        "junction",
        "bin-node",
        length_km=3.0,
    )

    # Longer alternative path.
    graph.add_edge(
        "depot",
        "bin-node",
        length_km=10.0,
    )

    road = RoadGraph(graph)

    bin_ = WasteBin(
        bin_id=0,
        road_node="bin-node",
        fill_percent=50.0,
        fill_rate_percent_per_hour=2.0,
        full_mass_kg=440.0,
    )

    truck = Truck(
        truck_id=0,
        current_node="depot",
        capacity_tonnes=10.0,
        speed_km_per_hour=30.0,
        fuel_capacity_litres=200.0,
        fuel_efficiency_km_per_litre=2.5,
    )

    depot = Depot(
        road_node="depot",
        unloading_bays=1,
        unload_time_minutes=11.0,
        refuel_rate_litres_per_minute=60.0,
    )

    return SimulationState(
        road_graph=road,
        bins={0: bin_},
        trucks={0: truck},
        depot=depot,
    )


def test_scheduling_travel_does_not_teleport_truck() -> None:
    state = make_state()
    queue = EventQueue()

    event = schedule_truck_travel(
        state=state,
        event_queue=queue,
        truck_id=0,
        destination_node="bin-node",
    )

    truck = state.trucks[0]

    assert truck.current_node == "depot"
    assert truck.destination_node == "bin-node"
    assert truck.status == TruckStatus.TRAVELLING

    assert event.event_type == EventType.TRUCK_ARRIVAL


def test_scheduled_trip_uses_shortest_road_distance() -> None:
    state = make_state()
    queue = EventQueue()

    event = schedule_truck_travel(
        state=state,
        event_queue=queue,
        truck_id=0,
        destination_node="bin-node",
    )

    assert event.payload["distance_km"] == pytest.approx(
        5.0
    )

    assert event.payload["path"] == [
        "depot",
        "junction",
        "bin-node",
    ]


def test_arrival_time_uses_physical_road_speed() -> None:
    state = make_state()
    queue = EventQueue()

    event = schedule_truck_travel(
        state=state,
        event_queue=queue,
        truck_id=0,
        destination_node="bin-node",
    )

    # 5 km / 30 km/h = 1/6 hour = 10 minutes.
    assert event.time_hours == pytest.approx(
        5.0 / 30.0
    )


def test_departure_consumes_correct_fuel() -> None:
    state = make_state()
    queue = EventQueue()

    schedule_truck_travel(
        state=state,
        event_queue=queue,
        truck_id=0,
        destination_node="bin-node",
    )

    truck = state.trucks[0]

    # 5 km / 2.5 km/L = 2 L.
    assert truck.fuel_remaining_litres == pytest.approx(
        198.0
    )

    assert truck.cumulative_fuel_used_litres == pytest.approx(
        2.0
    )


def test_distance_is_recorded_only_at_arrival() -> None:
    state = make_state()
    queue = EventQueue()

    event = schedule_truck_travel(
        state=state,
        event_queue=queue,
        truck_id=0,
        destination_node="bin-node",
    )

    truck = state.trucks[0]

    assert truck.cumulative_distance_km == pytest.approx(
        0.0
    )

    state.advance_to(
        event.time_hours
    )

    handle_truck_arrival(
        state,
        event,
    )

    assert truck.cumulative_distance_km == pytest.approx(
        5.0
    )


def test_engine_moves_truck_only_when_arrival_occurs() -> None:
    state = make_state()
    queue = EventQueue()

    schedule_truck_travel(
        state=state,
        event_queue=queue,
        truck_id=0,
        destination_node="bin-node",
    )

    engine = SimulationEngine(
        state=state,
        event_queue=queue,
    )

    register_movement_handlers(engine)

    assert state.trucks[0].current_node == "depot"

    engine.step()

    assert state.trucks[0].current_node == "bin-node"
    assert state.trucks[0].status == TruckStatus.IDLE


def test_bin_fill_advances_during_truck_travel() -> None:
    state = make_state()
    queue = EventQueue()

    schedule_truck_travel(
        state=state,
        event_queue=queue,
        truck_id=0,
        destination_node="bin-node",
    )

    engine = SimulationEngine(
        state=state,
        event_queue=queue,
    )

    register_movement_handlers(engine)

    engine.step()

    # Travel time = 1/6 hour.
    # Fill gain = 2%/h * 1/6 h = 1/3%.
    assert state.bins[0].fill_percent == pytest.approx(
        50.0 + (2.0 / 6.0)
    )

    assert state.current_time_hours == pytest.approx(
        5.0 / 30.0
    )


def test_truck_cannot_receive_second_trip_while_travelling() -> None:
    state = make_state()
    queue = EventQueue()

    schedule_truck_travel(
        state=state,
        event_queue=queue,
        truck_id=0,
        destination_node="bin-node",
    )

    with pytest.raises(MovementError):
        schedule_truck_travel(
            state=state,
            event_queue=queue,
            truck_id=0,
            destination_node="junction",
        )

    assert len(queue) == 1


def test_insufficient_fuel_does_not_schedule_event() -> None:
    state = make_state()
    queue = EventQueue()

    state.trucks[0].fuel_remaining_litres = 1.0

    with pytest.raises(ValueError):
        schedule_truck_travel(
            state=state,
            event_queue=queue,
            truck_id=0,
            destination_node="bin-node",
        )

    assert len(queue) == 0

    truck = state.trucks[0]

    assert truck.current_node == "depot"
    assert truck.status == TruckStatus.IDLE
    assert truck.fuel_remaining_litres == pytest.approx(
        1.0
    )


def test_arrival_event_destination_must_match_pending_trip() -> None:
    state = make_state()
    queue = EventQueue()

    event = schedule_truck_travel(
        state=state,
        event_queue=queue,
        truck_id=0,
        destination_node="bin-node",
    )

    event.payload["destination_node"] = "junction"

    state.advance_to(
        event.time_hours
    )

    with pytest.raises(MovementError):
        handle_truck_arrival(
            state,
            event,
        )

    assert state.trucks[0].current_node == "depot"
