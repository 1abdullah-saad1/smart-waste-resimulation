import networkx as nx
import pytest

from smart_waste.models.bin import WasteBin
from smart_waste.models.depot import Depot
from smart_waste.models.events import EventType
from smart_waste.models.truck import Truck
from smart_waste.movement.road_graph import RoadGraph
from smart_waste.simulation.engine import SimulationEngine
from smart_waste.simulation.event_queue import EventQueue
from smart_waste.simulation.state import (
    SimulationState,
    SimulationStateError,
)


def make_state() -> SimulationState:
    graph = nx.Graph()

    graph.add_edge(
        "depot",
        "bin-node",
        length_km=3.0,
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


def test_event_queue_orders_by_time() -> None:
    queue = EventQueue()

    queue.schedule(
        time_hours=2.0,
        event_type=EventType.TRUCK_ARRIVAL,
    )

    queue.schedule(
        time_hours=1.0,
        event_type=EventType.TRUCK_ARRIVAL,
    )

    assert queue.pop().time_hours == 1.0
    assert queue.pop().time_hours == 2.0


def test_equal_time_events_preserve_schedule_order() -> None:
    queue = EventQueue()

    first = queue.schedule(
        time_hours=1.0,
        event_type=EventType.TRUCK_ARRIVAL,
        entity_id=10,
    )

    second = queue.schedule(
        time_hours=1.0,
        event_type=EventType.TRUCK_ARRIVAL,
        entity_id=20,
    )

    assert first.sequence < second.sequence
    assert queue.pop().entity_id == 10
    assert queue.pop().entity_id == 20


def test_peek_does_not_remove_event() -> None:
    queue = EventQueue()

    queue.schedule(
        time_hours=1.0,
        event_type=EventType.TRUCK_ARRIVAL,
    )

    queue.peek()

    assert len(queue) == 1


def test_global_clock_advances_bin_physics() -> None:
    state = make_state()

    elapsed = state.advance_to(3.0)

    assert elapsed == pytest.approx(3.0)
    assert state.current_time_hours == pytest.approx(3.0)

    assert state.bins[0].fill_percent == pytest.approx(
        56.0
    )


def test_global_clock_cannot_move_backwards() -> None:
    state = make_state()

    state.advance_to(2.0)

    with pytest.raises(SimulationStateError):
        state.advance_to(1.0)


def test_zero_time_advance_changes_nothing() -> None:
    state = make_state()

    original_fill = state.bins[0].fill_percent

    elapsed = state.advance_to(0.0)

    assert elapsed == 0.0
    assert state.bins[0].fill_percent == original_fill


def test_state_rejects_missing_bin_road_node() -> None:
    state = make_state()

    state.bins[0].road_node = "missing"

    with pytest.raises(SimulationStateError):
        SimulationState(
            road_graph=state.road_graph,
            bins=state.bins,
            trucks=state.trucks,
            depot=state.depot,
        )


def test_state_rejects_mismatched_bin_key() -> None:
    state = make_state()

    with pytest.raises(SimulationStateError):
        SimulationState(
            road_graph=state.road_graph,
            bins={999: state.bins[0]},
            trucks=state.trucks,
            depot=state.depot,
        )


def test_engine_processes_earliest_event() -> None:
    state = make_state()
    queue = EventQueue()

    queue.schedule(
        time_hours=2.0,
        event_type=EventType.TRUCK_ARRIVAL,
    )

    queue.schedule(
        time_hours=1.0,
        event_type=EventType.TRUCK_ARRIVAL,
    )

    engine = SimulationEngine(
        state=state,
        event_queue=queue,
    )

    event = engine.step()

    assert event.time_hours == pytest.approx(1.0)
    assert state.current_time_hours == pytest.approx(1.0)
    assert engine.processed_events == 1


def test_engine_handler_runs_after_clock_advance() -> None:
    state = make_state()
    queue = EventQueue()

    queue.schedule(
        time_hours=2.0,
        event_type=EventType.TRUCK_ARRIVAL,
    )

    observed_times: list[float] = []

    def handler(
        current_state: SimulationState,
        _event,
    ) -> None:
        observed_times.append(
            current_state.current_time_hours
        )

    engine = SimulationEngine(
        state=state,
        event_queue=queue,
    )

    engine.register_handler(
        EventType.TRUCK_ARRIVAL,
        handler,
    )

    engine.step()

    assert observed_times == [2.0]


def test_engine_run_until_time() -> None:
    state = make_state()
    queue = EventQueue()

    for time_hours in (1.0, 2.0, 3.0):
        queue.schedule(
            time_hours=time_hours,
            event_type=EventType.TRUCK_ARRIVAL,
        )

    engine = SimulationEngine(
        state=state,
        event_queue=queue,
    )

    processed = engine.run(
        until_hours=2.0
    )

    assert processed == 2
    assert state.current_time_hours == pytest.approx(2.0)
    assert len(queue) == 1


def test_engine_max_events() -> None:
    state = make_state()
    queue = EventQueue()

    for time_hours in (1.0, 2.0, 3.0):
        queue.schedule(
            time_hours=time_hours,
            event_type=EventType.TRUCK_ARRIVAL,
        )

    engine = SimulationEngine(
        state=state,
        event_queue=queue,
    )

    processed = engine.run(
        max_events=1
    )

    assert processed == 1
    assert len(queue) == 2


def test_invariant_detects_invalid_truck_load() -> None:
    state = make_state()

    state.trucks[0].current_load_tonnes = 11.0

    with pytest.raises(SimulationStateError):
        state.validate_physical_invariants()


def test_multiple_trucks_share_one_global_clock() -> None:
    state = make_state()

    second = Truck(
        truck_id=1,
        current_node="depot",
        capacity_tonnes=10.0,
        speed_km_per_hour=30.0,
        fuel_capacity_litres=200.0,
        fuel_efficiency_km_per_litre=2.5,
    )

    state.trucks[1] = second

    queue = EventQueue()

    queue.schedule(
        time_hours=0.5,
        event_type=EventType.TRUCK_ARRIVAL,
        entity_id=1,
    )

    queue.schedule(
        time_hours=1.0,
        event_type=EventType.TRUCK_ARRIVAL,
        entity_id=0,
    )

    engine = SimulationEngine(
        state=state,
        event_queue=queue,
    )

    engine.step()

    assert state.current_time_hours == pytest.approx(0.5)

    engine.step()

    assert state.current_time_hours == pytest.approx(1.0)


def test_bin_fill_uses_actual_time_between_events() -> None:
    state = make_state()
    queue = EventQueue()

    queue.schedule(
        time_hours=1.0,
        event_type=EventType.SENSOR_SAMPLE,
    )

    queue.schedule(
        time_hours=4.0,
        event_type=EventType.SENSOR_SAMPLE,
    )

    engine = SimulationEngine(
        state=state,
        event_queue=queue,
    )

    engine.run()

    # Initial 50%, rate 2%/hour, 4 physical hours elapsed.
    assert state.bins[0].fill_percent == pytest.approx(
        58.0
    )

    assert state.current_time_hours == pytest.approx(4.0)
