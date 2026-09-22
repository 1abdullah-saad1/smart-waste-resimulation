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
from smart_waste.simulation.bin_service import (
    BinServiceError,
    handle_bin_service_complete,
    register_bin_service_handlers,
    schedule_bin_service,
    service_duration_hours,
)
from smart_waste.simulation.engine import SimulationEngine
from smart_waste.simulation.event_queue import EventQueue
from smart_waste.simulation.state import SimulationState


SERVICE_SECONDS = 36.0


def make_state(
    *,
    bin_fill_percent: float = 50.0,
    bin_fill_rate: float = 2.0,
    truck_load_tonnes: float = 0.0,
) -> SimulationState:
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
        fill_percent=bin_fill_percent,
        fill_rate_percent_per_hour=bin_fill_rate,
        full_mass_kg=440.0,
    )

    truck = Truck(
        truck_id=0,
        current_node="bin-node",
        capacity_tonnes=10.0,
        speed_km_per_hour=30.0,
        fuel_capacity_litres=200.0,
        fuel_efficiency_km_per_litre=2.5,
        current_load_tonnes=truck_load_tonnes,
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


def test_36_seconds_is_point_01_hours() -> None:
    assert service_duration_hours(
        36.0
    ) == pytest.approx(0.01)


def test_service_does_not_collect_immediately() -> None:
    state = make_state()
    queue = EventQueue()

    event = schedule_bin_service(
        state=state,
        event_queue=queue,
        truck_id=0,
        bin_id=0,
        service_time_seconds=SERVICE_SECONDS,
    )

    truck = state.trucks[0]
    bin_ = state.bins[0]

    assert event.event_type == EventType.BIN_SERVICE_COMPLETE
    assert truck.status == TruckStatus.SERVICING_BIN

    assert bin_.fill_percent == pytest.approx(50.0)
    assert truck.current_load_tonnes == pytest.approx(0.0)


def test_service_completion_time_is_physical() -> None:
    state = make_state()
    queue = EventQueue()

    event = schedule_bin_service(
        state=state,
        event_queue=queue,
        truck_id=0,
        bin_id=0,
        service_time_seconds=SERVICE_SECONDS,
    )

    assert event.time_hours == pytest.approx(
        0.01
    )


def test_bin_continues_filling_during_service() -> None:
    state = make_state(
        bin_fill_percent=50.0,
        bin_fill_rate=2.0,
    )
    queue = EventQueue()

    schedule_bin_service(
        state=state,
        event_queue=queue,
        truck_id=0,
        bin_id=0,
        service_time_seconds=SERVICE_SECONDS,
    )

    engine = SimulationEngine(
        state=state,
        event_queue=queue,
    )

    register_bin_service_handlers(engine)

    engine.step()

    # Immediately before collection at t=0.01 h:
    # fill = 50 + (2 * 0.01) = 50.02%.
    # Therefore collected mass includes physical accumulation
    # during the service interval.
    expected_mass = (
        440.0
        * 50.02
        / 100.0
        / 1000.0
    )

    assert state.trucks[0].current_load_tonnes == pytest.approx(
        expected_mass
    )

    assert state.bins[0].fill_percent == pytest.approx(
        0.0
    )


def test_bin_is_reset_only_at_completion() -> None:
    state = make_state()
    queue = EventQueue()

    event = schedule_bin_service(
        state=state,
        event_queue=queue,
        truck_id=0,
        bin_id=0,
        service_time_seconds=SERVICE_SECONDS,
    )

    state.advance_to(
        event.time_hours
    )

    assert state.bins[0].fill_percent > 50.0

    handle_bin_service_complete(
        state,
        event,
    )

    assert state.bins[0].fill_percent == 0.0
    assert state.bins[0].waste_age_hours == 0.0
    assert state.bins[0].collected_count == 1


def test_truck_returns_idle_after_service() -> None:
    state = make_state()
    queue = EventQueue()

    schedule_bin_service(
        state=state,
        event_queue=queue,
        truck_id=0,
        bin_id=0,
        service_time_seconds=SERVICE_SECONDS,
    )

    engine = SimulationEngine(
        state=state,
        event_queue=queue,
    )

    register_bin_service_handlers(engine)
    engine.step()

    truck = state.trucks[0]

    assert truck.status == TruckStatus.IDLE
    assert truck.next_event_time_hours is None


def test_service_requires_truck_at_bin() -> None:
    state = make_state()

    state.trucks[0].current_node = "depot"

    queue = EventQueue()

    with pytest.raises(BinServiceError):
        schedule_bin_service(
            state=state,
            event_queue=queue,
            truck_id=0,
            bin_id=0,
            service_time_seconds=SERVICE_SECONDS,
        )

    assert len(queue) == 0


def test_busy_truck_cannot_begin_bin_service() -> None:
    state = make_state()

    state.trucks[0].status = TruckStatus.TRAVELLING

    queue = EventQueue()

    with pytest.raises(BinServiceError):
        schedule_bin_service(
            state=state,
            event_queue=queue,
            truck_id=0,
            bin_id=0,
            service_time_seconds=SERVICE_SECONDS,
        )

    assert len(queue) == 0


def test_capacity_checked_before_service() -> None:
    state = make_state(
        bin_fill_percent=100.0,
        truck_load_tonnes=9.7,
    )

    queue = EventQueue()

    # Full bin = 0.44 t, but truck has only 0.30 t remaining.
    with pytest.raises(BinServiceError):
        schedule_bin_service(
            state=state,
            event_queue=queue,
            truck_id=0,
            bin_id=0,
            service_time_seconds=SERVICE_SECONDS,
        )

    assert len(queue) == 0


def test_service_capacity_accounts_for_fill_during_service() -> None:
    state = make_state(
        bin_fill_percent=50.0,
        bin_fill_rate=100.0,
    )

    # Initial mass = 0.22 t.
    # After 0.01 h at 100 %/h -> fill becomes 51%.
    # Completion mass = 0.2244 t.
    state.trucks[0].current_load_tonnes = (
        10.0 - 0.222
    )

    queue = EventQueue()

    with pytest.raises(BinServiceError):
        schedule_bin_service(
            state=state,
            event_queue=queue,
            truck_id=0,
            bin_id=0,
            service_time_seconds=SERVICE_SECONDS,
        )

    assert len(queue) == 0


def test_other_bins_advance_during_service() -> None:
    state = make_state()

    state.bins[1] = WasteBin(
        bin_id=1,
        road_node="depot",
        fill_percent=20.0,
        fill_rate_percent_per_hour=10.0,
        full_mass_kg=440.0,
    )

    queue = EventQueue()

    schedule_bin_service(
        state=state,
        event_queue=queue,
        truck_id=0,
        bin_id=0,
        service_time_seconds=SERVICE_SECONDS,
    )

    engine = SimulationEngine(
        state=state,
        event_queue=queue,
    )

    register_bin_service_handlers(engine)
    engine.step()

    # 10%/hour * 0.01 hour = +0.1%.
    assert state.bins[1].fill_percent == pytest.approx(
        20.1
    )


def test_service_event_timestamp_must_match_truck() -> None:
    state = make_state()
    queue = EventQueue()

    event = schedule_bin_service(
        state=state,
        event_queue=queue,
        truck_id=0,
        bin_id=0,
        service_time_seconds=SERVICE_SECONDS,
    )

    event = type(event)(
        time_hours=event.time_hours + 1.0,
        sequence=event.sequence,
        event_type=event.event_type,
        entity_id=event.entity_id,
        payload=event.payload,
    )

    state.advance_to(
        event.time_hours
    )

    with pytest.raises(BinServiceError):
        handle_bin_service_complete(
            state,
            event,
        )


def test_service_time_must_be_positive() -> None:
    with pytest.raises(ValueError):
        service_duration_hours(0.0)

    with pytest.raises(ValueError):
        service_duration_hours(-1.0)
