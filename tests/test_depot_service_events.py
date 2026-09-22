import networkx as nx
import pytest

from smart_waste.models.bin import WasteBin
from smart_waste.models.depot import Depot
from smart_waste.models.truck import (
    Truck,
    TruckStatus,
)
from smart_waste.movement.road_graph import RoadGraph
from smart_waste.simulation.depot_service import (
    DepotReturnReason,
    DepotServiceState,
    register_depot_handlers,
    schedule_depot_return,
)
from smart_waste.simulation.engine import SimulationEngine
from smart_waste.simulation.event_queue import EventQueue
from smart_waste.simulation.movement import (
    register_movement_handlers,
)
from smart_waste.simulation.state import SimulationState


def make_state(
    *,
    truck_count: int = 1,
    load_tonnes: float = 3.0,
    fuel_litres: float = 140.0,
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
        fill_percent=50.0,
        fill_rate_percent_per_hour=10.0,
        full_mass_kg=440.0,
    )

    trucks = {}

    for truck_id in range(truck_count):
        trucks[truck_id] = Truck(
            truck_id=truck_id,
            current_node="bin-node",
            capacity_tonnes=10.0,
            speed_km_per_hour=30.0,
            fuel_capacity_litres=200.0,
            fuel_efficiency_km_per_litre=2.5,
            current_load_tonnes=load_tonnes,
            fuel_remaining_litres=fuel_litres,
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
        trucks=trucks,
        depot=depot,
    )


def make_engine(
    state: SimulationState,
) -> tuple[
    SimulationEngine,
    DepotServiceState,
]:
    queue = EventQueue()

    engine = SimulationEngine(
        state=state,
        event_queue=queue,
    )

    depot_state = DepotServiceState()

    # Order is intentional.
    register_movement_handlers(engine)

    register_depot_handlers(
        engine,
        depot_state=depot_state,
    )

    return engine, depot_state


def test_depot_return_is_physical_travel() -> None:
    state = make_state()
    engine, _ = make_engine(state)

    event = schedule_depot_return(
        state=state,
        event_queue=engine.event_queue,
        truck_id=0,
        reason=DepotReturnReason.CAPACITY,
    )

    assert event.time_hours == pytest.approx(
        3.0 / 30.0
    )

    assert state.trucks[0].current_node == "bin-node"
    assert state.trucks[0].status == TruckStatus.TRAVELLING


def test_depot_arrival_enters_fcfs_queue() -> None:
    state = make_state()
    engine, depot_state = make_engine(state)

    schedule_depot_return(
        state=state,
        event_queue=engine.event_queue,
        truck_id=0,
        reason=DepotReturnReason.CAPACITY,
    )

    # DEPOT_ARRIVAL
    engine.step()

    truck = state.trucks[0]

    assert truck.current_node == "depot"
    assert truck.status == TruckStatus.WAITING_AT_DEPOT

    assert list(
        depot_state.waiting_truck_ids
    ) == [0]


def test_capacity_return_counter_recorded_at_arrival() -> None:
    state = make_state()
    engine, _ = make_engine(state)

    schedule_depot_return(
        state=state,
        event_queue=engine.event_queue,
        truck_id=0,
        reason=DepotReturnReason.CAPACITY,
    )

    assert state.trucks[0].depot_returns == 0

    engine.step()

    truck = state.trucks[0]

    assert truck.depot_returns == 1
    assert truck.capacity_returns == 1


def test_unload_takes_eleven_minutes() -> None:
    state = make_state(
        load_tonnes=3.0,
        fuel_litres=200.0,
    )

    # Put truck directly at depot so road travel takes zero time.
    state.trucks[0].current_node = "depot"

    engine, _ = make_engine(state)

    schedule_depot_return(
        state=state,
        event_queue=engine.event_queue,
        truck_id=0,
        reason=DepotReturnReason.ROUTINE,
    )

    # arrival
    engine.step()

    # service start
    engine.step()

    assert state.trucks[0].status == TruckStatus.UNLOADING

    # unload complete
    engine.step()

    assert state.current_time_hours == pytest.approx(
        11.0 / 60.0
    )

    assert state.trucks[0].current_load_tonnes == 0.0


def test_refuel_duration_follows_rate() -> None:
    state = make_state(
        load_tonnes=0.0,
        fuel_litres=140.0,
    )

    state.trucks[0].current_node = "depot"

    engine, _ = make_engine(state)

    schedule_depot_return(
        state=state,
        event_queue=engine.event_queue,
        truck_id=0,
        reason=DepotReturnReason.FUEL,
    )

    engine.run()

    # Missing 60 L / 60 L/min = 1 minute.
    assert state.current_time_hours == pytest.approx(
        1.0 / 60.0
    )

    assert state.trucks[0].fuel_remaining_litres == pytest.approx(
        200.0
    )


def test_unload_then_refuel_are_sequential() -> None:
    state = make_state(
        load_tonnes=3.0,
        fuel_litres=140.0,
    )

    state.trucks[0].current_node = "depot"

    engine, _ = make_engine(state)

    schedule_depot_return(
        state=state,
        event_queue=engine.event_queue,
        truck_id=0,
        reason=DepotReturnReason.COMBINED,
    )

    engine.run()

    # 11 minutes unloading + 1 minute refuelling.
    assert state.current_time_hours == pytest.approx(
        12.0 / 60.0
    )

    truck = state.trucks[0]

    assert truck.current_load_tonnes == 0.0
    assert truck.fuel_remaining_litres == pytest.approx(
        200.0
    )
    assert truck.status == TruckStatus.IDLE


def test_bins_continue_filling_during_depot_service() -> None:
    state = make_state(
        load_tonnes=3.0,
        fuel_litres=140.0,
    )

    state.trucks[0].current_node = "depot"

    engine, _ = make_engine(state)

    schedule_depot_return(
        state=state,
        event_queue=engine.event_queue,
        truck_id=0,
        reason=DepotReturnReason.COMBINED,
    )

    engine.run()

    # 12 min = 0.2 h, fill rate = 10 %/h.
    assert state.bins[0].fill_percent == pytest.approx(
        52.0
    )


def test_single_bay_is_fcfs() -> None:
    state = make_state(
        truck_count=2,
        load_tonnes=3.0,
        fuel_litres=200.0,
    )

    for truck in state.trucks.values():
        truck.current_node = "depot"

    engine, depot_state = make_engine(state)

    schedule_depot_return(
        state=state,
        event_queue=engine.event_queue,
        truck_id=0,
        reason=DepotReturnReason.ROUTINE,
    )

    schedule_depot_return(
        state=state,
        event_queue=engine.event_queue,
        truck_id=1,
        reason=DepotReturnReason.ROUTINE,
    )

    # Two zero-time arrivals.
    engine.step()
    engine.step()

    assert list(
        depot_state.waiting_truck_ids
    ) == [0, 1]

    # First service start.
    engine.step()

    assert state.trucks[0].status == TruckStatus.UNLOADING
    assert state.trucks[1].status == TruckStatus.WAITING_AT_DEPOT

    assert depot_state.active_truck_ids == {0}


def test_second_truck_starts_only_after_first_completes() -> None:
    state = make_state(
        truck_count=2,
        load_tonnes=3.0,
        fuel_litres=200.0,
    )

    for truck in state.trucks.values():
        truck.current_node = "depot"

    engine, _ = make_engine(state)

    for truck_id in (0, 1):
        schedule_depot_return(
            state=state,
            event_queue=engine.event_queue,
            truck_id=truck_id,
            reason=DepotReturnReason.ROUTINE,
        )

    engine.run()

    # 11 min per truck, one FCFS service channel.
    assert state.current_time_hours == pytest.approx(
        22.0 / 60.0
    )

    assert state.trucks[0].status == TruckStatus.IDLE
    assert state.trucks[1].status == TruckStatus.IDLE


def test_fuel_return_counter() -> None:
    state = make_state()
    engine, _ = make_engine(state)

    schedule_depot_return(
        state=state,
        event_queue=engine.event_queue,
        truck_id=0,
        reason=DepotReturnReason.FUEL,
    )

    engine.step()

    truck = state.trucks[0]

    assert truck.depot_returns == 1
    assert truck.fuel_returns == 1
    assert truck.capacity_returns == 0


def test_combined_return_counter() -> None:
    state = make_state()
    engine, _ = make_engine(state)

    schedule_depot_return(
        state=state,
        event_queue=engine.event_queue,
        truck_id=0,
        reason=DepotReturnReason.COMBINED,
    )

    engine.step()

    truck = state.trucks[0]

    assert truck.depot_returns == 1
    assert truck.combined_returns == 1


def test_road_travel_fuel_is_included_before_refuel() -> None:
    state = make_state(
        load_tonnes=0.0,
        fuel_litres=200.0,
    )

    engine, _ = make_engine(state)

    schedule_depot_return(
        state=state,
        event_queue=engine.event_queue,
        truck_id=0,
        reason=DepotReturnReason.ROUTINE,
    )

    # 3 km / 2.5 km/L = 1.2 L used before depot arrival.
    assert state.trucks[0].fuel_remaining_litres == pytest.approx(
        198.8
    )

    engine.run()

    assert state.trucks[0].fuel_remaining_litres == pytest.approx(
        200.0
    )

    assert state.trucks[0].cumulative_fuel_used_litres == pytest.approx(
        1.2
    )
