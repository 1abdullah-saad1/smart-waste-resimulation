import networkx as nx
import pytest

from smart_waste.collection.base import (
    CollectionPolicy,
    PolicyView,
)
from smart_waste.models.bin import WasteBin
from smart_waste.models.depot import Depot
from smart_waste.models.events import EventType
from smart_waste.models.truck import (
    Truck,
    TruckStatus,
)
from smart_waste.movement.road_graph import RoadGraph
from smart_waste.simulation.depot_service import (
    DepotReturnReason,
)
from smart_waste.simulation.dispatcher import (
    DispatchAction,
    DispatchError,
    dispatch_next_for_truck,
)
from smart_waste.simulation.event_queue import EventQueue
from smart_waste.simulation.reservations import BinReservationBook
from smart_waste.simulation.state import SimulationState


SERVICE_SECONDS = 36.0


class FixedBinPolicy(CollectionPolicy):
    def __init__(
        self,
        bin_id: int | None,
        *,
        complete: bool = False,
    ) -> None:
        self.bin_id = bin_id
        self.complete = complete

    def initialize(
        self,
        view: PolicyView,
    ) -> None:
        pass

    def select_next_bin(
        self,
        *,
        view: PolicyView,
        truck_id: int,
    ) -> int | None:
        return self.bin_id

    def on_service_complete(
        self,
        *,
        view: PolicyView,
        truck_id: int,
        bin_id: int,
    ) -> None:
        pass

    def is_complete(
        self,
        view: PolicyView,
    ) -> bool:
        return self.complete


def make_state(
    *,
    truck_node: str = "truck-node",
    truck_load_tonnes: float = 0.0,
    fuel_litres: float = 200.0,
    fill_percent: float = 100.0,
) -> SimulationState:
    graph = nx.Graph()

    graph.add_edge(
        "truck-node",
        "bin-node",
        length_km=20.0,
    )

    graph.add_edge(
        "bin-node",
        "depot",
        length_km=30.0,
    )

    # Direct safe return path.
    graph.add_edge(
        "truck-node",
        "depot",
        length_km=5.0,
    )

    road = RoadGraph(graph)

    bin_ = WasteBin(
        bin_id=0,
        road_node="bin-node",
        fill_percent=fill_percent,
        fill_rate_percent_per_hour=0.0,
        full_mass_kg=440.0,
    )

    truck = Truck(
        truck_id=0,
        current_node=truck_node,
        capacity_tonnes=10.0,
        speed_km_per_hour=30.0,
        fuel_capacity_litres=200.0,
        fuel_efficiency_km_per_litre=2.5,
        current_load_tonnes=truck_load_tonnes,
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
        trucks={0: truck},
        depot=depot,
    )


def test_feasible_policy_request_schedules_bin_trip() -> None:
    state = make_state()
    queue = EventQueue()

    result = dispatch_next_for_truck(
        state=state,
        event_queue=queue,
        policy=FixedBinPolicy(0),
        reservation_book=BinReservationBook(),
        truck_id=0,
        service_time_seconds=SERVICE_SECONDS,
    )

    assert result.action == DispatchAction.BIN_TRAVEL
    assert result.requested_bin_id == 0
    assert result.feasibility is not None
    assert result.feasibility.feasible

    assert len(queue) == 1
    assert result.event is not None
    assert result.event.event_type == EventType.TRUCK_ARRIVAL


def test_dispatch_does_not_teleport_truck() -> None:
    state = make_state()
    queue = EventQueue()

    dispatch_next_for_truck(
        state=state,
        event_queue=queue,
        policy=FixedBinPolicy(0),
        reservation_book=BinReservationBook(),
        truck_id=0,
        service_time_seconds=SERVICE_SECONDS,
    )

    truck = state.trucks[0]

    assert truck.current_node == "truck-node"
    assert truck.destination_node == "bin-node"
    assert truck.status == TruckStatus.TRAVELLING


def test_target_bin_id_is_preserved_in_event_metadata() -> None:
    state = make_state()
    queue = EventQueue()

    result = dispatch_next_for_truck(
        state=state,
        event_queue=queue,
        policy=FixedBinPolicy(0),
        reservation_book=BinReservationBook(),
        truck_id=0,
        service_time_seconds=SERVICE_SECONDS,
    )

    assert result.event is not None
    assert result.event.payload["target_bin_id"] == 0


def test_capacity_failure_forces_depot_return() -> None:
    state = make_state(
        truck_load_tonnes=9.7,
    )
    queue = EventQueue()

    result = dispatch_next_for_truck(
        state=state,
        event_queue=queue,
        policy=FixedBinPolicy(0),
        reservation_book=BinReservationBook(),
        truck_id=0,
        service_time_seconds=SERVICE_SECONDS,
    )

    assert result.action == DispatchAction.DEPOT_RETURN
    assert (
        result.depot_return_reason
        == DepotReturnReason.CAPACITY
    )

    assert result.event is not None
    assert result.event.event_type == EventType.DEPOT_ARRIVAL


def test_fuel_failure_forces_depot_return() -> None:
    # Candidate:
    # 20 km to bin + 25 km shortest bin->depot
    # because bin->truck-node->depot = 20 + 5 = 25 km.
    # Requirement = 45/2.5 = 18 L.
    #
    # Direct truck-node -> depot requires only 5/2.5 = 2 L.
    state = make_state(
        fuel_litres=10.0,
    )

    queue = EventQueue()

    result = dispatch_next_for_truck(
        state=state,
        event_queue=queue,
        policy=FixedBinPolicy(0),
        reservation_book=BinReservationBook(),
        truck_id=0,
        service_time_seconds=SERVICE_SECONDS,
    )

    assert result.action == DispatchAction.DEPOT_RETURN

    assert (
        result.depot_return_reason
        == DepotReturnReason.FUEL
    )


def test_combined_failure_forces_combined_return() -> None:
    state = make_state(
        truck_load_tonnes=9.7,
        fuel_litres=10.0,
    )

    queue = EventQueue()

    result = dispatch_next_for_truck(
        state=state,
        event_queue=queue,
        policy=FixedBinPolicy(0),
        reservation_book=BinReservationBook(),
        truck_id=0,
        service_time_seconds=SERVICE_SECONDS,
    )

    assert result.action == DispatchAction.DEPOT_RETURN

    assert (
        result.depot_return_reason
        == DepotReturnReason.COMBINED
    )


def test_policy_cannot_bypass_capacity_gate() -> None:
    state = make_state(
        truck_load_tonnes=9.7,
    )

    queue = EventQueue()

    result = dispatch_next_for_truck(
        state=state,
        event_queue=queue,
        policy=FixedBinPolicy(0),
        reservation_book=BinReservationBook(),
        truck_id=0,
        service_time_seconds=SERVICE_SECONDS,
    )

    assert result.action != DispatchAction.BIN_TRAVEL

    assert state.trucks[0].destination_node == "depot"


def test_policy_cannot_bypass_fuel_gate() -> None:
    state = make_state(
        fuel_litres=10.0,
    )

    queue = EventQueue()

    result = dispatch_next_for_truck(
        state=state,
        event_queue=queue,
        policy=FixedBinPolicy(0),
        reservation_book=BinReservationBook(),
        truck_id=0,
        service_time_seconds=SERVICE_SECONDS,
    )

    assert result.action != DispatchAction.BIN_TRAVEL

    assert state.trucks[0].destination_node == "depot"


def test_policy_complete_away_from_depot_schedules_final_return() -> None:
    state = make_state()
    queue = EventQueue()

    result = dispatch_next_for_truck(
        state=state,
        event_queue=queue,
        policy=FixedBinPolicy(
            None,
            complete=True,
        ),
        reservation_book=BinReservationBook(),
        truck_id=0,
        service_time_seconds=SERVICE_SECONDS,
    )

    assert result.action == DispatchAction.DEPOT_RETURN

    assert (
        result.depot_return_reason
        == DepotReturnReason.ROUTINE
    )

    assert result.event is not None
    assert result.event.event_type == EventType.DEPOT_ARRIVAL

    truck = state.trucks[0]

    assert truck.status == TruckStatus.TRAVELLING
    assert truck.current_node == "truck-node"
    assert truck.destination_node == "depot"

    assert len(queue) == 1

def test_none_candidate_does_not_schedule_physics() -> None:
    state = make_state()
    queue = EventQueue()

    result = dispatch_next_for_truck(
        state=state,
        event_queue=queue,
        policy=FixedBinPolicy(None),
        reservation_book=BinReservationBook(),
        truck_id=0,
        service_time_seconds=SERVICE_SECONDS,
    )

    assert result.action == DispatchAction.NO_CANDIDATE
    assert state.trucks[0].status == TruckStatus.IDLE
    assert len(queue) == 0


def test_unknown_bin_requested_by_policy_is_rejected() -> None:
    state = make_state()
    queue = EventQueue()

    with pytest.raises(DispatchError):
        dispatch_next_for_truck(
            state=state,
            event_queue=queue,
            policy=FixedBinPolicy(999),
            reservation_book=BinReservationBook(),
            truck_id=0,
            service_time_seconds=SERVICE_SECONDS,
        )

    assert len(queue) == 0


def test_busy_truck_cannot_be_dispatched() -> None:
    state = make_state()

    state.trucks[0].status = (
        TruckStatus.TRAVELLING
    )

    queue = EventQueue()

    with pytest.raises(DispatchError):
        dispatch_next_for_truck(
            state=state,
            event_queue=queue,
            policy=FixedBinPolicy(0),
            reservation_book=BinReservationBook(),
            truck_id=0,
            service_time_seconds=SERVICE_SECONDS,
        )


def test_truck_without_safe_depot_return_is_rejected() -> None:
    state = make_state(
        fuel_litres=1.0,
    )

    queue = EventQueue()

    with pytest.raises(
        DispatchError,
        match="cannot safely return to depot",
    ):
        dispatch_next_for_truck(
            state=state,
            event_queue=queue,
            policy=FixedBinPolicy(0),
            reservation_book=BinReservationBook(),
            truck_id=0,
            service_time_seconds=SERVICE_SECONDS,
        )

    assert len(queue) == 0


def test_empty_truck_at_depot_cannot_fix_capacity_impossibility() -> None:
    state = make_state(
        truck_node="depot",
        truck_load_tonnes=0.0,
    )

    # Artificially make one bin physically larger than truck capacity.
    state.bins[0].full_mass_kg = 11_000.0

    queue = EventQueue()

    with pytest.raises(
        DispatchError,
        match="exceeds truck capacity",
    ):
        dispatch_next_for_truck(
            state=state,
            event_queue=queue,
            policy=FixedBinPolicy(0),
            reservation_book=BinReservationBook(),
            truck_id=0,
            service_time_seconds=SERVICE_SECONDS,
        )


def test_full_tank_at_depot_cannot_fix_unreachable_candidate() -> None:
    state = make_state(
        truck_node="depot",
        fuel_litres=200.0,
    )

    # Full tank range:
    # 200 L * 2.5 km/L = 500 km.
    #
    # Candidate feasibility requires enough fuel for:
    # depot -> candidate -> depot.
    #
    # Make every available depot-to-bin route at least 300 km.
    # Therefore the round trip is at least 600 km:
    #
    #     600 / 2.5 = 240 L
    #
    # which exceeds the 200 L tank capacity.
    state.road_graph.graph["depot"]["bin-node"][
        "length_km"
    ] = 300.0

    state.road_graph.graph["truck-node"]["bin-node"][
        "length_km"
    ] = 300.0

    queue = EventQueue()

    with pytest.raises(
        DispatchError,
        match="full tank",
    ):
        dispatch_next_for_truck(
            state=state,
            event_queue=queue,
            policy=FixedBinPolicy(0),
            reservation_book=BinReservationBook(),
            truck_id=0,
            service_time_seconds=SERVICE_SECONDS,
        )

    assert len(queue) == 0
    assert state.trucks[0].status == TruckStatus.IDLE
    assert state.trucks[0].current_node == "depot"


def test_policy_complete_at_settled_depot_finishes_truck() -> None:
    state = make_state(
        truck_node="depot",
        truck_load_tonnes=0.0,
        fuel_litres=200.0,
    )

    queue = EventQueue()

    result = dispatch_next_for_truck(
        state=state,
        event_queue=queue,
        policy=FixedBinPolicy(
            None,
            complete=True,
        ),
        reservation_book=BinReservationBook(),
        truck_id=0,
        service_time_seconds=SERVICE_SECONDS,
    )

    assert result.action == DispatchAction.POLICY_COMPLETE
    assert state.trucks[0].status == TruckStatus.FINISHED
    assert len(queue) == 0
