import networkx as nx
import pytest

from smart_waste.collection.base import (
    CollectionPolicy,
    PolicyView,
)
from smart_waste.models.bin import WasteBin
from smart_waste.models.depot import Depot
from smart_waste.models.truck import (
    Truck,
    TruckStatus,
)
from smart_waste.movement.road_graph import RoadGraph
from smart_waste.simulation.dispatcher import (
    DispatchAction,
    DispatchError,
    dispatch_next_for_truck,
)
from smart_waste.simulation.engine import SimulationEngine
from smart_waste.simulation.event_queue import EventQueue
from smart_waste.simulation.orchestrator import (
    SimulationOrchestrator,
)
from smart_waste.simulation.policy_view import (
    build_policy_view,
)
from smart_waste.simulation.reservations import (
    BinReservationBook,
    ReservationError,
)
from smart_waste.simulation.state import SimulationState


SERVICE_SECONDS = 36.0


class FixedPolicy(CollectionPolicy):
    def __init__(self, bin_id: int) -> None:
        self.bin_id = bin_id

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
        return False


class ReservationAwarePolicy(CollectionPolicy):
    """
    Simple test policy that always chooses the lowest-ID
    unreserved, nonempty bin.
    """

    def __init__(self) -> None:
        self.completed: set[int] = set()
        self.callback_reservation_owner: list[
            int | None
        ] = []

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
        for bin_ in view.bins:
            if (
                bin_.bin_id not in self.completed
                and not bin_.is_reserved
                and bin_.fill_percent > 0.0
            ):
                return bin_.bin_id

        return None

    def on_service_complete(
        self,
        *,
        view: PolicyView,
        truck_id: int,
        bin_id: int,
    ) -> None:
        self.callback_reservation_owner.append(
            view.bin_by_id(
                bin_id
            ).reserved_by_truck_id
        )

        self.completed.add(
            bin_id
        )

    def is_complete(
        self,
        view: PolicyView,
    ) -> bool:
        return len(self.completed) >= 2


def make_state(
    *,
    truck_count: int = 2,
) -> SimulationState:
    graph = nx.Graph()

    graph.add_edge(
        "depot",
        "bin-0",
        length_km=3.0,
    )

    graph.add_edge(
        "depot",
        "bin-1",
        length_km=4.0,
    )

    graph.add_edge(
        "bin-0",
        "bin-1",
        length_km=2.0,
    )

    road = RoadGraph(graph)

    bins = {
        0: WasteBin(
            bin_id=0,
            road_node="bin-0",
            fill_percent=60.0,
            fill_rate_percent_per_hour=0.0,
            full_mass_kg=440.0,
        ),
        1: WasteBin(
            bin_id=1,
            road_node="bin-1",
            fill_percent=70.0,
            fill_rate_percent_per_hour=0.0,
            full_mass_kg=440.0,
        ),
    }

    trucks = {
        truck_id: Truck(
            truck_id=truck_id,
            current_node="depot",
            capacity_tonnes=10.0,
            speed_km_per_hour=30.0,
            fuel_capacity_litres=200.0,
            fuel_efficiency_km_per_litre=2.5,
        )
        for truck_id in range(
            truck_count
        )
    }

    depot = Depot(
        road_node="depot",
        unloading_bays=1,
        unload_time_minutes=11.0,
        refuel_rate_litres_per_minute=60.0,
    )

    return SimulationState(
        road_graph=road,
        bins=bins,
        trucks=trucks,
        depot=depot,
    )


def test_reserve_and_release() -> None:
    book = BinReservationBook()

    book.reserve(
        bin_id=4,
        truck_id=2,
    )

    assert book.owner(4) == 2
    assert book.bin_for_truck(2) == 4
    assert book.is_reserved(4)
    assert len(book) == 1

    book.release(
        bin_id=4,
        truck_id=2,
    )

    assert book.owner(4) is None
    assert book.bin_for_truck(2) is None
    assert len(book) == 0


def test_same_bin_cannot_be_reserved_by_two_trucks() -> None:
    book = BinReservationBook()

    book.reserve(
        bin_id=4,
        truck_id=1,
    )

    with pytest.raises(
        ReservationError,
        match="already reserved",
    ):
        book.reserve(
            bin_id=4,
            truck_id=2,
        )


def test_one_truck_cannot_reserve_two_bins() -> None:
    book = BinReservationBook()

    book.reserve(
        bin_id=4,
        truck_id=1,
    )

    with pytest.raises(
        ReservationError,
        match="already reserves",
    ):
        book.reserve(
            bin_id=5,
            truck_id=1,
        )


def test_wrong_truck_cannot_release_reservation() -> None:
    book = BinReservationBook()

    book.reserve(
        bin_id=4,
        truck_id=1,
    )

    with pytest.raises(
        ReservationError,
        match="not truck 2",
    ):
        book.release(
            bin_id=4,
            truck_id=2,
        )

    assert book.owner(4) == 1


def test_snapshot_cannot_mutate_reservation_book() -> None:
    book = BinReservationBook()

    book.reserve(
        bin_id=4,
        truck_id=1,
    )

    snapshot = book.snapshot()

    snapshot[4] = 99
    snapshot[5] = 2

    assert book.owner(4) == 1
    assert book.owner(5) is None


def test_policy_view_exposes_reservation_owner() -> None:
    state = make_state()
    book = BinReservationBook()

    book.reserve(
        bin_id=1,
        truck_id=0,
    )

    view = build_policy_view(
        state,
        reservations=book.snapshot(),
    )

    assert not view.bin_by_id(0).is_reserved

    assert view.bin_by_id(1).is_reserved
    assert (
        view.bin_by_id(1).reserved_by_truck_id
        == 0
    )
    assert view.bin_by_id(1).is_reserved_by(0)


def test_dispatcher_reserves_bin_before_travel() -> None:
    state = make_state(
        truck_count=1
    )

    queue = EventQueue()
    book = BinReservationBook()

    result = dispatch_next_for_truck(
        state=state,
        event_queue=queue,
        policy=FixedPolicy(0),
        reservation_book=book,
        truck_id=0,
        service_time_seconds=SERVICE_SECONDS,
    )

    assert result.action == DispatchAction.BIN_TRAVEL
    assert book.owner(0) == 0

    assert (
        state.trucks[0].status
        == TruckStatus.TRAVELLING
    )


def test_second_truck_cannot_bypass_existing_reservation() -> None:
    state = make_state()
    queue = EventQueue()
    book = BinReservationBook()

    first = dispatch_next_for_truck(
        state=state,
        event_queue=queue,
        policy=FixedPolicy(0),
        reservation_book=book,
        truck_id=0,
        service_time_seconds=SERVICE_SECONDS,
    )

    assert first.action == DispatchAction.BIN_TRAVEL

    with pytest.raises(
        DispatchError,
        match="already reserved by truck 0",
    ):
        dispatch_next_for_truck(
            state=state,
            event_queue=queue,
            policy=FixedPolicy(0),
            reservation_book=book,
            truck_id=1,
            service_time_seconds=SERVICE_SECONDS,
        )

    assert book.owner(0) == 0
    assert state.trucks[1].status == TruckStatus.IDLE

    # Only truck 0 has a scheduled trip.
    assert len(queue) == 1


def test_two_trucks_choose_different_unreserved_bins() -> None:
    state = make_state()
    policy = ReservationAwarePolicy()

    engine = SimulationEngine(
        state=state
    )

    orchestrator = SimulationOrchestrator(
        state=state,
        engine=engine,
        policy=policy,
        service_time_seconds=SERVICE_SECONDS,
    )

    results = orchestrator.start()

    assert [
        result.action
        for result in results
    ] == [
        DispatchAction.BIN_TRAVEL,
        DispatchAction.BIN_TRAVEL,
    ]

    assert orchestrator.reservation_book.owner(0) == 0
    assert orchestrator.reservation_book.owner(1) == 1

    assert state.trucks[0].destination_node == "bin-0"
    assert state.trucks[1].destination_node == "bin-1"


def test_reservation_remains_during_travel() -> None:
    state = make_state(
        truck_count=1
    )

    policy = ReservationAwarePolicy()

    engine = SimulationEngine(
        state=state
    )

    orchestrator = SimulationOrchestrator(
        state=state,
        engine=engine,
        policy=policy,
        service_time_seconds=SERVICE_SECONDS,
    )

    orchestrator.start(
        truck_ids=[0]
    )

    assert orchestrator.reservation_book.owner(0) == 0

    # Process physical arrival. Service has started but collection
    # has not completed yet.
    engine.step()

    assert (
        state.trucks[0].status
        == TruckStatus.SERVICING_BIN
    )

    assert orchestrator.reservation_book.owner(0) == 0


def test_reservation_released_only_after_service_completion() -> None:
    state = make_state(
        truck_count=1
    )

    policy = ReservationAwarePolicy()

    engine = SimulationEngine(
        state=state
    )

    orchestrator = SimulationOrchestrator(
        state=state,
        engine=engine,
        policy=policy,
        service_time_seconds=SERVICE_SECONDS,
    )

    orchestrator.start(
        truck_ids=[0]
    )

    # Arrival.
    engine.step()

    assert orchestrator.reservation_book.owner(0) == 0

    # Service completion.
    engine.step()

    assert orchestrator.reservation_book.owner(0) is None


def test_policy_callback_sees_released_reservation() -> None:
    state = make_state(
        truck_count=1
    )

    policy = ReservationAwarePolicy()

    engine = SimulationEngine(
        state=state
    )

    orchestrator = SimulationOrchestrator(
        state=state,
        engine=engine,
        policy=policy,
        service_time_seconds=SERVICE_SECONDS,
    )

    orchestrator.start(
        truck_ids=[0]
    )

    engine.step()
    engine.step()

    assert (
        policy.callback_reservation_owner
        == [None]
    )
