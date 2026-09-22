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
from smart_waste.simulation.dispatcher import (
    DispatchAction,
)
from smart_waste.simulation.engine import SimulationEngine
from smart_waste.simulation.orchestrator import (
    OrchestratorError,
    SimulationOrchestrator,
)
from smart_waste.simulation.state import SimulationState


SERVICE_SECONDS = 36.0


class OrderedBinsPolicy(CollectionPolicy):
    def __init__(
        self,
        bin_ids: list[int],
    ) -> None:
        self.pending = list(bin_ids)

        self.initialize_calls = 0
        self.select_calls = 0

        self.completed: list[int] = []
        self.callback_fill_levels: list[float] = []

    def initialize(
        self,
        view: PolicyView,
    ) -> None:
        self.initialize_calls += 1

    def select_next_bin(
        self,
        *,
        view: PolicyView,
        truck_id: int,
    ) -> int | None:
        self.select_calls += 1
        view.truck_by_id(
            truck_id
        )

        if not self.pending:
            return None

        return self.pending[0]

    def on_service_complete(
        self,
        *,
        view: PolicyView,
        truck_id: int,
        bin_id: int,
    ) -> None:
        if (
            not self.pending
            or self.pending[0] != bin_id
        ):
            raise RuntimeError(
                "unexpected service completion"
            )

        self.callback_fill_levels.append(
            view.bin_by_id(
                bin_id
            ).fill_percent
        )

        self.completed.append(
            bin_id
        )

        self.pending.pop(0)

    def is_complete(
        self,
        view: PolicyView,
    ) -> bool:
        return not self.pending


class NoCandidatePolicy(CollectionPolicy):
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
        return None

    def on_service_complete(
        self,
        *,
        view: PolicyView,
        truck_id: int,
        bin_id: int,
    ) -> None:
        raise AssertionError(
            "no bin should be serviced"
        )

    def is_complete(
        self,
        view: PolicyView,
    ) -> bool:
        return False


def make_state(
    *,
    initial_load_tonnes: float = 0.0,
    initial_fuel_litres: float = 200.0,
    bin0_fill_percent: float = 50.0,
) -> SimulationState:
    graph = nx.Graph()

    graph.add_edge(
        "depot",
        "bin-0",
        length_km=3.0,
    )

    graph.add_edge(
        "bin-0",
        "bin-1",
        length_km=2.0,
    )

    graph.add_edge(
        "depot",
        "bin-1",
        length_km=4.0,
    )

    road = RoadGraph(
        graph
    )

    bins = {
        0: WasteBin(
            bin_id=0,
            road_node="bin-0",
            fill_percent=bin0_fill_percent,
            fill_rate_percent_per_hour=0.0,
            full_mass_kg=440.0,
        ),
        1: WasteBin(
            bin_id=1,
            road_node="bin-1",
            fill_percent=60.0,
            fill_rate_percent_per_hour=0.0,
            full_mass_kg=440.0,
        ),
    }

    truck = Truck(
        truck_id=0,
        current_node="depot",
        capacity_tonnes=10.0,
        speed_km_per_hour=30.0,
        fuel_capacity_litres=200.0,
        fuel_efficiency_km_per_litre=2.5,
        current_load_tonnes=initial_load_tonnes,
        fuel_remaining_litres=initial_fuel_litres,
    )

    depot = Depot(
        road_node="depot",
        unloading_bays=1,
        unload_time_minutes=11.0,
        refuel_rate_litres_per_minute=60.0,
    )

    return SimulationState(
        road_graph=road,
        bins=bins,
        trucks={
            0: truck,
        },
        depot=depot,
    )


def make_orchestrator(
    *,
    state: SimulationState,
    policy: CollectionPolicy,
) -> SimulationOrchestrator:
    engine = SimulationEngine(
        state=state
    )

    return SimulationOrchestrator(
        state=state,
        engine=engine,
        policy=policy,
        service_time_seconds=SERVICE_SECONDS,
    )


def test_orchestrator_requires_same_engine_state() -> None:
    state_a = make_state()
    state_b = make_state()

    engine = SimulationEngine(
        state=state_a
    )

    with pytest.raises(
        OrchestratorError,
        match="same SimulationState",
    ):
        SimulationOrchestrator(
            state=state_b,
            engine=engine,
            policy=OrderedBinsPolicy([0]),
            service_time_seconds=SERVICE_SECONDS,
        )


def test_start_initializes_policy_once() -> None:
    state = make_state()
    policy = OrderedBinsPolicy([0])

    orchestrator = make_orchestrator(
        state=state,
        policy=policy,
    )

    orchestrator.start(
        truck_ids=[0]
    )

    assert policy.initialize_calls == 1

    orchestrator.initialize_policy()

    assert policy.initialize_calls == 1


def test_start_schedules_initial_bin_travel() -> None:
    state = make_state()
    policy = OrderedBinsPolicy([0])

    orchestrator = make_orchestrator(
        state=state,
        policy=policy,
    )

    results = orchestrator.start(
        truck_ids=[0]
    )

    assert len(results) == 1

    assert (
        results[0].action
        == DispatchAction.BIN_TRAVEL
    )

    assert (
        orchestrator.engine.event_queue.peek().event_type
        == EventType.TRUCK_ARRIVAL
    )

    assert state.trucks[0].status == TruckStatus.TRAVELLING


def test_arrival_automatically_starts_bin_service() -> None:
    state = make_state()
    policy = OrderedBinsPolicy([0])

    orchestrator = make_orchestrator(
        state=state,
        policy=policy,
    )

    orchestrator.start(
        truck_ids=[0]
    )

    orchestrator.engine.step()

    assert state.trucks[0].current_node == "bin-0"

    assert (
        state.trucks[0].status
        == TruckStatus.SERVICING_BIN
    )

    assert (
        orchestrator.engine.event_queue.peek().event_type
        == EventType.BIN_SERVICE_COMPLETE
    )


def test_policy_callback_occurs_after_physical_collection() -> None:
    state = make_state()
    policy = OrderedBinsPolicy([0])

    orchestrator = make_orchestrator(
        state=state,
        policy=policy,
    )

    orchestrator.start(
        truck_ids=[0]
    )

    # Arrival + bin-service completion.
    orchestrator.engine.step()
    orchestrator.engine.step()

    assert policy.completed == [0]

    # Policy callback sees post-collection physical state.
    assert policy.callback_fill_levels == [
        pytest.approx(0.0)
    ]

    assert state.bins[0].collected_count == 1


def test_service_completion_dispatches_next_bin() -> None:
    state = make_state()
    policy = OrderedBinsPolicy(
        [0, 1]
    )

    orchestrator = make_orchestrator(
        state=state,
        policy=policy,
    )

    orchestrator.start(
        truck_ids=[0]
    )

    # bin-0 arrival
    orchestrator.engine.step()

    # bin-0 service complete; orchestrator must dispatch bin-1.
    orchestrator.engine.step()

    truck = state.trucks[0]

    assert truck.status == TruckStatus.TRAVELLING
    assert truck.destination_node == "bin-1"

    assert (
        orchestrator.engine.event_queue.peek().event_type
        == EventType.TRUCK_ARRIVAL
    )


def test_one_bin_end_to_end_finishes_truck() -> None:
    state = make_state()
    policy = OrderedBinsPolicy([0])

    orchestrator = make_orchestrator(
        state=state,
        policy=policy,
    )

    orchestrator.start(
        truck_ids=[0]
    )

    processed = orchestrator.engine.run()

    assert processed == 6

    assert state.bins[0].collected_count == 1
    assert policy.completed == [0]

    assert (
        state.trucks[0].status
        == TruckStatus.FINISHED
    )

    assert state.trucks[0].current_node == "depot"
    assert state.trucks[0].current_load_tonnes == pytest.approx(0.0)
    assert state.trucks[0].fuel_remaining_litres == pytest.approx(200.0)

    assert orchestrator.engine.event_queue.is_empty


def test_two_bins_run_end_to_end_without_manual_policy_steps() -> None:
    state = make_state()
    policy = OrderedBinsPolicy(
        [0, 1]
    )

    orchestrator = make_orchestrator(
        state=state,
        policy=policy,
    )

    orchestrator.start(
        truck_ids=[0]
    )

    orchestrator.engine.run()

    assert policy.completed == [
        0,
        1,
    ]

    assert state.bins[0].collected_count == 1
    assert state.bins[1].collected_count == 1

    assert orchestrator.completed_services == [
        (0, 0),
        (0, 1),
    ]

    assert (
        state.trucks[0].status
        == TruckStatus.FINISHED
    )

    # Complete physical route:
    # depot -> bin0 = 3 km
    # bin0 -> bin1 = 2 km
    # bin1 -> depot = 4 km
    assert (
        state.trucks[0].cumulative_distance_km
        == pytest.approx(9.0)
    )

    # 9 km / 2.5 km/L = 3.6 L.
    assert (
        state.trucks[0].cumulative_fuel_used_litres
        == pytest.approx(3.6)
    )

    assert state.trucks[0].current_node == "depot"
    assert state.trucks[0].current_load_tonnes == pytest.approx(0.0)
    assert state.trucks[0].fuel_remaining_litres == pytest.approx(200.0)


def test_no_candidate_leaves_truck_idle() -> None:
    state = make_state()

    orchestrator = make_orchestrator(
        state=state,
        policy=NoCandidatePolicy(),
    )

    results = orchestrator.start(
        truck_ids=[0]
    )

    assert (
        results[0].action
        == DispatchAction.NO_CANDIDATE
    )

    assert state.trucks[0].status == TruckStatus.IDLE

    assert orchestrator.engine.event_queue.is_empty


def test_capacity_failure_returns_to_depot_and_resumes() -> None:
    state = make_state(
        initial_load_tonnes=9.7,
        bin0_fill_percent=100.0,
    )

    policy = OrderedBinsPolicy([0])

    orchestrator = make_orchestrator(
        state=state,
        policy=policy,
    )

    initial = orchestrator.start(
        truck_ids=[0]
    )

    assert (
        initial[0].action
        == DispatchAction.DEPOT_RETURN
    )

    orchestrator.engine.run()

    # Depot unload makes the previously infeasible candidate
    # feasible; the lifecycle resumes automatically.
    assert state.bins[0].collected_count == 1
    assert policy.completed == [0]

    assert (
        state.trucks[0].status
        == TruckStatus.FINISHED
    )

    actions = [
        item.action
        for item in orchestrator.dispatch_history
    ]

    assert actions == [
        DispatchAction.DEPOT_RETURN,
        DispatchAction.BIN_TRAVEL,
        DispatchAction.DEPOT_RETURN,
        DispatchAction.POLICY_COMPLETE,
    ]


def test_start_cannot_be_called_twice() -> None:
    state = make_state()

    orchestrator = make_orchestrator(
        state=state,
        policy=OrderedBinsPolicy([0]),
    )

    orchestrator.start(
        truck_ids=[0]
    )

    with pytest.raises(
        OrchestratorError,
        match="already started",
    ):
        orchestrator.start(
            truck_ids=[0]
        )


def test_dispatch_history_records_complete_lifecycle() -> None:
    state = make_state()
    policy = OrderedBinsPolicy(
        [0, 1]
    )

    orchestrator = make_orchestrator(
        state=state,
        policy=policy,
    )

    orchestrator.start(
        truck_ids=[0]
    )

    orchestrator.engine.run()

    actions = [
        result.action
        for result in orchestrator.dispatch_history
    ]

    assert actions == [
        DispatchAction.BIN_TRAVEL,
        DispatchAction.BIN_TRAVEL,
        DispatchAction.DEPOT_RETURN,
        DispatchAction.POLICY_COMPLETE,
    ]
