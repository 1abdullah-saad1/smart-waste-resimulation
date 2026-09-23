import networkx as nx
import pytest
import torch

from smart_waste.collection.dqn import (
    DQNCollectionPolicy,
)
from smart_waste.models.bin import (
    WasteBin,
)
from smart_waste.models.depot import (
    Depot,
)
from smart_waste.models.truck import (
    Truck,
    TruckStatus,
)
from smart_waste.movement.road_graph import (
    RoadGraph,
)
from smart_waste.rl.dqn import (
    DQNAgent,
    DQNConfig,
)
from smart_waste.rl.physical_training import (
    PhysicalDQNTrainingPolicy,
)
from smart_waste.simulation.dispatcher import (
    DispatchAction,
)
from smart_waste.simulation.engine import (
    SimulationEngine,
)
from smart_waste.simulation.orchestrator import (
    SimulationOrchestrator,
)
from smart_waste.simulation.state import (
    SimulationState,
)


SERVICE_SECONDS = 36.0


def zero_agent() -> DQNAgent:
    """
    One-bin DQN.

    Input:
        2 location
      + 1 fill
      + 1 hazard
      + 1 elapsed-full
      = 5 features

    Action dimension:
        1 bin
    """

    agent = DQNAgent(
        input_dim=5,
        action_dim=1,
        seed=123,
        config=DQNConfig(
            replay_warmup_transitions=1000,
        ),
    )

    with torch.no_grad():
        for parameter in (
            agent.online_network.parameters()
        ):
            parameter.zero_()

        for parameter in (
            agent.target_network.parameters()
        ):
            parameter.zero_()

    return agent


def make_policy(
    state: SimulationState,
) -> tuple[
    DQNAgent,
    DQNCollectionPolicy,
    PhysicalDQNTrainingPolicy,
]:
    agent = zero_agent()

    dqn_policy = DQNCollectionPolicy(
        road_graph=state.road_graph,
        agent=agent,
        epsilon=0.0,
    )

    training_policy = (
        PhysicalDQNTrainingPolicy(
            policy=dqn_policy,
            optimize_after_transition=False,
        )
    )

    return (
        agent,
        dqn_policy,
        training_policy,
    )


def make_orchestrator(
    state: SimulationState,
) -> tuple[
    SimulationOrchestrator,
    DQNAgent,
    PhysicalDQNTrainingPolicy,
]:
    (
        agent,
        _,
        training_policy,
    ) = make_policy(
        state
    )

    engine = SimulationEngine(
        state=state
    )

    orchestrator = SimulationOrchestrator(
        state=state,
        engine=engine,
        policy=training_policy,
        service_time_seconds=SERVICE_SECONDS,
    )

    return (
        orchestrator,
        agent,
        training_policy,
    )


def make_direct_state() -> SimulationState:
    """
    depot --2.5 km-- bin-0
    """

    graph = nx.Graph()

    graph.add_node(
        "depot",
        x=0.0,
        y=0.0,
    )

    graph.add_node(
        "bin-0",
        x=2.5,
        y=0.0,
    )

    graph.add_edge(
        "depot",
        "bin-0",
        length_km=2.5,
    )

    bin_ = WasteBin(
        bin_id=0,
        road_node="bin-0",
        fill_percent=95.0,
        fill_rate_percent_per_hour=0.0,
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
        road_graph=RoadGraph(
            graph
        ),
        bins={
            0: bin_,
        },
        trucks={
            0: truck,
        },
        depot=depot,
    )


def make_capacity_detour_state() -> SimulationState:
    """
    Truck begins away from depot with a nearly full load.

        start --2 km-- depot --3 km-- bin-0

    DQN requests bin-0.

    The Core must first force:

        start -> depot

    for capacity recovery, then:

        depot -> bin-0

    Therefore action-to-service physical distance = 5 km.
    """

    graph = nx.Graph()

    graph.add_node(
        "start",
        x=-2.0,
        y=0.0,
    )

    graph.add_node(
        "depot",
        x=0.0,
        y=0.0,
    )

    graph.add_node(
        "bin-0",
        x=3.0,
        y=0.0,
    )

    graph.add_edge(
        "start",
        "depot",
        length_km=2.0,
    )

    graph.add_edge(
        "depot",
        "bin-0",
        length_km=3.0,
    )

    bin_ = WasteBin(
        bin_id=0,
        road_node="bin-0",
        fill_percent=95.0,
        fill_rate_percent_per_hour=0.0,
        full_mass_kg=440.0,
    )

    truck = Truck(
        truck_id=0,
        current_node="start",
        capacity_tonnes=10.0,
        speed_km_per_hour=30.0,
        fuel_capacity_litres=200.0,
        fuel_efficiency_km_per_litre=2.5,
        current_load_tonnes=9.9,
    )

    depot = Depot(
        road_node="depot",
        unloading_bays=1,
        unload_time_minutes=11.0,
        refuel_rate_litres_per_minute=60.0,
    )

    return SimulationState(
        road_graph=RoadGraph(
            graph
        ),
        bins={
            0: bin_,
        },
        trucks={
            0: truck,
        },
        depot=depot,
    )


def test_physical_dqn_end_to_end_transition_uses_core_distance():
    state = make_direct_state()

    (
        orchestrator,
        agent,
        training,
    ) = make_orchestrator(
        state
    )

    initial = (
        orchestrator.start()
    )

    assert len(
        initial
    ) == 1

    assert (
        initial[
            0
        ].action
        == DispatchAction.BIN_TRAVEL
    )

    assert (
        initial[
            0
        ].requested_bin_id
        == 0
    )

    processed = (
        orchestrator.engine.run(
            max_events=100
        )
    )

    assert processed > 0

    assert (
        orchestrator.completed_services
        == [
            (
                0,
                0,
            )
        ]
    )

    assert (
        state.bins[
            0
        ].fill_percent
        == pytest.approx(
            0.0
        )
    )

    assert len(
        training.transitions
    ) == 1

    assert len(
        agent.replay_buffer
    ) == 1

    transition = (
        training.transitions[
            0
        ]
    )

    assert (
        transition.transition_kind
        == "service_complete"
    )

    # This value is not computed by the DQN.
    # It comes from Truck.cumulative_distance_km,
    # which is mutated only by the physical movement Core.
    assert (
        transition.distance_km
        == pytest.approx(
            2.5
        )
    )

    assert (
        transition.collection_fraction
        == pytest.approx(
            0.95
        )
    )

    # R =
    # 1.0 * 0.95
    # - 0.5 * 2.5
    # = -0.30
    assert (
        transition.reward
        == pytest.approx(
            -0.30
        )
    )

    # 2.5 km / 30 km/h + 36 s service.
    assert (
        transition.elapsed_hours
        == pytest.approx(
            (
                2.5 / 30.0
            )
            + (
                36.0 / 3600.0
            )
        )
    )

    truck = state.trucks[
        0
    ]

    # After policy completion the Core still performs
    # the terminal physical return to the depot.
    assert (
        truck.status
        == TruckStatus.FINISHED
    )

    assert (
        truck.current_node
        == "depot"
    )

    # Total episode distance includes:
    # depot -> bin = 2.5
    # bin -> depot = 2.5
    #
    # The action transition itself ends at physical service
    # completion; episode-level accounting retains the final
    # terminal return separately.
    assert (
        truck.cumulative_distance_km
        == pytest.approx(
            5.0
        )
    )

    assert (
        truck.cumulative_distance_km
        > transition.distance_km
    )


def test_capacity_forced_depot_detour_is_charged_to_pending_dqn_action():
    state = (
        make_capacity_detour_state()
    )

    (
        orchestrator,
        agent,
        training,
    ) = make_orchestrator(
        state
    )

    initial = (
        orchestrator.start()
    )

    assert len(
        initial
    ) == 1

    # DQN chose bin 0, but Core feasibility rejected
    # immediate service because the truck already carries
    # 9.9 t.
    assert (
        initial[
            0
        ].requested_bin_id
        == 0
    )

    assert (
        initial[
            0
        ].action
        == DispatchAction.DEPOT_RETURN
    )

    processed = (
        orchestrator.engine.run(
            max_events=200
        )
    )

    assert processed > 0

    assert (
        orchestrator.completed_services
        == [
            (
                0,
                0,
            )
        ]
    )

    assert len(
        training.transitions
    ) == 1

    assert len(
        agent.replay_buffer
    ) == 1

    transition = (
        training.transitions[
            0
        ]
    )

    assert (
        transition.transition_kind
        == "service_complete"
    )

    # Pending DQN action is retained across the Core-forced
    # capacity recovery:
    #
    # start -> depot = 2 km
    # depot -> bin-0 = 3 km
    #
    # Total action-to-service physical distance = 5 km.
    assert (
        transition.distance_km
        == pytest.approx(
            5.0
        )
    )

    assert (
        transition.collection_fraction
        == pytest.approx(
            0.95
        )
    )

    # R =
    # 0.95 - 0.5 * 5
    # = -1.55
    assert (
        transition.reward
        == pytest.approx(
            -1.55
        )
    )

    # Pure travel + service would be:
    #
    #   5 / 30 h + 36 s
    #
    # Actual elapsed transition time must be larger because
    # physical depot unloading/refuelling occurs in between.
    assert (
        transition.elapsed_hours
        > (
            (
                5.0
                / 30.0
            )
            + (
                36.0
                / 3600.0
            )
        )
    )

    truck = state.trucks[
        0
    ]

    assert (
        truck.status
        == TruckStatus.FINISHED
    )

    assert (
        truck.current_node
        == "depot"
    )

    # Episode total also includes the final 3-km return:
    #
    # 2 + 3 + 3 = 8 km.
    assert (
        truck.cumulative_distance_km
        == pytest.approx(
            8.0
        )
    )

    assert (
        truck.current_load_tonnes
        == pytest.approx(
            0.0
        )
    )

    assert (
        truck.fuel_remaining_litres
        == pytest.approx(
            200.0
        )
    )
