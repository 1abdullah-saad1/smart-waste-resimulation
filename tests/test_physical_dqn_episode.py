import networkx as nx
import pytest
import torch

from smart_waste.experiments.scenario_snapshot import (
    capture_physical_scenario,
)
from smart_waste.models.bin import (
    WasteBin,
)
from smart_waste.models.depot import (
    Depot,
)
from smart_waste.models.truck import (
    Truck,
)
from smart_waste.movement.road_graph import (
    RoadGraph,
)
from smart_waste.rl.dqn import (
    DQNAgent,
    DQNConfig,
)
from smart_waste.rl.physical_episode import (
    run_physical_dqn_episode,
)
from smart_waste.simulation.state import (
    SimulationState,
)


def make_snapshot():
    graph = nx.Graph()

    graph.add_node(
        "depot",
        x=0.0,
        y=0.0,
    )

    graph.add_node(
        "bin-0",
        x=2.0,
        y=0.0,
    )

    graph.add_edge(
        "depot",
        "bin-0",
        length_km=2.0,
    )

    state = SimulationState(
        road_graph=RoadGraph(
            graph
        ),
        bins={
            0: WasteBin(
                bin_id=0,
                road_node="bin-0",
                fill_percent=95.0,
                fill_rate_percent_per_hour=0.0,
                full_mass_kg=440.0,
            ),
        },
        trucks={
            0: Truck(
                truck_id=0,
                current_node="depot",
                capacity_tonnes=10.0,
                speed_km_per_hour=30.0,
                fuel_capacity_litres=200.0,
                fuel_efficiency_km_per_litre=2.5,
            ),
        },
        depot=Depot(
            road_node="depot",
            unloading_bays=1,
            unload_time_minutes=11.0,
            refuel_rate_litres_per_minute=60.0,
        ),
    )

    return capture_physical_scenario(
        state
    )


def zero_agent():
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


def test_episode_runner_reaches_natural_completion():
    snapshot = make_snapshot()

    agent = zero_agent()

    result = run_physical_dqn_episode(
        snapshot=snapshot,
        scenario_id="synthetic-one-bin",
        scenario_sha256="test-sha",
        agent=agent,
        episode_index=0,
        epsilon=0.0,
        optimize_after_transition=False,
    )

    assert (
        result.natural_completion
        is True
    )

    assert (
        result.policy_complete
        is True
    )

    assert (
        result.event_queue_empty
        is True
    )

    assert (
        result.all_trucks_finished
        is True
    )

    assert (
        result.reservations_remaining
        == 0
    )

    assert (
        result.completed_services
        == 1
    )

    assert (
        result.replay_transitions
        == 1
    )


def test_episode_result_separates_action_and_terminal_distance():
    snapshot = make_snapshot()

    agent = zero_agent()

    result = run_physical_dqn_episode(
        snapshot=snapshot,
        scenario_id="synthetic-one-bin",
        agent=agent,
        episode_index=0,
        epsilon=0.0,
        optimize_after_transition=False,
    )

    # Action transition:
    # depot -> bin = 2 km.
    assert (
        result.action_distance_km
        == pytest.approx(
            2.0
        )
    )

    # Complete physical episode:
    # depot -> bin -> depot = 4 km.
    assert (
        result.fleet_distance_km
        == pytest.approx(
            4.0
        )
    )

    assert (
        result.terminal_overhead_distance_km
        == pytest.approx(
            2.0
        )
    )

    assert (
        result.terminal_return_distance_km
        == pytest.approx(
            2.0
        )
    )

    assert (
        result.reward_distance_km
        == pytest.approx(
            4.0
        )
    )

    # 4 km / 2.5 km/L.
    assert (
        result.fuel_litres
        == pytest.approx(
            1.6
        )
    )


def test_fresh_physical_state_is_rebuilt_each_episode():
    snapshot = make_snapshot()

    agent = zero_agent()

    first = run_physical_dqn_episode(
        snapshot=snapshot,
        scenario_id="synthetic-one-bin",
        agent=agent,
        episode_index=0,
        epsilon=0.0,
        optimize_after_transition=False,
    )

    second = run_physical_dqn_episode(
        snapshot=snapshot,
        scenario_id="synthetic-one-bin",
        agent=agent,
        episode_index=1,
        epsilon=0.0,
        optimize_after_transition=False,
    )

    assert first.completed_services == 1
    assert second.completed_services == 1

    assert (
        first.fleet_distance_km
        == pytest.approx(
            second.fleet_distance_km
        )
    )

    assert (
        first.fuel_litres
        == pytest.approx(
            second.fuel_litres
        )
    )

    assert (
        first.completion_time_hours
        == pytest.approx(
            second.completion_time_hours
        )
    )


def test_shared_agent_replay_persists_across_episodes():
    snapshot = make_snapshot()

    agent = zero_agent()

    assert len(
        agent.replay_buffer
    ) == 0

    run_physical_dqn_episode(
        snapshot=snapshot,
        scenario_id="synthetic-one-bin",
        agent=agent,
        episode_index=0,
        epsilon=0.0,
        optimize_after_transition=False,
    )

    assert len(
        agent.replay_buffer
    ) == 1

    run_physical_dqn_episode(
        snapshot=snapshot,
        scenario_id="synthetic-one-bin",
        agent=agent,
        episode_index=1,
        epsilon=0.0,
        optimize_after_transition=False,
    )

    assert len(
        agent.replay_buffer
    ) == 2


def test_default_episode_epsilon_comes_from_agent_schedule():
    snapshot = make_snapshot()

    agent = zero_agent()

    result = run_physical_dqn_episode(
        snapshot=snapshot,
        scenario_id="synthetic-one-bin",
        agent=agent,
        episode_index=99,
        epsilon=None,
        optimize_after_transition=False,
    )

    assert (
        result.epsilon
        == pytest.approx(
            0.01
        )
    )


def test_episode_reward_uses_action_distance_not_terminal_return():
    snapshot = make_snapshot()

    agent = zero_agent()

    result = run_physical_dqn_episode(
        snapshot=snapshot,
        scenario_id="synthetic-one-bin",
        agent=agent,
        episode_index=0,
        epsilon=0.0,
        optimize_after_transition=False,
    )

    # Collection = 0.95
    # Outbound action distance = 2 km.
    # Terminal return = 2 km.
    #
    # Reward =
    # 0.95 - 0.5*(2 + 2)
    # = -1.05
    #
    # This preserves the terminal-distance behavior of the
    # legacy DQN environment while using the physical road Core.
    assert (
        result.total_reward
        == pytest.approx(
            -1.05
        )
    )

    assert (
        result.reward_distance_km
        == pytest.approx(
            result.fleet_distance_km
        )
    )
