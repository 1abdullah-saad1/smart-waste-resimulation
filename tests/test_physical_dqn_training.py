import networkx as nx
import numpy as np
import pytest
import torch

from smart_waste.collection.base import (
    BinPolicyView,
    PolicyView,
    TruckPolicyView,
)
from smart_waste.collection.dqn import (
    DQNCollectionPolicy,
)
from smart_waste.models.truck import (
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


def make_graph(
    num_bins: int,
) -> RoadGraph:
    graph = nx.Graph()

    for node in range(
        num_bins + 1
    ):
        graph.add_node(
            node,
            x_km=float(
                node
            ),
            y_km=0.0,
        )

    for node in range(
        num_bins
    ):
        graph.add_edge(
            node,
            node + 1,
            length_km=1.0,
        )

    return RoadGraph(
        graph
    )


def make_agent(
    *,
    input_dim: int,
    action_dim: int,
) -> DQNAgent:
    agent = DQNAgent(
        input_dim=input_dim,
        action_dim=action_dim,
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


def make_view(
    *,
    fills,
    rates,
    time_hours=0.0,
    truck_node=0,
    cumulative_distance_km=0.0,
    remaining_capacity_tonnes=10.0,
    reserved=None,
):
    if reserved is None:
        reserved = {}

    bins = tuple(
        BinPolicyView(
            bin_id=index,
            road_node=index + 1,
            fill_percent=float(
                fills[
                    index
                ]
            ),
            fill_rate_percent_per_hour=float(
                rates[
                    index
                ]
            ),
            waste_mass_tonnes=(
                0.44
                * float(
                    fills[
                        index
                    ]
                )
                / 100.0
            ),
            reported_fill_percent=float(
                fills[
                    index
                ]
            ),
            reserved_by_truck_id=(
                reserved.get(
                    index
                )
            ),
        )
        for index in range(
            len(
                fills
            )
        )
    )

    trucks = (
        TruckPolicyView(
            truck_id=0,
            current_node=truck_node,
            remaining_capacity_tonnes=float(
                remaining_capacity_tonnes
            ),
            fuel_remaining_litres=200.0,
            status=TruckStatus.IDLE,
            cumulative_distance_km=float(
                cumulative_distance_km
            ),
        ),
    )

    return PolicyView(
        current_time_hours=float(
            time_hours
        ),
        depot_node=0,
        bins=bins,
        trucks=trucks,
    )


def make_training_policy(
    *,
    graph,
    num_bins,
    hazards=None,
    initial_full_elapsed=None,
):
    agent = make_agent(
        input_dim=(
            2
            + 3 * num_bins
        ),
        action_dim=num_bins,
    )

    policy = DQNCollectionPolicy(
        road_graph=graph,
        agent=agent,
        epsilon=0.0,
        hazard_severity_by_bin=(
            hazards
        ),
        initial_full_elapsed_hours=(
            initial_full_elapsed
        ),
    )

    training = (
        PhysicalDQNTrainingPolicy(
            policy=policy,
        )
    )

    return (
        agent,
        policy,
        training,
    )


def test_successful_service_creates_replay_transition():
    graph = make_graph(
        2
    )

    (
        agent,
        _,
        training,
    ) = make_training_policy(
        graph=graph,
        num_bins=2,
    )

    initial = make_view(
        fills=[
            95.0,
            20.0,
        ],
        rates=[
            0.0,
            0.0,
        ],
    )

    training.initialize(
        initial
    )

    action = (
        training.select_next_bin(
            view=initial,
            truck_id=0,
        )
    )

    assert action == 0

    completed = make_view(
        fills=[
            0.0,
            20.0,
        ],
        rates=[
            0.0,
            0.0,
        ],
        time_hours=1.0,
        truck_node=1,
        cumulative_distance_km=1.0,
        remaining_capacity_tonnes=(
            10.0
            - 0.44 * 0.95
        ),
    )

    training.on_service_complete(
        view=completed,
        truck_id=0,
        bin_id=0,
    )

    assert len(
        agent.replay_buffer
    ) == 1

    assert len(
        training.transitions
    ) == 1


def test_reward_uses_exact_core_distance_and_physical_collection():
    graph = make_graph(
        2
    )

    (
        _,
        _,
        training,
    ) = make_training_policy(
        graph=graph,
        num_bins=2,
    )

    initial = make_view(
        fills=[
            95.0,
            20.0,
        ],
        rates=[
            0.0,
            0.0,
        ],
    )

    training.initialize(
        initial
    )

    assert (
        training.select_next_bin(
            view=initial,
            truck_id=0,
        )
        == 0
    )

    completed = make_view(
        fills=[
            0.0,
            20.0,
        ],
        rates=[
            0.0,
            0.0,
        ],
        time_hours=1.0,
        truck_node=1,
        cumulative_distance_km=1.25,
    )

    training.on_service_complete(
        view=completed,
        truck_id=0,
        bin_id=0,
    )

    record = (
        training.transitions[
            0
        ]
    )

    assert record.distance_km == pytest.approx(
        1.25
    )

    assert (
        record.collection_fraction
        == pytest.approx(
            0.95
        )
    )

    assert (
        record.distance_km
        == pytest.approx(
            1.25
        )
    )

    assert (
        record.terminal_return_distance_km
        == pytest.approx(
            1.0
        )
    )

    assert (
        record.reward_distance_km
        == pytest.approx(
            2.25
        )
    )

    # 1*0.95 - 0.5*(1.25 + 1.0)
    assert record.reward == pytest.approx(
        -0.175
    )


def test_collection_fraction_uses_physical_fill_growth():
    graph = make_graph(
        1
    )

    (
        _,
        _,
        training,
    ) = make_training_policy(
        graph=graph,
        num_bins=1,
    )

    initial = make_view(
        fills=[
            90.0,
        ],
        rates=[
            5.0,
        ],
    )

    training.initialize(
        initial
    )

    assert (
        training.select_next_bin(
            view=initial,
            truck_id=0,
        )
        == 0
    )

    completed = make_view(
        fills=[
            0.0,
        ],
        rates=[
            5.0,
        ],
        time_hours=1.0,
        truck_node=1,
        cumulative_distance_km=1.0,
    )

    training.on_service_complete(
        view=completed,
        truck_id=0,
        bin_id=0,
    )

    assert (
        training.transitions[
            0
        ].collection_fraction
        == pytest.approx(
            0.95
        )
    )


def test_depot_recovery_keeps_same_pending_action():
    graph = make_graph(
        2
    )

    (
        agent,
        _,
        training,
    ) = make_training_policy(
        graph=graph,
        num_bins=2,
    )

    initial = make_view(
        fills=[
            95.0,
            20.0,
        ],
        rates=[
            0.0,
            0.0,
        ],
    )

    training.initialize(
        initial
    )

    assert (
        training.select_next_bin(
            view=initial,
            truck_id=0,
        )
        == 0
    )

    # Core has physically returned truck to depot.
    after_depot = make_view(
        fills=[
            95.0,
            20.0,
        ],
        rates=[
            0.0,
            0.0,
        ],
        time_hours=2.0,
        truck_node=0,
        cumulative_distance_km=2.0,
    )

    # Same unresolved action is retained.
    assert (
        training.select_next_bin(
            view=after_depot,
            truck_id=0,
        )
        == 0
    )

    assert len(
        agent.replay_buffer
    ) == 0

    completed = make_view(
        fills=[
            0.0,
            20.0,
        ],
        rates=[
            0.0,
            0.0,
        ],
        time_hours=3.0,
        truck_node=1,
        cumulative_distance_km=3.0,
    )

    training.on_service_complete(
        view=completed,
        truck_id=0,
        bin_id=0,
    )

    record = (
        training.transitions[
            0
        ]
    )

    assert record.elapsed_hours == pytest.approx(
        3.0
    )

    assert record.distance_km == pytest.approx(
        3.0
    )

    assert (
        record.transition_kind
        == "service_complete"
    )


def test_unavailable_pending_target_is_closed_without_collection():
    graph = make_graph(
        2
    )

    (
        agent,
        _,
        training,
    ) = make_training_policy(
        graph=graph,
        num_bins=2,
    )

    initial = make_view(
        fills=[
            95.0,
            95.0,
        ],
        rates=[
            0.0,
            0.0,
        ],
    )

    training.initialize(
        initial
    )

    assert (
        training.select_next_bin(
            view=initial,
            truck_id=0,
        )
        == 0
    )

    changed = make_view(
        fills=[
            95.0,
            95.0,
        ],
        rates=[
            0.0,
            0.0,
        ],
        time_hours=1.0,
        truck_node=0,
        cumulative_distance_km=1.0,
        reserved={
            0: 1,
        },
    )

    # Pending bin 0 is no longer selectable, so its transition
    # closes with zero collection and a new action is selected.
    new_action = (
        training.select_next_bin(
            view=changed,
            truck_id=0,
        )
    )

    assert new_action == 1

    assert len(
        agent.replay_buffer
    ) == 1

    record = (
        training.transitions[
            0
        ]
    )

    assert (
        record.transition_kind
        == "interrupted_before_service"
    )

    assert (
        record.collection_fraction
        == pytest.approx(
            0.0
        )
    )

    assert (
        record.distance_km
        == pytest.approx(
            1.0
        )
    )


def test_hazard_delay_uses_actual_elapsed_time_and_remaining_hazards():
    graph = make_graph(
        2
    )

    (
        _,
        _,
        training,
    ) = make_training_policy(
        graph=graph,
        num_bins=2,
        hazards={
            0: 1.0,
            1: (
                2.0
                / 3.0
            ),
        },
    )

    initial = make_view(
        fills=[
            20.0,
            20.0,
        ],
        rates=[
            0.0,
            0.0,
        ],
    )

    training.initialize(
        initial
    )

    # Emergency hazard on bin 0 forces action 0.
    assert (
        training.select_next_bin(
            view=initial,
            truck_id=0,
        )
        == 0
    )

    completed = make_view(
        fills=[
            0.0,
            20.0,
        ],
        rates=[
            0.0,
            0.0,
        ],
        time_hours=1.5,
        truck_node=1,
        cumulative_distance_km=1.0,
    )

    training.on_service_complete(
        view=completed,
        truck_id=0,
        bin_id=0,
    )

    record = (
        training.transitions[
            0
        ]
    )

    # Bin 0 hazard is cleared at service; bin 1 remains at 2/3.
    assert (
        record.hazard_delay_term
        == pytest.approx(
            (
                2.0
                / 3.0
            )
            * 1.5
        )
    )


def test_sla_violation_is_taken_from_physical_elapsed_state():
    graph = make_graph(
        2
    )

    (
        _,
        _,
        training,
    ) = make_training_policy(
        graph=graph,
        num_bins=2,
        hazards={
            0: 1.0,
        },
        initial_full_elapsed={
            1: 6.5,
        },
    )

    initial = make_view(
        fills=[
            20.0,
            100.0,
        ],
        rates=[
            0.0,
            0.0,
        ],
    )

    training.initialize(
        initial
    )

    assert (
        training.select_next_bin(
            view=initial,
            truck_id=0,
        )
        == 0
    )

    completed = make_view(
        fills=[
            0.0,
            100.0,
        ],
        rates=[
            0.0,
            0.0,
        ],
        time_hours=0.5,
        truck_node=1,
        cumulative_distance_km=1.0,
    )

    training.on_service_complete(
        view=completed,
        truck_id=0,
        bin_id=0,
    )

    assert (
        training.transitions[
            0
        ].sla_violation_term
        == pytest.approx(
            1.0
        )
    )
