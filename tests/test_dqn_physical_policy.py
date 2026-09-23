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
from smart_waste.rl.state_adapter import (
    DQNPhysicalStateAdapter,
)

import networkx as nx


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


def make_view(
    *,
    fills,
    rates,
    time_hours=0.0,
    reserved=None,
):
    if reserved is None:
        reserved = {}

    bins = tuple(
        BinPolicyView(
            bin_id=index,
            road_node=index + 1,
            fill_percent=float(
                fill
            ),
            fill_rate_percent_per_hour=float(
                rates[
                    index
                ]
            ),
            waste_mass_tonnes=(
                0.44
                * float(
                    fill
                )
                / 100.0
            ),
            reported_fill_percent=float(
                fill
            ),
            reserved_by_truck_id=(
                reserved.get(
                    index
                )
            ),
        )
        for index, fill in enumerate(
            fills
        )
    )

    trucks = (
        TruckPolicyView(
            truck_id=0,
            current_node=0,
            remaining_capacity_tonnes=10.0,
            fuel_remaining_litres=200.0,
            status=TruckStatus.IDLE,
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


def zero_agent(
    *,
    input_dim: int,
    action_dim: int,
) -> DQNAgent:
    agent = DQNAgent(
        input_dim=input_dim,
        action_dim=action_dim,
        seed=123,
        config=DQNConfig(),
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


def test_default_learning_rate_matches_frozen_manuscript_value():
    assert (
        DQNConfig().learning_rate
        == pytest.approx(
            1e-4
        )
    )


def test_primary_state_dimension_is_3002():
    assert (
        DQNPhysicalStateAdapter.expected_dimension(
            1000
        )
        == 3002
    )


def test_state_adapter_builds_expected_small_state():
    graph = make_graph(
        3
    )

    view = make_view(
        fills=[
            90.0,
            80.0,
            50.0,
        ],
        rates=[
            0.0,
            0.0,
            0.0,
        ],
    )

    adapter = (
        DQNPhysicalStateAdapter(
            road_graph=graph,
            initial_hazard_severity={
                1: (
                    2.0
                    / 3.0
                ),
            },
        )
    )

    adapter.initialize(
        view
    )

    observation = (
        adapter.observation(
            view=view,
            truck_id=0,
        )
    )

    assert observation.shape == (
        11,
    )

    # L_t
    assert observation[
        0
    ] == pytest.approx(
        0.0
    )

    assert observation[
        1
    ] == pytest.approx(
        0.0
    )

    # F_t
    assert observation[
        2:5
    ].tolist() == pytest.approx(
        [
            0.9,
            0.8,
            0.5,
        ]
    )

    # H_t
    assert observation[
        5:8
    ].tolist() == pytest.approx(
        [
            0.0,
            2.0 / 3.0,
            0.0,
        ]
    )


def test_full_bin_timer_uses_elapsed_physical_time():
    graph = make_graph(
        1
    )

    initial = make_view(
        fills=[
            90.0,
        ],
        rates=[
            5.0,
        ],
        time_hours=0.0,
    )

    adapter = (
        DQNPhysicalStateAdapter(
            road_graph=graph,
        )
    )

    adapter.initialize(
        initial
    )

    later = make_view(
        fills=[
            100.0,
        ],
        rates=[
            5.0,
        ],
        time_hours=4.0,
    )

    adapter.update(
        later
    )

    # 90 -> 100 at 5 %/h requires 2 hours.
    # Therefore it has been full for the remaining 2 hours.
    assert (
        adapter.full_elapsed_hours(
            0
        )
        == pytest.approx(
            2.0
        )
    )


def test_emergency_hazard_overrides_ordinary_candidates():
    graph = make_graph(
        3
    )

    view = make_view(
        fills=[
            95.0,
            80.0,
            20.0,
        ],
        rates=[
            0.0,
            5.0,
            0.0,
        ],
    )

    agent = zero_agent(
        input_dim=11,
        action_dim=3,
    )

    policy = DQNCollectionPolicy(
        road_graph=graph,
        agent=agent,
        hazard_severity_by_bin={
            2: 1.0,
        },
    )

    policy.initialize(
        view
    )

    assert policy.action_mask(
        view
    ).tolist() == [
        False,
        False,
        True,
    ]


def test_near_full_and_predicted_overflow_are_candidates():
    graph = make_graph(
        3
    )

    view = make_view(
        fills=[
            95.0,
            80.0,
            20.0,
        ],
        rates=[
            0.0,
            5.0,
            0.0,
        ],
    )

    agent = zero_agent(
        input_dim=11,
        action_dim=3,
    )

    policy = DQNCollectionPolicy(
        road_graph=graph,
        agent=agent,
    )

    policy.initialize(
        view
    )

    assert policy.action_mask(
        view
    ).tolist() == [
        True,
        True,
        False,
    ]


def test_reserved_candidate_is_masked_but_not_complete():
    graph = make_graph(
        2
    )

    view = make_view(
        fills=[
            95.0,
            20.0,
        ],
        rates=[
            0.0,
            0.0,
        ],
        reserved={
            0: 1,
        },
    )

    agent = zero_agent(
        input_dim=8,
        action_dim=2,
    )

    policy = DQNCollectionPolicy(
        road_graph=graph,
        agent=agent,
    )

    policy.initialize(
        view
    )

    assert policy.action_mask(
        view
    ).tolist() == [
        False,
        False,
    ]

    # The reserved eligible bin is still unfinished work.
    assert not policy.is_complete(
        view
    )


def test_unverified_bin_is_hidden_and_masked():
    graph = make_graph(
        2
    )

    view = make_view(
        fills=[
            100.0,
            95.0,
        ],
        rates=[
            0.0,
            0.0,
        ],
    )

    agent = zero_agent(
        input_dim=8,
        action_dim=2,
    )

    policy = DQNCollectionPolicy(
        road_graph=graph,
        agent=agent,
        verified_by_bin={
            0: False,
            1: True,
        },
    )

    policy.initialize(
        view
    )

    observation = (
        policy.state_adapter.observation(
            view=view,
            truck_id=0,
        )
    )

    assert policy.action_mask(
        view
    ).tolist() == [
        False,
        True,
    ]

    # First fill feature belongs to unverified bin 0.
    assert observation[
        2
    ] == pytest.approx(
        0.0
    )


def test_greedy_policy_returns_lowest_action_when_q_values_tie():
    graph = make_graph(
        3
    )

    view = make_view(
        fills=[
            95.0,
            95.0,
            20.0,
        ],
        rates=[
            0.0,
            0.0,
            0.0,
        ],
    )

    agent = zero_agent(
        input_dim=11,
        action_dim=3,
    )

    policy = DQNCollectionPolicy(
        road_graph=graph,
        agent=agent,
        epsilon=0.0,
    )

    policy.initialize(
        view
    )

    selected = (
        policy.select_next_bin(
            view=view,
            truck_id=0,
        )
    )

    assert selected == 0
