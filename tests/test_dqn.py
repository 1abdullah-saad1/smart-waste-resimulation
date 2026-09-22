import numpy as np
import pytest
import torch
from torch import nn

from smart_waste.rl.dqn import (
    DQNAgent,
    DQNConfig,
    QNetwork,
    ReplayBuffer,
    linear_epsilon,
)


def test_network_output_shape():
    network = QNetwork(
        input_dim=11,
        action_dim=3,
    )

    x = torch.zeros(
        (4, 11)
    )

    output = network(x)

    assert output.shape == (
        4,
        3,
    )


def test_network_has_manuscript_hidden_layers():
    network = QNetwork(
        input_dim=11,
        action_dim=3,
    )

    linear_layers = [
        layer
        for layer in network.network
        if isinstance(
            layer,
            nn.Linear,
        )
    ]

    dimensions = [
        (
            layer.in_features,
            layer.out_features,
        )
        for layer in linear_layers
    ]

    assert dimensions == [
        (11, 256),
        (256, 128),
        (128, 64),
        (64, 3),
    ]


def test_epsilon_starts_at_one():
    assert linear_epsilon(
        0
    ) == pytest.approx(
        1.0
    )


def test_epsilon_ends_at_point_zero_one():
    assert linear_epsilon(
        99
    ) == pytest.approx(
        0.01
    )


def test_replay_buffer_sample_shapes():
    buffer = ReplayBuffer(
        capacity=10,
        action_dim=3,
    )

    rng = np.random.default_rng(
        123
    )

    for index in range(4):

        buffer.append(
            state=np.array(
                [index, 0.0],
                dtype=np.float32,
            ),
            action=index % 3,
            reward=float(index),
            next_state=np.array(
                [index + 1, 0.0],
                dtype=np.float32,
            ),
            terminal=False,
            next_action_mask=np.array(
                [True, False, True]
            ),
        )

    batch = buffer.sample(
        batch_size=2,
        rng=rng,
    )

    assert batch[
        "states"
    ].shape == (
        2,
        2,
    )

    assert batch[
        "next_action_masks"
    ].shape == (
        2,
        3,
    )


def test_replay_buffer_respects_capacity():
    buffer = ReplayBuffer(
        capacity=3,
        action_dim=2,
    )

    for index in range(10):

        buffer.append(
            state=np.array(
                [index],
                dtype=np.float32,
            ),
            action=0,
            reward=0.0,
            next_state=np.array(
                [index + 1],
                dtype=np.float32,
            ),
            terminal=False,
            next_action_mask=np.array(
                [True, True]
            ),
        )

    assert len(buffer) == 3


def test_masked_greedy_never_selects_invalid_action():
    config = DQNConfig(
        replay_warmup_transitions=2,
        batch_size=2,
    )

    agent = DQNAgent(
        input_dim=4,
        action_dim=3,
        seed=123,
        config=config,
    )

    action = agent.select_action(
        np.zeros(
            4,
            dtype=np.float32,
        ),
        np.array(
            [False, True, False]
        ),
        epsilon=0.0,
    )

    assert action == 1


def test_random_policy_only_uses_valid_actions():
    agent = DQNAgent(
        input_dim=4,
        action_dim=4,
        seed=123,
    )

    mask = np.array(
        [
            False,
            True,
            False,
            True,
        ]
    )

    actions = {
        agent.select_action(
            np.zeros(
                4,
                dtype=np.float32,
            ),
            mask,
            epsilon=1.0,
        )
        for _ in range(100)
    }

    assert actions.issubset(
        {1, 3}
    )


def test_optimize_returns_finite_loss():
    config = DQNConfig(
        replay_capacity=20,
        batch_size=2,
        replay_warmup_transitions=2,
        target_update_steps=10,
    )

    agent = DQNAgent(
        input_dim=4,
        action_dim=3,
        seed=123,
        config=config,
    )

    for index in range(4):

        state = np.array(
            [
                index,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )

        next_state = np.array(
            [
                index + 1,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )

        agent.remember(
            state=state,
            action=index % 3,
            reward=1.0,
            next_state=next_state,
            terminal=False,
            next_action_mask=np.array(
                [True, True, True]
            ),
        )

    loss = agent.optimize()

    assert loss is not None
    assert np.isfinite(loss)


def test_target_network_can_be_synchronized():
    config = DQNConfig(
        replay_capacity=10,
        batch_size=2,
        replay_warmup_transitions=2,
        target_update_steps=1,
    )

    agent = DQNAgent(
        input_dim=4,
        action_dim=2,
        seed=123,
        config=config,
    )

    for index in range(2):

        agent.remember(
            state=np.zeros(
                4,
                dtype=np.float32,
            ),
            action=0,
            reward=1.0,
            next_state=np.ones(
                4,
                dtype=np.float32,
            ),
            terminal=True,
            next_action_mask=np.array(
                [False, False]
            ),
        )

    agent.optimize()

    online = list(
        agent.online_network.parameters()
    )

    target = list(
        agent.target_network.parameters()
    )

    assert all(
        torch.equal(a, b)
        for a, b in zip(
            online,
            target,
        )
    )


def test_zero_epsilon_does_not_advance_exploration_rng():

    first = DQNAgent(
        input_dim=4,
        action_dim=4,
        seed=12345,
    )

    second = DQNAgent(
        input_dim=4,
        action_dim=4,
        seed=12345,
    )

    observation = np.zeros(
        4,
        dtype=np.float32,
    )

    mask = np.ones(
        4,
        dtype=bool,
    )

    # Greedy evaluation must not consume exploration RNG.
    first.select_action(
        observation,
        mask,
        epsilon=0.0,
    )

    first_random = [
        first.select_action(
            observation,
            mask,
            epsilon=1.0,
        )
        for _ in range(20)
    ]

    second_random = [
        second.select_action(
            observation,
            mask,
            epsilon=1.0,
        )
        for _ in range(20)
    ]

    assert first_random == second_random
