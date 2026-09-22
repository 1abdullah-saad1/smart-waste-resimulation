import numpy as np

from smart_waste.rl.dqn import (
    DQNAgent,
    DQNConfig,
)
from smart_waste.rl.environment import (
    SmartWasteRoutingEnv,
)
from smart_waste.rl.training import (
    train_dqn,
)


def make_training_env():
    return SmartWasteRoutingEnv(
        bin_xy=np.array(
            [
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
                [4.0, 0.0],
            ],
            dtype=np.float64,
        ),
        depot_xy=np.array(
            [0.0, 0.0],
            dtype=np.float64,
        ),
        initial_fill_percent=np.array(
            [
                95.0,
                92.0,
                91.0,
                90.0,
            ],
            dtype=np.float64,
        ),
        fill_rate_percent_per_hour=np.zeros(
            4,
            dtype=np.float64,
        ),
        hazard_severity=np.zeros(
            4,
            dtype=np.float64,
        ),
        prediction_horizon_hours=4.0,
        decision_interval_hours=1.0,
        num_trucks=1,
        max_steps=10,
    )


def make_agent(env):
    config = DQNConfig(
        replay_capacity=100,
        batch_size=2,
        replay_warmup_transitions=2,
        target_update_steps=5,
        epsilon_decay_episodes=5,
    )

    return DQNAgent(
        input_dim=(
            env.observation_space.shape[0]
        ),
        action_dim=(
            env.action_space.n
        ),
        seed=20260922,
        config=config,
        device="cpu",
    )


def test_training_returns_requested_episode_count():

    env = make_training_env()
    agent = make_agent(env)

    result = train_dqn(
        env=env,
        agent=agent,
        episodes=5,
        master_seed=20260922,
    )

    assert len(
        result.episodes
    ) == 5


def test_training_episode_metrics_are_finite():

    env = make_training_env()
    agent = make_agent(env)

    result = train_dqn(
        env=env,
        agent=agent,
        episodes=5,
        master_seed=20260922,
    )

    for episode in result.episodes:

        assert np.isfinite(
            episode.total_reward
        )

        assert np.isfinite(
            episode.distance_km
        )

        assert np.isfinite(
            episode.fuel_litres
        )

        assert episode.steps > 0


def test_epsilon_decays_across_training():

    env = make_training_env()
    agent = make_agent(env)

    result = train_dqn(
        env=env,
        agent=agent,
        episodes=5,
        master_seed=20260922,
    )

    epsilons = [
        episode.epsilon
        for episode
        in result.episodes
    ]

    assert epsilons[0] == 1.0
    assert epsilons[-1] == 0.01

    assert all(
        earlier >= later
        for earlier, later
        in zip(
            epsilons,
            epsilons[1:],
        )
    )


def test_training_performs_optimization():

    env = make_training_env()
    agent = make_agent(env)

    result = train_dqn(
        env=env,
        agent=agent,
        episodes=5,
        master_seed=20260922,
    )

    total_updates = sum(
        episode.optimization_steps
        for episode
        in result.episodes
    )

    assert total_updates > 0


def test_training_records_can_be_serialized():

    env = make_training_env()
    agent = make_agent(env)

    result = train_dqn(
        env=env,
        agent=agent,
        episodes=2,
        master_seed=20260922,
    )

    records = result.to_records()

    assert len(records) == 2

    assert "episode" in records[0]
    assert "epsilon" in records[0]
    assert "total_reward" in records[0]
    assert "mean_loss" in records[0]
    assert "fuel_litres" in records[0]