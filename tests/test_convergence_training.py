import numpy as np

from smart_waste.rl.convergence_training import (
    GreedyEvaluation,
    train_dqn_until_convergence,
)
from smart_waste.rl.dqn import (
    DQNAgent,
    DQNConfig,
)
from smart_waste.rl.environment import (
    SmartWasteRoutingEnv,
)


def make_env():

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
                94.0,
                93.0,
                92.0,
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
        prediction_horizon_hours=1.0,
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
        epsilon_decay_episodes=4,
    )

    return DQNAgent(
        input_dim=(
            env.observation_space.shape[0]
        ),
        action_dim=(
            env.action_space.n
        ),
        seed=123,
        config=config,
        device="cpu",
    )


def stable_evaluator(
    agent,
    episodes_completed,
):

    return GreedyEvaluation(
        episode_completed=(
            episodes_completed
        ),
        steps=4,
        total_reward=1.0,
        distance_km=10.0,
        fuel_litres=4.0,
        simulated_time_hours=4.0,
        terminated=True,
        truncated=False,
    )


def changing_evaluator(
    agent,
    episodes_completed,
):

    return GreedyEvaluation(
        episode_completed=(
            episodes_completed
        ),
        steps=4,
        total_reward=1.0,
        distance_km=float(
            episodes_completed
        ),
        fuel_litres=float(
            episodes_completed
        ),
        simulated_time_hours=4.0,
        terminated=True,
        truncated=False,
    )


def test_convergence_can_stop_before_maximum():

    env = make_env()
    agent = make_agent(env)

    result = train_dqn_until_convergence(
        env=env,
        agent=agent,
        master_seed=123,
        minimum_episodes=4,
        maximum_episodes=10,
        window_size=2,
        patience_windows=2,
        reward_absolute_slope_threshold=1000.0,
        loss_absolute_slope_threshold=1000.0,
        greedy_evaluation_interval_episodes=2,
        greedy_relative_change_threshold=0.01,
        greedy_evaluator=stable_evaluator,
    )

    assert result.converged
    assert result.episodes_completed == 6

    assert (
        result.stop_reason
        == "convergence_criteria_satisfied"
    )


def test_nonconverged_training_reaches_maximum():

    env = make_env()
    agent = make_agent(env)

    result = train_dqn_until_convergence(
        env=env,
        agent=agent,
        master_seed=123,
        minimum_episodes=4,
        maximum_episodes=8,
        window_size=2,
        patience_windows=2,
        reward_absolute_slope_threshold=1000.0,
        loss_absolute_slope_threshold=1000.0,
        greedy_evaluation_interval_episodes=2,
        greedy_relative_change_threshold=0.000001,
        greedy_evaluator=changing_evaluator,
    )

    assert not result.converged
    assert result.episodes_completed == 8

    assert (
        result.stop_reason
        == "maximum_episodes_reached"
    )


def test_checkpoint_evaluations_are_recorded():

    env = make_env()
    agent = make_agent(env)

    result = train_dqn_until_convergence(
        env=env,
        agent=agent,
        master_seed=123,
        minimum_episodes=4,
        maximum_episodes=6,
        window_size=2,
        patience_windows=2,
        reward_absolute_slope_threshold=1000.0,
        loss_absolute_slope_threshold=1000.0,
        greedy_evaluation_interval_episodes=2,
        greedy_relative_change_threshold=0.01,
        greedy_evaluator=stable_evaluator,
    )

    completed = [
        item.episode_completed
        for item in result.evaluations
    ]

    assert completed == [
        2,
        4,
        6,
    ]

    assert len(
        result.convergence_checks
    ) == 2