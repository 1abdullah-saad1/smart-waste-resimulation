from __future__ import annotations

import numpy as np

from smart_waste.rl.dqn import (
    DQNAgent,
    DQNConfig,
)
from smart_waste.rl.environment import (
    SmartWasteRoutingEnv,
)


def main():

    rng = np.random.default_rng(20260922)

    num_bins = 12

    bin_xy = rng.uniform(
        0.0,
        5.0,
        size=(num_bins, 2),
    )

    fill = np.array(
        [
            95.0,
            92.0,
            96.0,
            91.0,
            94.0,
            93.0,
            97.0,
            90.0,
            95.0,
            92.0,
            98.0,
            91.0,
        ],
        dtype=np.float64,
    )

    fill_rates = np.zeros(
        num_bins,
        dtype=np.float64,
    )

    hazards = np.zeros(
        num_bins,
        dtype=np.float64,
    )

    # One emergency hazard to test hazard override.
    hazards[5] = 1.0

    env = SmartWasteRoutingEnv(
        bin_xy=bin_xy,
        depot_xy=np.array(
            [2.5, 2.5],
            dtype=np.float64,
        ),
        initial_fill_percent=fill,
        fill_rate_percent_per_hour=fill_rates,
        hazard_severity=hazards,
        prediction_horizon_hours=4.0,
        decision_interval_hours=1.0,
        num_trucks=2,
        max_steps=30,
    )

    observation, info = env.reset(
        seed=20260922
    )

    config = DQNConfig(
        replay_capacity=100,
        batch_size=4,
        replay_warmup_transitions=8,
        target_update_steps=5,
        epsilon_decay_episodes=10,
    )

    agent = DQNAgent(
        input_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        seed=20260922,
        config=config,
        device="cpu",
    )

    total_reward = 0.0
    losses = []

    terminated = False
    truncated = False

    step = 0

    while not (
        terminated
        or truncated
    ):

        epsilon = 0.50

        action = agent.select_action(
            observation,
            info["action_mask"],
            epsilon=epsilon,
        )

        (
            next_observation,
            reward,
            terminated,
            truncated,
            next_info,
        ) = env.step(action)

        agent.remember(
            state=observation,
            action=action,
            reward=reward,
            next_state=next_observation,
            terminal=(
                terminated
                or truncated
            ),
            next_action_mask=(
                next_info[
                    "action_mask"
                ]
            ),
        )

        loss = agent.optimize()

        if loss is not None:
            losses.append(loss)

        total_reward += reward

        observation = next_observation
        info = next_info

        step += 1

        print(
            f"step={step:02d} "
            f"action={action:02d} "
            f"reward={reward:9.4f} "
            f"candidates={info['candidate_count']:02d} "
            f"fuel={info['total_fuel_litres']:.4f} L "
            f"loss={loss}"
        )

    print()
    print("=" * 80)
    print("SMALL DQN SMOKE TEST PASSED")
    print("=" * 80)

    print(
        "Observation dimension:",
        env.observation_space.shape[0],
    )

    print(
        "Action dimension:",
        env.action_space.n,
    )

    print(
        "Steps:",
        step,
    )

    print(
        "Replay transitions:",
        len(agent.replay_buffer),
    )

    print(
        "Optimization steps:",
        agent.optimization_steps,
    )

    print(
        f"Total reward: {total_reward:.4f}"
    )

    print(
        f"Total distance: "
        f"{env.total_distance_km:.4f} km"
    )

    print(
        f"Total fuel: "
        f"{env.total_fuel_litres:.4f} L"
    )

    if losses:
        print(
            f"Last loss: {losses[-1]:.6f}"
        )

    assert step > 0
    assert len(agent.replay_buffer) > 0
    assert agent.optimization_steps > 0
    assert np.isfinite(total_reward)

    if losses:
        assert np.all(
            np.isfinite(losses)
        )


if __name__ == "__main__":
    main()