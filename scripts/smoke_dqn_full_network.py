from __future__ import annotations

import time

import numpy as np
import torch

from smart_waste.rl.dqn import (
    DQNAgent,
    DQNConfig,
)
from smart_waste.rl.environment import (
    SmartWasteRoutingEnv,
)


MASTER_SEED = 20260922
NUM_BINS = 1000


def main():

    rng = np.random.default_rng(
        MASTER_SEED
    )

    side_km = np.sqrt(50.0)

    bin_xy = rng.uniform(
        0.0,
        side_km,
        size=(NUM_BINS, 2),
    )

    depot_xy = np.array(
        [
            side_km / 2.0,
            side_km / 2.0,
        ],
        dtype=np.float64,
    )

    initial_fill = rng.uniform(
        0.0,
        100.0,
        size=NUM_BINS,
    )

    # Smoke-test values only.
    # These are NOT final experimental parameters.
    fill_rates = np.zeros(
        NUM_BINS,
        dtype=np.float64,
    )

    hazards = np.zeros(
        NUM_BINS,
        dtype=np.float64,
    )

    verified = np.ones(
        NUM_BINS,
        dtype=bool,
    )

    env = SmartWasteRoutingEnv(
        bin_xy=bin_xy,
        depot_xy=depot_xy,
        initial_fill_percent=initial_fill,
        fill_rate_percent_per_hour=fill_rates,
        hazard_severity=hazards,
        verified_mask=verified,
        prediction_horizon_hours=4.0,
        decision_interval_hours=1.0,
        num_trucks=10,
        truck_capacity_tonnes=10.0,
        fuel_efficiency_km_per_litre=2.5,
        max_steps=1000,
    )

    observation, info = env.reset(
        seed=MASTER_SEED
    )

    config = DQNConfig(
        replay_capacity=1000,
        batch_size=64,
        replay_warmup_transitions=64,
        target_update_steps=250,
    )

    agent = DQNAgent(
        input_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        seed=MASTER_SEED,
        config=config,
        device="cpu",
    )

    print("=" * 84)
    print(
        "FULL 1000-BIN DQN NETWORK SMOKE TEST"
    )
    print("=" * 84)

    print(
        "Observation dimension:",
        env.observation_space.shape[0],
    )

    print(
        "Action dimension:",
        env.action_space.n,
    )

    print(
        "Initial candidate count:",
        info["candidate_count"],
    )

    parameter_count = sum(
        parameter.numel()
        for parameter
        in agent.online_network.parameters()
    )

    print(
        "Online-network parameters:",
        f"{parameter_count:,}",
    )

    network_mb = (
        parameter_count
        * 4
        / 1024**2
    )

    print(
        "Approx. FP32 parameters/network:",
        f"{network_mb:.2f} MiB",
    )

    # --------------------------------------------------
    # Forward + masked action test
    # --------------------------------------------------

    start = time.perf_counter()

    action = agent.select_action(
        observation,
        info["action_mask"],
        epsilon=0.0,
    )

    forward_seconds = (
        time.perf_counter()
        - start
    )

    assert info[
        "action_mask"
    ][action]

    print(
        "Greedy valid action:",
        action,
    )

    print(
        "Greedy action time:",
        f"{forward_seconds:.4f} s",
    )

    # --------------------------------------------------
    # One environment transition
    # --------------------------------------------------

    (
        next_observation,
        reward,
        terminated,
        truncated,
        next_info,
    ) = env.step(action)

    print(
        "First-step reward:",
        f"{reward:.6f}",
    )

    print(
        "First-step fuel:",
        f"{next_info['step_fuel_litres']:.6f} L",
    )

    # --------------------------------------------------
    # Populate exactly one full training batch.
    # --------------------------------------------------

    for index in range(
        config.batch_size
    ):

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

    assert agent.can_optimize()

    # --------------------------------------------------
    # Full forward/backward optimization smoke test
    # --------------------------------------------------

    start = time.perf_counter()

    loss = agent.optimize()

    optimize_seconds = (
        time.perf_counter()
        - start
    )

    print(
        "Optimization loss:",
        loss,
    )

    print(
        "One batch optimization time:",
        f"{optimize_seconds:.4f} s",
    )

    assert loss is not None
    assert np.isfinite(loss)

    print()

    print(
        "Torch threads:",
        torch.get_num_threads(),
    )

    print(
        "CUDA available:",
        torch.cuda.is_available(),
    )

    print("=" * 84)
    print(
        "FULL 1000-BIN DQN SMOKE TEST PASSED"
    )
    print("=" * 84)


if __name__ == "__main__":
    main()