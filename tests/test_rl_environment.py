import numpy as np
import pytest

from smart_waste.rl.environment import (
    SmartWasteRoutingEnv,
)


def make_env(
    *,
    hazards=None,
    verified=None,
):
    if hazards is None:
        hazards = np.array(
            [0.0, 0.0, 0.0]
        )

    if verified is None:
        verified = np.array(
            [True, True, True]
        )

    return SmartWasteRoutingEnv(
        bin_xy=np.array(
            [
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
            ]
        ),
        depot_xy=np.array(
            [0.0, 0.0]
        ),
        initial_fill_percent=np.array(
            [95.0, 80.0, 50.0]
        ),
        fill_rate_percent_per_hour=np.array(
            [0.0, 5.0, 0.0]
        ),
        hazard_severity=hazards,
        verified_mask=verified,
        prediction_horizon_hours=4.0,
        decision_interval_hours=1.0,
        num_trucks=1,
        max_steps=20,
    )


def test_observation_shape():
    env = make_env()

    observation, info = env.reset()

    assert observation.shape == (11,)
    assert env.observation_space.contains(
        observation
    )


def test_action_space_matches_number_of_bins():
    env = make_env()

    assert env.action_space.n == 3


def test_near_full_and_predicted_overflow_are_candidates():
    env = make_env()

    _, info = env.reset()

    mask = info["action_mask"]

    assert mask.tolist() == [
        True,
        True,
        False,
    ]


def test_unverified_bin_is_excluded():
    env = make_env(
        verified=np.array(
            [False, True, True]
        )
    )

    _, info = env.reset()

    assert info[
        "action_mask"
    ].tolist() == [
        False,
        True,
        False,
    ]


def test_emergency_hazard_overrides_ordinary_candidates():
    env = make_env(
        hazards=np.array(
            [0.0, 0.0, 1.0]
        )
    )

    _, info = env.reset()

    assert info[
        "action_mask"
    ].tolist() == [
        False,
        False,
        True,
    ]


def test_invalid_masked_action_is_rejected():
    env = make_env()

    env.reset()

    with pytest.raises(ValueError):
        env.step(2)


def test_service_consumes_distance_and_fuel():
    env = make_env()

    env.reset()

    _, _, _, _, info = env.step(0)

    assert info[
        "dispatch_distance_km"
    ] == pytest.approx(1.0)

    assert info[
        "step_fuel_litres"
    ] == pytest.approx(
        1.0 / 2.5
    )


def test_reset_is_deterministic():
    env = make_env()

    first, first_info = env.reset()

    env.step(0)

    second, second_info = env.reset()

    assert np.array_equal(
        first,
        second,
    )

    assert np.array_equal(
        first_info["action_mask"],
        second_info["action_mask"],
    )

def test_decision_interval_is_distributed_across_fleet():

    env = SmartWasteRoutingEnv(
        bin_xy=np.array(
            [
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
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
                95.0,
                50.0,
            ],
            dtype=np.float64,
        ),
        fill_rate_percent_per_hour=np.array(
            [
                0.0,
                0.0,
                4.0,
            ],
            dtype=np.float64,
        ),
        hazard_severity=np.zeros(
            3,
            dtype=np.float64,
        ),
        prediction_horizon_hours=1.0,
        decision_interval_hours=1.0,
        num_trucks=2,
        max_steps=10,
    )

    _, info = env.reset()

    assert info[
        "simulated_time_hours"
    ] == pytest.approx(0.0)

    _, _, _, _, info = env.step(0)

    # One action of a two-truck sequential fleet represents
    # half of the one-hour fleet decision epoch.
    assert info[
        "simulated_time_hours"
    ] == pytest.approx(0.5)

    # The unserviced third bin should therefore grow by
    # 4 %/h × 0.5 h = 2 percentage points.
    assert env.fill_percent[2] == pytest.approx(
        52.0
    )