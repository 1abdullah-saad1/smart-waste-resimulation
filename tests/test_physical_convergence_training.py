import pytest

from smart_waste.rl.physical_convergence_training import (
    _curriculum_condition,
    _derive_agent_seed,
    _matched_validation_change,
    _normalized_slope,
    ValidationRunRecord,
)


def test_curriculum_contains_all_32_conditions_exactly_once():
    observed = [
        _curriculum_condition(
            episode
        )
        for episode in range(
            32
        )
    ]

    assert len(
        observed
    ) == 32

    assert len(
        set(
            observed
        )
    ) == 32

    assert observed[
        0
    ] == (
        0,
        "H0",
    )

    assert observed[
        3
    ] == (
        0,
        "H3",
    )

    assert observed[
        4
    ] == (
        1,
        "H0",
    )

    assert observed[
        31
    ] == (
        7,
        "H3",
    )

    assert (
        _curriculum_condition(
            32
        )
        == observed[
            0
        ]
    )


def test_topology_agent_seed_is_deterministic_and_distinct():
    topologies = (
        "manhattan",
        "superblock",
        "hex",
        "radial_concentric",
    )

    seeds = [
        _derive_agent_seed(
            master_seed=20260924,
            topology_id=topology,
        )
        for topology in topologies
    ]

    assert len(
        set(
            seeds
        )
    ) == 4

    assert seeds == [
        _derive_agent_seed(
            master_seed=20260924,
            topology_id=topology,
        )
        for topology in topologies
    ]


def test_normalized_slope_is_scale_invariant():
    first = [
        100.0,
        101.0,
        102.0,
        103.0,
    ]

    second = [
        value * 10.0
        for value in first
    ]

    assert (
        _normalized_slope(
            first
        )
        == pytest.approx(
            _normalized_slope(
                second
            )
        )
    )


def make_validation(
    *,
    fuel,
    reward,
):
    return ValidationRunRecord(
        checkpoint_episode=32,
        validation_slot_id=0,
        challenge_id="H0",
        total_reward=reward,
        reward_distance_km=10.0,
        fleet_distance_km=10.0,
        fuel_litres=fuel,
        completion_time_hours=1.0,
        completed_services=1,
        natural_completion=True,
    )


def test_matched_validation_change_uses_condition_identity():
    previous = [
        make_validation(
            fuel=100.0,
            reward=-50.0,
        )
    ]

    current = [
        make_validation(
            fuel=101.0,
            reward=-50.5,
        )
    ]

    fuel_change = (
        _matched_validation_change(
            current,
            previous,
            field_name="fuel_litres",
        )
    )

    reward_change = (
        _matched_validation_change(
            current,
            previous,
            field_name="total_reward",
        )
    )

    assert fuel_change == pytest.approx(
        0.01
    )

    assert reward_change == pytest.approx(
        0.01
    )
