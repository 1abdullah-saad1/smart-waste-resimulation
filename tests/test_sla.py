import pytest

from smart_waste.core.sla import (
    SLAState,
    advance_sla_state,
    evaluate_sla,
    normalized_full_bin_time,
)


def test_initial_sla_state_has_zero_elapsed_time():
    state = SLAState(bin_id=1)

    assert state.full_bin_elapsed_hours == pytest.approx(0.0)
    assert state.hazard_elapsed_hours == pytest.approx(0.0)


def test_full_bin_timer_accumulates():
    state = SLAState(bin_id=1)

    state = advance_sla_state(
        state,
        current_fill_percent=100.0,
        hazard_active=False,
        elapsed_hours=2.0,
    )

    state = advance_sla_state(
        state,
        current_fill_percent=100.0,
        hazard_active=False,
        elapsed_hours=3.0,
    )

    assert state.full_bin_elapsed_hours == pytest.approx(5.0)


def test_full_bin_timer_resets_below_100_percent():
    state = SLAState(
        bin_id=1,
        full_bin_elapsed_hours=5.0,
    )

    state = advance_sla_state(
        state,
        current_fill_percent=99.0,
        hazard_active=False,
        elapsed_hours=1.0,
    )

    assert state.full_bin_elapsed_hours == pytest.approx(0.0)


def test_exactly_six_hours_is_not_yet_violation():
    state = SLAState(
        bin_id=1,
        full_bin_elapsed_hours=6.0,
    )

    result = evaluate_sla(state)

    assert not result.full_bin_violation


def test_more_than_six_hours_is_violation():
    state = SLAState(
        bin_id=1,
        full_bin_elapsed_hours=6.01,
    )

    result = evaluate_sla(state)

    assert result.full_bin_violation
    assert result.any_violation
    assert result.violation_indicator == pytest.approx(1.0)


def test_normalized_full_bin_time():
    assert normalized_full_bin_time(
        3.0
    ) == pytest.approx(0.5)


def test_normalized_full_bin_time_is_capped():
    assert normalized_full_bin_time(
        12.0
    ) == pytest.approx(1.0)


def test_hazard_timer_accumulates_and_resets():
    state = SLAState(bin_id=1)

    state = advance_sla_state(
        state,
        current_fill_percent=50.0,
        hazard_active=True,
        elapsed_hours=2.0,
    )

    assert state.hazard_elapsed_hours == pytest.approx(2.0)

    state = advance_sla_state(
        state,
        current_fill_percent=50.0,
        hazard_active=False,
        elapsed_hours=1.0,
    )

    assert state.hazard_elapsed_hours == pytest.approx(0.0)


def test_unspecified_hazard_limit_is_not_invented():
    state = SLAState(
        bin_id=1,
        hazard_elapsed_hours=10.0,
        hazard_active=True,
    )

    result = evaluate_sla(
        state,
        hazard_limit_hours=None,
    )

    assert result.hazard_violation is None


def test_explicit_hazard_limit_can_be_evaluated():
    state = SLAState(
        bin_id=1,
        hazard_elapsed_hours=4.0,
        hazard_active=True,
    )

    result = evaluate_sla(
        state,
        hazard_limit_hours=3.0,
    )

    assert result.hazard_violation is True
    assert result.any_violation