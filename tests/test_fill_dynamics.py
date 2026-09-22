import numpy as np
import pytest

from smart_waste.core.fill_dynamics import (
    BinFillState,
    advance_fill,
    generate_fill_rates,
    is_predictive_75_80_candidate,
    predict_fill,
    predicts_overflow,
    select_proactive_candidate_ids,
)


def test_fill_advances_linearly():
    result = advance_fill(
        70.0,
        fill_rate_percent_per_hour=5.0,
        elapsed_hours=2.0,
    )

    assert result == pytest.approx(80.0)


def test_fill_is_capped_at_100():
    result = advance_fill(
        95.0,
        fill_rate_percent_per_hour=10.0,
        elapsed_hours=2.0,
    )

    assert result == pytest.approx(100.0)


def test_predict_fill():
    result = predict_fill(
        75.0,
        fill_rate_percent_per_hour=5.0,
        prediction_horizon_hours=4.0,
    )

    assert result == pytest.approx(95.0)


def test_overflow_prediction_true():
    assert predicts_overflow(
        80.0,
        fill_rate_percent_per_hour=5.0,
        prediction_horizon_hours=4.0,
    )


def test_overflow_prediction_false():
    assert not predicts_overflow(
        70.0,
        fill_rate_percent_per_hour=2.0,
        prediction_horizon_hours=4.0,
    )


def test_75_80_predictive_candidate():
    state = BinFillState(
        bin_id=5,
        current_fill_percent=80.0,
        fill_rate_percent_per_hour=5.0,
    )

    assert is_predictive_75_80_candidate(
        state,
        prediction_horizon_hours=4.0,
    )


def test_proactive_selection_combines_near_full_and_prediction():
    states = (
        BinFillState(
            bin_id=0,
            current_fill_percent=95.0,
            fill_rate_percent_per_hour=0.5,
        ),
        BinFillState(
            bin_id=1,
            current_fill_percent=80.0,
            fill_rate_percent_per_hour=5.0,
        ),
        BinFillState(
            bin_id=2,
            current_fill_percent=50.0,
            fill_rate_percent_per_hour=1.0,
        ),
    )

    selected = select_proactive_candidate_ids(
        states,
        prediction_horizon_hours=4.0,
        near_full_threshold_percent=90.0,
    )

    assert selected == (0, 1)


def test_fill_rate_generation_is_reproducible():
    a = generate_fill_rates(
        num_bins=1000,
        master_seed=20260922,
        minimum_rate_percent_per_hour=0.5,
        maximum_rate_percent_per_hour=5.0,
        replicate_id=0,
    )

    b = generate_fill_rates(
        num_bins=1000,
        master_seed=20260922,
        minimum_rate_percent_per_hour=0.5,
        maximum_rate_percent_per_hour=5.0,
        replicate_id=0,
    )

    assert np.array_equal(a, b)