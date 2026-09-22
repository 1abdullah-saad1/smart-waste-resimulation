import pytest

from smart_waste.security.five_day_fdi import (
    simulate_five_day_unauthenticated_fdi,
)


def test_five_day_test_injects_exactly_50_alerts():
    result = simulate_five_day_unauthenticated_fdi()

    assert result.total_injected == 50


def test_cumulative_alerts_match_manuscript_figure():
    result = simulate_five_day_unauthenticated_fdi()

    actual = [
        row.cumulative_injected
        for row in result.daily_results
    ]

    assert actual == [
        10,
        20,
        30,
        40,
        50,
    ]


def test_unprotected_processes_all_forged_alerts():
    result = simulate_five_day_unauthenticated_fdi()

    actual = [
        row.unprotected_cumulative_processed
        for row in result.daily_results
    ]

    assert actual == [
        10,
        20,
        30,
        40,
        50,
    ]

    assert result.total_unprotected_processed == 50


def test_poa_processes_zero_forged_alerts():
    result = simulate_five_day_unauthenticated_fdi()

    assert result.total_poa_processed == 0

    assert all(
        row.poa_cumulative_processed == 0
        for row in result.daily_results
    )


def test_poa_rejects_all_50_alerts():
    result = simulate_five_day_unauthenticated_fdi()

    assert result.total_poa_rejected == 50


def test_unprotected_cloud_penetration_is_100_percent():
    result = simulate_five_day_unauthenticated_fdi()

    assert (
        result.unprotected_cloud_penetration_rate
        == pytest.approx(1.0)
    )


def test_poa_cloud_penetration_is_zero_percent():
    result = simulate_five_day_unauthenticated_fdi()

    assert (
        result.poa_cloud_penetration_rate
        == pytest.approx(0.0)
    )


def test_poa_rejection_rate_is_100_percent():
    result = simulate_five_day_unauthenticated_fdi()

    assert (
        result.poa_rejection_rate
        == pytest.approx(1.0)
    )