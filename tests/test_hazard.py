import pytest

from smart_waste.core.hazard import (
    evaluate_hazard,
)


def test_normal_bin_has_zero_hazard():
    state = evaluate_hazard(
        bin_id=1,
        temperature_c=25.0,
        humidity_percent=40.0,
    )

    assert not state.event_required
    assert state.severity_score == pytest.approx(0.0)
    assert state.normalized_severity == pytest.approx(0.0)


def test_70_c_is_emergency_fire():
    state = evaluate_hazard(
        bin_id=1,
        temperature_c=70.0,
        humidity_percent=40.0,
    )

    assert state.emergency_fire
    assert state.event_required


def test_below_70_c_is_not_emergency_fire():
    state = evaluate_hazard(
        bin_id=1,
        temperature_c=69.9,
        humidity_percent=40.0,
    )

    assert not state.emergency_fire


def test_predicted_combustion_risk_requires_event():
    state = evaluate_hazard(
        bin_id=1,
        temperature_c=55.0,
        humidity_percent=45.0,
        predicted_fire_risk_high=True,
    )

    assert state.predicted_combustion_risk
    assert state.event_required
    assert state.preemptive_fire_collection
    assert state.severity_score == pytest.approx(2.0)


def test_excess_moisture_requires_event():
    state = evaluate_hazard(
        bin_id=1,
        temperature_c=25.0,
        humidity_percent=90.0,
        excess_moisture_detected=True,
    )

    assert state.excess_moisture
    assert state.event_required
    assert state.severity_score == pytest.approx(1.0)


def test_emergency_fire_has_highest_severity():
    state = evaluate_hazard(
        bin_id=1,
        temperature_c=75.0,
        humidity_percent=95.0,
        predicted_fire_risk_high=True,
        excess_moisture_detected=True,
    )

    assert state.emergency_fire
    assert state.predicted_combustion_risk
    assert state.excess_moisture
    assert state.severity_score == pytest.approx(3.0)


def test_normalized_emergency_severity_is_one():
    state = evaluate_hazard(
        bin_id=1,
        temperature_c=80.0,
        humidity_percent=40.0,
    )

    assert state.normalized_severity == pytest.approx(1.0)


def test_invalid_humidity_is_rejected():
    with pytest.raises(ValueError):
        evaluate_hazard(
            bin_id=1,
            temperature_c=25.0,
            humidity_percent=101.0,
        )


def test_invalid_bin_id_is_rejected():
    with pytest.raises(ValueError):
        evaluate_hazard(
            bin_id=-1,
            temperature_c=25.0,
            humidity_percent=40.0,
        )