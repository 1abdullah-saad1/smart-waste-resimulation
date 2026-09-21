import pytest

from smart_waste.statistics.descriptive import summarize


def test_summary_count():
    result = summarize(
        [1, 2, 3, 4, 5]
    )

    assert result.n == 5


def test_summary_mean_and_median():
    result = summarize(
        [1, 2, 3, 4, 5]
    )

    assert result.mean == pytest.approx(3.0)
    assert result.median == pytest.approx(3.0)


def test_summary_std_is_sample_std():
    result = summarize(
        [1, 2, 3, 4, 5]
    )

    assert result.std == pytest.approx(
        1.5811388300841898
    )


def test_confidence_interval_contains_mean():
    result = summarize(
        [1, 2, 3, 4, 5]
    )

    assert result.ci95_low < result.mean
    assert result.ci95_high > result.mean


def test_single_observation_rejected():
    with pytest.raises(ValueError):
        summarize([1.0])


def test_constant_values_have_zero_width_ci():
    result = summarize(
        [5.0, 5.0, 5.0, 5.0]
    )

    assert result.std == pytest.approx(0.0)
    assert result.ci95_low == pytest.approx(5.0)
    assert result.ci95_high == pytest.approx(5.0)