import pytest

from smart_waste.core.bin import WasteBin


def test_bin_service_threshold():
    bin_ = WasteBin(
        bin_id=1,
        x_km=1.0,
        y_km=2.0,
        fill_percent=80.0,
    )

    assert bin_.requires_service(80.0)


def test_bin_below_service_threshold():
    bin_ = WasteBin(
        bin_id=1,
        x_km=1.0,
        y_km=2.0,
        fill_percent=79.99,
    )

    assert not bin_.requires_service(80.0)


def test_invalid_fill_level_rejected():
    with pytest.raises(ValueError):
        WasteBin(
            bin_id=1,
            x_km=1.0,
            y_km=2.0,
            fill_percent=101.0,
        )