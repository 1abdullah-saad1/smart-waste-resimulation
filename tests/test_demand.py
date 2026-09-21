import pytest

from smart_waste.core.bin import WasteBin


def test_full_bin_mass_nominal():
    bin_ = WasteBin(
        bin_id=0,
        x_km=0.0,
        y_km=0.0,
        fill_percent=100.0,
    )

    mass = bin_.waste_mass_kg(
        bin_volume_m3=1.1,
        waste_density_kg_per_m3=229.088,
    )

    assert mass == pytest.approx(
        251.9968,
        rel=1e-6,
    )


def test_half_full_bin_has_half_mass():
    bin_ = WasteBin(
        bin_id=0,
        x_km=0.0,
        y_km=0.0,
        fill_percent=50.0,
    )

    mass = bin_.waste_mass_kg(
        bin_volume_m3=1.1,
        waste_density_kg_per_m3=229.088,
    )

    assert mass == pytest.approx(
        125.9984,
        rel=1e-6,
    )


def test_mass_conversion_to_tonnes():
    bin_ = WasteBin(
        bin_id=0,
        x_km=0.0,
        y_km=0.0,
        fill_percent=100.0,
    )

    tonnes = bin_.waste_mass_tonnes(
        bin_volume_m3=1.1,
        waste_density_kg_per_m3=229.088,
    )

    assert tonnes == pytest.approx(
        0.2519968,
        rel=1e-6,
    )


def test_invalid_density_rejected():
    bin_ = WasteBin(
        bin_id=0,
        x_km=0.0,
        y_km=0.0,
        fill_percent=50.0,
    )

    with pytest.raises(ValueError):
        bin_.waste_mass_kg(
            bin_volume_m3=1.1,
            waste_density_kg_per_m3=0.0,
        )