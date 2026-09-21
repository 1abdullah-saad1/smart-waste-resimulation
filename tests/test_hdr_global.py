import pytest

from smart_waste.core.city import City
from smart_waste.routing.hdr_global import (
    simulate_global_greedy_hdr,
)


def make_city():
    return City.generate(
        area_km2=50.0,
        num_bins=1000,
        master_seed=20260922,
    )


def run_hdr():
    return simulate_global_greedy_hdr(
        make_city(),
        num_trucks=10,
        truck_capacity_tonnes=10.0,
        fuel_efficiency_km_per_litre=2.5,
        bin_volume_m3=1.1,
        waste_density_kg_per_m3=229.088,
        threshold_percent=80.0,
    )


def test_global_hdr_visits_186_bins():
    result = run_hdr()

    assert result.total_bins_visited == 186


def test_global_hdr_fuel_matches_distance():
    result = run_hdr()

    assert result.total_fuel_litres == pytest.approx(
        result.total_distance_km / 2.5
    )


def test_global_hdr_is_reproducible():
    a = run_hdr()
    b = run_hdr()

    assert a.total_distance_km == pytest.approx(
        b.total_distance_km
    )

    assert a.total_fuel_litres == pytest.approx(
        b.total_fuel_litres
    )