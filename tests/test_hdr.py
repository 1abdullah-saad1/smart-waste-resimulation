import pytest

from smart_waste.core.city import City
from smart_waste.routing.hdr import (
    select_hdr_candidates,
    simulate_hdr,
)


def make_city():
    return City.generate(
        area_km2=50.0,
        num_bins=1000,
        master_seed=20260922,
    )


def run_hdr(city, reported_fill_percent=None):
    return simulate_hdr(
        city,
        num_trucks=10,
        truck_capacity_tonnes=10.0,
        fuel_efficiency_km_per_litre=2.5,
        bin_volume_m3=1.1,
        waste_density_kg_per_m3=229.088,
        threshold_percent=80.0,
        reported_fill_percent=reported_fill_percent,
    )


def test_hdr_candidates_match_service_threshold():
    city = make_city()

    candidate_ids = select_hdr_candidates(
        city,
        threshold_percent=80.0,
    )

    expected_ids = tuple(
        bin_.bin_id
        for bin_ in city.bins
        if bin_.fill_percent >= 80.0
    )

    assert candidate_ids == expected_ids


def test_hdr_visits_only_eligible_bins():
    city = make_city()

    result = run_hdr(city)

    visited = {
        bin_id
        for route in result.truck_routes
        for bin_id in route.route
    }

    expected = {
        bin_.bin_id
        for bin_ in city.bins
        if bin_.fill_percent >= 80.0
    }

    assert visited == expected


def test_hdr_visits_fewer_bins_than_tsr():
    city = make_city()

    result = run_hdr(city)

    assert result.total_bins_visited < 1000


def test_hdr_collects_correct_true_mass():
    city = make_city()

    expected_mass = sum(
        bin_.waste_mass_tonnes(
            bin_volume_m3=1.1,
            waste_density_kg_per_m3=229.088,
        )
        for bin_ in city.bins
        if bin_.fill_percent >= 80.0
    )

    result = run_hdr(city)

    assert (
        result.total_collected_mass_tonnes
        == pytest.approx(expected_mass)
    )


def test_hdr_fuel_matches_distance():
    city = make_city()

    result = run_hdr(city)

    assert result.total_fuel_litres == pytest.approx(
        result.total_distance_km / 2.5
    )


def test_hdr_is_reproducible():
    result_a = run_hdr(make_city())
    result_b = run_hdr(make_city())

    assert (
        result_a.total_distance_km
        == pytest.approx(
            result_b.total_distance_km
        )
    )

    assert (
        result_a.total_fuel_litres
        == pytest.approx(
            result_b.total_fuel_litres
        )
    )


def test_reported_fill_can_create_false_candidate():
    city = make_city()

    low_bin = next(
        bin_
        for bin_ in city.bins
        if bin_.fill_percent < 80.0
    )

    normal_candidates = set(
        select_hdr_candidates(
            city,
            threshold_percent=80.0,
        )
    )

    assert low_bin.bin_id not in normal_candidates

    attacked_candidates = set(
        select_hdr_candidates(
            city,
            threshold_percent=80.0,
            reported_fill_percent={
                low_bin.bin_id: 100.0
            },
        )
    )

    assert low_bin.bin_id in attacked_candidates