import pytest

from smart_waste.core.city import City
from smart_waste.routing.tsr import (
    assign_bins_to_static_sectors,
    simulate_tsr,
)


def make_city():
    return City.generate(
        area_km2=50.0,
        num_bins=1000,
        master_seed=20260922,
    )


def test_tsr_assigns_every_bin_once():
    city = make_city()

    assignments = assign_bins_to_static_sectors(
        city,
        num_trucks=10,
    )

    all_ids = [
        bin_id
        for assignment in assignments
        for bin_id in assignment
    ]

    assert len(all_ids) == 1000

    assert len(set(all_ids)) == 1000

    assert sorted(all_ids) == list(range(1000))


def test_tsr_has_ten_assignments():
    city = make_city()

    assignments = assign_bins_to_static_sectors(
        city,
        num_trucks=10,
    )

    assert len(assignments) == 10


def test_tsr_sector_sizes_are_balanced():
    city = make_city()

    assignments = assign_bins_to_static_sectors(
        city,
        num_trucks=10,
    )

    sizes = [
        len(assignment)
        for assignment in assignments
    ]

    assert max(sizes) - min(sizes) <= 1


def test_tsr_visits_all_bins():
    city = make_city()

    result = simulate_tsr(
        city,
        num_trucks=10,
        truck_capacity_tonnes=10.0,
        fuel_efficiency_km_per_litre=2.5,
        bin_volume_m3=1.1,
        waste_density_kg_per_m3=229.088,
    )

    assert result.total_bins_visited == 1000


def test_tsr_collects_all_generated_waste():
    city = make_city()

    expected = sum(
        bin_.waste_mass_tonnes(
            bin_volume_m3=1.1,
            waste_density_kg_per_m3=229.088,
        )
        for bin_ in city.bins
    )

    result = simulate_tsr(
        city,
        num_trucks=10,
        truck_capacity_tonnes=10.0,
        fuel_efficiency_km_per_litre=2.5,
        bin_volume_m3=1.1,
        waste_density_kg_per_m3=229.088,
    )

    assert (
        result.total_collected_mass_tonnes
        == pytest.approx(expected)
    )


def test_tsr_fuel_matches_distance():
    city = make_city()

    result = simulate_tsr(
        city,
        num_trucks=10,
        truck_capacity_tonnes=10.0,
        fuel_efficiency_km_per_litre=2.5,
        bin_volume_m3=1.1,
        waste_density_kg_per_m3=229.088,
    )

    assert result.total_fuel_litres == pytest.approx(
        result.total_distance_km / 2.5
    )


def test_tsr_is_exactly_reproducible():
    city_a = make_city()
    city_b = make_city()

    result_a = simulate_tsr(
        city_a,
        num_trucks=10,
        truck_capacity_tonnes=10.0,
        fuel_efficiency_km_per_litre=2.5,
        bin_volume_m3=1.1,
        waste_density_kg_per_m3=229.088,
    )

    result_b = simulate_tsr(
        city_b,
        num_trucks=10,
        truck_capacity_tonnes=10.0,
        fuel_efficiency_km_per_litre=2.5,
        bin_volume_m3=1.1,
        waste_density_kg_per_m3=229.088,
    )

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