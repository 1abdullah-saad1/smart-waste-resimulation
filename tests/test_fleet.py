import pytest

from smart_waste.core.city import City
from smart_waste.core.fleet import create_fleet


def make_city():
    return City.generate(
        area_km2=50.0,
        num_bins=1000,
        master_seed=20260922,
    )


def test_create_ten_trucks():
    city = make_city()

    fleet = create_fleet(
        city,
        num_trucks=10,
        truck_capacity_tonnes=10.0,
        fuel_efficiency_km_per_litre=2.5,
    )

    assert len(fleet) == 10


def test_all_trucks_start_at_depot():
    city = make_city()

    fleet = create_fleet(
        city,
        num_trucks=10,
        truck_capacity_tonnes=10.0,
        fuel_efficiency_km_per_litre=2.5,
    )

    for truck in fleet:
        assert truck.location == pytest.approx(
            city.depot_location
        )


def test_all_trucks_have_manuscript_capacity():
    city = make_city()

    fleet = create_fleet(
        city,
        num_trucks=10,
        truck_capacity_tonnes=10.0,
        fuel_efficiency_km_per_litre=2.5,
    )

    assert all(
        truck.capacity_tonnes == 10.0
        for truck in fleet
    )


def test_all_trucks_have_manuscript_fuel_efficiency():
    city = make_city()

    fleet = create_fleet(
        city,
        num_trucks=10,
        truck_capacity_tonnes=10.0,
        fuel_efficiency_km_per_litre=2.5,
    )

    assert all(
        truck.fuel_efficiency_km_per_litre == 2.5
        for truck in fleet
    )