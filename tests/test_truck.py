import pytest

from smart_waste.core.truck import Truck


def test_truck_remaining_capacity():
    truck = Truck(
        truck_id=0,
        capacity_tonnes=10.0,
        fuel_efficiency_km_per_litre=2.5,
        x_km=0.0,
        y_km=0.0,
        current_load_tonnes=4.0,
    )

    assert truck.remaining_capacity_tonnes == 6.0


def test_truck_fuel_consumption():
    truck = Truck(
        truck_id=0,
        capacity_tonnes=10.0,
        fuel_efficiency_km_per_litre=2.5,
        x_km=0.0,
        y_km=0.0,
        distance_travelled_km=100.0,
    )

    assert truck.fuel_consumed_litres == pytest.approx(40.0)


def test_overloaded_truck_rejected():
    with pytest.raises(ValueError):
        Truck(
            truck_id=0,
            capacity_tonnes=10.0,
            fuel_efficiency_km_per_litre=2.5,
            x_km=0.0,
            y_km=0.0,
            current_load_tonnes=11.0,
        )