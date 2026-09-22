import heapq

import pytest

from smart_waste.models.bin import WasteBin
from smart_waste.models.depot import Depot
from smart_waste.models.events import (
    EventType,
    SimulationEvent,
)
from smart_waste.models.fuel import (
    candidate_fuel_requirement_litres,
    fuel_required_litres,
    is_candidate_fuel_feasible,
    refuel_time_minutes,
)
from smart_waste.models.truck import (
    Truck,
    TruckStatus,
)


def make_truck() -> Truck:
    return Truck(
        truck_id=0,
        current_node="depot",
        capacity_tonnes=10.0,
        speed_km_per_hour=30.0,
        fuel_capacity_litres=200.0,
        fuel_efficiency_km_per_litre=2.5,
    )


def test_full_bin_mass_mapping() -> None:
    bin_ = WasteBin(
        bin_id=0,
        road_node=1,
        fill_percent=100.0,
        fill_rate_percent_per_hour=1.0,
        full_mass_kg=440.0,
    )

    assert bin_.waste_mass_kg == pytest.approx(440.0)
    assert bin_.waste_mass_tonnes == pytest.approx(0.44)


def test_half_bin_mass_mapping() -> None:
    bin_ = WasteBin(
        bin_id=0,
        road_node=1,
        fill_percent=50.0,
        fill_rate_percent_per_hour=1.0,
        full_mass_kg=440.0,
    )

    assert bin_.waste_mass_kg == pytest.approx(220.0)


def test_bin_fill_advances_with_physical_time() -> None:
    bin_ = WasteBin(
        bin_id=0,
        road_node=1,
        fill_percent=50.0,
        fill_rate_percent_per_hour=2.0,
        full_mass_kg=440.0,
    )

    bin_.advance(3.0)

    assert bin_.fill_percent == pytest.approx(56.0)
    assert bin_.waste_age_hours == pytest.approx(3.0)


def test_collection_resets_bin() -> None:
    bin_ = WasteBin(
        bin_id=0,
        road_node=1,
        fill_percent=50.0,
        fill_rate_percent_per_hour=1.0,
        full_mass_kg=440.0,
        waste_age_hours=12.0,
    )

    collected = bin_.collect()

    assert collected == pytest.approx(0.22)
    assert bin_.fill_percent == 0.0
    assert bin_.waste_age_hours == 0.0
    assert bin_.collected_count == 1


def test_fuel_consumption() -> None:
    assert fuel_required_litres(
        25.0,
        efficiency_km_per_litre=2.5,
    ) == pytest.approx(10.0)


def test_dynamic_return_fuel_requirement() -> None:
    required = candidate_fuel_requirement_litres(
        distance_to_candidate_km=20.0,
        candidate_to_depot_km=30.0,
        efficiency_km_per_litre=2.5,
    )

    assert required == pytest.approx(20.0)


def test_candidate_fuel_feasibility() -> None:
    assert is_candidate_fuel_feasible(
        fuel_remaining_litres=20.0,
        distance_to_candidate_km=20.0,
        candidate_to_depot_km=30.0,
        efficiency_km_per_litre=2.5,
    )

    assert not is_candidate_fuel_feasible(
        fuel_remaining_litres=19.9,
        distance_to_candidate_km=20.0,
        candidate_to_depot_km=30.0,
        efficiency_km_per_litre=2.5,
    )


def test_refuel_time() -> None:
    assert refuel_time_minutes(
        current_litres=140.0,
        tank_capacity_litres=200.0,
        rate_litres_per_minute=60.0,
    ) == pytest.approx(1.0)


def test_truck_starts_with_full_tank() -> None:
    truck = make_truck()

    assert truck.fuel_remaining_litres == 200.0


def test_truck_travel_requires_arrival_completion() -> None:
    truck = make_truck()

    fuel_used = truck.begin_travel(
        destination_node=1,
        distance_km=30.0,
    )

    assert fuel_used == pytest.approx(12.0)

    # Departure does not teleport the truck.
    assert truck.current_node == "depot"
    assert truck.destination_node == 1
    assert truck.status == TruckStatus.TRAVELLING

    assert truck.cumulative_distance_km == pytest.approx(0.0)
    assert truck.cumulative_fuel_used_litres == pytest.approx(12.0)
    assert truck.fuel_remaining_litres == pytest.approx(188.0)

    completed_distance = truck.complete_travel()

    assert completed_distance == pytest.approx(30.0)
    assert truck.current_node == 1
    assert truck.destination_node is None
    assert truck.status == TruckStatus.IDLE
    assert truck.cumulative_distance_km == pytest.approx(30.0)


def test_truck_cannot_begin_unfuelled_travel() -> None:
    truck = make_truck()
    truck.fuel_remaining_litres = 1.0

    with pytest.raises(ValueError):
        truck.begin_travel(
            destination_node=1,
            distance_km=10.0,
        )

    assert truck.current_node == "depot"
    assert truck.status == TruckStatus.IDLE
    assert truck.fuel_remaining_litres == pytest.approx(1.0)


def test_capacity_invariant() -> None:
    truck = make_truck()

    truck.load_waste(9.0)

    assert not truck.can_load(1.1)

    with pytest.raises(ValueError):
        truck.load_waste(1.1)


def test_refuel_full() -> None:
    truck = make_truck()
    truck.fuel_remaining_litres = 125.0

    added = truck.refuel_full()

    assert added == pytest.approx(75.0)
    assert truck.fuel_remaining_litres == pytest.approx(200.0)
    assert truck.cumulative_refuelled_litres == pytest.approx(75.0)


def test_depot_service_is_sequential() -> None:
    truck = make_truck()
    truck.current_load_tonnes = 3.0
    truck.fuel_remaining_litres = 140.0

    depot = Depot(
        road_node="depot",
        unloading_bays=1,
        unload_time_minutes=11.0,
        refuel_rate_litres_per_minute=60.0,
    )

    assert depot.service_duration_minutes(
        truck
    ) == pytest.approx(12.0)


def test_empty_truck_skips_unload_time() -> None:
    truck = make_truck()
    truck.fuel_remaining_litres = 140.0

    depot = Depot(
        road_node="depot",
        unloading_bays=1,
        unload_time_minutes=11.0,
        refuel_rate_litres_per_minute=60.0,
    )

    assert depot.service_duration_minutes(
        truck
    ) == pytest.approx(1.0)


def test_events_are_deterministically_ordered() -> None:
    queue = [
        SimulationEvent(
            time_hours=2.0,
            sequence=1,
            event_type=EventType.TRUCK_ARRIVAL,
        ),
        SimulationEvent(
            time_hours=1.0,
            sequence=2,
            event_type=EventType.TRUCK_ARRIVAL,
        ),
        SimulationEvent(
            time_hours=1.0,
            sequence=1,
            event_type=EventType.TRUCK_ARRIVAL,
        ),
    ]

    heapq.heapify(queue)

    first = heapq.heappop(queue)
    second = heapq.heappop(queue)

    assert (first.time_hours, first.sequence) == (1.0, 1)
    assert (second.time_hours, second.sequence) == (1.0, 2)
