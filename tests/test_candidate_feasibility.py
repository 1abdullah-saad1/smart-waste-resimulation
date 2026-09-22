import networkx as nx
import pytest

from smart_waste.models.bin import WasteBin
from smart_waste.models.depot import Depot
from smart_waste.models.truck import (
    Truck,
    TruckStatus,
)
from smart_waste.movement.road_graph import RoadGraph
from smart_waste.simulation.feasibility import (
    FeasibilityError,
    FeasibilityFailure,
    can_return_to_depot,
    evaluate_candidate_feasibility,
    fuel_required_to_depot_litres,
)
from smart_waste.simulation.state import SimulationState


SERVICE_SECONDS = 36.0


def make_state(
    *,
    truck_load_tonnes: float = 0.0,
    fuel_litres: float = 200.0,
    fill_percent: float = 100.0,
    fill_rate: float = 0.0,
) -> SimulationState:
    graph = nx.Graph()

    graph.add_edge(
        "truck-node",
        "bin-node",
        length_km=20.0,
    )

    graph.add_edge(
        "bin-node",
        "depot",
        length_km=30.0,
    )

    graph.add_edge(
        "truck-node",
        "depot",
        length_km=40.0,
    )

    road = RoadGraph(graph)

    bin_ = WasteBin(
        bin_id=0,
        road_node="bin-node",
        fill_percent=fill_percent,
        fill_rate_percent_per_hour=fill_rate,
        full_mass_kg=440.0,
    )

    truck = Truck(
        truck_id=0,
        current_node="truck-node",
        capacity_tonnes=10.0,
        speed_km_per_hour=30.0,
        fuel_capacity_litres=200.0,
        fuel_efficiency_km_per_litre=2.5,
        current_load_tonnes=truck_load_tonnes,
        fuel_remaining_litres=fuel_litres,
    )

    depot = Depot(
        road_node="depot",
        unloading_bays=1,
        unload_time_minutes=11.0,
        refuel_rate_litres_per_minute=60.0,
    )

    return SimulationState(
        road_graph=road,
        bins={0: bin_},
        trucks={0: truck},
        depot=depot,
    )


def test_fully_feasible_candidate() -> None:
    state = make_state()

    result = evaluate_candidate_feasibility(
        state=state,
        truck_id=0,
        bin_id=0,
        service_time_seconds=SERVICE_SECONDS,
    )

    assert result.feasible
    assert result.failure == FeasibilityFailure.NONE


def test_fuel_requirement_includes_return_to_depot() -> None:
    state = make_state()

    result = evaluate_candidate_feasibility(
        state=state,
        truck_id=0,
        bin_id=0,
        service_time_seconds=SERVICE_SECONDS,
    )

    # 20 km to candidate + 30 km to depot = 50 km.
    # 50 / 2.5 = 20 L.
    assert result.required_fuel_litres == pytest.approx(
        20.0
    )


def test_exact_dynamic_fuel_reserve_is_feasible() -> None:
    state = make_state(
        fuel_litres=20.0
    )

    result = evaluate_candidate_feasibility(
        state=state,
        truck_id=0,
        bin_id=0,
        service_time_seconds=SERVICE_SECONDS,
    )

    assert result.fuel_feasible
    assert result.feasible


def test_below_dynamic_fuel_reserve_is_rejected() -> None:
    state = make_state(
        fuel_litres=19.999
    )

    result = evaluate_candidate_feasibility(
        state=state,
        truck_id=0,
        bin_id=0,
        service_time_seconds=SERVICE_SECONDS,
    )

    assert not result.fuel_feasible
    assert not result.feasible
    assert result.failure == FeasibilityFailure.FUEL


def test_capacity_failure() -> None:
    state = make_state(
        truck_load_tonnes=9.7,
        fill_percent=100.0,
    )

    result = evaluate_candidate_feasibility(
        state=state,
        truck_id=0,
        bin_id=0,
        service_time_seconds=SERVICE_SECONDS,
    )

    assert not result.capacity_feasible
    assert result.fuel_feasible
    assert result.failure == FeasibilityFailure.CAPACITY


def test_combined_capacity_and_fuel_failure() -> None:
    state = make_state(
        truck_load_tonnes=9.7,
        fuel_litres=10.0,
    )

    result = evaluate_candidate_feasibility(
        state=state,
        truck_id=0,
        bin_id=0,
        service_time_seconds=SERVICE_SECONDS,
    )

    assert not result.feasible
    assert (
        result.failure
        == FeasibilityFailure.CAPACITY_AND_FUEL
    )


def test_capacity_uses_projected_service_mass() -> None:
    state = make_state(
        truck_load_tonnes=9.778,
        fill_percent=50.0,
        fill_rate=100.0,
    )

    result = evaluate_candidate_feasibility(
        state=state,
        truck_id=0,
        bin_id=0,
        service_time_seconds=SERVICE_SECONDS,
    )

    # Initial bin mass = 0.2200 t.
    # At service completion fill = 51%, mass = 0.2244 t.
    assert (
        result.projected_collection_mass_tonnes
        == pytest.approx(0.2244)
    )

    assert not result.capacity_feasible


def test_feasibility_uses_road_distance() -> None:
    state = make_state()

    result = evaluate_candidate_feasibility(
        state=state,
        truck_id=0,
        bin_id=0,
        service_time_seconds=SERVICE_SECONDS,
    )

    assert result.distance_to_candidate_km == pytest.approx(
        20.0
    )

    assert result.candidate_to_depot_km == pytest.approx(
        30.0
    )


def test_immediate_depot_return_requirement() -> None:
    state = make_state()

    # Shortest truck-node -> depot is 40 km.
    # 40 / 2.5 = 16 L.
    assert fuel_required_to_depot_litres(
        state=state,
        truck_id=0,
    ) == pytest.approx(16.0)


def test_can_return_to_depot_at_exact_required_fuel() -> None:
    state = make_state(
        fuel_litres=16.0
    )

    assert can_return_to_depot(
        state=state,
        truck_id=0,
    )


def test_cannot_return_to_depot_below_required_fuel() -> None:
    state = make_state(
        fuel_litres=15.999
    )

    assert not can_return_to_depot(
        state=state,
        truck_id=0,
    )


def test_busy_truck_cannot_be_evaluated() -> None:
    state = make_state()

    state.trucks[0].status = (
        TruckStatus.TRAVELLING
    )

    with pytest.raises(FeasibilityError):
        evaluate_candidate_feasibility(
            state=state,
            truck_id=0,
            bin_id=0,
            service_time_seconds=SERVICE_SECONDS,
        )


def test_unknown_candidate_is_rejected() -> None:
    state = make_state()

    with pytest.raises(FeasibilityError):
        evaluate_candidate_feasibility(
            state=state,
            truck_id=0,
            bin_id=999,
            service_time_seconds=SERVICE_SECONDS,
        )
