from dataclasses import FrozenInstanceError

import networkx as nx
import pytest

from smart_waste.collection.base import (
    CollectionPolicy,
    PolicyView,
)
from smart_waste.models.bin import WasteBin
from smart_waste.models.depot import Depot
from smart_waste.models.truck import Truck
from smart_waste.movement.road_graph import RoadGraph
from smart_waste.simulation.policy_view import (
    build_policy_view,
)
from smart_waste.simulation.state import SimulationState


def make_state() -> SimulationState:
    graph = nx.Graph()

    graph.add_edge(
        "depot",
        "bin-0",
        length_km=1.0,
    )

    graph.add_edge(
        "depot",
        "bin-1",
        length_km=2.0,
    )

    road = RoadGraph(graph)

    bins = {
        1: WasteBin(
            bin_id=1,
            road_node="bin-1",
            fill_percent=80.0,
            fill_rate_percent_per_hour=2.0,
            full_mass_kg=440.0,
        ),
        0: WasteBin(
            bin_id=0,
            road_node="bin-0",
            fill_percent=50.0,
            fill_rate_percent_per_hour=1.0,
            full_mass_kg=440.0,
        ),
    }

    trucks = {
        1: Truck(
            truck_id=1,
            current_node="depot",
            capacity_tonnes=10.0,
            speed_km_per_hour=30.0,
            fuel_capacity_litres=200.0,
            fuel_efficiency_km_per_litre=2.5,
        ),
        0: Truck(
            truck_id=0,
            current_node="depot",
            capacity_tonnes=10.0,
            speed_km_per_hour=30.0,
            fuel_capacity_litres=200.0,
            fuel_efficiency_km_per_litre=2.5,
        ),
    }

    depot = Depot(
        road_node="depot",
        unloading_bays=1,
        unload_time_minutes=11.0,
        refuel_rate_litres_per_minute=60.0,
    )

    return SimulationState(
        road_graph=road,
        bins=bins,
        trucks=trucks,
        depot=depot,
    )


def test_collection_policy_is_abstract() -> None:
    with pytest.raises(TypeError):
        CollectionPolicy()


def test_policy_view_contains_global_context() -> None:
    state = make_state()
    state.advance_to(2.0)

    view = build_policy_view(state)

    assert view.current_time_hours == pytest.approx(2.0)
    assert view.depot_node == "depot"


def test_policy_view_is_sorted_by_stable_ids() -> None:
    state = make_state()

    view = build_policy_view(state)

    assert [
        bin_.bin_id
        for bin_ in view.bins
    ] == [0, 1]

    assert [
        truck.truck_id
        for truck in view.trucks
    ] == [0, 1]


def test_policy_view_is_frozen() -> None:
    state = make_state()
    view = build_policy_view(state)

    with pytest.raises(FrozenInstanceError):
        view.current_time_hours = 99.0  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        view.bins[0].fill_percent = 99.0  # type: ignore[misc]


def test_policy_snapshot_does_not_change_with_state() -> None:
    state = make_state()

    view = build_policy_view(state)

    original_fill = (
        view.bin_by_id(0).fill_percent
    )

    state.advance_to(5.0)

    assert view.bin_by_id(0).fill_percent == pytest.approx(
        original_fill
    )

    new_view = build_policy_view(state)

    assert new_view.bin_by_id(0).fill_percent > original_fill


def test_policy_view_lookup_helpers() -> None:
    view = build_policy_view(
        make_state()
    )

    assert view.bin_by_id(1).bin_id == 1
    assert view.truck_by_id(0).truck_id == 0

    with pytest.raises(KeyError):
        view.bin_by_id(999)

    with pytest.raises(KeyError):
        view.truck_by_id(999)


def test_policy_can_choose_bin_without_mutating_state() -> None:
    class FirstBinPolicy(CollectionPolicy):
        def initialize(
            self,
            view: PolicyView,
        ) -> None:
            pass

        def select_next_bin(
            self,
            *,
            view: PolicyView,
            truck_id: int,
        ) -> int | None:
            view.truck_by_id(truck_id)

            if not view.bins:
                return None

            return view.bins[0].bin_id

        def on_service_complete(
            self,
            *,
            view: PolicyView,
            truck_id: int,
            bin_id: int,
        ) -> None:
            pass

        def is_complete(
            self,
            view: PolicyView,
        ) -> bool:
            return False

    state = make_state()
    policy = FirstBinPolicy()

    view = build_policy_view(state)

    policy.initialize(view)

    candidate = policy.select_next_bin(
        view=view,
        truck_id=0,
    )

    assert candidate == 0

    # Policy received only snapshot values.
    assert state.bins[0].fill_percent == pytest.approx(
        50.0
    )


def test_policy_lifecycle_callbacks_are_supported() -> None:
    class TrackingPolicy(CollectionPolicy):
        def __init__(self) -> None:
            self.initialized = False
            self.completed_bins: list[int] = []

        def initialize(
            self,
            view: PolicyView,
        ) -> None:
            self.initialized = True

        def select_next_bin(
            self,
            *,
            view: PolicyView,
            truck_id: int,
        ) -> int | None:
            return None

        def on_service_complete(
            self,
            *,
            view: PolicyView,
            truck_id: int,
            bin_id: int,
        ) -> None:
            self.completed_bins.append(bin_id)

        def is_complete(
            self,
            view: PolicyView,
        ) -> bool:
            return bool(self.completed_bins)

    state = make_state()
    view = build_policy_view(state)

    policy = TrackingPolicy()

    policy.initialize(view)

    assert policy.initialized
    assert not policy.is_complete(view)

    policy.on_service_complete(
        view=view,
        truck_id=0,
        bin_id=1,
    )

    assert policy.completed_bins == [1]
    assert policy.is_complete(view)


def test_policy_view_exposes_read_only_physical_distance_counter():
    """
    Training/reward logic may observe the Core's cumulative
    distance counter without taking responsibility for movement.
    """
    from smart_waste.collection.base import (
        TruckPolicyView,
    )
    from smart_waste.models.truck import (
        TruckStatus,
    )

    truck = TruckPolicyView(
        truck_id=0,
        current_node=1,
        remaining_capacity_tonnes=9.0,
        fuel_remaining_litres=190.0,
        status=TruckStatus.IDLE,
        cumulative_distance_km=12.5,
    )

    assert (
        truck.cumulative_distance_km
        == 12.5
    )
