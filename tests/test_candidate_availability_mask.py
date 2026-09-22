import networkx as nx
import pytest

from smart_waste.collection.base import (
    CandidateMask,
)
from smart_waste.models.bin import WasteBin
from smart_waste.models.depot import Depot
from smart_waste.models.truck import Truck
from smart_waste.movement.road_graph import RoadGraph
from smart_waste.simulation.policy_view import (
    build_policy_view,
)
from smart_waste.simulation.reservations import (
    BinReservationBook,
)
from smart_waste.simulation.state import (
    SimulationState,
)


def make_state(
    *,
    truck_load_tonnes: float = 0.0,
    fuel_litres: float = 200.0,
) -> SimulationState:
    graph = nx.Graph()

    graph.add_edge(
        "depot",
        "bin-0",
        length_km=2.0,
    )

    graph.add_edge(
        "depot",
        "bin-1",
        length_km=3.0,
    )

    graph.add_edge(
        "depot",
        "bin-2",
        length_km=4.0,
    )

    road = RoadGraph(graph)

    bins = {
        2: WasteBin(
            bin_id=2,
            road_node="bin-2",
            fill_percent=70.0,
            fill_rate_percent_per_hour=0.0,
            full_mass_kg=440.0,
        ),
        0: WasteBin(
            bin_id=0,
            road_node="bin-0",
            fill_percent=50.0,
            fill_rate_percent_per_hour=0.0,
            full_mass_kg=440.0,
        ),
        1: WasteBin(
            bin_id=1,
            road_node="bin-1",
            fill_percent=60.0,
            fill_rate_percent_per_hour=0.0,
            full_mass_kg=440.0,
        ),
    }

    truck = Truck(
        truck_id=0,
        current_node="depot",
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
        bins=bins,
        trucks={
            0: truck,
        },
        depot=depot,
    )


def test_candidate_mask_rejects_length_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="equal length",
    ):
        CandidateMask(
            bin_ids=(0, 1),
            selectable=(True,),
        )


def test_candidate_mask_rejects_duplicate_ids() -> None:
    with pytest.raises(
        ValueError,
        match="unique",
    ):
        CandidateMask(
            bin_ids=(0, 0),
            selectable=(
                True,
                False,
            ),
        )


def test_unreserved_bins_are_selectable() -> None:
    view = build_policy_view(
        make_state()
    )

    mask = (
        view.reservation_availability_mask()
    )

    assert mask.bin_ids == (
        0,
        1,
        2,
    )

    assert mask.selectable == (
        True,
        True,
        True,
    )

    assert mask.selectable_bin_ids == (
        0,
        1,
        2,
    )


def test_reserved_bin_is_masked_out() -> None:
    state = make_state()
    book = BinReservationBook()

    book.reserve(
        bin_id=1,
        truck_id=7,
    )

    view = build_policy_view(
        state,
        reservations=book.snapshot(),
    )

    mask = (
        view.reservation_availability_mask()
    )

    assert mask.selectable == (
        True,
        False,
        True,
    )

    assert mask.selectable_bin_ids == (
        0,
        2,
    )

    assert mask.is_selectable(0)
    assert not mask.is_selectable(1)
    assert mask.is_selectable(2)


def test_available_bins_follow_stable_id_order() -> None:
    state = make_state()
    book = BinReservationBook()

    book.reserve(
        bin_id=1,
        truck_id=7,
    )

    view = build_policy_view(
        state,
        reservations=book.snapshot(),
    )

    assert [
        bin_.bin_id
        for bin_ in view.available_bins()
    ] == [
        0,
        2,
    ]

    assert view.available_bin_ids() == (
        0,
        2,
    )


def test_all_reserved_bins_produce_empty_availability() -> None:
    state = make_state()

    view = build_policy_view(
        state,
        reservations={
            0: 10,
            1: 11,
            2: 12,
        },
    )

    mask = (
        view.reservation_availability_mask()
    )

    assert mask.selectable == (
        False,
        False,
        False,
    )

    assert mask.selectable_bin_ids == ()
    assert view.available_bins() == ()
    assert view.available_bin_ids() == ()


def test_mask_lookup_rejects_unknown_bin() -> None:
    view = build_policy_view(
        make_state()
    )

    mask = (
        view.reservation_availability_mask()
    )

    with pytest.raises(KeyError):
        mask.is_selectable(999)


def test_availability_mask_does_not_encode_capacity_physics() -> None:
    """
    Reservation availability must remain separate from physical
    feasibility.

    Even a nearly full truck still sees an unreserved bin as
    policy-selectable. The dispatcher is responsible for forcing
    a depot return if capacity is insufficient.
    """

    state = make_state(
        truck_load_tonnes=9.9,
    )

    view = build_policy_view(
        state
    )

    mask = (
        view.reservation_availability_mask()
    )

    assert mask.selectable == (
        True,
        True,
        True,
    )


def test_availability_mask_does_not_encode_fuel_physics() -> None:
    """
    Fuel constraints remain exclusively under the dispatcher.
    """

    state = make_state(
        fuel_litres=0.0,
    )

    view = build_policy_view(
        state
    )

    mask = (
        view.reservation_availability_mask()
    )

    assert mask.selectable == (
        True,
        True,
        True,
    )


def test_policy_snapshot_mask_is_independent_of_later_reservations() -> None:
    state = make_state()
    book = BinReservationBook()

    original_view = build_policy_view(
        state,
        reservations=book.snapshot(),
    )

    book.reserve(
        bin_id=1,
        truck_id=7,
    )

    original_mask = (
        original_view.reservation_availability_mask()
    )

    assert original_mask.selectable == (
        True,
        True,
        True,
    )

    new_view = build_policy_view(
        state,
        reservations=book.snapshot(),
    )

    assert (
        new_view
        .reservation_availability_mask()
        .selectable
    ) == (
        True,
        False,
        True,
    )
