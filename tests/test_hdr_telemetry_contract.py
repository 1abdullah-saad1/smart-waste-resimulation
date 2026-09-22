import networkx as nx
import pytest

from smart_waste.collection.base import (
    BinPolicyView,
    PolicyView,
)
from smart_waste.collection.hdr import (
    HDRConfigurationError,
    HDREligibilitySnapshot,
    HDRTelemetryError,
    build_hdr_eligibility_snapshot,
)
from smart_waste.models.bin import WasteBin
from smart_waste.models.depot import Depot
from smart_waste.models.truck import Truck
from smart_waste.movement.road_graph import RoadGraph
from smart_waste.simulation.policy_view import (
    PolicyViewError,
    build_policy_view,
)
from smart_waste.simulation.state import (
    SimulationState,
)


def make_state() -> SimulationState:
    graph = nx.Graph()

    graph.add_node(
        "depot",
        x=0.0,
        y=0.0,
    )

    fills = {
        0: 79.0,
        1: 80.0,
        2: 95.0,
    }

    bins = {}

    for bin_id, fill in fills.items():
        node = f"bin-{bin_id}"

        graph.add_node(
            node,
            x=float(bin_id + 1),
            y=0.0,
        )

        graph.add_edge(
            "depot",
            node,
            length_km=1.0,
        )

        bins[bin_id] = WasteBin(
            bin_id=bin_id,
            road_node=node,
            fill_percent=fill,
            fill_rate_percent_per_hour=0.0,
            full_mass_kg=440.0,
        )

    truck = Truck(
        truck_id=0,
        current_node="depot",
        capacity_tonnes=10.0,
        speed_km_per_hour=30.0,
        fuel_capacity_litres=200.0,
        fuel_efficiency_km_per_litre=2.5,
    )

    return SimulationState(
        road_graph=RoadGraph(
            graph
        ),
        bins=bins,
        trucks={
            0: truck,
        },
        depot=Depot(
            road_node="depot",
            unloading_bays=1,
            unload_time_minutes=11.0,
            refuel_rate_litres_per_minute=60.0,
        ),
    )


def test_nominal_policy_view_reports_true_fill() -> None:
    state = make_state()

    view = build_policy_view(
        state
    )

    for bin_view in view.bins:
        assert (
            bin_view.reported_fill_percent
            == pytest.approx(
                bin_view.fill_percent
            )
        )

        assert (
            bin_view.routing_fill_percent
            == pytest.approx(
                bin_view.true_fill_percent
            )
        )


def test_reported_override_does_not_mutate_physical_truth() -> None:
    state = make_state()

    original_true_fill = (
        state.bins[0].fill_percent
    )

    original_mass = (
        state.bins[0].waste_mass_tonnes
    )

    view = build_policy_view(
        state,
        reported_fill_percent={
            0: 100.0,
        },
    )

    bin_view = view.bin_by_id(
        0
    )

    assert bin_view.true_fill_percent == pytest.approx(
        79.0
    )

    assert bin_view.routing_fill_percent == pytest.approx(
        100.0
    )

    assert state.bins[0].fill_percent == pytest.approx(
        original_true_fill
    )

    assert (
        state.bins[0].waste_mass_tonnes
        == pytest.approx(
            original_mass
        )
    )


def test_partial_reported_override_falls_back_to_true_fill() -> None:
    state = make_state()

    view = build_policy_view(
        state,
        reported_fill_percent={
            0: 100.0,
        },
    )

    assert (
        view.bin_by_id(0).routing_fill_percent
        == pytest.approx(100.0)
    )

    assert (
        view.bin_by_id(1).routing_fill_percent
        == pytest.approx(80.0)
    )

    assert (
        view.bin_by_id(2).routing_fill_percent
        == pytest.approx(95.0)
    )


def test_unknown_reported_bin_is_rejected() -> None:
    state = make_state()

    with pytest.raises(
        PolicyViewError,
        match="unknown bin IDs",
    ):
        build_policy_view(
            state,
            reported_fill_percent={
                999: 100.0,
            },
        )


def test_out_of_range_reported_fill_is_rejected() -> None:
    state = make_state()

    with pytest.raises(
        PolicyViewError,
        match="between 0 and 100",
    ):
        build_policy_view(
            state,
            reported_fill_percent={
                0: 101.0,
            },
        )


def test_hdr_threshold_is_inclusive_at_80_percent() -> None:
    state = make_state()

    snapshot = (
        build_hdr_eligibility_snapshot(
            view=build_policy_view(
                state
            ),
            threshold_percent=80.0,
        )
    )

    assert snapshot.eligible_bin_ids == (
        1,
        2,
    )

    assert not snapshot.is_eligible(
        0
    )

    assert snapshot.is_eligible(
        1
    )


def test_hdr_eligibility_uses_reported_not_true_fill() -> None:
    state = make_state()

    view = build_policy_view(
        state,
        reported_fill_percent={
            # Physically 79%, but reported as full.
            0: 100.0,

            # Physically 95%, but reported below threshold.
            2: 20.0,
        },
    )

    snapshot = (
        build_hdr_eligibility_snapshot(
            view=view
        )
    )

    assert snapshot.eligible_bin_ids == (
        0,
        1,
    )

    assert (
        state.bins[0].fill_percent
        == pytest.approx(79.0)
    )

    assert (
        state.bins[2].fill_percent
        == pytest.approx(95.0)
    )


def test_hdr_snapshot_is_sorted_by_bin_id() -> None:
    view = PolicyView(
        current_time_hours=0.0,
        depot_node="depot",
        bins=(
            BinPolicyView(
                bin_id=9,
                road_node="n9",
                fill_percent=90.0,
                fill_rate_percent_per_hour=0.0,
                waste_mass_tonnes=0.396,
                reported_fill_percent=90.0,
            ),
            BinPolicyView(
                bin_id=2,
                road_node="n2",
                fill_percent=90.0,
                fill_rate_percent_per_hour=0.0,
                waste_mass_tonnes=0.396,
                reported_fill_percent=90.0,
            ),
            BinPolicyView(
                bin_id=5,
                road_node="n5",
                fill_percent=90.0,
                fill_rate_percent_per_hour=0.0,
                waste_mass_tonnes=0.396,
                reported_fill_percent=90.0,
            ),
        ),
        trucks=(),
    )

    snapshot = (
        build_hdr_eligibility_snapshot(
            view=view
        )
    )

    assert snapshot.eligible_bin_ids == (
        2,
        5,
        9,
    )

    assert tuple(
        bin_id
        for (
            bin_id,
            _,
        ) in snapshot.reported_fill_by_bin
    ) == (
        2,
        5,
        9,
    )


def test_hdr_snapshot_can_be_empty() -> None:
    view = PolicyView(
        current_time_hours=0.0,
        depot_node="depot",
        bins=(
            BinPolicyView(
                bin_id=0,
                road_node="n0",
                fill_percent=10.0,
                fill_rate_percent_per_hour=0.0,
                waste_mass_tonnes=0.044,
                reported_fill_percent=10.0,
            ),
        ),
        trucks=(),
    )

    snapshot = (
        build_hdr_eligibility_snapshot(
            view=view
        )
    )

    assert snapshot.eligible_bin_ids == ()


def test_hdr_snapshot_is_immutable_from_later_views() -> None:
    state = make_state()

    first_view = build_policy_view(
        state
    )

    snapshot = (
        build_hdr_eligibility_snapshot(
            view=first_view
        )
    )

    second_view = build_policy_view(
        state,
        reported_fill_percent={
            0: 100.0,
            1: 0.0,
            2: 0.0,
        },
    )

    assert (
        build_hdr_eligibility_snapshot(
            view=second_view
        ).eligible_bin_ids
        == (
            0,
        )
    )

    # Original frozen snapshot must not change.
    assert snapshot.eligible_bin_ids == (
        1,
        2,
    )


def test_invalid_hdr_threshold_is_rejected() -> None:
    state = make_state()

    with pytest.raises(
        HDRConfigurationError,
        match="between 0 and 100",
    ):
        build_hdr_eligibility_snapshot(
            view=build_policy_view(
                state
            ),
            threshold_percent=120.0,
        )


def test_direct_policy_view_without_reported_value_falls_back_to_true() -> None:
    bin_view = BinPolicyView(
        bin_id=7,
        road_node="n7",
        fill_percent=82.0,
        fill_rate_percent_per_hour=0.0,
        waste_mass_tonnes=0.3608,
    )

    assert bin_view.reported_fill_percent is None

    assert (
        bin_view.routing_fill_percent
        == pytest.approx(82.0)
    )

    snapshot = (
        build_hdr_eligibility_snapshot(
            view=PolicyView(
                current_time_hours=0.0,
                depot_node="depot",
                bins=(
                    bin_view,
                ),
                trucks=(),
            )
        )
    )

    assert snapshot.eligible_bin_ids == (
        7,
    )
