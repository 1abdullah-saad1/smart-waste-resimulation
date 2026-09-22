import networkx as nx
import pytest

from smart_waste.collection.base import (
    BinPolicyView,
    PolicyView,
    TruckPolicyView,
)
from smart_waste.collection.tsr import (
    balance_adjacent_zones,
    build_longitudinal_zones,
    build_tsr_distance_oracle,
    validate_primary_tsr_partition,
)
from smart_waste.models.truck import TruckStatus
from smart_waste.movement.road_graph import RoadGraph


BIN_COUNT = 1000
TRUCK_COUNT = 10
BINS_PER_TRUCK = 100


def make_primary_scale_fixture() -> tuple[
    PolicyView,
    RoadGraph,
]:
    """
    Construct a benchmark-scale synthetic road network.

    Ten independent 100-bin road rays share one central depot.

    Coordinates are intentionally used only for deterministic
    longitudinal zoning. Physical TSR routing uses graph
    length_km exclusively.
    """

    graph = nx.Graph()

    graph.add_node(
        "depot",
        x=-1.0,
        y=0.0,
    )

    bins: list[
        BinPolicyView
    ] = []

    for truck_id in range(
        TRUCK_COUNT
    ):
        previous_node = "depot"

        for local_index in range(
            BINS_PER_TRUCK
        ):
            bin_id = (
                truck_id
                * BINS_PER_TRUCK
                + local_index
            )

            node = f"bin-{bin_id}"

            graph.add_node(
                node,
                x=float(bin_id),
                y=float(truck_id),
            )

            graph.add_edge(
                previous_node,
                node,
                length_km=1.0,
            )

            previous_node = node

            bins.append(
                BinPolicyView(
                    bin_id=bin_id,
                    road_node=node,
                    fill_percent=50.0,
                    fill_rate_percent_per_hour=0.0,
                    waste_mass_tonnes=0.22,
                )
            )

    road_graph = RoadGraph(
        graph
    )

    trucks = tuple(
        TruckPolicyView(
            truck_id=truck_id,
            current_node="depot",
            remaining_capacity_tonnes=10.0,
            fuel_remaining_litres=200.0,
            status=TruckStatus.IDLE,
        )
        for truck_id in range(
            TRUCK_COUNT
        )
    )

    view = PolicyView(
        current_time_hours=0.0,
        depot_node="depot",
        bins=tuple(
            reversed(
                bins
            )
        ),
        trucks=trucks,
    )

    return (
        view,
        road_graph,
    )


def test_primary_1000_bin_10_truck_tsr_acceptance() -> None:
    view, road_graph = (
        make_primary_scale_fixture()
    )

    assert len(view.bins) == BIN_COUNT
    assert len(view.trucks) == TRUCK_COUNT

    zones = build_longitudinal_zones(
        view=view,
        road_graph=road_graph,
        truck_ids=range(
            TRUCK_COUNT
        ),
    )

    validate_primary_tsr_partition(
        zones
    )

    assert len(zones) == TRUCK_COUNT

    assert all(
        len(zone.bin_ids)
        == BINS_PER_TRUCK
        for zone in zones
    )

    # Exact deterministic initial longitudinal partition.
    for truck_id, zone in enumerate(
        zones
    ):
        expected = tuple(
            range(
                truck_id
                * BINS_PER_TRUCK,
                (
                    truck_id
                    + 1
                )
                * BINS_PER_TRUCK,
            )
        )

        assert zone.truck_id == truck_id
        assert zone.bin_ids == expected

    oracle = build_tsr_distance_oracle(
        view=view,
        road_graph=road_graph,
    )

    result = balance_adjacent_zones(
        zones=zones,
        view=view,
        road_graph=road_graph,
        candidate_count=10,
        target_relative_range=0.10,
        two_opt_epsilon=1.0e-12,
        distance_oracle=oracle,
    )

    # All ten synthetic rays are physically identical, therefore
    # the initial partition is already perfectly balanced.
    assert result.target_met
    assert result.exchanges == ()
    assert result.iterations == 0

    assert (
        result.initial_objective.relative_range
        == pytest.approx(0.0)
    )

    assert (
        result.final_objective.relative_range
        == pytest.approx(0.0)
    )

    assert (
        result.final_zones
        == result.initial_zones
    )

    # Each 100-bin ray:
    #
    # depot -> first bin       =   1 km
    # 99 consecutive bin legs =  99 km
    # last bin -> depot        = 100 km
    #
    # closed route            = 200 km
    assert len(
        result.final_routes
    ) == TRUCK_COUNT

    for route in result.final_routes:
        assert len(
            route.optimized_bin_ids
        ) == BINS_PER_TRUCK

        assert (
            route.optimized_distance_km
            == pytest.approx(200.0)
        )

        assert (
            route.optimized_distance_km
            <= route.initial_nn_distance_km
            + 1.0e-12
        )

    # Global population preservation.
    final_bin_ids = [
        bin_id
        for zone in result.final_zones
        for bin_id in zone.bin_ids
    ]

    assert len(
        final_bin_ids
    ) == BIN_COUNT

    assert len(
        set(final_bin_ids)
    ) == BIN_COUNT

    assert sorted(
        final_bin_ids
    ) == list(
        range(
            BIN_COUNT
        )
    )

    # The same invariant must hold for finalized routes.
    routed_bin_ids = [
        bin_id
        for route in result.final_routes
        for bin_id in route.optimized_bin_ids
    ]

    assert len(
        routed_bin_ids
    ) == BIN_COUNT

    assert len(
        set(routed_bin_ids)
    ) == BIN_COUNT

    assert sorted(
        routed_bin_ids
    ) == list(
        range(
            BIN_COUNT
        )
    )

    # Primary-scale distance-cache acceptance:
    #
    # one central depot + 1000 unique bin road nodes.
    assert len(
        oracle.service_nodes
    ) == 1001

    # Every required source may execute Dijkstra at most once.
    assert (
        oracle.source_search_count
        <= len(
            oracle.service_nodes
        )
    )

    assert (
        oracle.cached_source_count
        == oracle.source_search_count
    )

    # For ten 100-bin matrices, all bin sources and the depot are
    # required, so this synthetic case should populate all 1001
    # sources exactly once.
    assert (
        oracle.source_search_count
        == 1001
    )
