import networkx as nx
import pytest

from smart_waste.collection.base import (
    BinPolicyView,
    PolicyView,
)
from smart_waste.collection.tsr import (
    TSRZone,
    balance_adjacent_zones,
)
from smart_waste.movement.road_graph import RoadGraph


def make_star_case(
    *,
    weights: dict[int, float],
    x_coordinates: dict[int, float],
    zones: tuple[TSRZone, ...],
) -> tuple[
    PolicyView,
    RoadGraph,
    tuple[TSRZone, ...],
]:
    """
    Star road network.

    depot -> bin_i edge = weight_i

    Therefore for any zone:

        closed route distance
        = 2 * sum(zone edge weights)

    independent of route ordering.

    This makes balancing-objective tests exact and transparent.
    """

    graph = nx.Graph()

    graph.add_node(
        "depot",
        x=-1.0,
        y=0.0,
    )

    for bin_id, weight in weights.items():
        node = f"bin-{bin_id}"

        graph.add_node(
            node,
            x=x_coordinates[bin_id],
            y=0.0,
        )

        graph.add_edge(
            "depot",
            node,
            length_km=weight,
        )

    road = RoadGraph(
        graph
    )

    view = PolicyView(
        current_time_hours=0.0,
        depot_node="depot",
        bins=tuple(
            BinPolicyView(
                bin_id=bin_id,
                road_node=f"bin-{bin_id}",
                fill_percent=50.0,
                fill_rate_percent_per_hour=0.0,
                waste_mass_tonnes=0.22,
            )
            for bin_id in sorted(
                weights
            )
        ),
        trucks=(),
    )

    return (
        view,
        road,
        zones,
    )


def balanced_two_zone_case():
    return make_star_case(
        weights={
            0: 10.0,
            1: 10.0,
            2: 1.0,
            3: 1.0,
        },
        x_coordinates={
            0: 0.0,
            1: 1.0,
            2: 2.0,
            3: 3.0,
        },
        zones=(
            TSRZone(
                truck_id=0,
                bin_ids=(
                    0,
                    1,
                ),
            ),
            TSRZone(
                truck_id=1,
                bin_ids=(
                    2,
                    3,
                ),
            ),
        ),
    )


def test_balancing_strictly_improves_distance_range() -> None:
    view, road, zones = (
        balanced_two_zone_case()
    )

    result = balance_adjacent_zones(
        zones=zones,
        view=view,
        road_graph=road,
        candidate_count=2,
        target_relative_range=0.10,
    )

    assert (
        result.final_objective.distance_range_km
        < result.initial_objective.distance_range_km
    )

    assert (
        result.final_objective.total_distance_km
        <= result.initial_objective.total_distance_km
        + 1.0e-12
    )


def test_balancing_preserves_zone_sizes_and_bin_population() -> None:
    view, road, zones = (
        balanced_two_zone_case()
    )

    result = balance_adjacent_zones(
        zones=zones,
        view=view,
        road_graph=road,
        candidate_count=2,
        target_relative_range=0.10,
    )

    assert [
        len(zone.bin_ids)
        for zone in result.final_zones
    ] == [
        2,
        2,
    ]

    final_bins = sorted(
        bin_id
        for zone in result.final_zones
        for bin_id in zone.bin_ids
    )

    assert final_bins == [
        0,
        1,
        2,
        3,
    ]


def test_balancing_can_meet_target() -> None:
    view, road, zones = (
        balanced_two_zone_case()
    )

    result = balance_adjacent_zones(
        zones=zones,
        view=view,
        road_graph=road,
        candidate_count=2,
        target_relative_range=0.10,
    )

    assert result.target_met

    assert (
        result.final_objective.relative_range
        <= 0.10 + 1.0e-12
    )

    # One 10-weight bin exchanged with one 1-weight bin gives
    # both zones total edge weight 11 -> closed route 22 km.
    distances = sorted(
        route.optimized_distance_km
        for route in result.final_routes
    )

    assert distances == pytest.approx(
        [
            22.0,
            22.0,
        ]
    )


def test_equal_objective_swap_is_not_accepted() -> None:
    view, road, zones = make_star_case(
        weights={
            0: 10.0,
            1: 1.0,
        },
        x_coordinates={
            0: 0.0,
            1: 1.0,
        },
        zones=(
            TSRZone(
                truck_id=0,
                bin_ids=(0,),
            ),
            TSRZone(
                truck_id=1,
                bin_ids=(1,),
            ),
        ),
    )

    result = balance_adjacent_zones(
        zones=zones,
        view=view,
        road_graph=road,
        candidate_count=1,
        target_relative_range=0.10,
    )

    # Swapping the only bins merely swaps route lengths.
    # Global objective is unchanged, so no exchange is accepted.
    assert result.exchanges == ()
    assert not result.target_met

    assert (
        result.final_objective
        == result.initial_objective
    )

    assert (
        result.final_zones
        == result.initial_zones
    )


def test_accepted_exchanges_are_adjacent_only() -> None:
    view, road, zones = make_star_case(
        weights={
            0: 10.0,
            1: 10.0,
            2: 1.0,
            3: 1.0,
            4: 30.0,
            5: 30.0,
        },
        x_coordinates={
            0: 0.0,
            1: 1.0,
            2: 2.0,
            3: 3.0,
            4: 4.0,
            5: 5.0,
        },
        zones=(
            TSRZone(
                truck_id=0,
                bin_ids=(
                    0,
                    1,
                ),
            ),
            TSRZone(
                truck_id=1,
                bin_ids=(
                    2,
                    3,
                ),
            ),
            TSRZone(
                truck_id=2,
                bin_ids=(
                    4,
                    5,
                ),
            ),
        ),
    )

    result = balance_adjacent_zones(
        zones=zones,
        view=view,
        road_graph=road,
        candidate_count=2,
        target_relative_range=0.10,
    )

    allowed_pairs = {
        (0, 1),
        (1, 2),
    }

    assert all(
        (
            exchange.left_truck_id,
            exchange.right_truck_id,
        )
        in allowed_pairs
        for exchange in result.exchanges
    )


def test_balancing_is_exactly_deterministic() -> None:
    view, road, zones = (
        balanced_two_zone_case()
    )

    first = balance_adjacent_zones(
        zones=zones,
        view=view,
        road_graph=road,
        candidate_count=2,
        target_relative_range=0.10,
    )

    second = balance_adjacent_zones(
        zones=zones,
        view=view,
        road_graph=road,
        candidate_count=2,
        target_relative_range=0.10,
    )

    assert first == second


def test_final_routes_match_final_zone_memberships() -> None:
    view, road, zones = (
        balanced_two_zone_case()
    )

    result = balance_adjacent_zones(
        zones=zones,
        view=view,
        road_graph=road,
        candidate_count=2,
        target_relative_range=0.10,
    )

    zones_by_truck = {
        zone.truck_id: set(
            zone.bin_ids
        )
        for zone in result.final_zones
    }

    for route in result.final_routes:
        assert set(
            route.optimized_bin_ids
        ) == zones_by_truck[
            route.truck_id
        ]


def test_already_balanced_case_performs_no_exchange() -> None:
    view, road, zones = make_star_case(
        weights={
            0: 5.0,
            1: 5.0,
            2: 5.0,
            3: 5.0,
        },
        x_coordinates={
            0: 0.0,
            1: 1.0,
            2: 2.0,
            3: 3.0,
        },
        zones=(
            TSRZone(
                truck_id=0,
                bin_ids=(
                    0,
                    1,
                ),
            ),
            TSRZone(
                truck_id=1,
                bin_ids=(
                    2,
                    3,
                ),
            ),
        ),
    )

    result = balance_adjacent_zones(
        zones=zones,
        view=view,
        road_graph=road,
        candidate_count=2,
        target_relative_range=0.10,
    )

    assert result.target_met
    assert result.exchanges == ()
    assert result.iterations == 0


def test_balancing_rejects_nonpositive_candidate_count() -> None:
    view, road, zones = (
        balanced_two_zone_case()
    )

    with pytest.raises(
        ValueError,
        match="candidate_count",
    ):
        balance_adjacent_zones(
            zones=zones,
            view=view,
            road_graph=road,
            candidate_count=0,
        )
