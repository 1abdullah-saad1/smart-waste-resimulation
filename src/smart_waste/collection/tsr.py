from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from numbers import Real
from typing import Sequence

import networkx as nx

from smart_waste.collection.base import PolicyView
from smart_waste.movement.road_graph import RoadGraph


class TSRPlanningError(RuntimeError):
    """Raised when deterministic TSR planning cannot be constructed."""


@dataclass(frozen=True)
class TSRZone:
    """
    Static longitudinal service zone assigned to one truck.
    """

    truck_id: int
    bin_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.bin_ids:
            raise ValueError(
                "TSR zone cannot be empty"
            )

        if len(set(self.bin_ids)) != len(
            self.bin_ids
        ):
            raise ValueError(
                "TSR zone contains duplicate bin IDs"
            )


@dataclass(frozen=True)
class TSRDistanceMatrix:
    """
    Shortest-road distances for one static TSR zone.

    Index ordering follows bin_ids exactly.

    All route distances are graph-road distances and include
    depot departure/return when route_distance_km() is used.
    """

    bin_ids: tuple[int, ...]
    depot_distances_km: tuple[float, ...]
    pairwise_distances_km: tuple[
        tuple[float, ...],
        ...,
    ]

    def __post_init__(self) -> None:
        n = len(self.bin_ids)

        if n == 0:
            raise ValueError(
                "distance matrix cannot be empty"
            )

        if len(set(self.bin_ids)) != n:
            raise ValueError(
                "distance matrix bin IDs must be unique"
            )

        if len(self.depot_distances_km) != n:
            raise ValueError(
                "depot distance vector has wrong size"
            )

        if len(self.pairwise_distances_km) != n:
            raise ValueError(
                "pairwise matrix has wrong row count"
            )

        if any(
            len(row) != n
            for row in self.pairwise_distances_km
        ):
            raise ValueError(
                "pairwise matrix must be square"
            )

    def _index(
        self,
        bin_id: int,
    ) -> int:
        try:
            return self.bin_ids.index(
                bin_id
            )
        except ValueError as exc:
            raise KeyError(
                f"unknown bin_id: {bin_id}"
            ) from exc

    def depot_to_bin_km(
        self,
        bin_id: int,
    ) -> float:
        return self.depot_distances_km[
            self._index(bin_id)
        ]

    def between_bins_km(
        self,
        first_bin_id: int,
        second_bin_id: int,
    ) -> float:
        i = self._index(
            first_bin_id
        )

        j = self._index(
            second_bin_id
        )

        return self.pairwise_distances_km[
            i
        ][j]

    def leg_distance_km(
        self,
        first_bin_id: int | None,
        second_bin_id: int | None,
    ) -> float:
        """
        None represents the central depot.
        """

        if (
            first_bin_id is None
            and second_bin_id is None
        ):
            return 0.0

        if first_bin_id is None:
            assert second_bin_id is not None

            return self.depot_to_bin_km(
                second_bin_id
            )

        if second_bin_id is None:
            return self.depot_to_bin_km(
                first_bin_id
            )

        return self.between_bins_km(
            first_bin_id,
            second_bin_id,
        )

    def route_distance_km(
        self,
        route_bin_ids: Sequence[int],
    ) -> float:
        """
        Closed route:

            depot -> bins in supplied order -> depot
        """

        route = tuple(
            route_bin_ids
        )

        if not route:
            return 0.0

        if len(set(route)) != len(route):
            raise ValueError(
                "route contains duplicate bin IDs"
            )

        unknown = (
            set(route)
            - set(self.bin_ids)
        )

        if unknown:
            raise KeyError(
                f"route contains unknown bin IDs: "
                f"{sorted(unknown)}"
            )

        total = self.depot_to_bin_km(
            route[0]
        )

        for first_bin, second_bin in zip(
            route,
            route[1:],
        ):
            total += self.between_bins_km(
                first_bin,
                second_bin,
            )

        total += self.depot_to_bin_km(
            route[-1]
        )

        return float(total)


@dataclass(frozen=True)
class TSRRoute:
    truck_id: int

    initial_nn_bin_ids: tuple[int, ...]
    optimized_bin_ids: tuple[int, ...]

    initial_nn_distance_km: float
    optimized_distance_km: float

    @property
    def improvement_km(self) -> float:
        return (
            self.initial_nn_distance_km
            - self.optimized_distance_km
        )


def _road_node_xy(
    road_graph: RoadGraph,
    road_node: int | str,
) -> tuple[float, float]:
    if road_node not in road_graph.graph:
        raise TSRPlanningError(
            f"road node does not exist: {road_node!r}"
        )

    attributes = (
        road_graph.graph.nodes[
            road_node
        ]
    )

    if (
        "x" not in attributes
        or "y" not in attributes
    ):
        raise TSRPlanningError(
            f"road node {road_node!r} is missing x/y coordinates"
        )

    x = attributes["x"]
    y = attributes["y"]

    if (
        not isinstance(x, Real)
        or not isinstance(y, Real)
    ):
        raise TSRPlanningError(
            f"road node {road_node!r} has nonnumeric x/y coordinates"
        )

    x_value = float(x)
    y_value = float(y)

    if (
        not isfinite(x_value)
        or not isfinite(y_value)
    ):
        raise TSRPlanningError(
            f"road node {road_node!r} has nonfinite x/y coordinates"
        )

    return (
        x_value,
        y_value,
    )


def build_longitudinal_zones(
    *,
    view: PolicyView,
    road_graph: RoadGraph,
    truck_ids: Sequence[int],
) -> tuple[TSRZone, ...]:
    """
    Deterministic initial TSR partition.

    Bins are sorted lexicographically by:

        (x, y, bin_id)

    then split into contiguous equal-sized zones.

    For the primary benchmark:
        1000 bins / 10 trucks = 100 bins per truck.
    """

    ordered_truck_ids = tuple(
        sorted(
            truck_ids
        )
    )

    if not ordered_truck_ids:
        raise TSRPlanningError(
            "TSR requires at least one truck"
        )

    if len(set(ordered_truck_ids)) != len(
        ordered_truck_ids
    ):
        raise TSRPlanningError(
            "truck IDs must be unique"
        )

    bin_count = len(
        view.bins
    )

    truck_count = len(
        ordered_truck_ids
    )

    if bin_count == 0:
        raise TSRPlanningError(
            "TSR requires at least one bin"
        )

    if bin_count % truck_count != 0:
        raise TSRPlanningError(
            "bin count must be exactly divisible by truck count"
        )

    bins_per_truck = (
        bin_count
        // truck_count
    )

    sortable: list[
        tuple[
            float,
            float,
            int,
        ]
    ] = []

    for bin_ in view.bins:
        x, y = _road_node_xy(
            road_graph,
            bin_.road_node,
        )

        sortable.append(
            (
                x,
                y,
                bin_.bin_id,
            )
        )

    sortable.sort()

    ordered_bin_ids = tuple(
        item[2]
        for item in sortable
    )

    zones: list[
        TSRZone
    ] = []

    for zone_index, truck_id in enumerate(
        ordered_truck_ids
    ):
        start = (
            zone_index
            * bins_per_truck
        )

        end = (
            start
            + bins_per_truck
        )

        zones.append(
            TSRZone(
                truck_id=truck_id,
                bin_ids=ordered_bin_ids[
                    start:end
                ],
            )
        )

    assigned = tuple(
        bin_id
        for zone in zones
        for bin_id in zone.bin_ids
    )

    if len(assigned) != bin_count:
        raise TSRPlanningError(
            "TSR zone assignment lost bins"
        )

    if len(set(assigned)) != bin_count:
        raise TSRPlanningError(
            "TSR zone assignment duplicated bins"
        )

    return tuple(
        zones
    )


def validate_primary_tsr_partition(
    zones: Sequence[TSRZone],
) -> None:
    """
    Validate the fixed primary benchmark definition:

        10 trucks
        exactly 100 bins per truck
        exactly 1000 bins total
    """

    zones_tuple = tuple(
        zones
    )

    if len(zones_tuple) != 10:
        raise TSRPlanningError(
            "primary TSR benchmark requires exactly 10 zones"
        )

    if any(
        len(zone.bin_ids) != 100
        for zone in zones_tuple
    ):
        raise TSRPlanningError(
            "primary TSR benchmark requires exactly "
            "100 bins per truck"
        )

    all_bin_ids = tuple(
        bin_id
        for zone in zones_tuple
        for bin_id in zone.bin_ids
    )

    if len(all_bin_ids) != 1000:
        raise TSRPlanningError(
            "primary TSR benchmark requires exactly 1000 bins"
        )

    if len(set(all_bin_ids)) != 1000:
        raise TSRPlanningError(
            "primary TSR benchmark bins must be unique"
        )


def build_zone_distance_matrix(
    *,
    zone: TSRZone,
    view: PolicyView,
    road_graph: RoadGraph,
) -> TSRDistanceMatrix:
    """
    Build shortest-road distance matrix for one zone.

    Dijkstra is run once per unique physical road node rather than
    once per bin pair.
    """

    bin_nodes = {
        bin_id: view.bin_by_id(
            bin_id
        ).road_node
        for bin_id in zone.bin_ids
    }

    service_nodes = {
        view.depot_node,
        *bin_nodes.values(),
    }

    try:
        road_graph.validate_service_nodes(
            service_nodes
        )
    except ValueError as exc:
        raise TSRPlanningError(
            str(exc)
        ) from exc

    graph = road_graph.graph
    weight = road_graph.length_attribute

    unique_sources = tuple(
        dict.fromkeys(
            (
                view.depot_node,
                *(
                    bin_nodes[
                        bin_id
                    ]
                    for bin_id in zone.bin_ids
                ),
            )
        )
    )

    distance_cache: dict[
        int | str,
        dict[
            int | str,
            float,
        ],
    ] = {}

    for source in unique_sources:
        lengths = (
            nx.single_source_dijkstra_path_length(
                graph,
                source,
                weight=weight,
            )
        )

        missing = (
            service_nodes
            - set(lengths)
        )

        if missing:
            raise TSRPlanningError(
                f"road network does not connect source "
                f"{source!r} to service nodes: "
                f"{sorted(missing, key=str)}"
            )

        distance_cache[
            source
        ] = {
            target: float(distance)
            for target, distance in lengths.items()
        }

    depot_distances = tuple(
        distance_cache[
            view.depot_node
        ][
            bin_nodes[
                bin_id
            ]
        ]
        for bin_id in zone.bin_ids
    )

    pairwise = tuple(
        tuple(
            distance_cache[
                bin_nodes[
                    first_bin_id
                ]
            ][
                bin_nodes[
                    second_bin_id
                ]
            ]
            for second_bin_id in zone.bin_ids
        )
        for first_bin_id in zone.bin_ids
    )

    return TSRDistanceMatrix(
        bin_ids=zone.bin_ids,
        depot_distances_km=(
            depot_distances
        ),
        pairwise_distances_km=(
            pairwise
        ),
    )


def nearest_neighbor_route(
    matrix: TSRDistanceMatrix,
) -> tuple[int, ...]:
    """
    Deterministic depot-origin nearest-neighbor route.

    Equal-distance ties are broken by ascending bin_id.
    """

    remaining = set(
        matrix.bin_ids
    )

    route: list[int] = []

    current_bin_id: int | None = None

    while remaining:
        next_bin_id = min(
            remaining,
            key=lambda candidate: (
                matrix.leg_distance_km(
                    current_bin_id,
                    candidate,
                ),
                candidate,
            ),
        )

        route.append(
            next_bin_id
        )

        remaining.remove(
            next_bin_id
        )

        current_bin_id = (
            next_bin_id
        )

    return tuple(
        route
    )


def deterministic_two_opt(
    matrix: TSRDistanceMatrix,
    route_bin_ids: Sequence[int],
    *,
    epsilon: float = 1.0e-12,
) -> tuple[int, ...]:
    """
    Deterministic best-improvement 2-opt for a closed depot route.

    Internal segment reversal preserves its internal cost because
    the road graph is undirected.

    At each iteration:
    1. examine every possible reversal,
    2. choose the greatest distance reduction,
    3. for numerically tied improvements choose the
       lexicographically smallest resulting route.

    Stop when no improvement greater than epsilon remains.
    """

    if epsilon < 0.0:
        raise ValueError(
            "epsilon cannot be negative"
        )

    route = tuple(
        route_bin_ids
    )

    if len(route) != len(
        matrix.bin_ids
    ):
        raise ValueError(
            "2-opt route must contain every matrix bin exactly once"
        )

    if (
        len(set(route))
        != len(route)
        or set(route)
        != set(matrix.bin_ids)
    ):
        raise ValueError(
            "2-opt route must be a permutation of matrix bin IDs"
        )

    if len(route) < 2:
        return route

    while True:
        best_delta = 0.0
        best_route: tuple[
            int,
            ...,
        ] | None = None

        n = len(route)

        for i in range(n - 1):
            previous_bin = (
                None
                if i == 0
                else route[i - 1]
            )

            first_reversed_bin = (
                route[i]
            )

            for j in range(
                i + 1,
                n,
            ):
                last_reversed_bin = (
                    route[j]
                )

                following_bin = (
                    None
                    if j == n - 1
                    else route[j + 1]
                )

                old_boundary = (
                    matrix.leg_distance_km(
                        previous_bin,
                        first_reversed_bin,
                    )
                    + matrix.leg_distance_km(
                        last_reversed_bin,
                        following_bin,
                    )
                )

                new_boundary = (
                    matrix.leg_distance_km(
                        previous_bin,
                        last_reversed_bin,
                    )
                    + matrix.leg_distance_km(
                        first_reversed_bin,
                        following_bin,
                    )
                )

                delta = (
                    new_boundary
                    - old_boundary
                )

                if delta >= -epsilon:
                    continue

                candidate = (
                    route[:i]
                    + tuple(
                        reversed(
                            route[
                                i:j + 1
                            ]
                        )
                    )
                    + route[
                        j + 1:
                    ]
                )

                if (
                    best_route is None
                    or delta
                    < best_delta
                    - epsilon
                    or (
                        abs(
                            delta
                            - best_delta
                        )
                        <= epsilon
                        and candidate
                        < best_route
                    )
                ):
                    best_delta = delta
                    best_route = candidate

        if best_route is None:
            break

        route = best_route

    return route


def plan_zone_route(
    *,
    zone: TSRZone,
    view: PolicyView,
    road_graph: RoadGraph,
    two_opt_epsilon: float = 1.0e-12,
) -> TSRRoute:
    matrix = build_zone_distance_matrix(
        zone=zone,
        view=view,
        road_graph=road_graph,
    )

    initial_route = (
        nearest_neighbor_route(
            matrix
        )
    )

    initial_distance = (
        matrix.route_distance_km(
            initial_route
        )
    )

    optimized_route = (
        deterministic_two_opt(
            matrix,
            initial_route,
            epsilon=two_opt_epsilon,
        )
    )

    optimized_distance = (
        matrix.route_distance_km(
            optimized_route
        )
    )

    if (
        optimized_distance
        > initial_distance
        + two_opt_epsilon
    ):
        raise TSRPlanningError(
            "2-opt increased TSR route distance"
        )

    return TSRRoute(
        truck_id=zone.truck_id,
        initial_nn_bin_ids=(
            initial_route
        ),
        optimized_bin_ids=(
            optimized_route
        ),
        initial_nn_distance_km=(
            initial_distance
        ),
        optimized_distance_km=(
            optimized_distance
        ),
    )


def plan_all_zone_routes(
    *,
    zones: Sequence[TSRZone],
    view: PolicyView,
    road_graph: RoadGraph,
    two_opt_epsilon: float = 1.0e-12,
) -> tuple[TSRRoute, ...]:
    """
    Construct one deterministic optimized route per static zone.
    """

    ordered_zones = tuple(
        sorted(
            zones,
            key=lambda zone: (
                zone.truck_id
            ),
        )
    )

    if len(
        {
            zone.truck_id
            for zone in ordered_zones
        }
    ) != len(ordered_zones):
        raise TSRPlanningError(
            "each TSR zone must belong to a unique truck"
        )

    all_bin_ids = tuple(
        bin_id
        for zone in ordered_zones
        for bin_id in zone.bin_ids
    )

    if len(set(all_bin_ids)) != len(
        all_bin_ids
    ):
        raise TSRPlanningError(
            "TSR zones overlap"
        )

    return tuple(
        plan_zone_route(
            zone=zone,
            view=view,
            road_graph=road_graph,
            two_opt_epsilon=(
                two_opt_epsilon
            ),
        )
        for zone in ordered_zones
    )


@dataclass(frozen=True, order=True)
class TSRBalanceObjective:
    """
    Lexicographic TSR load-balancing objective.

    Comparison order:
    1. minimize route-distance range,
    2. minimize total route distance.

    mean_distance_km and relative_range are reported metrics and
    do not participate in objective ordering.
    """

    distance_range_km: float
    total_distance_km: float

    mean_distance_km: float = field(
        compare=False
    )

    relative_range: float = field(
        compare=False
    )

    def target_met(
        self,
        target_relative_range: float,
        *,
        epsilon: float = 1.0e-12,
    ) -> bool:
        if target_relative_range < 0.0:
            raise ValueError(
                "target_relative_range cannot be negative"
            )

        if epsilon < 0.0:
            raise ValueError(
                "epsilon cannot be negative"
            )

        return (
            self.relative_range
            <= target_relative_range
            + epsilon
        )


@dataclass(frozen=True)
class TSRBoundary:
    """
    Fixed longitudinal boundary between two initially adjacent
    TSR zones.

    x_coordinate is a zoning coordinate only. It is never used
    as a physical truck-travel distance.
    """

    left_truck_id: int
    right_truck_id: int
    x_coordinate: float


@dataclass(frozen=True)
class TSRBoundaryCandidates:
    """
    Deterministic candidate bins nearest one longitudinal boundary.
    """

    boundary: TSRBoundary
    left_bin_ids: tuple[int, ...]
    right_bin_ids: tuple[int, ...]


def tsr_balance_objective(
    routes: Sequence[TSRRoute],
) -> TSRBalanceObjective:
    """
    Compute the predeclared lexicographic balancing objective.
    """

    route_tuple = tuple(
        routes
    )

    if not route_tuple:
        raise TSRPlanningError(
            "balance objective requires at least one TSR route"
        )

    distances = tuple(
        float(
            route.optimized_distance_km
        )
        for route in route_tuple
    )

    if any(
        distance < 0.0
        or not isfinite(distance)
        for distance in distances
    ):
        raise TSRPlanningError(
            "TSR route distances must be finite and non-negative"
        )

    minimum = min(
        distances
    )

    maximum = max(
        distances
    )

    total = sum(
        distances
    )

    mean = (
        total
        / len(distances)
    )

    distance_range = (
        maximum
        - minimum
    )

    if mean <= 1.0e-15:
        relative_range = 0.0
    else:
        relative_range = (
            distance_range
            / mean
        )

    return TSRBalanceObjective(
        distance_range_km=float(
            distance_range
        ),
        total_distance_km=float(
            total
        ),
        mean_distance_km=float(
            mean
        ),
        relative_range=float(
            relative_range
        ),
    )


def _zone_x_coordinates(
    *,
    zone: TSRZone,
    view: PolicyView,
    road_graph: RoadGraph,
) -> tuple[
    tuple[int, float, float],
    ...,
]:
    """
    Return deterministic (bin_id, x, y) records for one zone.
    """

    records = []

    for bin_id in zone.bin_ids:
        bin_ = view.bin_by_id(
            bin_id
        )

        x, y = _road_node_xy(
            road_graph,
            bin_.road_node,
        )

        records.append(
            (
                bin_id,
                x,
                y,
            )
        )

    return tuple(
        records
    )


def build_longitudinal_boundaries(
    *,
    zones: Sequence[TSRZone],
    view: PolicyView,
    road_graph: RoadGraph,
) -> tuple[TSRBoundary, ...]:
    """
    Freeze longitudinal boundaries from the initial contiguous
    zone partition.

    For adjacent zones j and j+1:

        boundary_x =
            (max_x(left) + min_x(right)) / 2

    These boundaries remain fixed during Phase 4B exchanges to
    prevent iterative zone-boundary drift.
    """

    ordered_zones = tuple(
        sorted(
            zones,
            key=lambda zone: zone.truck_id,
        )
    )

    if len(ordered_zones) < 2:
        return ()

    if len(
        {
            zone.truck_id
            for zone in ordered_zones
        }
    ) != len(ordered_zones):
        raise TSRPlanningError(
            "TSR zone truck IDs must be unique"
        )

    all_bin_ids = tuple(
        bin_id
        for zone in ordered_zones
        for bin_id in zone.bin_ids
    )

    if len(set(all_bin_ids)) != len(
        all_bin_ids
    ):
        raise TSRPlanningError(
            "TSR zones overlap"
        )

    boundaries: list[
        TSRBoundary
    ] = []

    for left_zone, right_zone in zip(
        ordered_zones,
        ordered_zones[1:],
    ):
        left_records = _zone_x_coordinates(
            zone=left_zone,
            view=view,
            road_graph=road_graph,
        )

        right_records = _zone_x_coordinates(
            zone=right_zone,
            view=view,
            road_graph=road_graph,
        )

        left_max_x = max(
            record[1]
            for record in left_records
        )

        right_min_x = min(
            record[1]
            for record in right_records
        )

        boundary_x = (
            left_max_x
            + right_min_x
        ) / 2.0

        boundaries.append(
            TSRBoundary(
                left_truck_id=(
                    left_zone.truck_id
                ),
                right_truck_id=(
                    right_zone.truck_id
                ),
                x_coordinate=float(
                    boundary_x
                ),
            )
        )

    return tuple(
        boundaries
    )


def closest_boundary_bins(
    *,
    zone: TSRZone,
    boundary: TSRBoundary,
    view: PolicyView,
    road_graph: RoadGraph,
    candidate_count: int = 10,
) -> tuple[int, ...]:
    """
    Select bins geometrically nearest a frozen longitudinal zone
    boundary.

    Ordering:
        1. absolute x-distance to boundary,
        2. y coordinate,
        3. bin_id.

    Coordinates are used only for zoning/boundary selection.
    They are never substituted for graph-road travel distance.
    """

    if candidate_count <= 0:
        raise ValueError(
            "candidate_count must be positive"
        )

    if zone.truck_id not in {
        boundary.left_truck_id,
        boundary.right_truck_id,
    }:
        raise TSRPlanningError(
            "zone is not adjacent to supplied boundary"
        )

    records = _zone_x_coordinates(
        zone=zone,
        view=view,
        road_graph=road_graph,
    )

    ordered = sorted(
        records,
        key=lambda record: (
            abs(
                record[1]
                - boundary.x_coordinate
            ),
            record[2],
            record[0],
        ),
    )

    return tuple(
        record[0]
        for record in ordered[
            :min(
                candidate_count,
                len(ordered),
            )
        ]
    )


def build_boundary_candidates(
    *,
    zones: Sequence[TSRZone],
    boundaries: Sequence[TSRBoundary],
    view: PolicyView,
    road_graph: RoadGraph,
    candidate_count: int = 10,
) -> tuple[TSRBoundaryCandidates, ...]:
    """
    Build the fixed-size candidate sets used by adjacent-zone
    exchange search.
    """

    if candidate_count <= 0:
        raise ValueError(
            "candidate_count must be positive"
        )

    zone_by_truck = {
        zone.truck_id: zone
        for zone in zones
    }

    results: list[
        TSRBoundaryCandidates
    ] = []

    for boundary in boundaries:
        try:
            left_zone = zone_by_truck[
                boundary.left_truck_id
            ]

            right_zone = zone_by_truck[
                boundary.right_truck_id
            ]
        except KeyError as exc:
            raise TSRPlanningError(
                "boundary references unknown TSR zone"
            ) from exc

        results.append(
            TSRBoundaryCandidates(
                boundary=boundary,
                left_bin_ids=closest_boundary_bins(
                    zone=left_zone,
                    boundary=boundary,
                    view=view,
                    road_graph=road_graph,
                    candidate_count=(
                        candidate_count
                    ),
                ),
                right_bin_ids=closest_boundary_bins(
                    zone=right_zone,
                    boundary=boundary,
                    view=view,
                    road_graph=road_graph,
                    candidate_count=(
                        candidate_count
                    ),
                ),
            )
        )

    return tuple(
        results
    )
