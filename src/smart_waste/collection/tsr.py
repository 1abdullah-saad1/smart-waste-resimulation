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


@dataclass
class TSRRoadDistanceOracle:
    """
    Scenario-local shortest-road distance cache.

    Dijkstra is executed at most once for each unique service-road
    source node. Subsequent TSR route planning and balancing reuse
    the cached road distances.

    This changes computational cost only; it does not change the
    TSR objective, routing rule, or physical distance definition.
    """

    road_graph: RoadGraph
    service_nodes: tuple[int | str, ...]

    _distance_cache: dict[
        int | str,
        dict[int | str, float],
    ] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    _source_search_count: int = field(
        default=0,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        unique_nodes = tuple(
            dict.fromkeys(
                self.service_nodes
            )
        )

        if not unique_nodes:
            raise TSRPlanningError(
                "distance oracle requires service nodes"
            )

        self.service_nodes = unique_nodes

        try:
            self.road_graph.validate_service_nodes(
                set(
                    self.service_nodes
                )
            )
        except ValueError as exc:
            raise TSRPlanningError(
                str(exc)
            ) from exc

    @property
    def source_search_count(self) -> int:
        return self._source_search_count

    @property
    def cached_source_count(self) -> int:
        return len(
            self._distance_cache
        )

    def _validate_registered_node(
        self,
        node: int | str,
    ) -> None:
        if node not in self.service_nodes:
            raise TSRPlanningError(
                f"road node {node!r} is not registered "
                "in the TSR distance oracle"
            )

    def _ensure_source(
        self,
        source: int | str,
    ) -> None:
        self._validate_registered_node(
            source
        )

        if source in self._distance_cache:
            return

        lengths = (
            nx.single_source_dijkstra_path_length(
                self.road_graph.graph,
                source,
                weight=(
                    self.road_graph.length_attribute
                ),
            )
        )

        missing = [
            node
            for node in self.service_nodes
            if node not in lengths
        ]

        if missing:
            raise TSRPlanningError(
                f"road network does not connect source "
                f"{source!r} to service nodes: "
                f"{sorted(missing, key=str)}"
            )

        self._distance_cache[source] = {
            node: float(
                lengths[node]
            )
            for node in self.service_nodes
        }

        self._source_search_count += 1

    def distance_km(
        self,
        source: int | str,
        target: int | str,
    ) -> float:
        self._validate_registered_node(
            source
        )

        self._validate_registered_node(
            target
        )

        if source == target:
            return 0.0

        self._ensure_source(
            source
        )

        return self._distance_cache[
            source
        ][target]


def build_tsr_distance_oracle(
    *,
    view: PolicyView,
    road_graph: RoadGraph,
) -> TSRRoadDistanceOracle:
    """
    Register only the central depot and physical road nodes used by
    bins in this policy snapshot.

    Multiple bins snapped to the same road node therefore share one
    Dijkstra source.
    """

    ordered_bins = tuple(
        sorted(
            view.bins,
            key=lambda bin_: bin_.bin_id,
        )
    )

    service_nodes = tuple(
        dict.fromkeys(
            (
                view.depot_node,
                *(
                    bin_.road_node
                    for bin_ in ordered_bins
                ),
            )
        )
    )

    return TSRRoadDistanceOracle(
        road_graph=road_graph,
        service_nodes=service_nodes,
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
    distance_oracle: TSRRoadDistanceOracle | None = None,
) -> TSRDistanceMatrix:
    """
    Build one zone matrix from shortest-road distances.

    When a shared distance_oracle is supplied, repeated route
    planning and balancing reuse previously computed Dijkstra
    searches.
    """

    oracle = (
        distance_oracle
        if distance_oracle is not None
        else build_tsr_distance_oracle(
            view=view,
            road_graph=road_graph,
        )
    )

    if oracle.road_graph is not road_graph:
        raise TSRPlanningError(
            "distance oracle belongs to a different road graph"
        )

    bin_nodes = {
        bin_id: view.bin_by_id(
            bin_id
        ).road_node
        for bin_id in zone.bin_ids
    }

    depot_distances = tuple(
        oracle.distance_km(
            view.depot_node,
            bin_nodes[
                bin_id
            ],
        )
        for bin_id in zone.bin_ids
    )

    pairwise = tuple(
        tuple(
            oracle.distance_km(
                bin_nodes[
                    first_bin_id
                ],
                bin_nodes[
                    second_bin_id
                ],
            )
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
    distance_oracle: TSRRoadDistanceOracle | None = None,
) -> TSRRoute:
    matrix = build_zone_distance_matrix(
        zone=zone,
        view=view,
        road_graph=road_graph,
        distance_oracle=distance_oracle,
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
    distance_oracle: TSRRoadDistanceOracle | None = None,
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

    oracle = (
        distance_oracle
        if distance_oracle is not None
        else build_tsr_distance_oracle(
            view=view,
            road_graph=road_graph,
        )
    )

    return tuple(
        plan_zone_route(
            zone=zone,
            view=view,
            road_graph=road_graph,
            two_opt_epsilon=(
                two_opt_epsilon
            ),
            distance_oracle=oracle,
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


@dataclass(frozen=True)
class TSRZoneExchange:
    """One accepted deterministic adjacent-zone exchange."""

    iteration: int
    boundary_index: int

    left_truck_id: int
    right_truck_id: int

    left_bin_to_right: int
    right_bin_to_left: int

    objective_before: TSRBalanceObjective
    objective_after: TSRBalanceObjective


@dataclass(frozen=True)
class TSRBalanceResult:
    """
    Complete deterministic result of TSR adjacent-zone balancing.
    """

    initial_zones: tuple[TSRZone, ...]
    final_zones: tuple[TSRZone, ...]

    boundaries: tuple[TSRBoundary, ...]

    initial_routes: tuple[TSRRoute, ...]
    final_routes: tuple[TSRRoute, ...]

    initial_objective: TSRBalanceObjective
    final_objective: TSRBalanceObjective

    target_relative_range: float
    target_met: bool

    exchanges: tuple[
        TSRZoneExchange,
        ...,
    ]

    @property
    def iterations(self) -> int:
        return len(
            self.exchanges
        )


def _objective_is_strictly_better(
    candidate: TSRBalanceObjective,
    incumbent: TSRBalanceObjective,
    *,
    epsilon: float,
) -> bool:
    """
    Compare objectives using the predeclared lexicographic rule
    while protecting against floating-point noise.
    """

    if epsilon < 0.0:
        raise ValueError(
            "epsilon cannot be negative"
        )

    if (
        candidate.distance_range_km
        < incumbent.distance_range_km
        - epsilon
    ):
        return True

    if (
        abs(
            candidate.distance_range_km
            - incumbent.distance_range_km
        )
        <= epsilon
        and candidate.total_distance_km
        < incumbent.total_distance_km
        - epsilon
    ):
        return True

    return False


def _objectives_equivalent(
    first: TSRBalanceObjective,
    second: TSRBalanceObjective,
    *,
    epsilon: float,
) -> bool:
    return (
        abs(
            first.distance_range_km
            - second.distance_range_km
        )
        <= epsilon
        and abs(
            first.total_distance_km
            - second.total_distance_km
        )
        <= epsilon
    )


def _exchange_zone_bins(
    *,
    left_zone: TSRZone,
    right_zone: TSRZone,
    left_bin_id: int,
    right_bin_id: int,
) -> tuple[
    TSRZone,
    TSRZone,
]:
    """
    Perform a one-for-one exchange while preserving zone sizes.
    """

    if left_bin_id not in left_zone.bin_ids:
        raise TSRPlanningError(
            f"bin {left_bin_id} does not belong to "
            f"truck {left_zone.truck_id}"
        )

    if right_bin_id not in right_zone.bin_ids:
        raise TSRPlanningError(
            f"bin {right_bin_id} does not belong to "
            f"truck {right_zone.truck_id}"
        )

    if left_bin_id == right_bin_id:
        raise TSRPlanningError(
            "cannot exchange a bin with itself"
        )

    left_ids = [
        bin_id
        for bin_id in left_zone.bin_ids
        if bin_id != left_bin_id
    ]

    left_ids.append(
        right_bin_id
    )

    right_ids = [
        bin_id
        for bin_id in right_zone.bin_ids
        if bin_id != right_bin_id
    ]

    right_ids.append(
        left_bin_id
    )

    new_left = TSRZone(
        truck_id=left_zone.truck_id,
        bin_ids=tuple(
            sorted(
                left_ids
            )
        ),
    )

    new_right = TSRZone(
        truck_id=right_zone.truck_id,
        bin_ids=tuple(
            sorted(
                right_ids
            )
        ),
    )

    if (
        len(new_left.bin_ids)
        != len(left_zone.bin_ids)
        or len(new_right.bin_ids)
        != len(right_zone.bin_ids)
    ):
        raise TSRPlanningError(
            "TSR exchange changed zone size"
        )

    if set(
        new_left.bin_ids
    ) & set(
        new_right.bin_ids
    ):
        raise TSRPlanningError(
            "TSR exchange created overlapping zones"
        )

    return (
        new_left,
        new_right,
    )


def _replace_two_zones(
    *,
    zones: Sequence[TSRZone],
    left_zone: TSRZone,
    right_zone: TSRZone,
) -> tuple[TSRZone, ...]:
    replacements = {
        left_zone.truck_id: left_zone,
        right_zone.truck_id: right_zone,
    }

    result = tuple(
        replacements.get(
            zone.truck_id,
            zone,
        )
        for zone in zones
    )

    return tuple(
        sorted(
            result,
            key=lambda zone: zone.truck_id,
        )
    )


def _replace_two_routes(
    *,
    routes: Sequence[TSRRoute],
    left_route: TSRRoute,
    right_route: TSRRoute,
) -> tuple[TSRRoute, ...]:
    replacements = {
        left_route.truck_id: left_route,
        right_route.truck_id: right_route,
    }

    result = tuple(
        replacements.get(
            route.truck_id,
            route,
        )
        for route in routes
    )

    return tuple(
        sorted(
            result,
            key=lambda route: route.truck_id,
        )
    )


def _validate_balanced_zone_invariants(
    *,
    initial_zones: Sequence[TSRZone],
    candidate_zones: Sequence[TSRZone],
) -> None:
    """
    Ensure balancing changes assignment only, never cardinality
    or bin population.
    """

    initial = tuple(
        sorted(
            initial_zones,
            key=lambda zone: zone.truck_id,
        )
    )

    candidate = tuple(
        sorted(
            candidate_zones,
            key=lambda zone: zone.truck_id,
        )
    )

    if len(initial) != len(candidate):
        raise TSRPlanningError(
            "balancing changed number of TSR zones"
        )

    if tuple(
        zone.truck_id
        for zone in initial
    ) != tuple(
        zone.truck_id
        for zone in candidate
    ):
        raise TSRPlanningError(
            "balancing changed TSR truck identities"
        )

    initial_sizes = {
        zone.truck_id: len(
            zone.bin_ids
        )
        for zone in initial
    }

    candidate_sizes = {
        zone.truck_id: len(
            zone.bin_ids
        )
        for zone in candidate
    }

    if initial_sizes != candidate_sizes:
        raise TSRPlanningError(
            "balancing changed bins-per-truck"
        )

    initial_bins = sorted(
        bin_id
        for zone in initial
        for bin_id in zone.bin_ids
    )

    candidate_bins = sorted(
        bin_id
        for zone in candidate
        for bin_id in zone.bin_ids
    )

    if initial_bins != candidate_bins:
        raise TSRPlanningError(
            "balancing changed global bin population"
        )

    if len(candidate_bins) != len(
        set(candidate_bins)
    ):
        raise TSRPlanningError(
            "balancing duplicated bin assignments"
        )


def balance_adjacent_zones(
    *,
    zones: Sequence[TSRZone],
    view: PolicyView,
    road_graph: RoadGraph,
    candidate_count: int = 10,
    target_relative_range: float = 0.10,
    two_opt_epsilon: float = 1.0e-12,
    distance_oracle: TSRRoadDistanceOracle | None = None,
) -> TSRBalanceResult:
    """
    Deterministically balance static TSR zones.

    Algorithm
    ---------
    1. Freeze longitudinal boundaries from the initial partition.
    2. Compute NN + deterministic 2-opt route for every zone.
    3. For each adjacent boundary, identify the nearest
       candidate_count bins from both current zones.
    4. Evaluate every one-for-one candidate exchange.
    5. Re-plan only the two affected routes.
    6. Evaluate the GLOBAL lexicographic objective:
           a) minimum max-min route-distance range,
           b) minimum total route distance.
    7. Apply exactly one globally best strict improvement.
    8. Refresh boundary candidates around the same frozen
       boundaries and repeat.
    9. Stop when:
           - the <= target_relative_range target is achieved, or
           - no strict improving exchange exists.

    If the engineering balance target cannot be achieved,
    target_met is False and the deterministic best solution
    reached by the declared search is retained.
    """

    if candidate_count <= 0:
        raise ValueError(
            "candidate_count must be positive"
        )

    if target_relative_range < 0.0:
        raise ValueError(
            "target_relative_range cannot be negative"
        )

    if two_opt_epsilon < 0.0:
        raise ValueError(
            "two_opt_epsilon cannot be negative"
        )

    initial_zones = tuple(
        sorted(
            zones,
            key=lambda zone: zone.truck_id,
        )
    )

    if not initial_zones:
        raise TSRPlanningError(
            "balancing requires at least one TSR zone"
        )

    oracle = (
        distance_oracle
        if distance_oracle is not None
        else build_tsr_distance_oracle(
            view=view,
            road_graph=road_graph,
        )
    )

    initial_routes = plan_all_zone_routes(
        zones=initial_zones,
        view=view,
        road_graph=road_graph,
        two_opt_epsilon=two_opt_epsilon,
        distance_oracle=oracle,
    )

    initial_objective = tsr_balance_objective(
        initial_routes
    )

    boundaries = build_longitudinal_boundaries(
        zones=initial_zones,
        view=view,
        road_graph=road_graph,
    )

    current_zones = initial_zones
    current_routes = initial_routes
    current_objective = initial_objective

    exchanges: list[
        TSRZoneExchange
    ] = []

    if current_objective.target_met(
        target_relative_range,
        epsilon=two_opt_epsilon,
    ):
        return TSRBalanceResult(
            initial_zones=initial_zones,
            final_zones=current_zones,
            boundaries=boundaries,
            initial_routes=initial_routes,
            final_routes=current_routes,
            initial_objective=initial_objective,
            final_objective=current_objective,
            target_relative_range=(
                target_relative_range
            ),
            target_met=True,
            exchanges=(),
        )

    while True:
        candidate_groups = (
            build_boundary_candidates(
                zones=current_zones,
                boundaries=boundaries,
                view=view,
                road_graph=road_graph,
                candidate_count=candidate_count,
            )
        )

        zone_by_truck = {
            zone.truck_id: zone
            for zone in current_zones
        }

        best_zones: tuple[
            TSRZone,
            ...,
        ] | None = None

        best_routes: tuple[
            TSRRoute,
            ...,
        ] | None = None

        best_objective: (
            TSRBalanceObjective
            | None
        ) = None

        best_key: (
            tuple[int, int, int]
            | None
        ) = None

        best_exchange_data: (
            tuple[
                int,
                int,
                int,
                int,
                int,
            ]
            | None
        ) = None

        for (
            boundary_index,
            candidate_group,
        ) in enumerate(
            candidate_groups
        ):
            boundary = (
                candidate_group.boundary
            )

            left_zone = zone_by_truck[
                boundary.left_truck_id
            ]

            right_zone = zone_by_truck[
                boundary.right_truck_id
            ]

            for left_bin_id in (
                candidate_group.left_bin_ids
            ):
                for right_bin_id in (
                    candidate_group.right_bin_ids
                ):
                    (
                        candidate_left_zone,
                        candidate_right_zone,
                    ) = _exchange_zone_bins(
                        left_zone=left_zone,
                        right_zone=right_zone,
                        left_bin_id=left_bin_id,
                        right_bin_id=right_bin_id,
                    )

                    candidate_left_route = (
                        plan_zone_route(
                            zone=(
                                candidate_left_zone
                            ),
                            view=view,
                            road_graph=road_graph,
                            two_opt_epsilon=(
                                two_opt_epsilon
                            ),
                            distance_oracle=oracle,
                        )
                    )

                    candidate_right_route = (
                        plan_zone_route(
                            zone=(
                                candidate_right_zone
                            ),
                            view=view,
                            road_graph=road_graph,
                            two_opt_epsilon=(
                                two_opt_epsilon
                            ),
                            distance_oracle=oracle,
                        )
                    )

                    candidate_zones = (
                        _replace_two_zones(
                            zones=current_zones,
                            left_zone=(
                                candidate_left_zone
                            ),
                            right_zone=(
                                candidate_right_zone
                            ),
                        )
                    )

                    candidate_routes = (
                        _replace_two_routes(
                            routes=current_routes,
                            left_route=(
                                candidate_left_route
                            ),
                            right_route=(
                                candidate_right_route
                            ),
                        )
                    )

                    _validate_balanced_zone_invariants(
                        initial_zones=(
                            initial_zones
                        ),
                        candidate_zones=(
                            candidate_zones
                        ),
                    )

                    candidate_objective = (
                        tsr_balance_objective(
                            candidate_routes
                        )
                    )

                    if not _objective_is_strictly_better(
                        candidate_objective,
                        current_objective,
                        epsilon=(
                            two_opt_epsilon
                        ),
                    ):
                        continue

                    candidate_key = (
                        boundary_index,
                        left_bin_id,
                        right_bin_id,
                    )

                    choose_candidate = False

                    if best_objective is None:
                        choose_candidate = True

                    elif _objective_is_strictly_better(
                        candidate_objective,
                        best_objective,
                        epsilon=(
                            two_opt_epsilon
                        ),
                    ):
                        choose_candidate = True

                    elif (
                        _objectives_equivalent(
                            candidate_objective,
                            best_objective,
                            epsilon=(
                                two_opt_epsilon
                            ),
                        )
                        and (
                            best_key is None
                            or candidate_key
                            < best_key
                        )
                    ):
                        choose_candidate = True

                    if choose_candidate:
                        best_zones = (
                            candidate_zones
                        )

                        best_routes = (
                            candidate_routes
                        )

                        best_objective = (
                            candidate_objective
                        )

                        best_key = (
                            candidate_key
                        )

                        best_exchange_data = (
                            boundary_index,
                            boundary.left_truck_id,
                            boundary.right_truck_id,
                            left_bin_id,
                            right_bin_id,
                        )

        if (
            best_zones is None
            or best_routes is None
            or best_objective is None
            or best_exchange_data is None
        ):
            break

        previous_objective = (
            current_objective
        )

        (
            boundary_index,
            left_truck_id,
            right_truck_id,
            left_bin_id,
            right_bin_id,
        ) = best_exchange_data

        current_zones = (
            best_zones
        )

        current_routes = (
            best_routes
        )

        current_objective = (
            best_objective
        )

        exchanges.append(
            TSRZoneExchange(
                iteration=len(
                    exchanges
                ) + 1,
                boundary_index=(
                    boundary_index
                ),
                left_truck_id=(
                    left_truck_id
                ),
                right_truck_id=(
                    right_truck_id
                ),
                left_bin_to_right=(
                    left_bin_id
                ),
                right_bin_to_left=(
                    right_bin_id
                ),
                objective_before=(
                    previous_objective
                ),
                objective_after=(
                    current_objective
                ),
            )
        )

        if current_objective.target_met(
            target_relative_range,
            epsilon=two_opt_epsilon,
        ):
            break

    _validate_balanced_zone_invariants(
        initial_zones=initial_zones,
        candidate_zones=current_zones,
    )

    return TSRBalanceResult(
        initial_zones=initial_zones,
        final_zones=current_zones,
        boundaries=boundaries,
        initial_routes=initial_routes,
        final_routes=current_routes,
        initial_objective=initial_objective,
        final_objective=current_objective,
        target_relative_range=(
            target_relative_range
        ),
        target_met=(
            current_objective.target_met(
                target_relative_range,
                epsilon=two_opt_epsilon,
            )
        ),
        exchanges=tuple(
            exchanges
        ),
    )
