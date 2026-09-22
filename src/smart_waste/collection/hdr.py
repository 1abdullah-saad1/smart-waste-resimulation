from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

import networkx as nx

from smart_waste.collection.base import (
    CollectionPolicy,
    PolicyView,
)
from smart_waste.movement.road_graph import (
    RoadGraph,
)


class HDRConfigurationError(ValueError):
    """Raised when the HDR benchmark definition is invalid."""


class HDRTelemetryError(RuntimeError):
    """Raised when reported routing telemetry is invalid."""


@dataclass(frozen=True)
class HDREligibilitySnapshot:
    """
    Immutable HDR candidate snapshot.

    Eligibility is determined from REPORTED fill, not physical
    waste truth.

    The snapshot is intentionally frozen at policy initialization
    for the revised benchmark. Route selection may later be dynamic
    among this pending set, but membership is reproducible and does
    not silently change because simulation time advances.
    """

    threshold_percent: float

    reported_fill_by_bin: tuple[
        tuple[int, float],
        ...,
    ]

    eligible_bin_ids: tuple[
        int,
        ...,
    ]

    def reported_fill_percent(
        self,
        bin_id: int,
    ) -> float:
        for (
            candidate_bin_id,
            reported_fill,
        ) in self.reported_fill_by_bin:
            if candidate_bin_id == bin_id:
                return reported_fill

        raise KeyError(
            f"unknown HDR bin_id: {bin_id}"
        )

    def is_eligible(
        self,
        bin_id: int,
    ) -> bool:
        if not any(
            candidate_bin_id == bin_id
            for (
                candidate_bin_id,
                _,
            ) in self.reported_fill_by_bin
        ):
            raise KeyError(
                f"unknown HDR bin_id: {bin_id}"
            )

        return (
            bin_id
            in self.eligible_bin_ids
        )


def _validated_threshold(
    threshold_percent: float,
) -> float:
    try:
        threshold = float(
            threshold_percent
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise HDRConfigurationError(
            "HDR threshold must be numeric"
        ) from exc

    if not isfinite(
        threshold
    ):
        raise HDRConfigurationError(
            "HDR threshold must be finite"
        )

    if not (
        0.0
        <= threshold
        <= 100.0
    ):
        raise HDRConfigurationError(
            "HDR threshold must be between 0 and 100"
        )

    return threshold


def _validated_reported_fill(
    *,
    bin_id: int,
    value: float,
) -> float:
    try:
        fill = float(
            value
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise HDRTelemetryError(
            f"reported fill for bin {bin_id} must be numeric"
        ) from exc

    if not isfinite(
        fill
    ):
        raise HDRTelemetryError(
            f"reported fill for bin {bin_id} must be finite"
        )

    if not (
        0.0
        <= fill
        <= 100.0
    ):
        raise HDRTelemetryError(
            f"reported fill for bin {bin_id} must be "
            "between 0 and 100"
        )

    return fill


def build_hdr_eligibility_snapshot(
    *,
    view: PolicyView,
    threshold_percent: float = 80.0,
) -> HDREligibilitySnapshot:
    """
    Freeze HDR service eligibility from one policy snapshot.

    Rule:
        eligible iff reported_fill >= threshold

    Inclusive >=80% reproduces the baseline definition while
    keeping physical fill and routing telemetry separate.

    The function deliberately performs no:
    - road-distance calculation,
    - truck assignment,
    - capacity check,
    - fuel check,
    - depot-return decision.
    """

    threshold = _validated_threshold(
        threshold_percent
    )

    reported_records: list[
        tuple[int, float]
    ] = []

    eligible: list[
        int
    ] = []

    for bin_ in sorted(
        view.bins,
        key=lambda item: item.bin_id,
    ):
        reported_fill = (
            _validated_reported_fill(
                bin_id=bin_.bin_id,
                value=(
                    bin_.routing_fill_percent
                ),
            )
        )

        reported_records.append(
            (
                bin_.bin_id,
                reported_fill,
            )
        )

        if (
            reported_fill
            >= threshold
        ):
            eligible.append(
                bin_.bin_id
            )

    return HDREligibilitySnapshot(
        threshold_percent=(
            threshold
        ),
        reported_fill_by_bin=tuple(
            reported_records
        ),
        eligible_bin_ids=tuple(
            eligible
        ),
    )


class HDRPolicyError(RuntimeError):
    """Raised when the revised HDR policy lifecycle is violated."""


@dataclass
class HDRRoadDistanceOracle:
    """
    Scenario-local shortest-road distance cache for HDR.

    Dijkstra is executed at most once for each unique source road
    node requested by dynamic HDR selection.

    The cache changes computational cost only. It does not change
    candidate eligibility or physical movement semantics.
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
        self.service_nodes = tuple(
            dict.fromkeys(
                self.service_nodes
            )
        )

        if not self.service_nodes:
            raise HDRPolicyError(
                "HDR distance oracle requires service nodes"
            )

        try:
            self.road_graph.validate_service_nodes(
                set(
                    self.service_nodes
                )
            )
        except ValueError as exc:
            raise HDRPolicyError(
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

    def _validate_node(
        self,
        node: int | str,
    ) -> None:
        if node not in self.service_nodes:
            raise HDRPolicyError(
                f"road node {node!r} is not registered "
                "in the HDR distance oracle"
            )

    def _ensure_source(
        self,
        source: int | str,
    ) -> None:
        self._validate_node(
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
            raise HDRPolicyError(
                f"road network does not connect source "
                f"{source!r} to HDR service nodes: "
                f"{sorted(missing, key=str)}"
            )

        self._distance_cache[
            source
        ] = {
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
        self._validate_node(
            source
        )

        self._validate_node(
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


def build_hdr_distance_oracle(
    *,
    view: PolicyView,
    road_graph: RoadGraph,
) -> HDRRoadDistanceOracle:
    """
    Register depot, all bin nodes, and all current truck nodes.

    Multiple bins snapped to one physical road node share one
    Dijkstra source.
    """

    ordered_bins = tuple(
        sorted(
            view.bins,
            key=lambda bin_: bin_.bin_id,
        )
    )

    ordered_trucks = tuple(
        sorted(
            view.trucks,
            key=lambda truck: truck.truck_id,
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
                *(
                    truck.current_node
                    for truck in ordered_trucks
                ),
            )
        )
    )

    return HDRRoadDistanceOracle(
        road_graph=road_graph,
        service_nodes=service_nodes,
    )


class HDRCollectionPolicy(CollectionPolicy):
    """
    Revised Heuristic Dynamic Routing baseline.

    Eligibility
    -----------
    Candidate membership is frozen once during initialize():

        reported fill >= threshold

    Routing
    -------
    At every policy decision, the active truck selects the nearest
    currently unreserved bin from the remaining eligible pool using
    shortest-road distance.

    Equal-distance ties are resolved by ascending bin_id.

    After successful service, that bin is removed from the pending
    pool. The next selection is recomputed from the truck's current
    physical road node.

    The policy deliberately does not make capacity, fuel, travel,
    unloading, refuelling, or depot-return decisions. Those remain
    responsibilities of the central simulation dispatcher.
    """

    def __init__(
        self,
        *,
        road_graph: RoadGraph,
        threshold_percent: float = 80.0,
    ) -> None:
        self._road_graph = road_graph

        self._threshold_percent = (
            _validated_threshold(
                threshold_percent
            )
        )

        self._eligibility_snapshot: (
            HDREligibilitySnapshot
            | None
        ) = None

        self._distance_oracle: (
            HDRRoadDistanceOracle
            | None
        ) = None

        self._pending_bin_ids: set[
            int
        ] = set()

        self._completed_bin_ids: set[
            int
        ] = set()

    @property
    def initialized(self) -> bool:
        return (
            self._eligibility_snapshot
            is not None
        )

    @property
    def eligibility_snapshot(
        self,
    ) -> HDREligibilitySnapshot:
        if self._eligibility_snapshot is None:
            raise HDRPolicyError(
                "HDR policy has not been initialized"
            )

        return self._eligibility_snapshot

    @property
    def distance_oracle(
        self,
    ) -> HDRRoadDistanceOracle:
        if self._distance_oracle is None:
            raise HDRPolicyError(
                "HDR policy has not been initialized"
            )

        return self._distance_oracle

    @property
    def pending_bin_ids(
        self,
    ) -> tuple[int, ...]:
        return tuple(
            sorted(
                self._pending_bin_ids
            )
        )

    @property
    def completed_bin_ids(
        self,
    ) -> tuple[int, ...]:
        return tuple(
            sorted(
                self._completed_bin_ids
            )
        )

    def initialize(
        self,
        view: PolicyView,
    ) -> None:
        if self.initialized:
            raise HDRPolicyError(
                "HDR policy is already initialized"
            )

        snapshot = (
            build_hdr_eligibility_snapshot(
                view=view,
                threshold_percent=(
                    self._threshold_percent
                ),
            )
        )

        oracle = build_hdr_distance_oracle(
            view=view,
            road_graph=self._road_graph,
        )

        self._eligibility_snapshot = snapshot
        self._distance_oracle = oracle

        self._pending_bin_ids = set(
            snapshot.eligible_bin_ids
        )

        self._completed_bin_ids.clear()

    def _truck_current_node(
        self,
        *,
        view: PolicyView,
        truck_id: int,
    ) -> int | str:
        for truck in view.trucks:
            if truck.truck_id == truck_id:
                return truck.current_node

        raise HDRPolicyError(
            f"unknown HDR truck_id: {truck_id}"
        )

    def select_next_bin(
        self,
        *,
        view: PolicyView,
        truck_id: int,
    ) -> int | None:
        if not self.initialized:
            raise HDRPolicyError(
                "HDR policy must be initialized before selection"
            )

        current_node = self._truck_current_node(
            view=view,
            truck_id=truck_id,
        )

        candidates = []

        for bin_id in sorted(
            self._pending_bin_ids
        ):
            bin_view = view.bin_by_id(
                bin_id
            )

            if bin_view.is_reserved:
                continue

            distance = (
                self.distance_oracle.distance_km(
                    current_node,
                    bin_view.road_node,
                )
            )

            candidates.append(
                (
                    distance,
                    bin_id,
                )
            )

        if not candidates:
            return None

        _, selected_bin_id = min(
            candidates
        )

        return selected_bin_id

    def on_service_complete(
        self,
        *,
        view: PolicyView,
        truck_id: int,
        bin_id: int,
    ) -> None:
        if not self.initialized:
            raise HDRPolicyError(
                "HDR policy must be initialized before "
                "service completion"
            )

        # Validate truck identity through the current policy view.
        self._truck_current_node(
            view=view,
            truck_id=truck_id,
        )

        if bin_id not in self._pending_bin_ids:
            if bin_id in self._completed_bin_ids:
                raise HDRPolicyError(
                    f"HDR bin {bin_id} was completed twice"
                )

            raise HDRPolicyError(
                f"bin {bin_id} is not pending in the "
                "frozen HDR eligibility set"
            )

        bin_view = view.bin_by_id(
            bin_id
        )

        # Orchestrator releases reservation before policy callback.
        if bin_view.is_reserved:
            raise HDRPolicyError(
                f"completed HDR bin {bin_id} still appears reserved"
            )

        self._pending_bin_ids.remove(
            bin_id
        )

        self._completed_bin_ids.add(
            bin_id
        )

    def is_complete(
        self,
        view: PolicyView,
    ) -> bool:
        if not self.initialized:
            return False

        return not self._pending_bin_ids
