from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite

import numpy as np

from smart_waste.collection.base import (
    PolicyView,
)
from smart_waste.movement.road_graph import (
    RoadGraph,
)


class DQNStateAdapterError(RuntimeError):
    """Raised when the physical DQN state cannot be built safely."""


@dataclass
class DQNPhysicalStateAdapter:
    """
    Build the manuscript/reconstruction DQN state:

        s_t = (L_t, F_t, H_t, T_t)

    For N bins:

        dimension = 2 + 3*N

    For the primary benchmark:

        N = 1000
        dimension = 3002

    Important
    ---------
    This adapter does not perform:
    - movement,
    - routing-distance calculation,
    - truck-capacity logic,
    - fuel logic,
    - depot logic,
    - physical-time advancement.

    It only derives neural-network features from PolicyView.
    """

    road_graph: RoadGraph

    sla_limit_hours: float = 6.0

    verified_by_bin: (
        Mapping[int, bool]
        | None
    ) = None

    initial_hazard_severity: (
        Mapping[int, float]
        | None
    ) = None

    initial_full_elapsed_hours: (
        Mapping[int, float]
        | None
    ) = None

    _initialized: bool = field(
        default=False,
        init=False,
    )

    _bin_ids: tuple[int, ...] = field(
        default=(),
        init=False,
    )

    _verified: dict[int, bool] = field(
        default_factory=dict,
        init=False,
    )

    _hazard: dict[int, float] = field(
        default_factory=dict,
        init=False,
    )

    _full_elapsed: dict[int, float] = field(
        default_factory=dict,
        init=False,
    )

    _last_routing_fill: dict[
        int,
        float,
    ] = field(
        default_factory=dict,
        init=False,
    )

    _last_time_hours: float = field(
        default=0.0,
        init=False,
    )

    _coordinate_min: np.ndarray = field(
        default_factory=lambda: np.zeros(
            2,
            dtype=np.float64,
        ),
        init=False,
        repr=False,
    )

    _coordinate_span: np.ndarray = field(
        default_factory=lambda: np.ones(
            2,
            dtype=np.float64,
        ),
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.sla_limit_hours <= 0.0:
            raise DQNStateAdapterError(
                "sla_limit_hours must be positive"
            )

        self._initialize_coordinate_bounds()

    @staticmethod
    def expected_dimension(
        num_bins: int,
    ) -> int:
        if num_bins <= 0:
            raise ValueError(
                "num_bins must be positive"
            )

        return (
            2
            + 3 * num_bins
        )

    def _node_xy(
        self,
        node: int | str,
    ) -> np.ndarray:
        if node not in self.road_graph.graph:
            raise DQNStateAdapterError(
                f"road node {node!r} is missing"
            )

        data = self.road_graph.graph.nodes[
            node
        ]

        coordinate_pairs = (
            (
                "x_km",
                "y_km",
            ),
            (
                "x",
                "y",
            ),
        )

        for x_key, y_key in coordinate_pairs:
            if (
                x_key in data
                and y_key in data
            ):
                try:
                    x_value = float(
                        data[
                            x_key
                        ]
                    )
                    y_value = float(
                        data[
                            y_key
                        ]
                    )
                except (
                    TypeError,
                    ValueError,
                ) as exc:
                    raise DQNStateAdapterError(
                        f"road node {node!r} has "
                        "nonnumeric coordinates"
                    ) from exc

                if not (
                    isfinite(
                        x_value
                    )
                    and isfinite(
                        y_value
                    )
                ):
                    raise DQNStateAdapterError(
                        f"road node {node!r} has "
                        "non-finite coordinates"
                    )

                return np.asarray(
                    [
                        x_value,
                        y_value,
                    ],
                    dtype=np.float64,
                )

        raise DQNStateAdapterError(
            f"road node {node!r} has no supported "
            "coordinate attributes; expected "
            "(x_km, y_km) or (x, y)"
        )

    def _initialize_coordinate_bounds(
        self,
    ) -> None:
        coordinates = np.vstack(
            [
                self._node_xy(
                    node
                )
                for node in (
                    self.road_graph.graph.nodes
                )
            ]
        )

        self._coordinate_min = (
            coordinates.min(
                axis=0
            )
        )

        coordinate_max = (
            coordinates.max(
                axis=0
            )
        )

        self._coordinate_span = (
            coordinate_max
            - self._coordinate_min
        )

        self._coordinate_span[
            self._coordinate_span
            == 0.0
        ] = 1.0

    def _normalize_node(
        self,
        node: int | str,
    ) -> np.ndarray:
        xy = self._node_xy(
            node
        )

        normalized = (
            (
                xy
                - self._coordinate_min
            )
            / self._coordinate_span
        )

        return np.clip(
            normalized,
            0.0,
            1.0,
        )

    def initialize(
        self,
        view: PolicyView,
    ) -> None:
        if self._initialized:
            raise DQNStateAdapterError(
                "DQN state adapter is already initialized"
            )

        ordered_bins = tuple(
            sorted(
                view.bins,
                key=lambda item: item.bin_id,
            )
        )

        bin_ids = tuple(
            bin_.bin_id
            for bin_ in ordered_bins
        )

        # The DQN action index is deliberately identical
        # to bin_id.
        if bin_ids != tuple(
            range(
                len(
                    bin_ids
                )
            )
        ):
            raise DQNStateAdapterError(
                "DQN requires contiguous bin IDs "
                "0..N-1"
            )

        self._bin_ids = (
            bin_ids
        )

        known = set(
            bin_ids
        )

        configured_verified = (
            {}
            if self.verified_by_bin
            is None
            else dict(
                self.verified_by_bin
            )
        )

        configured_hazard = (
            {}
            if self.initial_hazard_severity
            is None
            else dict(
                self.initial_hazard_severity
            )
        )

        configured_elapsed = (
            {}
            if self.initial_full_elapsed_hours
            is None
            else dict(
                self.initial_full_elapsed_hours
            )
        )

        for mapping_name, mapping in (
            (
                "verified_by_bin",
                configured_verified,
            ),
            (
                "initial_hazard_severity",
                configured_hazard,
            ),
            (
                "initial_full_elapsed_hours",
                configured_elapsed,
            ),
        ):
            unknown = (
                set(
                    mapping
                )
                - known
            )

            if unknown:
                raise DQNStateAdapterError(
                    f"{mapping_name} references "
                    "unknown bins: "
                    f"{sorted(unknown)}"
                )

        for bin_ in ordered_bins:
            bin_id = (
                bin_.bin_id
            )

            self._verified[
                bin_id
            ] = bool(
                configured_verified.get(
                    bin_id,
                    True,
                )
            )

            severity = float(
                configured_hazard.get(
                    bin_id,
                    0.0,
                )
            )

            if not (
                isfinite(
                    severity
                )
                and 0.0
                <= severity
                <= 1.0
            ):
                raise DQNStateAdapterError(
                    f"hazard severity for bin "
                    f"{bin_id} must be in [0, 1]"
                )

            self._hazard[
                bin_id
            ] = severity

            elapsed = float(
                configured_elapsed.get(
                    bin_id,
                    0.0,
                )
            )

            if not (
                isfinite(
                    elapsed
                )
                and elapsed
                >= 0.0
            ):
                raise DQNStateAdapterError(
                    f"full-bin elapsed time for "
                    f"bin {bin_id} must be "
                    "finite and non-negative"
                )

            self._full_elapsed[
                bin_id
            ] = elapsed

            self._last_routing_fill[
                bin_id
            ] = float(
                bin_.routing_fill_percent
            )

        self._last_time_hours = float(
            view.current_time_hours
        )

        self._initialized = True

    def is_verified(
        self,
        bin_id: int,
    ) -> bool:
        if not self._initialized:
            raise DQNStateAdapterError(
                "state adapter is not initialized"
            )

        return self._verified[
            bin_id
        ]

    def hazard_severity(
        self,
        bin_id: int,
    ) -> float:
        if not self._initialized:
            raise DQNStateAdapterError(
                "state adapter is not initialized"
            )

        return self._hazard[
            bin_id
        ]

    def clear_hazard(
        self,
        bin_id: int,
    ) -> None:
        if not self._initialized:
            raise DQNStateAdapterError(
                "state adapter is not initialized"
            )

        self._hazard[
            bin_id
        ] = 0.0

    def full_elapsed_hours(
        self,
        bin_id: int,
    ) -> float:
        if not self._initialized:
            raise DQNStateAdapterError(
                "state adapter is not initialized"
            )

        return self._full_elapsed[
            bin_id
        ]

    def update(
        self,
        view: PolicyView,
    ) -> None:
        if not self._initialized:
            raise DQNStateAdapterError(
                "state adapter is not initialized"
            )

        current_time = float(
            view.current_time_hours
        )

        if (
            current_time
            < self._last_time_hours
        ):
            raise DQNStateAdapterError(
                "policy-view time moved backwards"
            )

        elapsed_hours = (
            current_time
            - self._last_time_hours
        )

        current_ids = tuple(
            bin_.bin_id
            for bin_ in sorted(
                view.bins,
                key=lambda item: item.bin_id,
            )
        )

        if (
            current_ids
            != self._bin_ids
        ):
            raise DQNStateAdapterError(
                "bin identity changed after "
                "DQN initialization"
            )

        for bin_ in view.bins:
            bin_id = (
                bin_.bin_id
            )

            previous_fill = (
                self._last_routing_fill[
                    bin_id
                ]
            )

            current_fill = float(
                bin_.routing_fill_percent
            )

            if current_fill < 100.0:
                full_elapsed = 0.0

            elif previous_fill >= 100.0:
                full_elapsed = (
                    self._full_elapsed[
                        bin_id
                    ]
                    + elapsed_hours
                )

            else:
                rate = float(
                    bin_.fill_rate_percent_per_hour
                )

                if rate > 0.0:
                    time_to_full = max(
                        0.0,
                        (
                            100.0
                            - previous_fill
                        )
                        / rate,
                    )

                    full_elapsed = max(
                        0.0,
                        elapsed_hours
                        - time_to_full,
                    )
                else:
                    # A discontinuous telemetry jump cannot
                    # legitimately back-date the SLA timer.
                    full_elapsed = 0.0

            self._full_elapsed[
                bin_id
            ] = full_elapsed

            self._last_routing_fill[
                bin_id
            ] = current_fill

        self._last_time_hours = (
            current_time
        )

    def observation(
        self,
        *,
        view: PolicyView,
        truck_id: int,
    ) -> np.ndarray:
        self.update(
            view
        )

        truck = view.truck_by_id(
            truck_id
        )

        truck_location = (
            self._normalize_node(
                truck.current_node
            )
        )

        fill_state = np.zeros(
            len(
                self._bin_ids
            ),
            dtype=np.float64,
        )

        hazard_state = np.zeros_like(
            fill_state
        )

        time_state = np.zeros_like(
            fill_state
        )

        for bin_ in view.bins:
            bin_id = (
                bin_.bin_id
            )

            # Unverified telemetry is not exposed to
            # the DQN feature vector.
            if not self._verified[
                bin_id
            ]:
                continue

            fill_state[
                bin_id
            ] = (
                float(
                    bin_.routing_fill_percent
                )
                / 100.0
            )

            hazard_state[
                bin_id
            ] = self._hazard[
                bin_id
            ]

            time_state[
                bin_id
            ] = min(
                1.0,
                self._full_elapsed[
                    bin_id
                ]
                / self.sla_limit_hours,
            )

        observation = np.concatenate(
            (
                truck_location,
                fill_state,
                hazard_state,
                time_state,
            )
        ).astype(
            np.float32
        )

        expected = self.expected_dimension(
            len(
                self._bin_ids
            )
        )

        if observation.shape != (
            expected,
        ):
            raise DQNStateAdapterError(
                "unexpected DQN observation shape"
            )

        return observation
