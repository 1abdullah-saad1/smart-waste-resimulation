from __future__ import annotations

from collections.abc import Mapping
from math import isclose

import numpy as np

from smart_waste.collection.base import (
    CollectionPolicy,
    PolicyView,
)
from smart_waste.movement.road_graph import (
    RoadGraph,
)
from smart_waste.rl.dqn import (
    DQNAgent,
)
from smart_waste.rl.state_adapter import (
    DQNPhysicalStateAdapter,
)


class DQNCollectionPolicyError(RuntimeError):
    """Raised when the physical DQN policy contract is violated."""


class DQNCollectionPolicy(CollectionPolicy):
    """
    Physical-core DQN collection policy.

    The DQN chooses only a bin_id.

    It deliberately does not:
    - move trucks,
    - calculate physical travel,
    - consume fuel,
    - unload trucks,
    - refuel trucks,
    - mutate bins,
    - advance simulation time,
    - bypass central feasibility checks.

    All physical operations remain responsibilities of the
    simulation dispatcher/orchestrator.
    """

    def __init__(
        self,
        *,
        road_graph: RoadGraph,
        agent: DQNAgent,
        near_full_threshold_percent: float = 90.0,
        prediction_horizon_hours: float = 4.0,
        sla_limit_hours: float = 6.0,
        epsilon: float = 0.0,
        verified_by_bin: (
            Mapping[int, bool]
            | None
        ) = None,
        hazard_severity_by_bin: (
            Mapping[int, float]
            | None
        ) = None,
        initial_full_elapsed_hours: (
            Mapping[int, float]
            | None
        ) = None,
    ) -> None:
        if not (
            0.0
            <= near_full_threshold_percent
            <= 100.0
        ):
            raise ValueError(
                "near_full_threshold_percent "
                "must be in [0, 100]"
            )

        if prediction_horizon_hours <= 0.0:
            raise ValueError(
                "prediction_horizon_hours must be positive"
            )

        if sla_limit_hours <= 0.0:
            raise ValueError(
                "sla_limit_hours must be positive"
            )

        if not (
            0.0
            <= epsilon
            <= 1.0
        ):
            raise ValueError(
                "epsilon must be in [0, 1]"
            )

        self._road_graph = (
            road_graph
        )

        self._agent = agent

        self._near_full_threshold_percent = (
            float(
                near_full_threshold_percent
            )
        )

        self._prediction_horizon_hours = (
            float(
                prediction_horizon_hours
            )
        )

        self._epsilon = float(
            epsilon
        )

        self._state_adapter = (
            DQNPhysicalStateAdapter(
                road_graph=road_graph,
                sla_limit_hours=(
                    sla_limit_hours
                ),
                verified_by_bin=(
                    verified_by_bin
                ),
                initial_hazard_severity=(
                    hazard_severity_by_bin
                ),
                initial_full_elapsed_hours=(
                    initial_full_elapsed_hours
                ),
            )
        )

        self._initialized = False

        self._completed_bin_ids: set[
            int
        ] = set()

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def completed_bin_ids(
        self,
    ) -> tuple[int, ...]:
        return tuple(
            sorted(
                self._completed_bin_ids
            )
        )

    @property
    def state_adapter(
        self,
    ) -> DQNPhysicalStateAdapter:
        return self._state_adapter

    @property
    def agent(
        self,
    ) -> DQNAgent:
        return self._agent

    @property
    def epsilon(
        self,
    ) -> float:
        return self._epsilon

    def set_epsilon(
        self,
        epsilon: float,
    ) -> None:
        if not (
            0.0
            <= epsilon
            <= 1.0
        ):
            raise ValueError(
                "epsilon must be in [0, 1]"
            )

        self._epsilon = float(
            epsilon
        )

    def initialize(
        self,
        view: PolicyView,
    ) -> None:
        if self._initialized:
            raise DQNCollectionPolicyError(
                "DQN collection policy "
                "is already initialized"
            )

        ordered_bins = tuple(
            sorted(
                view.bins,
                key=lambda item: item.bin_id,
            )
        )

        num_bins = len(
            ordered_bins
        )

        expected_input_dim = (
            DQNPhysicalStateAdapter.expected_dimension(
                num_bins
            )
        )

        if (
            self._agent.input_dim
            != expected_input_dim
        ):
            raise DQNCollectionPolicyError(
                "DQN input dimension mismatch: "
                f"agent={self._agent.input_dim}, "
                f"required={expected_input_dim}"
            )

        if (
            self._agent.action_dim
            != num_bins
        ):
            raise DQNCollectionPolicyError(
                "DQN action dimension mismatch: "
                f"agent={self._agent.action_dim}, "
                f"required={num_bins}"
            )

        self._state_adapter.initialize(
            view
        )

        self._completed_bin_ids.clear()

        self._initialized = True

    def _candidate_mask(
        self,
        *,
        view: PolicyView,
        include_reserved: bool,
    ) -> np.ndarray:
        if not self._initialized:
            raise DQNCollectionPolicyError(
                "DQN policy must be initialized"
            )

        self._state_adapter.update(
            view
        )

        num_bins = len(
            view.bins
        )

        candidates = np.zeros(
            num_bins,
            dtype=bool,
        )

        emergency = np.zeros(
            num_bins,
            dtype=bool,
        )

        for bin_ in view.bins:
            bin_id = (
                bin_.bin_id
            )

            if (
                bin_id
                in self._completed_bin_ids
            ):
                continue

            if not self._state_adapter.is_verified(
                bin_id
            ):
                continue

            routing_fill = float(
                bin_.routing_fill_percent
            )

            predicted_fill = (
                routing_fill
                + (
                    float(
                        bin_.fill_rate_percent_per_hour
                    )
                    * self._prediction_horizon_hours
                )
            )

            severity = (
                self._state_adapter.hazard_severity(
                    bin_id
                )
            )

            eligible = (
                routing_fill
                >= self._near_full_threshold_percent
                or predicted_fill
                >= 100.0
                or severity
                > 0.0
            )

            if not eligible:
                continue

            candidates[
                bin_id
            ] = True

            if isclose(
                severity,
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                emergency[
                    bin_id
                ] = True

        # Preserve the existing DQN emergency override:
        # emergency fire excludes ordinary candidates.
        if np.any(
            emergency
        ):
            candidates = emergency

        if not include_reserved:
            for bin_ in view.bins:
                if bin_.is_reserved:
                    candidates[
                        bin_.bin_id
                    ] = False

        return candidates

    def action_mask(
        self,
        view: PolicyView,
    ) -> np.ndarray:
        return self._candidate_mask(
            view=view,
            include_reserved=False,
        )

    def select_next_bin(
        self,
        *,
        view: PolicyView,
        truck_id: int,
    ) -> int | None:
        if not self._initialized:
            raise DQNCollectionPolicyError(
                "DQN policy must be initialized "
                "before selection"
            )

        mask = self.action_mask(
            view
        )

        if not np.any(
            mask
        ):
            return None

        observation = (
            self._state_adapter.observation(
                view=view,
                truck_id=truck_id,
            )
        )

        action = (
            self._agent.select_action(
                observation,
                mask,
                epsilon=(
                    self._epsilon
                ),
            )
        )

        if not mask[
            action
        ]:
            raise DQNCollectionPolicyError(
                "DQN selected a masked action"
            )

        return int(
            action
        )

    def on_service_complete(
        self,
        *,
        view: PolicyView,
        truck_id: int,
        bin_id: int,
    ) -> None:
        if not self._initialized:
            raise DQNCollectionPolicyError(
                "DQN policy must be initialized "
                "before service completion"
            )

        # Validate identities through PolicyView.
        view.truck_by_id(
            truck_id
        )

        completed_bin = view.bin_by_id(
            bin_id
        )

        if completed_bin.is_reserved:
            raise DQNCollectionPolicyError(
                f"completed DQN bin {bin_id} "
                "still appears reserved"
            )

        if (
            bin_id
            in self._completed_bin_ids
        ):
            raise DQNCollectionPolicyError(
                f"DQN bin {bin_id} "
                "was completed twice"
            )

        self._completed_bin_ids.add(
            bin_id
        )

        # Servicing clears the routing hazard attached to
        # this collection episode.
        self._state_adapter.clear_hazard(
            bin_id
        )

        self._state_adapter.update(
            view
        )

    def is_complete(
        self,
        view: PolicyView,
    ) -> bool:
        if not self._initialized:
            return False

        # Reservations must not falsely indicate completion.
        # A reserved eligible bin is still unfinished work.
        remaining = self._candidate_mask(
            view=view,
            include_reserved=True,
        )

        return not np.any(
            remaining
        )
