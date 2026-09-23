from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from smart_waste.collection.base import (
    CollectionPolicy,
    PolicyView,
)
from smart_waste.collection.dqn import (
    DQNCollectionPolicy,
)


class PhysicalDQNTrainingError(RuntimeError):
    """Raised when a physical DQN transition is invalid."""


@dataclass(frozen=True)
class PhysicalRewardWeights:
    """
    Manuscript reward coefficients:

        R = wc*C - wd*D - wh*H - ws*S
    """

    collection: float = 1.0
    distance: float = 0.5
    hazard_delay: float = 5.0
    sla_violation: float = 3.0


@dataclass(frozen=True)
class PendingPhysicalDecision:
    truck_id: int

    action_bin_id: int

    state: np.ndarray

    start_time_hours: float

    start_cumulative_distance_km: float

    start_true_fill_percent: float

    fill_rate_percent_per_hour: float


@dataclass(frozen=True)
class PhysicalTransitionRecord:
    truck_id: int

    action_bin_id: int

    transition_kind: str

    start_time_hours: float
    end_time_hours: float
    elapsed_hours: float

    distance_km: float

    collection_fraction: float

    hazard_delay_term: float

    sla_violation_term: float

    reward: float

    terminal: bool

    optimization_loss: float | None


class PhysicalDQNTrainingPolicy(
    CollectionPolicy
):
    """
    Training wrapper around DQNCollectionPolicy.

    The wrapped DQN remains a destination-selection policy only.

    Physical movement, capacity, fuel, depot returns, unloading,
    refuelling and global time remain entirely in the central Core.

    A transition starts when the DQN selects a bin and normally
    ends when that same bin is physically serviced.

    If the Core first sends the truck to the depot for capacity or
    fuel recovery, the SAME pending action is retained and retried
    after depot service whenever it is still a valid candidate.

    This is important: the resulting reward then includes the
    physical depot detour in distance/time rather than silently
    treating depot recovery as a free operation.
    """

    def __init__(
        self,
        *,
        policy: DQNCollectionPolicy,
        reward_weights: (
            PhysicalRewardWeights
            | None
        ) = None,
        optimize_after_transition: bool = True,
    ) -> None:
        self._policy = policy

        self._reward_weights = (
            reward_weights
            if reward_weights is not None
            else PhysicalRewardWeights()
        )

        self._optimize_after_transition = bool(
            optimize_after_transition
        )

        self._pending: dict[
            int,
            PendingPhysicalDecision,
        ] = {}

        self._transitions: list[
            PhysicalTransitionRecord
        ] = []

        self._initialized = False

    @property
    def policy(
        self,
    ) -> DQNCollectionPolicy:
        return self._policy

    @property
    def transitions(
        self,
    ) -> tuple[
        PhysicalTransitionRecord,
        ...,
    ]:
        return tuple(
            self._transitions
        )

    @property
    def pending_truck_ids(
        self,
    ) -> tuple[int, ...]:
        return tuple(
            sorted(
                self._pending
            )
        )

    def set_epsilon(
        self,
        epsilon: float,
    ) -> None:
        self._policy.set_epsilon(
            epsilon
        )

    def initialize(
        self,
        view: PolicyView,
    ) -> None:
        if self._initialized:
            raise PhysicalDQNTrainingError(
                "physical DQN training policy "
                "is already initialized"
            )

        self._policy.initialize(
            view
        )

        self._pending.clear()
        self._transitions.clear()

        self._initialized = True

    def _validated_distance_delta(
        self,
        *,
        pending: PendingPhysicalDecision,
        view: PolicyView,
    ) -> float:
        truck = view.truck_by_id(
            pending.truck_id
        )

        distance = (
            float(
                truck.cumulative_distance_km
            )
            - pending.start_cumulative_distance_km
        )

        if (
            not isfinite(
                distance
            )
            or distance < -1e-10
        ):
            raise PhysicalDQNTrainingError(
                "physical cumulative distance "
                "moved backwards"
            )

        return max(
            0.0,
            distance,
        )

    def _elapsed_hours(
        self,
        *,
        pending: PendingPhysicalDecision,
        view: PolicyView,
    ) -> float:
        elapsed = (
            float(
                view.current_time_hours
            )
            - pending.start_time_hours
        )

        if (
            not isfinite(
                elapsed
            )
            or elapsed < -1e-12
        ):
            raise PhysicalDQNTrainingError(
                "physical transition time "
                "moved backwards"
            )

        return max(
            0.0,
            elapsed,
        )

    def _physical_collection_fraction(
        self,
        *,
        pending: PendingPhysicalDecision,
        elapsed_hours: float,
    ) -> float:
        """
        Reconstruct physical fill at service completion.

        WasteBin dynamics use the same constant per-run
        fill_rate_percent_per_hour and cap fill at 100%.

        This uses TRUE physical fill cached at action selection,
        not routing/spoofed telemetry.
        """

        fill_at_service = min(
            100.0,
            (
                pending.start_true_fill_percent
                + (
                    pending.fill_rate_percent_per_hour
                    * elapsed_hours
                )
            ),
        )

        return max(
            0.0,
            fill_at_service / 100.0,
        )

    def _hazard_delay_term(
        self,
        *,
        view: PolicyView,
        elapsed_hours: float,
    ) -> float:
        """
        Preserve the legacy DQN reward semantics while replacing
        fixed decision time with actual physical elapsed time.

        The serviced bin's hazard is cleared before this function
        is called. Remaining verified hazard severity therefore
        represents unresolved hazard exposure after the action.

        This is a DQN training signal, not the final reported
        physical hazard-response metric.
        """

        total_severity = 0.0

        for bin_ in view.bins:
            bin_id = (
                bin_.bin_id
            )

            if not (
                self._policy
                .state_adapter
                .is_verified(
                    bin_id
                )
            ):
                continue

            total_severity += (
                self._policy
                .state_adapter
                .hazard_severity(
                    bin_id
                )
            )

        return (
            total_severity
            * elapsed_hours
        )

    def _sla_violation_term(
        self,
        *,
        view: PolicyView,
    ) -> float:
        limit = (
            self._policy
            .state_adapter
            .sla_limit_hours
        )

        for bin_ in view.bins:
            bin_id = (
                bin_.bin_id
            )

            if not (
                self._policy
                .state_adapter
                .is_verified(
                    bin_id
                )
            ):
                continue

            if (
                self._policy
                .state_adapter
                .full_elapsed_hours(
                    bin_id
                )
                > limit
            ):
                return 1.0

        return 0.0

    def _reward(
        self,
        *,
        collection_fraction: float,
        distance_km: float,
        hazard_delay_term: float,
        sla_violation_term: float,
    ) -> float:
        weights = (
            self._reward_weights
        )

        return float(
            (
                weights.collection
                * collection_fraction
            )
            - (
                weights.distance
                * distance_km
            )
            - (
                weights.hazard_delay
                * hazard_delay_term
            )
            - (
                weights.sla_violation
                * sla_violation_term
            )
        )

    def _next_transition_state(
        self,
        *,
        view: PolicyView,
        truck_id: int,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        bool,
    ]:
        next_state = (
            self._policy
            .state_adapter
            .observation(
                view=view,
                truck_id=truck_id,
            )
        )

        next_mask = (
            self._policy.action_mask(
                view
            )
        )

        terminal = (
            self._policy.is_complete(
                view
            )
        )

        return (
            next_state,
            next_mask,
            terminal,
        )

    def _store_transition(
        self,
        *,
        pending: PendingPhysicalDecision,
        view: PolicyView,
        transition_kind: str,
        service_completed: bool,
    ) -> PhysicalTransitionRecord:
        elapsed = (
            self._elapsed_hours(
                pending=pending,
                view=view,
            )
        )

        distance = (
            self._validated_distance_delta(
                pending=pending,
                view=view,
            )
        )

        if service_completed:
            collection_fraction = (
                self._physical_collection_fraction(
                    pending=pending,
                    elapsed_hours=elapsed,
                )
            )
        else:
            collection_fraction = 0.0

        (
            next_state,
            next_mask,
            terminal,
        ) = self._next_transition_state(
            view=view,
            truck_id=(
                pending.truck_id
            ),
        )

        hazard_delay_term = (
            self._hazard_delay_term(
                view=view,
                elapsed_hours=elapsed,
            )
        )

        sla_violation_term = (
            self._sla_violation_term(
                view=view
            )
        )

        reward = self._reward(
            collection_fraction=(
                collection_fraction
            ),
            distance_km=distance,
            hazard_delay_term=(
                hazard_delay_term
            ),
            sla_violation_term=(
                sla_violation_term
            ),
        )

        self._policy.agent.remember(
            state=pending.state,
            action=(
                pending.action_bin_id
            ),
            reward=reward,
            next_state=next_state,
            terminal=terminal,
            next_action_mask=(
                next_mask
            ),
        )

        loss = None

        if self._optimize_after_transition:
            loss = (
                self._policy
                .agent
                .optimize()
            )

        record = (
            PhysicalTransitionRecord(
                truck_id=(
                    pending.truck_id
                ),
                action_bin_id=(
                    pending.action_bin_id
                ),
                transition_kind=(
                    transition_kind
                ),
                start_time_hours=(
                    pending.start_time_hours
                ),
                end_time_hours=float(
                    view.current_time_hours
                ),
                elapsed_hours=elapsed,
                distance_km=distance,
                collection_fraction=(
                    collection_fraction
                ),
                hazard_delay_term=(
                    hazard_delay_term
                ),
                sla_violation_term=(
                    sla_violation_term
                ),
                reward=reward,
                terminal=terminal,
                optimization_loss=(
                    loss
                ),
            )
        )

        self._transitions.append(
            record
        )

        return record

    def _pending_action_still_selectable(
        self,
        *,
        view: PolicyView,
        pending: PendingPhysicalDecision,
    ) -> bool:
        mask = (
            self._policy.action_mask(
                view
            )
        )

        action = (
            pending.action_bin_id
        )

        return bool(
            0
            <= action
            < len(
                mask
            )
            and mask[
                action
            ]
        )

    def select_next_bin(
        self,
        *,
        view: PolicyView,
        truck_id: int,
    ) -> int | None:
        if not self._initialized:
            raise PhysicalDQNTrainingError(
                "physical DQN training policy "
                "must be initialized"
            )

        existing = (
            self._pending.get(
                truck_id
            )
        )

        if existing is not None:
            # A prior action may have triggered a physical
            # depot return rather than immediate bin travel.
            #
            # If the selected bin is still valid, retain the
            # original action so its eventual transition includes
            # depot travel, queue/unload/refuel time, and the
            # subsequent trip to the bin.
            if self._pending_action_still_selectable(
                view=view,
                pending=existing,
            ):
                return (
                    existing.action_bin_id
                )

            # The target became unavailable or was superseded by
            # an emergency while the truck was recovering.
            #
            # Close the old transition with zero collection,
            # preserving all physical distance/time already spent.
            self._store_transition(
                pending=existing,
                view=view,
                transition_kind=(
                    "interrupted_before_service"
                ),
                service_completed=False,
            )

            del self._pending[
                truck_id
            ]

        mask = (
            self._policy.action_mask(
                view
            )
        )

        if not np.any(
            mask
        ):
            return None

        state = (
            self._policy
            .state_adapter
            .observation(
                view=view,
                truck_id=truck_id,
            )
        )

        action = (
            self._policy.select_next_bin(
                view=view,
                truck_id=truck_id,
            )
        )

        if action is None:
            return None

        truck = view.truck_by_id(
            truck_id
        )

        selected_bin = (
            view.bin_by_id(
                action
            )
        )

        self._pending[
            truck_id
        ] = (
            PendingPhysicalDecision(
                truck_id=truck_id,
                action_bin_id=(
                    action
                ),
                state=state.copy(),
                start_time_hours=float(
                    view.current_time_hours
                ),
                start_cumulative_distance_km=float(
                    truck.cumulative_distance_km
                ),
                start_true_fill_percent=float(
                    selected_bin.true_fill_percent
                ),
                fill_rate_percent_per_hour=float(
                    selected_bin.fill_rate_percent_per_hour
                ),
            )
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
            raise PhysicalDQNTrainingError(
                "physical DQN training policy "
                "must be initialized"
            )

        pending = self._pending.get(
            truck_id
        )

        if pending is None:
            raise PhysicalDQNTrainingError(
                f"truck {truck_id} completed service "
                "without a pending DQN decision"
            )

        if (
            pending.action_bin_id
            != bin_id
        ):
            raise PhysicalDQNTrainingError(
                f"truck {truck_id} serviced bin "
                f"{bin_id}, but pending DQN action "
                f"was {pending.action_bin_id}"
            )

        # First update DQN policy bookkeeping and clear the
        # serviced routing hazard.
        self._policy.on_service_complete(
            view=view,
            truck_id=truck_id,
            bin_id=bin_id,
        )

        self._store_transition(
            pending=pending,
            view=view,
            transition_kind=(
                "service_complete"
            ),
            service_completed=True,
        )

        del self._pending[
            truck_id
        ]

    def is_complete(
        self,
        view: PolicyView,
    ) -> bool:
        if not self._initialized:
            return False

        # An unresolved selected action is still policy work even
        # if the truck is temporarily at the depot.
        if self._pending:
            return False

        return (
            self._policy.is_complete(
                view
            )
        )
