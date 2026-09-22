from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from smart_waste.collection.base import CollectionPolicy
from smart_waste.models.events import SimulationEvent
from smart_waste.models.truck import TruckStatus
from smart_waste.simulation.depot_service import (
    DepotReturnReason,
    schedule_depot_return,
)
from smart_waste.simulation.event_queue import EventQueue
from smart_waste.simulation.feasibility import (
    CandidateFeasibility,
    FeasibilityFailure,
    can_return_to_depot,
    evaluate_candidate_feasibility,
)
from smart_waste.simulation.movement import (
    schedule_truck_travel,
)
from smart_waste.simulation.policy_view import (
    build_policy_view,
)
from smart_waste.simulation.state import SimulationState


class DispatchError(RuntimeError):
    """Raised when a policy request cannot be dispatched safely."""


class DispatchAction(str, Enum):
    BIN_TRAVEL = "bin_travel"
    DEPOT_RETURN = "depot_return"
    POLICY_COMPLETE = "policy_complete"
    NO_CANDIDATE = "no_candidate"


@dataclass(frozen=True)
class DispatchResult:
    truck_id: int
    action: DispatchAction

    requested_bin_id: int | None = None

    feasibility: CandidateFeasibility | None = None

    depot_return_reason: DepotReturnReason | None = None

    event: SimulationEvent | None = None


def _depot_reason_from_failure(
    failure: FeasibilityFailure,
) -> DepotReturnReason:
    if failure == FeasibilityFailure.CAPACITY:
        return DepotReturnReason.CAPACITY

    if failure == FeasibilityFailure.FUEL:
        return DepotReturnReason.FUEL

    if failure == FeasibilityFailure.CAPACITY_AND_FUEL:
        return DepotReturnReason.COMBINED

    raise DispatchError(
        f"cannot map feasibility failure to depot return: {failure}"
    )


def _validate_recoverable_depot_return(
    *,
    state: SimulationState,
    truck_id: int,
    failure: FeasibilityFailure,
) -> None:
    """
    Ensure an infeasible candidate can be resolved safely by
    returning to the depot.

    This prevents the dispatcher from scheduling a depot return
    when the truck itself no longer has enough fuel to reach depot.
    """

    truck = state.trucks[truck_id]

    if not can_return_to_depot(
        state=state,
        truck_id=truck_id,
    ):
        raise DispatchError(
            f"truck {truck_id} cannot safely return to depot"
        )

    at_depot = (
        truck.current_node
        == state.depot.road_node
    )

    if not at_depot:
        return

    capacity_unrecoverable = (
        failure
        in {
            FeasibilityFailure.CAPACITY,
            FeasibilityFailure.CAPACITY_AND_FUEL,
        }
        and truck.current_load_tonnes == 0.0
    )

    fuel_unrecoverable = (
        failure
        in {
            FeasibilityFailure.FUEL,
            FeasibilityFailure.CAPACITY_AND_FUEL,
        }
        and truck.fuel_remaining_litres
        >= truck.fuel_capacity_litres
    )

    if capacity_unrecoverable:
        raise DispatchError(
            "candidate exceeds truck capacity even with "
            "an empty truck at depot"
        )

    if fuel_unrecoverable:
        raise DispatchError(
            "candidate violates dynamic fuel reserve even with "
            "a full tank at depot"
        )


def dispatch_next_for_truck(
    *,
    state: SimulationState,
    event_queue: EventQueue,
    policy: CollectionPolicy,
    truck_id: int,
    service_time_seconds: float,
) -> DispatchResult:
    """
    Request and safely dispatch one policy decision.

    Collection policy chooses only the desired bin.

    The dispatcher retains exclusive authority over:
    - candidate validation,
    - capacity feasibility,
    - dynamic fuel reserve,
    - depot-return decisions,
    - physical movement scheduling.
    """

    if truck_id not in state.trucks:
        raise DispatchError(
            f"unknown truck_id: {truck_id}"
        )

    truck = state.trucks[truck_id]

    if truck.status != TruckStatus.IDLE:
        raise DispatchError(
            f"truck {truck_id} must be idle before dispatch"
        )

    view = build_policy_view(state)

    if policy.is_complete(view):
        truck.status = TruckStatus.FINISHED

        return DispatchResult(
            truck_id=truck_id,
            action=DispatchAction.POLICY_COMPLETE,
        )

    requested_bin_id = policy.select_next_bin(
        view=view,
        truck_id=truck_id,
    )

    if requested_bin_id is None:
        return DispatchResult(
            truck_id=truck_id,
            action=DispatchAction.NO_CANDIDATE,
        )

    if not isinstance(requested_bin_id, int):
        raise DispatchError(
            "policy must return integer bin_id or None"
        )

    if requested_bin_id not in state.bins:
        raise DispatchError(
            f"policy requested unknown bin_id: "
            f"{requested_bin_id}"
        )

    feasibility = evaluate_candidate_feasibility(
        state=state,
        truck_id=truck_id,
        bin_id=requested_bin_id,
        service_time_seconds=service_time_seconds,
    )

    if feasibility.feasible:
        target_bin = state.bins[
            requested_bin_id
        ]

        event = schedule_truck_travel(
            state=state,
            event_queue=event_queue,
            truck_id=truck_id,
            destination_node=target_bin.road_node,
            metadata={
                "target_bin_id": requested_bin_id,
            },
        )

        return DispatchResult(
            truck_id=truck_id,
            action=DispatchAction.BIN_TRAVEL,
            requested_bin_id=requested_bin_id,
            feasibility=feasibility,
            event=event,
        )

    _validate_recoverable_depot_return(
        state=state,
        truck_id=truck_id,
        failure=feasibility.failure,
    )

    depot_reason = (
        _depot_reason_from_failure(
            feasibility.failure
        )
    )

    event = schedule_depot_return(
        state=state,
        event_queue=event_queue,
        truck_id=truck_id,
        reason=depot_reason,
    )

    return DispatchResult(
        truck_id=truck_id,
        action=DispatchAction.DEPOT_RETURN,
        requested_bin_id=requested_bin_id,
        feasibility=feasibility,
        depot_return_reason=depot_reason,
        event=event,
    )
