from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isclose

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
from smart_waste.simulation.reservations import (
    BinReservationBook,
    ReservationError,
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


def _truck_is_terminally_settled(
    *,
    state: SimulationState,
    truck_id: int,
) -> bool:
    """
    A completed truck is terminal only after it is physically
    back at depot, unloaded and fully refuelled.
    """

    truck = state.trucks[truck_id]

    at_depot = (
        truck.current_node
        == state.depot.road_node
    )

    empty = isclose(
        truck.current_load_tonnes,
        0.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    )

    full_tank = isclose(
        truck.fuel_remaining_litres,
        truck.fuel_capacity_litres,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )

    return (
        at_depot
        and empty
        and full_tank
    )


def _validate_recoverable_depot_return(
    *,
    state: SimulationState,
    truck_id: int,
    failure: FeasibilityFailure,
) -> None:
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
        and isclose(
            truck.fuel_remaining_litres,
            truck.fuel_capacity_litres,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
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
    reservation_book: BinReservationBook,
    truck_id: int,
    service_time_seconds: float,
) -> DispatchResult:
    """
    Execute one policy decision through the physical gate.

    Policy completion does not mean instantaneous truck
    termination. A truck must first return to depot, unload and
    refuel before FINISHED can be assigned.
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

    existing_reservation = (
        reservation_book.bin_for_truck(
            truck_id
        )
    )

    if existing_reservation is not None:
        raise DispatchError(
            f"truck {truck_id} already reserves "
            f"bin {existing_reservation}"
        )

    view = build_policy_view(
        state,
        reservations=(
            reservation_book.snapshot()
        ),
    )

    if policy.is_complete(view):
        if _truck_is_terminally_settled(
            state=state,
            truck_id=truck_id,
        ):
            truck.status = TruckStatus.FINISHED

            return DispatchResult(
                truck_id=truck_id,
                action=DispatchAction.POLICY_COMPLETE,
            )

        if not can_return_to_depot(
            state=state,
            truck_id=truck_id,
        ):
            raise DispatchError(
                f"truck {truck_id} completed policy work "
                "but cannot safely return to depot"
            )

        event = schedule_depot_return(
            state=state,
            event_queue=event_queue,
            truck_id=truck_id,
            reason=DepotReturnReason.ROUTINE,
        )

        return DispatchResult(
            truck_id=truck_id,
            action=DispatchAction.DEPOT_RETURN,
            depot_return_reason=DepotReturnReason.ROUTINE,
            event=event,
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

    reservation_owner = (
        reservation_book.owner(
            requested_bin_id
        )
    )

    if reservation_owner is not None:
        raise DispatchError(
            f"policy requested bin {requested_bin_id}, "
            f"already reserved by truck {reservation_owner}"
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

        try:
            reservation_book.reserve(
                bin_id=requested_bin_id,
                truck_id=truck_id,
            )

            event = schedule_truck_travel(
                state=state,
                event_queue=event_queue,
                truck_id=truck_id,
                destination_node=target_bin.road_node,
                metadata={
                    "target_bin_id": requested_bin_id,
                },
            )
        except Exception:
            if (
                reservation_book.owner(
                    requested_bin_id
                )
                == truck_id
            ):
                reservation_book.release(
                    bin_id=requested_bin_id,
                    truck_id=truck_id,
                )

            raise

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
