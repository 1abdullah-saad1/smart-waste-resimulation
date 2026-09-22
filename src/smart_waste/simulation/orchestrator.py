from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from smart_waste.collection.base import CollectionPolicy
from smart_waste.models.events import (
    EventType,
    SimulationEvent,
)
from smart_waste.models.truck import TruckStatus
from smart_waste.simulation.bin_service import (
    register_bin_service_handlers,
    schedule_bin_service,
    service_duration_hours,
)
from smart_waste.simulation.depot_service import (
    DepotServiceState,
    register_depot_handlers,
)
from smart_waste.simulation.dispatcher import (
    DispatchResult,
    dispatch_next_for_truck,
)
from smart_waste.simulation.engine import SimulationEngine
from smart_waste.simulation.movement import (
    register_movement_handlers,
)
from smart_waste.simulation.policy_view import (
    build_policy_view,
)
from smart_waste.simulation.state import SimulationState


class OrchestratorError(RuntimeError):
    """Raised when simulation lifecycle orchestration is invalid."""


@dataclass
class SimulationOrchestrator:
    """
    Connect collection-policy decisions to the physical
    discrete-event simulation.

    Registration order is intentional:

    1. movement completes physical arrival,
    2. bin/depot physics mutate physical state,
    3. orchestrator reacts to the completed physical event.

    Policies never directly mutate physical state.
    """

    state: SimulationState
    engine: SimulationEngine
    policy: CollectionPolicy
    service_time_seconds: float

    depot_state: DepotServiceState = field(
        default_factory=DepotServiceState
    )

    dispatch_history: list[DispatchResult] = field(
        default_factory=list,
        init=False,
    )

    completed_services: list[
        tuple[int, int]
    ] = field(
        default_factory=list,
        init=False,
    )

    _handlers_registered: bool = field(
        default=False,
        init=False,
    )

    _policy_initialized: bool = field(
        default=False,
        init=False,
    )

    _started: bool = field(
        default=False,
        init=False,
    )

    def __post_init__(self) -> None:
        if self.engine.state is not self.state:
            raise OrchestratorError(
                "engine and orchestrator must reference "
                "the same SimulationState"
            )

        # Reuse the canonical validation.
        service_duration_hours(
            self.service_time_seconds
        )

    def register_handlers(self) -> None:
        """
        Register the complete physical event chain exactly once.
        """

        if self._handlers_registered:
            return

        # Registration order is part of the simulation contract.
        register_movement_handlers(
            self.engine
        )

        register_bin_service_handlers(
            self.engine
        )

        register_depot_handlers(
            self.engine,
            depot_state=self.depot_state,
        )

        # These handlers run after the corresponding physical
        # handlers above.
        self.engine.register_handler(
            EventType.TRUCK_ARRIVAL,
            self._handle_bin_arrival,
        )

        self.engine.register_handler(
            EventType.BIN_SERVICE_COMPLETE,
            self._handle_bin_service_complete,
        )

        self.engine.register_handler(
            EventType.DEPOT_SERVICE_COMPLETE,
            self._handle_depot_service_complete,
        )

        self._handlers_registered = True

    def initialize_policy(self) -> None:
        """Initialize policy state exactly once."""

        if self._policy_initialized:
            return

        self.policy.initialize(
            build_policy_view(
                self.state
            )
        )

        self._policy_initialized = True

    def dispatch_truck(
        self,
        truck_id: int,
    ) -> DispatchResult:
        """
        Execute one policy decision through the central dispatcher.
        """

        if not self._handlers_registered:
            raise OrchestratorError(
                "orchestrator handlers are not registered"
            )

        if not self._policy_initialized:
            raise OrchestratorError(
                "collection policy is not initialized"
            )

        result = dispatch_next_for_truck(
            state=self.state,
            event_queue=self.engine.event_queue,
            policy=self.policy,
            truck_id=truck_id,
            service_time_seconds=(
                self.service_time_seconds
            ),
        )

        self.dispatch_history.append(
            result
        )

        return result

    def start(
        self,
        truck_ids: Iterable[int] | None = None,
    ) -> tuple[DispatchResult, ...]:
        """
        Initialize the lifecycle and dispatch selected trucks.

        If truck_ids is omitted, all trucks are started in stable
        truck_id order.
        """

        if self._started:
            raise OrchestratorError(
                "simulation orchestrator has already started"
            )

        self.register_handlers()
        self.initialize_policy()

        if truck_ids is None:
            selected_ids = tuple(
                sorted(
                    self.state.trucks
                )
            )
        else:
            selected_ids = tuple(
                truck_ids
            )

        if len(set(selected_ids)) != len(
            selected_ids
        ):
            raise OrchestratorError(
                "truck_ids must not contain duplicates"
            )

        for truck_id in selected_ids:
            if truck_id not in self.state.trucks:
                raise OrchestratorError(
                    f"unknown truck_id: {truck_id}"
                )

            if (
                self.state.trucks[
                    truck_id
                ].status
                != TruckStatus.IDLE
            ):
                raise OrchestratorError(
                    f"truck {truck_id} must be idle "
                    "when simulation starts"
                )

        self._started = True

        results = tuple(
            self.dispatch_truck(
                truck_id
            )
            for truck_id in selected_ids
        )

        return results

    def _handle_bin_arrival(
        self,
        state: SimulationState,
        event: SimulationEvent,
    ) -> None:
        """
        After movement completes, automatically start physical
        service of the target bin.
        """

        if not isinstance(
            event.entity_id,
            int,
        ):
            raise OrchestratorError(
                "TRUCK_ARRIVAL requires integer truck_id"
            )

        target_bin_id = event.payload.get(
            "target_bin_id"
        )

        if not isinstance(
            target_bin_id,
            int,
        ):
            raise OrchestratorError(
                "dispatched bin arrival is missing target_bin_id"
            )

        schedule_bin_service(
            state=state,
            event_queue=self.engine.event_queue,
            truck_id=event.entity_id,
            bin_id=target_bin_id,
            service_time_seconds=(
                self.service_time_seconds
            ),
        )

    def _handle_bin_service_complete(
        self,
        state: SimulationState,
        event: SimulationEvent,
    ) -> None:
        """
        Notify policy only after physical collection has completed,
        then immediately request the next policy decision.
        """

        if not isinstance(
            event.entity_id,
            int,
        ):
            raise OrchestratorError(
                "BIN_SERVICE_COMPLETE requires integer truck_id"
            )

        bin_id = event.payload.get(
            "bin_id"
        )

        if not isinstance(
            bin_id,
            int,
        ):
            raise OrchestratorError(
                "BIN_SERVICE_COMPLETE requires integer bin_id"
            )

        truck_id = event.entity_id

        self.completed_services.append(
            (
                truck_id,
                bin_id,
            )
        )

        self.policy.on_service_complete(
            view=build_policy_view(
                state
            ),
            truck_id=truck_id,
            bin_id=bin_id,
        )

        self.dispatch_truck(
            truck_id
        )

    def _handle_depot_service_complete(
        self,
        state: SimulationState,
        event: SimulationEvent,
    ) -> None:
        """
        Resume collection automatically after unloading/refuelling.
        """

        if not isinstance(
            event.entity_id,
            int,
        ):
            raise OrchestratorError(
                "DEPOT_SERVICE_COMPLETE requires integer truck_id"
            )

        truck_id = event.entity_id

        if (
            state.trucks[
                truck_id
            ].status
            != TruckStatus.IDLE
        ):
            raise OrchestratorError(
                "depot physical service must complete before "
                "collection can resume"
            )

        self.dispatch_truck(
            truck_id
        )
