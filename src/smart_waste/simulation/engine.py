from __future__ import annotations

from collections.abc import Callable

from smart_waste.models.events import (
    EventType,
    SimulationEvent,
)
from smart_waste.simulation.event_queue import EventQueue
from smart_waste.simulation.state import SimulationState


EventHandler = Callable[
    [SimulationState, SimulationEvent],
    None,
]


class SimulationEngine:
    """
    Deterministic discrete-event simulation engine.

    The engine exclusively owns temporal progression.
    Multiple handlers may observe/process the same event in
    registration order.
    """

    def __init__(
        self,
        *,
        state: SimulationState,
        event_queue: EventQueue | None = None,
    ) -> None:
        self.state = state

        self.event_queue = (
            EventQueue()
            if event_queue is None
            else event_queue
        )

        self._handlers: dict[
            EventType,
            list[EventHandler],
        ] = {}

        self.processed_events = 0

    def register_handler(
        self,
        event_type: EventType,
        handler: EventHandler,
    ) -> None:
        self._handlers.setdefault(
            event_type,
            [],
        ).append(handler)

    def step(self) -> SimulationEvent:
        """Process exactly one earliest event."""

        event = self.event_queue.pop()

        self.state.advance_to(
            event.time_hours
        )

        handlers = self._handlers.get(
            event.event_type,
            (),
        )

        for handler in handlers:
            handler(
                self.state,
                event,
            )

        self.state.validate_physical_invariants()

        self.processed_events += 1

        return event

    def run(
        self,
        *,
        until_hours: float | None = None,
        max_events: int | None = None,
    ) -> int:
        """
        Process events chronologically.

        Returns the number of events processed during this call.
        """

        if (
            until_hours is not None
            and until_hours < 0.0
        ):
            raise ValueError(
                "until_hours cannot be negative"
            )

        if (
            max_events is not None
            and max_events < 0
        ):
            raise ValueError(
                "max_events cannot be negative"
            )

        processed_at_start = (
            self.processed_events
        )

        while not self.event_queue.is_empty:
            if max_events is not None:
                processed_this_run = (
                    self.processed_events
                    - processed_at_start
                )

                if processed_this_run >= max_events:
                    break

            next_event = (
                self.event_queue.peek()
            )

            if (
                until_hours is not None
                and next_event.time_hours > until_hours
            ):
                break

            self.step()

        return (
            self.processed_events
            - processed_at_start
        )
