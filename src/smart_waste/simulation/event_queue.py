from __future__ import annotations

import heapq
from typing import Any

from smart_waste.models.events import (
    EventType,
    SimulationEvent,
)


class EventQueue:
    """
    Deterministic priority queue for simulation events.

    Events are ordered by:
    1. simulation timestamp,
    2. monotonically increasing sequence number.
    """

    def __init__(self) -> None:
        self._heap: list[SimulationEvent] = []
        self._next_sequence = 0

    def __len__(self) -> int:
        return len(self._heap)

    @property
    def is_empty(self) -> bool:
        return not self._heap

    def schedule(
        self,
        *,
        time_hours: float,
        event_type: EventType,
        entity_id: int | str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> SimulationEvent:
        if time_hours < 0.0:
            raise ValueError(
                "time_hours cannot be negative"
            )

        event = SimulationEvent(
            time_hours=float(time_hours),
            sequence=self._next_sequence,
            event_type=event_type,
            entity_id=entity_id,
            payload={} if payload is None else dict(payload),
        )

        self._next_sequence += 1
        heapq.heappush(self._heap, event)

        return event

    def peek(self) -> SimulationEvent:
        if not self._heap:
            raise IndexError("event queue is empty")

        return self._heap[0]

    def pop(self) -> SimulationEvent:
        if not self._heap:
            raise IndexError("event queue is empty")

        return heapq.heappop(self._heap)

    def clear(self) -> None:
        self._heap.clear()
        self._next_sequence = 0
