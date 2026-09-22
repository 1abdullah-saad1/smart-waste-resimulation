from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventType(str, Enum):
    TRUCK_ARRIVAL = "truck_arrival"
    BIN_SERVICE_COMPLETE = "bin_service_complete"
    DEPOT_ARRIVAL = "depot_arrival"
    DEPOT_SERVICE_START = "depot_service_start"
    DEPOT_UNLOAD_COMPLETE = "depot_unload_complete"
    DEPOT_SERVICE_COMPLETE = "depot_service_complete"
    SENSOR_SAMPLE = "sensor_sample"
    HAZARD = "hazard"
    RADIO = "radio"


@dataclass(order=True, frozen=True)
class SimulationEvent:
    """
    Event ordered first by simulation time and then by sequence.

    Sequence provides deterministic ordering when two events have
    the same timestamp.
    """

    time_hours: float
    sequence: int

    event_type: EventType = field(compare=False)
    entity_id: int | str | None = field(
        default=None,
        compare=False,
    )
    payload: dict[str, Any] = field(
        default_factory=dict,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.time_hours < 0.0:
            raise ValueError(
                "time_hours cannot be negative"
            )

        if self.sequence < 0:
            raise ValueError(
                "sequence must be non-negative"
            )
