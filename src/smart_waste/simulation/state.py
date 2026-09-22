from __future__ import annotations

from dataclasses import dataclass

from smart_waste.models.bin import WasteBin
from smart_waste.models.depot import Depot
from smart_waste.models.truck import Truck
from smart_waste.movement.road_graph import RoadGraph


class SimulationStateError(ValueError):
    """Raised when global simulation state is invalid."""


@dataclass
class SimulationState:
    """
    Complete mutable physical state of one simulation run.

    A single global clock is shared by every truck, bin,
    depot operation and future subsystem.
    """

    road_graph: RoadGraph
    bins: dict[int, WasteBin]
    trucks: dict[int, Truck]
    depot: Depot
    current_time_hours: float = 0.0

    def __post_init__(self) -> None:
        if self.current_time_hours < 0.0:
            raise SimulationStateError(
                "current_time_hours cannot be negative"
            )

        if not self.bins:
            raise SimulationStateError(
                "simulation must contain at least one bin"
            )

        if not self.trucks:
            raise SimulationStateError(
                "simulation must contain at least one truck"
            )

        self._validate_identity_consistency()
        self._validate_road_nodes()
        self.road_graph.validate_connected()

    def _validate_identity_consistency(self) -> None:
        for key, bin_ in self.bins.items():
            if key != bin_.bin_id:
                raise SimulationStateError(
                    "bin dictionary key does not match bin_id"
                )

        for key, truck in self.trucks.items():
            if key != truck.truck_id:
                raise SimulationStateError(
                    "truck dictionary key does not match truck_id"
                )

    def _validate_road_nodes(self) -> None:
        service_nodes: list[int | str] = [
            self.depot.road_node
        ]

        service_nodes.extend(
            bin_.road_node
            for bin_ in self.bins.values()
        )

        service_nodes.extend(
            truck.current_node
            for truck in self.trucks.values()
        )

        try:
            self.road_graph.validate_service_nodes(
                tuple(service_nodes)
            )
        except ValueError as exc:
            raise SimulationStateError(
                str(exc)
            ) from exc

    def advance_to(self, target_time_hours: float) -> float:
        """
        Advance the global physical clock.

        All time-dependent physical bin states advance by exactly
        the same elapsed duration.

        Returns elapsed simulation time in hours.
        """

        if target_time_hours < self.current_time_hours:
            raise SimulationStateError(
                "simulation time cannot move backwards"
            )

        elapsed = (
            target_time_hours
            - self.current_time_hours
        )

        if elapsed == 0.0:
            return 0.0

        for bin_ in self.bins.values():
            bin_.advance(elapsed)

        self.current_time_hours = float(
            target_time_hours
        )

        return elapsed

    def validate_physical_invariants(self) -> None:
        """
        Validate invariants that must hold throughout a run.
        """

        for bin_ in self.bins.values():
            if not 0.0 <= bin_.fill_percent <= 100.0:
                raise SimulationStateError(
                    f"bin {bin_.bin_id} fill invariant violated"
                )

        for truck in self.trucks.values():
            if not (
                0.0
                <= truck.current_load_tonnes
                <= truck.capacity_tonnes
            ):
                raise SimulationStateError(
                    f"truck {truck.truck_id} capacity invariant violated"
                )

            if not (
                0.0
                <= truck.fuel_remaining_litres
                <= truck.fuel_capacity_litres
            ):
                raise SimulationStateError(
                    f"truck {truck.truck_id} fuel invariant violated"
                )

            if truck.current_node not in self.road_graph.graph:
                raise SimulationStateError(
                    f"truck {truck.truck_id} is not on road network"
                )
