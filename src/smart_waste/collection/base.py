from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from smart_waste.models.truck import TruckStatus


@dataclass(frozen=True)
class BinPolicyView:
    """
    Read-only bin information exposed to collection policies.

    Policies may choose bins, but cannot mutate physical bin state.
    """

    bin_id: int
    road_node: int | str
    fill_percent: float
    fill_rate_percent_per_hour: float
    waste_mass_tonnes: float


@dataclass(frozen=True)
class TruckPolicyView:
    """
    Read-only truck information exposed to collection policies.
    """

    truck_id: int
    current_node: int | str
    remaining_capacity_tonnes: float
    fuel_remaining_litres: float
    status: TruckStatus


@dataclass(frozen=True)
class PolicyView:
    """
    Immutable snapshot of policy-visible simulation state.

    The policy receives values, not the mutable SimulationState.
    """

    current_time_hours: float
    depot_node: int | str
    bins: tuple[BinPolicyView, ...]
    trucks: tuple[TruckPolicyView, ...]

    def bin_by_id(
        self,
        bin_id: int,
    ) -> BinPolicyView:
        for bin_ in self.bins:
            if bin_.bin_id == bin_id:
                return bin_

        raise KeyError(
            f"unknown bin_id: {bin_id}"
        )

    def truck_by_id(
        self,
        truck_id: int,
    ) -> TruckPolicyView:
        for truck in self.trucks:
            if truck.truck_id == truck_id:
                return truck

        raise KeyError(
            f"unknown truck_id: {truck_id}"
        )


class CollectionPolicy(ABC):
    """
    Interface implemented by TSR, HDR, DQN and future policies.

    A collection policy decides *which bin it wants*.

    It does not:
    - move trucks,
    - consume fuel,
    - change truck loads,
    - empty bins,
    - advance time,
    - bypass depot service,
    - decide physical feasibility.
    """

    @abstractmethod
    def initialize(
        self,
        view: PolicyView,
    ) -> None:
        """Initialize policy-specific internal state."""

    @abstractmethod
    def select_next_bin(
        self,
        *,
        view: PolicyView,
        truck_id: int,
    ) -> int | None:
        """
        Return requested bin_id.

        None means this policy currently has no bin to request
        for the specified truck.
        """

    @abstractmethod
    def on_service_complete(
        self,
        *,
        view: PolicyView,
        truck_id: int,
        bin_id: int,
    ) -> None:
        """
        Notify policy after physical service has completed.
        """

    @abstractmethod
    def is_complete(
        self,
        view: PolicyView,
    ) -> bool:
        """Return whether this policy has completed its work."""
