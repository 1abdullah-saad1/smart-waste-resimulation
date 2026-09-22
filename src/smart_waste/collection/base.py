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
    reserved_by_truck_id: int | None = None

    @property
    def is_reserved(self) -> bool:
        return self.reserved_by_truck_id is not None

    def is_reserved_by(
        self,
        truck_id: int,
    ) -> bool:
        return (
            self.reserved_by_truck_id
            == truck_id
        )


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
class CandidateMask:
    """
    Immutable policy-side coordination mask.

    This mask expresses reservation availability only.

    It does NOT certify:
    - truck capacity,
    - fuel feasibility,
    - road reachability,
    - service-time feasibility.

    Those remain simulation-core responsibilities.
    """

    bin_ids: tuple[int, ...]
    selectable: tuple[bool, ...]

    def __post_init__(self) -> None:
        if len(self.bin_ids) != len(self.selectable):
            raise ValueError(
                "bin_ids and selectable must have equal length"
            )

        if len(set(self.bin_ids)) != len(self.bin_ids):
            raise ValueError(
                "CandidateMask bin_ids must be unique"
            )

    @property
    def selectable_bin_ids(self) -> tuple[int, ...]:
        return tuple(
            bin_id
            for bin_id, allowed in zip(
                self.bin_ids,
                self.selectable,
                strict=True,
            )
            if allowed
        )

    def is_selectable(
        self,
        bin_id: int,
    ) -> bool:
        try:
            index = self.bin_ids.index(
                bin_id
            )
        except ValueError as exc:
            raise KeyError(
                f"unknown bin_id: {bin_id}"
            ) from exc

        return self.selectable[index]


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

    def reservation_availability_mask(
        self,
    ) -> CandidateMask:
        """
        Return a deterministic mask aligned with self.bins.

        True means the bin is currently unreserved and may be
        requested by a collection policy.

        This is coordination availability, not physical
        feasibility.
        """

        return CandidateMask(
            bin_ids=tuple(
                bin_.bin_id
                for bin_ in self.bins
            ),
            selectable=tuple(
                not bin_.is_reserved
                for bin_ in self.bins
            ),
        )

    def available_bins(
        self,
    ) -> tuple[BinPolicyView, ...]:
        """
        Return currently unreserved bins in stable bin-id order.
        """

        return tuple(
            bin_
            for bin_ in self.bins
            if not bin_.is_reserved
        )

    def available_bin_ids(
        self,
    ) -> tuple[int, ...]:
        return tuple(
            bin_.bin_id
            for bin_ in self.available_bins()
        )

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
