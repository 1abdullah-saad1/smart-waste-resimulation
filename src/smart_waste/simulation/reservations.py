from __future__ import annotations

from dataclasses import dataclass, field


class ReservationError(RuntimeError):
    """Raised when bin reservation invariants are violated."""


@dataclass
class BinReservationBook:
    """
    Deterministic one-bin-per-truck reservation registry.

    This is coordination state, not physical bin state.

    Invariants:
    - one bin can be reserved by at most one truck,
    - one truck can reserve at most one bin.
    """

    _bin_to_truck: dict[int, int] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    _truck_to_bin: dict[int, int] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __len__(self) -> int:
        return len(self._bin_to_truck)

    def owner(
        self,
        bin_id: int,
    ) -> int | None:
        return self._bin_to_truck.get(
            bin_id
        )

    def bin_for_truck(
        self,
        truck_id: int,
    ) -> int | None:
        return self._truck_to_bin.get(
            truck_id
        )

    def is_reserved(
        self,
        bin_id: int,
    ) -> bool:
        return bin_id in self._bin_to_truck

    def reserve(
        self,
        *,
        bin_id: int,
        truck_id: int,
    ) -> None:
        existing_owner = self.owner(
            bin_id
        )

        if existing_owner is not None:
            raise ReservationError(
                f"bin {bin_id} is already reserved by "
                f"truck {existing_owner}"
            )

        existing_bin = self.bin_for_truck(
            truck_id
        )

        if existing_bin is not None:
            raise ReservationError(
                f"truck {truck_id} already reserves "
                f"bin {existing_bin}"
            )

        self._bin_to_truck[
            bin_id
        ] = truck_id

        self._truck_to_bin[
            truck_id
        ] = bin_id

    def release(
        self,
        *,
        bin_id: int,
        truck_id: int,
    ) -> None:
        owner = self.owner(
            bin_id
        )

        if owner is None:
            raise ReservationError(
                f"bin {bin_id} is not reserved"
            )

        if owner != truck_id:
            raise ReservationError(
                f"bin {bin_id} is reserved by truck {owner}, "
                f"not truck {truck_id}"
            )

        del self._bin_to_truck[
            bin_id
        ]

        del self._truck_to_bin[
            truck_id
        ]

    def snapshot(
        self,
    ) -> dict[int, int]:
        """
        Return an independent deterministic snapshot.

        Callers cannot mutate the reservation book through it.
        """

        return dict(
            sorted(
                self._bin_to_truck.items()
            )
        )
