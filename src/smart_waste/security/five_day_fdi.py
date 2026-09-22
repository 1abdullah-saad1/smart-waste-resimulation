from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DailyFDIResult:
    day: int

    injected_today: int
    cumulative_injected: int

    unprotected_processed_today: int
    unprotected_cumulative_processed: int

    poa_processed_today: int
    poa_cumulative_processed: int

    poa_rejected_today: int
    poa_cumulative_rejected: int


@dataclass(frozen=True)
class FiveDayFDIResult:
    daily_results: tuple[DailyFDIResult, ...]

    @property
    def total_injected(self) -> int:
        return sum(
            row.injected_today
            for row in self.daily_results
        )

    @property
    def total_unprotected_processed(self) -> int:
        return sum(
            row.unprotected_processed_today
            for row in self.daily_results
        )

    @property
    def total_poa_processed(self) -> int:
        return sum(
            row.poa_processed_today
            for row in self.daily_results
        )

    @property
    def total_poa_rejected(self) -> int:
        return sum(
            row.poa_rejected_today
            for row in self.daily_results
        )

    @property
    def unprotected_cloud_penetration_rate(self) -> float:
        if self.total_injected == 0:
            return 0.0

        return (
            self.total_unprotected_processed
            / self.total_injected
        )

    @property
    def poa_cloud_penetration_rate(self) -> float:
        if self.total_injected == 0:
            return 0.0

        return (
            self.total_poa_processed
            / self.total_injected
        )

    @property
    def poa_rejection_rate(self) -> float:
        if self.total_injected == 0:
            return 0.0

        return (
            self.total_poa_rejected
            / self.total_injected
        )


def simulate_five_day_unauthenticated_fdi(
    *,
    alerts_per_day: int = 10,
    number_of_days: int = 5,
) -> FiveDayFDIResult:
    """
    Reproduce the manuscript's five-day unauthenticated FDI test.

    Figure 14 depicts 10 new forged bin-full alerts per day,
    producing cumulative counts of 10, 20, 30, 40 and 50.

    Unprotected systems accept every forged alert.

    The proposed PoA/signature-verification path rejects every
    injected event because the evaluated attacker does not possess
    valid edge-node signing credentials.
    """

    if alerts_per_day <= 0:
        raise ValueError(
            "alerts_per_day must be greater than zero"
        )

    if number_of_days <= 0:
        raise ValueError(
            "number_of_days must be greater than zero"
        )

    rows: list[DailyFDIResult] = []

    cumulative_injected = 0
    cumulative_unprotected = 0
    cumulative_poa_processed = 0
    cumulative_poa_rejected = 0

    for day in range(1, number_of_days + 1):

        injected_today = alerts_per_day

        # Unprotected baseline trusts all forged telemetry.
        unprotected_today = injected_today

        # Manuscript primary threat model:
        # forged events have no valid signing credentials.
        poa_processed_today = 0
        poa_rejected_today = injected_today

        cumulative_injected += injected_today
        cumulative_unprotected += unprotected_today
        cumulative_poa_processed += poa_processed_today
        cumulative_poa_rejected += poa_rejected_today

        rows.append(
            DailyFDIResult(
                day=day,
                injected_today=injected_today,
                cumulative_injected=cumulative_injected,
                unprotected_processed_today=(
                    unprotected_today
                ),
                unprotected_cumulative_processed=(
                    cumulative_unprotected
                ),
                poa_processed_today=(
                    poa_processed_today
                ),
                poa_cumulative_processed=(
                    cumulative_poa_processed
                ),
                poa_rejected_today=(
                    poa_rejected_today
                ),
                poa_cumulative_rejected=(
                    cumulative_poa_rejected
                ),
            )
        )

    return FiveDayFDIResult(
        daily_results=tuple(rows)
    )