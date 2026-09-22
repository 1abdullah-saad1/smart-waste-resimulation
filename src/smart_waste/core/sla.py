from __future__ import annotations

from dataclasses import dataclass


FULL_BIN_THRESHOLD_PERCENT = 100.0
FULL_BIN_SLA_LIMIT_HOURS = 6.0


@dataclass(frozen=True)
class SLAState:
    """
    Time-dependent SLA state for one bin.

    T_t in the manuscript corresponds to the elapsed time
    for which the bin has remained full.

    Hazard elapsed time is also tracked because the manuscript
    contains an unresolved-hazard SLA branch.
    """

    bin_id: int
    full_bin_elapsed_hours: float = 0.0
    hazard_elapsed_hours: float = 0.0
    hazard_active: bool = False

    def __post_init__(self) -> None:
        if self.bin_id < 0:
            raise ValueError(
                "bin_id must be non-negative"
            )

        if self.full_bin_elapsed_hours < 0.0:
            raise ValueError(
                "full_bin_elapsed_hours cannot be negative"
            )

        if self.hazard_elapsed_hours < 0.0:
            raise ValueError(
                "hazard_elapsed_hours cannot be negative"
            )


@dataclass(frozen=True)
class SLAEvaluation:
    """
    Result of evaluating the smart-contract SLA conditions.

    hazard_violation is None when the hazard time limit
    is unspecified by the experimental configuration.
    """

    full_bin_violation: bool
    hazard_violation: bool | None

    @property
    def any_violation(self) -> bool:
        return (
            self.full_bin_violation
            or self.hazard_violation is True
        )

    @property
    def violation_indicator(self) -> float:
        """
        Binary S_t representation for the DQN reward.
        """

        return 1.0 if self.any_violation else 0.0


def advance_sla_state(
    previous: SLAState,
    *,
    current_fill_percent: float,
    hazard_active: bool,
    elapsed_hours: float,
    full_threshold_percent: float = (
        FULL_BIN_THRESHOLD_PERCENT
    ),
) -> SLAState:
    """
    Advance SLA timers.

    Full-bin timer:
      fill >= 100% -> accumulate time
      fill < 100%  -> reset timer

    Hazard timer:
      active hazard -> accumulate time
      no hazard     -> reset timer
    """

    if not 0.0 <= current_fill_percent <= 100.0:
        raise ValueError(
            "current_fill_percent must be between 0 and 100"
        )

    if elapsed_hours < 0.0:
        raise ValueError(
            "elapsed_hours cannot be negative"
        )

    if current_fill_percent >= full_threshold_percent:
        full_elapsed = (
            previous.full_bin_elapsed_hours
            + elapsed_hours
        )
    else:
        full_elapsed = 0.0

    if hazard_active:
        hazard_elapsed = (
            previous.hazard_elapsed_hours
            + elapsed_hours
        )
    else:
        hazard_elapsed = 0.0

    return SLAState(
        bin_id=previous.bin_id,
        full_bin_elapsed_hours=full_elapsed,
        hazard_elapsed_hours=hazard_elapsed,
        hazard_active=hazard_active,
    )


def evaluate_sla(
    state: SLAState,
    *,
    full_bin_limit_hours: float = (
        FULL_BIN_SLA_LIMIT_HOURS
    ),
    hazard_limit_hours: float | None = None,
) -> SLAEvaluation:
    """
    Evaluate manuscript SLA conditions.

    Full-bin rule is explicit in the manuscript:
        full duration > 6 hours -> violation.

    The manuscript refers to SLA_HAZARD_LIMIT but does not
    provide its numerical value. Therefore hazard violation
    remains unevaluated (None) unless an explicit value is
    supplied by a later reconstruction experiment.
    """

    if full_bin_limit_hours <= 0.0:
        raise ValueError(
            "full_bin_limit_hours must be positive"
        )

    full_violation = (
        state.full_bin_elapsed_hours
        > full_bin_limit_hours
    )

    if hazard_limit_hours is None:
        hazard_violation = None
    else:
        if hazard_limit_hours <= 0.0:
            raise ValueError(
                "hazard_limit_hours must be positive"
            )

        hazard_violation = (
            state.hazard_active
            and state.hazard_elapsed_hours
            > hazard_limit_hours
        )

    return SLAEvaluation(
        full_bin_violation=full_violation,
        hazard_violation=hazard_violation,
    )


def normalized_full_bin_time(
    elapsed_hours: float,
    *,
    sla_limit_hours: float = (
        FULL_BIN_SLA_LIMIT_HOURS
    ),
) -> float:
    """
    Normalize T_t to [0, 1] for neural-network input.

    Values at or beyond the SLA limit map to 1.0.
    The separate SLA violation indicator preserves whether
    the limit has actually been exceeded.
    """

    if elapsed_hours < 0.0:
        raise ValueError(
            "elapsed_hours cannot be negative"
        )

    if sla_limit_hours <= 0.0:
        raise ValueError(
            "sla_limit_hours must be positive"
        )

    return min(
        1.0,
        elapsed_hours / sla_limit_hours,
    )