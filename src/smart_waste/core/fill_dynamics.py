from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from smart_waste.core.seed_manager import (
    create_rng,
    derive_seed,
)


@dataclass(frozen=True)
class BinFillState:
    bin_id: int
    current_fill_percent: float
    fill_rate_percent_per_hour: float

    def __post_init__(self) -> None:
        if self.bin_id < 0:
            raise ValueError(
                "bin_id must be non-negative"
            )

        if not 0.0 <= self.current_fill_percent <= 100.0:
            raise ValueError(
                "current_fill_percent must be between 0 and 100"
            )

        if self.fill_rate_percent_per_hour < 0.0:
            raise ValueError(
                "fill_rate_percent_per_hour cannot be negative"
            )


def advance_fill(
    current_fill_percent: float,
    *,
    fill_rate_percent_per_hour: float,
    elapsed_hours: float,
) -> float:
    """
    Advance a physical bin fill level over time.

    Fill is capped at 100%.
    """

    if not 0.0 <= current_fill_percent <= 100.0:
        raise ValueError(
            "current_fill_percent must be between 0 and 100"
        )

    if fill_rate_percent_per_hour < 0.0:
        raise ValueError(
            "fill_rate_percent_per_hour cannot be negative"
        )

    if elapsed_hours < 0.0:
        raise ValueError(
            "elapsed_hours cannot be negative"
        )

    predicted = (
        current_fill_percent
        + fill_rate_percent_per_hour
        * elapsed_hours
    )

    return min(100.0, predicted)


def predict_fill(
    current_fill_percent: float,
    *,
    fill_rate_percent_per_hour: float,
    prediction_horizon_hours: float,
) -> float:
    """
    Predict fill level at the end of the prediction horizon.

    At this stage the predictor is a transparent linear extrapolation
    model. It is a reconstruction assumption, not a value claimed
    by the manuscript.
    """

    return advance_fill(
        current_fill_percent,
        fill_rate_percent_per_hour=(
            fill_rate_percent_per_hour
        ),
        elapsed_hours=prediction_horizon_hours,
    )


def predicts_overflow(
    current_fill_percent: float,
    *,
    fill_rate_percent_per_hour: float,
    prediction_horizon_hours: float,
    overflow_threshold_percent: float = 100.0,
) -> bool:
    """
    Return True if the bin is predicted to reach the overflow
    threshold before the next collection cycle.
    """

    predicted = predict_fill(
        current_fill_percent,
        fill_rate_percent_per_hour=(
            fill_rate_percent_per_hour
        ),
        prediction_horizon_hours=(
            prediction_horizon_hours
        ),
    )

    return predicted >= overflow_threshold_percent


def is_near_full(
    fill_percent: float,
    *,
    threshold_percent: float = 90.0,
) -> bool:
    return fill_percent >= threshold_percent


def is_predictive_75_80_candidate(
    state: BinFillState,
    *,
    prediction_horizon_hours: float,
    minimum_percent: float = 75.0,
    maximum_percent: float = 80.0,
) -> bool:
    """
    Reproduce the representative manuscript scenario in which
    75-80% bins are added when predicted to overflow.
    """

    if not (
        minimum_percent
        <= state.current_fill_percent
        <= maximum_percent
    ):
        return False

    return predicts_overflow(
        state.current_fill_percent,
        fill_rate_percent_per_hour=(
            state.fill_rate_percent_per_hour
        ),
        prediction_horizon_hours=(
            prediction_horizon_hours
        ),
    )


def select_proactive_candidate_ids(
    states: tuple[BinFillState, ...],
    *,
    prediction_horizon_hours: float,
    near_full_threshold_percent: float = 90.0,
) -> tuple[int, ...]:
    """
    Select bins for proactive routing.

    A bin is eligible if:
    1. it is already near the manuscript's ~90% threshold, OR
    2. it is predicted to overflow before the next collection cycle.

    This general rule follows the Layer-5 description.
    """

    selected: list[int] = []

    for state in states:

        near_full = is_near_full(
            state.current_fill_percent,
            threshold_percent=(
                near_full_threshold_percent
            ),
        )

        predicted_overflow = predicts_overflow(
            state.current_fill_percent,
            fill_rate_percent_per_hour=(
                state.fill_rate_percent_per_hour
            ),
            prediction_horizon_hours=(
                prediction_horizon_hours
            ),
        )

        if near_full or predicted_overflow:
            selected.append(state.bin_id)

    return tuple(selected)


def generate_fill_rates(
    *,
    num_bins: int,
    master_seed: int,
    minimum_rate_percent_per_hour: float,
    maximum_rate_percent_per_hour: float,
    replicate_id: int = 0,
) -> np.ndarray:
    """
    Generate deterministic bin-specific fill rates.

    The distribution bounds are explicit reconstruction parameters.
    No manuscript-derived default is imposed.
    """

    if num_bins <= 0:
        raise ValueError(
            "num_bins must be greater than zero"
        )

    if minimum_rate_percent_per_hour < 0.0:
        raise ValueError(
            "minimum rate cannot be negative"
        )

    if (
        maximum_rate_percent_per_hour
        < minimum_rate_percent_per_hour
    ):
        raise ValueError(
            "maximum rate must be >= minimum rate"
        )

    if replicate_id < 0:
        raise ValueError(
            "replicate_id must be non-negative"
        )

    seed = derive_seed(
        master_seed,
        f"fill-rates:replicate:{replicate_id}",
    )

    rng = create_rng(seed)

    return rng.uniform(
        low=minimum_rate_percent_per_hour,
        high=maximum_rate_percent_per_hour,
        size=num_bins,
    )