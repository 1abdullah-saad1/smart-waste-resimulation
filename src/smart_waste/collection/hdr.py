from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from smart_waste.collection.base import (
    PolicyView,
)


class HDRConfigurationError(ValueError):
    """Raised when the HDR benchmark definition is invalid."""


class HDRTelemetryError(RuntimeError):
    """Raised when reported routing telemetry is invalid."""


@dataclass(frozen=True)
class HDREligibilitySnapshot:
    """
    Immutable HDR candidate snapshot.

    Eligibility is determined from REPORTED fill, not physical
    waste truth.

    The snapshot is intentionally frozen at policy initialization
    for the revised benchmark. Route selection may later be dynamic
    among this pending set, but membership is reproducible and does
    not silently change because simulation time advances.
    """

    threshold_percent: float

    reported_fill_by_bin: tuple[
        tuple[int, float],
        ...,
    ]

    eligible_bin_ids: tuple[
        int,
        ...,
    ]

    def reported_fill_percent(
        self,
        bin_id: int,
    ) -> float:
        for (
            candidate_bin_id,
            reported_fill,
        ) in self.reported_fill_by_bin:
            if candidate_bin_id == bin_id:
                return reported_fill

        raise KeyError(
            f"unknown HDR bin_id: {bin_id}"
        )

    def is_eligible(
        self,
        bin_id: int,
    ) -> bool:
        if not any(
            candidate_bin_id == bin_id
            for (
                candidate_bin_id,
                _,
            ) in self.reported_fill_by_bin
        ):
            raise KeyError(
                f"unknown HDR bin_id: {bin_id}"
            )

        return (
            bin_id
            in self.eligible_bin_ids
        )


def _validated_threshold(
    threshold_percent: float,
) -> float:
    try:
        threshold = float(
            threshold_percent
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise HDRConfigurationError(
            "HDR threshold must be numeric"
        ) from exc

    if not isfinite(
        threshold
    ):
        raise HDRConfigurationError(
            "HDR threshold must be finite"
        )

    if not (
        0.0
        <= threshold
        <= 100.0
    ):
        raise HDRConfigurationError(
            "HDR threshold must be between 0 and 100"
        )

    return threshold


def _validated_reported_fill(
    *,
    bin_id: int,
    value: float,
) -> float:
    try:
        fill = float(
            value
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise HDRTelemetryError(
            f"reported fill for bin {bin_id} must be numeric"
        ) from exc

    if not isfinite(
        fill
    ):
        raise HDRTelemetryError(
            f"reported fill for bin {bin_id} must be finite"
        )

    if not (
        0.0
        <= fill
        <= 100.0
    ):
        raise HDRTelemetryError(
            f"reported fill for bin {bin_id} must be "
            "between 0 and 100"
        )

    return fill


def build_hdr_eligibility_snapshot(
    *,
    view: PolicyView,
    threshold_percent: float = 80.0,
) -> HDREligibilitySnapshot:
    """
    Freeze HDR service eligibility from one policy snapshot.

    Rule:
        eligible iff reported_fill >= threshold

    Inclusive >=80% reproduces the baseline definition while
    keeping physical fill and routing telemetry separate.

    The function deliberately performs no:
    - road-distance calculation,
    - truck assignment,
    - capacity check,
    - fuel check,
    - depot-return decision.
    """

    threshold = _validated_threshold(
        threshold_percent
    )

    reported_records: list[
        tuple[int, float]
    ] = []

    eligible: list[
        int
    ] = []

    for bin_ in sorted(
        view.bins,
        key=lambda item: item.bin_id,
    ):
        reported_fill = (
            _validated_reported_fill(
                bin_id=bin_.bin_id,
                value=(
                    bin_.routing_fill_percent
                ),
            )
        )

        reported_records.append(
            (
                bin_.bin_id,
                reported_fill,
            )
        )

        if (
            reported_fill
            >= threshold
        ):
            eligible.append(
                bin_.bin_id
            )

    return HDREligibilitySnapshot(
        threshold_percent=(
            threshold
        ),
        reported_fill_by_bin=tuple(
            reported_records
        ),
        eligible_bin_ids=tuple(
            eligible
        ),
    )
