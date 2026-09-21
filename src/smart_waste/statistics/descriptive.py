from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class DescriptiveStats:
    n: int
    mean: float
    median: float
    std: float
    q1: float
    q3: float
    iqr: float
    minimum: float
    maximum: float
    ci95_low: float
    ci95_high: float


def summarize(
    values: list[float] | np.ndarray,
    confidence: float = 0.95,
) -> DescriptiveStats:
    """
    Compute descriptive statistics and a Student-t
    confidence interval for the population mean.
    """

    array = np.asarray(values, dtype=np.float64)

    if array.ndim != 1:
        raise ValueError("values must be one-dimensional")

    if len(array) < 2:
        raise ValueError(
            "at least two observations are required"
        )

    if not 0.0 < confidence < 1.0:
        raise ValueError(
            "confidence must be between 0 and 1"
        )

    n = len(array)

    mean = float(np.mean(array))
    median = float(np.median(array))

    std = float(
        np.std(
            array,
            ddof=1,
        )
    )

    q1 = float(np.percentile(array, 25))
    q3 = float(np.percentile(array, 75))

    if std == 0.0:
        ci_low = mean
        ci_high = mean
    else:
        sem = stats.sem(array)

        ci_low, ci_high = stats.t.interval(
            confidence,
            df=n - 1,
            loc=mean,
            scale=sem,
        )

    return DescriptiveStats(
        n=n,
        mean=mean,
        median=median,
        std=std,
        q1=q1,
        q3=q3,
        iqr=q3 - q1,
        minimum=float(np.min(array)),
        maximum=float(np.max(array)),
        ci95_low=float(ci_low),
        ci95_high=float(ci_high),
    )