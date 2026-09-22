from __future__ import annotations

import hashlib
import json

from dataclasses import dataclass
from math import isfinite

from smart_waste.core.fill_dynamics import (
    generate_fill_rates,
)
from smart_waste.core.seed_manager import (
    create_rng,
    derive_seed,
)


class WorkloadError(ValueError):
    """Raised when a physical workload specification is invalid."""


@dataclass(frozen=True)
class PhysicalWorkloadSpec:
    """
    Topology-independent physical workload specification.

    All distribution bounds are explicit reconstruction/benchmark
    parameters. No manuscript-derived defaults are imposed here.
    """

    num_bins: int

    master_seed: int
    replicate_id: int

    minimum_initial_fill_percent: float
    maximum_initial_fill_percent: float

    minimum_fill_rate_percent_per_hour: float
    maximum_fill_rate_percent_per_hour: float

    def __post_init__(self) -> None:
        if self.num_bins <= 0:
            raise WorkloadError(
                "num_bins must be positive"
            )

        if self.master_seed < 0:
            raise WorkloadError(
                "master_seed must be non-negative"
            )

        if self.replicate_id < 0:
            raise WorkloadError(
                "replicate_id must be non-negative"
            )

        values = (
            self.minimum_initial_fill_percent,
            self.maximum_initial_fill_percent,
            self.minimum_fill_rate_percent_per_hour,
            self.maximum_fill_rate_percent_per_hour,
        )

        if not all(
            isfinite(value)
            for value in values
        ):
            raise WorkloadError(
                "workload bounds must be finite"
            )

        if not (
            0.0
            <= self.minimum_initial_fill_percent
            <= self.maximum_initial_fill_percent
            <= 100.0
        ):
            raise WorkloadError(
                "initial fill range must satisfy "
                "0 <= minimum <= maximum <= 100"
            )

        if (
            self.minimum_fill_rate_percent_per_hour
            < 0.0
        ):
            raise WorkloadError(
                "minimum fill rate cannot be negative"
            )

        if (
            self.maximum_fill_rate_percent_per_hour
            < self.minimum_fill_rate_percent_per_hour
        ):
            raise WorkloadError(
                "maximum fill rate must be >= minimum fill rate"
            )


@dataclass(frozen=True)
class PhysicalWorkloadRealization:
    """
    Immutable topology-independent physical workload.

    Element i always belongs to bin_id i. Therefore a realization
    can be reused unchanged across road topologies whose canonical
    bin IDs are 0..N-1.
    """

    schema_version: str

    spec: PhysicalWorkloadSpec

    initial_fill_percent: tuple[
        float,
        ...,
    ]

    fill_rate_percent_per_hour: tuple[
        float,
        ...,
    ]

    sha256: str
    workload_id: str


def _canonical_payload(
    *,
    spec: PhysicalWorkloadSpec,
    initial_fill_percent: tuple[
        float,
        ...,
    ],
    fill_rate_percent_per_hour: tuple[
        float,
        ...,
    ],
) -> dict[str, object]:
    return {
        "schema_version": "physical-workload-v1",
        "num_bins": spec.num_bins,
        "master_seed": spec.master_seed,
        "replicate_id": spec.replicate_id,
        "distribution": {
            "initial_fill": {
                "type": "uniform",
                "minimum_percent": (
                    spec.minimum_initial_fill_percent
                ),
                "maximum_percent": (
                    spec.maximum_initial_fill_percent
                ),
            },
            "fill_rate": {
                "type": "uniform",
                "minimum_percent_per_hour": (
                    spec.minimum_fill_rate_percent_per_hour
                ),
                "maximum_percent_per_hour": (
                    spec.maximum_fill_rate_percent_per_hour
                ),
            },
        },
        "initial_fill_percent": list(
            initial_fill_percent
        ),
        "fill_rate_percent_per_hour": list(
            fill_rate_percent_per_hour
        ),
    }


def _payload_sha256(
    payload: dict[str, object],
) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        serialized
    ).hexdigest()


def generate_physical_workload(
    spec: PhysicalWorkloadSpec,
) -> PhysicalWorkloadRealization:
    """
    Generate one deterministic workload realization.

    The RNG namespaces intentionally contain no topology identity.
    This guarantees that one replicate can be reused across every
    benchmark road topology.
    """

    initial_fill_seed = derive_seed(
        spec.master_seed,
        (
            "initial-fills:"
            f"replicate:{spec.replicate_id}"
        ),
    )

    initial_fill_rng = create_rng(
        initial_fill_seed
    )

    initial_fill_array = (
        initial_fill_rng.uniform(
            low=(
                spec.minimum_initial_fill_percent
            ),
            high=(
                spec.maximum_initial_fill_percent
            ),
            size=spec.num_bins,
        )
    )

    fill_rate_array = (
        generate_fill_rates(
            num_bins=spec.num_bins,
            master_seed=spec.master_seed,
            minimum_rate_percent_per_hour=(
                spec.minimum_fill_rate_percent_per_hour
            ),
            maximum_rate_percent_per_hour=(
                spec.maximum_fill_rate_percent_per_hour
            ),
            replicate_id=spec.replicate_id,
        )
    )

    initial_fill = tuple(
        float(value)
        for value in initial_fill_array
    )

    fill_rates = tuple(
        float(value)
        for value in fill_rate_array
    )

    if (
        len(initial_fill)
        != spec.num_bins
        or len(fill_rates)
        != spec.num_bins
    ):
        raise RuntimeError(
            "generated workload length invariant violated"
        )

    payload = _canonical_payload(
        spec=spec,
        initial_fill_percent=initial_fill,
        fill_rate_percent_per_hour=fill_rates,
    )

    sha256 = _payload_sha256(
        payload
    )

    return PhysicalWorkloadRealization(
        schema_version="physical-workload-v1",
        spec=spec,
        initial_fill_percent=initial_fill,
        fill_rate_percent_per_hour=fill_rates,
        sha256=sha256,
        workload_id=(
            f"workload-{sha256[:16]}"
        ),
    )
