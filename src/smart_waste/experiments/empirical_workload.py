from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


EMPIRICAL_WORKLOAD_SCHEMA = (
    "empirical-wyndham-workload-v1"
)

DEFAULT_PROFILE_LIBRARY = Path(
    "data/processed/workload/"
    "wyndham_empirical_profiles.csv"
)

EXPECTED_PROFILE_LIBRARY_SHA256 = (
    "0a7fdda9659d6eafc54aff2c965162d4"
    "a2cec1216f33ccdd420b0e96124c6bbb"
)


@dataclass(frozen=True)
class EmpiricalWorkloadSpec:
    num_bins: int
    master_seed: int
    replicate_id: int

    profile_library_path: str = (
        str(
            DEFAULT_PROFILE_LIBRARY
        )
    )

    profile_library_sha256: str = (
        EXPECTED_PROFILE_LIBRARY_SHA256
    )

    def __post_init__(self) -> None:
        if self.num_bins <= 0:
            raise ValueError(
                "num_bins must be positive"
            )

        if self.master_seed < 0:
            raise ValueError(
                "master_seed must be nonnegative"
            )

        if self.replicate_id < 0:
            raise ValueError(
                "replicate_id must be nonnegative"
            )

        if not self.profile_library_sha256:
            raise ValueError(
                "profile_library_sha256 "
                "must not be empty"
            )


@dataclass(frozen=True)
class EmpiricalPhysicalWorkloadRealization:
    schema_version: str

    spec: EmpiricalWorkloadSpec

    source_profile_sha256: str

    sampling_seed: int

    profile_ids: tuple[int, ...]

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

    @property
    def replicate_id(self) -> int:
        return (
            self.spec.replicate_id
        )

    @property
    def num_bins(self) -> int:
        return (
            self.spec.num_bins
        )


def _sha256_file(
    path: Path,
) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def _canonical_json_bytes(
    value: object,
) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        ensure_ascii=False,
        allow_nan=False,
    ).encode(
        "utf-8"
    )


def _derive_sampling_seed(
    *,
    master_seed: int,
    replicate_id: int,
) -> int:
    """
    Deterministic namespace-separated seed.

    This seed derivation is part of the frozen empirical-workload
    contract and does not depend on Python's process hash seed.
    """

    payload = (
        "smart-waste|"
        "empirical-wyndham-workload-v1|"
        f"master={master_seed}|"
        f"replicate={replicate_id}"
    ).encode(
        "utf-8"
    )

    digest = hashlib.sha256(
        payload
    ).digest()

    return int.from_bytes(
        digest[
            :8
        ],
        byteorder="big",
        signed=False,
    )


def load_empirical_profile_library(
    spec: EmpiricalWorkloadSpec,
) -> pd.DataFrame:
    path = Path(
        spec.profile_library_path
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Empirical profile library not found: {path}"
        )

    actual_sha256 = (
        _sha256_file(
            path
        )
    )

    if (
        actual_sha256
        != spec.profile_library_sha256
    ):
        raise RuntimeError(
            "Empirical profile-library SHA-256 mismatch.\n"
            f"expected={spec.profile_library_sha256}\n"
            f"actual={actual_sha256}"
        )

    frame = pd.read_csv(
        path
    )

    required_columns = {
        "profile_id",
        "initial_fill_percent",
        "fill_rate_percent_per_hour",
    }

    missing = (
        required_columns
        - set(
            frame.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Empirical profile library missing columns: "
            f"{sorted(missing)}"
        )

    if len(
        frame
    ) < spec.num_bins:
        raise RuntimeError(
            "Empirical profile library is too small "
            "for sampling without replacement"
        )

    if frame[
        "profile_id"
    ].isna().any():
        raise RuntimeError(
            "profile_id contains missing values"
        )

    if not frame[
        "profile_id"
    ].is_unique:
        raise RuntimeError(
            "profile_id must be unique"
        )

    profile_ids = tuple(
        int(value)
        for value in frame[
            "profile_id"
        ]
    )

    if profile_ids != tuple(
        range(
            len(
                frame
            )
        )
    ):
        raise RuntimeError(
            "profile_id must be contiguous "
            "from 0 to N-1"
        )

    initial_fill = pd.to_numeric(
        frame[
            "initial_fill_percent"
        ],
        errors="raise",
    )

    fill_rate = pd.to_numeric(
        frame[
            "fill_rate_percent_per_hour"
        ],
        errors="raise",
    )

    if not (
        initial_fill.between(
            0.0,
            100.0,
            inclusive="both",
        )
    ).all():
        raise RuntimeError(
            "initial fill outside [0, 100]"
        )

    if (
        fill_rate
        < 0.0
    ).any():
        raise RuntimeError(
            "negative empirical fill rate found"
        )

    return frame


def generate_empirical_workload(
    spec: EmpiricalWorkloadSpec,
) -> EmpiricalPhysicalWorkloadRealization:
    profiles = (
        load_empirical_profile_library(
            spec
        )
    )

    sampling_seed = (
        _derive_sampling_seed(
            master_seed=(
                spec.master_seed
            ),
            replicate_id=(
                spec.replicate_id
            ),
        )
    )

    rng = np.random.default_rng(
        sampling_seed
    )

    selected_indices = (
        rng.choice(
            len(
                profiles
            ),
            size=(
                spec.num_bins
            ),
            replace=False,
        )
    )

    selected = (
        profiles.iloc[
            selected_indices
        ]
        .reset_index(
            drop=True
        )
    )

    profile_ids = tuple(
        int(value)
        for value in selected[
            "profile_id"
        ]
    )

    if len(
        set(
            profile_ids
        )
    ) != spec.num_bins:
        raise RuntimeError(
            "Sampling without replacement "
            "produced duplicate profiles"
        )

    initial_fill_percent = tuple(
        float(value)
        for value in selected[
            "initial_fill_percent"
        ]
    )

    fill_rate_percent_per_hour = tuple(
        float(value)
        for value in selected[
            "fill_rate_percent_per_hour"
        ]
    )

    identity_payload = {
        "schema_version": (
            EMPIRICAL_WORKLOAD_SCHEMA
        ),
        "profile_library_sha256": (
            spec.profile_library_sha256
        ),
        "num_bins": (
            spec.num_bins
        ),
        "master_seed": (
            spec.master_seed
        ),
        "replicate_id": (
            spec.replicate_id
        ),
        "sampling_method": (
            "deterministic_without_replacement"
        ),
        "sampling_seed": (
            sampling_seed
        ),
        # Ordered profile IDs define the bin_id -> empirical
        # profile mapping. The profile-library SHA fixes the
        # physical values associated with each profile.
        "profile_ids": list(
            profile_ids
        ),
    }

    workload_sha256 = hashlib.sha256(
        _canonical_json_bytes(
            identity_payload
        )
    ).hexdigest()

    workload_id = (
        "empirical-workload-"
        + workload_sha256[
            :16
        ]
    )

    return (
        EmpiricalPhysicalWorkloadRealization(
            schema_version=(
                EMPIRICAL_WORKLOAD_SCHEMA
            ),
            spec=spec,
            source_profile_sha256=(
                spec.profile_library_sha256
            ),
            sampling_seed=(
                sampling_seed
            ),
            profile_ids=(
                profile_ids
            ),
            initial_fill_percent=(
                initial_fill_percent
            ),
            fill_rate_percent_per_hour=(
                fill_rate_percent_per_hour
            ),
            sha256=(
                workload_sha256
            ),
            workload_id=(
                workload_id
            ),
        )
    )
