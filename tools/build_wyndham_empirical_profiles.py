from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


RAW_PATH = Path(
    "data/external/wyndham/"
    "wyndham_smartbin_filllevel.json"
)

OUTPUT_CSV = Path(
    "data/processed/workload/"
    "wyndham_empirical_profiles.csv"
)

OUTPUT_MANIFEST = Path(
    "data/processed/workload/"
    "wyndham_empirical_profiles_manifest.json"
)

EXPECTED_RAW_SHA256 = (
    "b2d7f95bb5b235c16c93a14d6d13f736"
    "ad682dd6707ec3b2b1841ef99f364ebb"
)

PROFILE_SCHEMA_VERSION = (
    "wyndham-empirical-profile-v1"
)

SOURCE_DATASET = (
    "Wyndham Smart Bin Fill Level Historical"
)

SOURCE_URL = (
    "https://data.gov.au/data/dataset/"
    "660a87c3-480e-498b-bfc3-ec84dc504b1c/"
    "resource/58ef5329-3141-4019-be41-24f2584256bc/"
    "download/wyndham_smartbin_filllevel.json"
)


def sha256_bytes(
    payload: bytes,
) -> str:
    return hashlib.sha256(
        payload
    ).hexdigest()


def sha256_file(
    path: Path,
) -> str:
    return sha256_bytes(
        path.read_bytes()
    )


def load_raw() -> pd.DataFrame:
    raw_bytes = (
        RAW_PATH.read_bytes()
    )

    raw_sha = sha256_bytes(
        raw_bytes
    )

    if raw_sha != EXPECTED_RAW_SHA256:
        raise RuntimeError(
            "Raw Wyndham dataset SHA-256 mismatch.\n"
            f"expected={EXPECTED_RAW_SHA256}\n"
            f"actual={raw_sha}"
        )

    data = json.loads(
        raw_bytes.decode(
            "utf-8-sig"
        )
    )

    features = data.get(
        "features"
    )

    if not isinstance(
        features,
        list,
    ):
        raise RuntimeError(
            "GeoJSON features must be a list"
        )

    rows = []

    for feature_index, feature in enumerate(
        features
    ):
        properties = (
            feature.get(
                "properties"
            )
            or {}
        )

        rows.append(
            {
                "source_feature_index": (
                    feature_index
                ),
                "serial_number": (
                    properties.get(
                        "serialNumber"
                    )
                ),
                "description": (
                    properties.get(
                        "description"
                    )
                ),
                "timestamp": (
                    properties.get(
                        "timestamp"
                    )
                ),
                "latest_fullness": (
                    properties.get(
                        "latestFullness"
                    )
                ),
                "fullness_threshold": (
                    properties.get(
                        "fullnessThreshold"
                    )
                ),
                "reason": (
                    properties.get(
                        "reason"
                    )
                ),
            }
        )

    frame = pd.DataFrame(
        rows
    )

    frame[
        "timestamp"
    ] = pd.to_datetime(
        frame[
            "timestamp"
        ],
        errors="raise",
    )

    frame[
        "latest_fullness"
    ] = pd.to_numeric(
        frame[
            "latest_fullness"
        ],
        errors="raise",
    )

    if frame[
        "serial_number"
    ].isna().any():
        raise RuntimeError(
            "serial_number contains missing values"
        )

    if frame[
        "timestamp"
    ].isna().any():
        raise RuntimeError(
            "timestamp contains missing values"
        )

    if frame[
        "latest_fullness"
    ].isna().any():
        raise RuntimeError(
            "latest_fullness contains missing values"
        )

    observed_levels = set(
        frame[
            "latest_fullness"
        ].unique()
    )

    expected_levels = {
        0,
        2,
        4,
        6,
        8,
        10,
    }

    if not observed_levels.issubset(
        expected_levels
    ):
        raise RuntimeError(
            "Unexpected source fullness levels: "
            f"{sorted(observed_levels)}"
        )

    return frame


def resolve_duplicates(
    frame: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    dict,
]:
    """
    Resolve duplicate serial/date observations conservatively.

    Rules
    -----
    1. Same serial/date + same fullness:
       collapse deterministically to the earliest source feature.

    2. Same serial/date + conflicting fullness:
       exclude the entire serial/date key from empirical calibration.

    No averaging, maximum selection, minimum selection, or arbitrary
    first-value selection is permitted for conflicting measurements.
    """

    key_columns = [
        "serial_number",
        "timestamp",
    ]

    duplicate_mask = (
        frame.duplicated(
            subset=key_columns,
            keep=False,
        )
    )

    duplicate_rows = (
        frame.loc[
            duplicate_mask
        ]
        .copy()
    )

    duplicate_key_count = int(
        duplicate_rows[
            key_columns
        ]
        .drop_duplicates()
        .shape[0]
    )

    conflicting_key_set = set()
    conflict_records = []

    for key, group in (
        duplicate_rows.groupby(
            key_columns,
            sort=True,
        )
    ):
        fullness_values = tuple(
            sorted(
                {
                    float(value)
                    for value
                    in group[
                        "latest_fullness"
                    ].dropna()
                }
            )
        )

        if len(
            fullness_values
        ) > 1:
            serial_number = key[0]
            timestamp = pd.Timestamp(
                key[1]
            )

            conflicting_key_set.add(
                (
                    serial_number,
                    timestamp,
                )
            )

            conflict_records.append(
                {
                    "serial_number": (
                        serial_number
                    ),
                    "timestamp": (
                        timestamp
                    ),
                    "fullness_values": (
                        "|".join(
                            str(value)
                            for value
                            in fullness_values
                        )
                    ),
                    "row_count": (
                        int(
                            len(
                                group
                            )
                        )
                    ),
                }
            )

    conflict_mask = pd.Series(
        [
            (
                serial_number,
                pd.Timestamp(
                    timestamp
                ),
            )
            in conflicting_key_set
            for serial_number, timestamp
            in zip(
                frame[
                    "serial_number"
                ],
                frame[
                    "timestamp"
                ],
            )
        ],
        index=frame.index,
        dtype=bool,
    )

    conflicting_rows = (
        frame.loc[
            conflict_mask
        ]
        .copy()
        .sort_values(
            [
                "serial_number",
                "timestamp",
                "source_feature_index",
            ]
        )
    )

    conflict_output = Path(
        "data/processed/workload/"
        "wyndham_conflicting_duplicate_rows.csv"
    )

    conflict_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    conflicting_rows.to_csv(
        conflict_output,
        index=False,
        lineterminator="\n",
        date_format="%Y-%m-%d",
    )

    # Remove every row belonging to a conflicting serial/date key.
    nonconflicting = (
        frame.loc[
            ~conflict_mask
        ]
        .copy()
    )

    before_same_value_collapse = (
        len(
            nonconflicting
        )
    )

    deduplicated = (
        nonconflicting
        .sort_values(
            [
                "serial_number",
                "timestamp",
                "source_feature_index",
            ]
        )
        .drop_duplicates(
            subset=key_columns,
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    same_value_duplicate_rows_collapsed = (
        before_same_value_collapse
        - len(
            deduplicated
        )
    )

    total_rows_removed = (
        len(
            frame
        )
        - len(
            deduplicated
        )
    )

    duplicate_statistics = {
        "duplicate_key_count_total": (
            duplicate_key_count
        ),
        "conflicting_duplicate_key_count": (
            len(
                conflicting_key_set
            )
        ),
        "conflicting_duplicate_rows_excluded": (
            int(
                len(
                    conflicting_rows
                )
            )
        ),
        "same_value_duplicate_rows_collapsed": (
            int(
                same_value_duplicate_rows_collapsed
            )
        ),
        "total_rows_removed_before_transition_building": (
            int(
                total_rows_removed
            )
        ),
        "deduplicated_record_count": (
            int(
                len(
                    deduplicated
                )
            )
        ),
        "conflict_audit_file": (
            str(
                conflict_output
            )
        ),
        "resolution_rule": (
            "collapse identical-fullness duplicate serial/date keys; "
            "exclude entire serial/date key when duplicate fullness "
            "values conflict"
        ),
        "conflicting_keys": [
            {
                "serial_number": (
                    int(
                        record[
                            "serial_number"
                        ]
                    )
                ),
                "timestamp": (
                    pd.Timestamp(
                        record[
                            "timestamp"
                        ]
                    ).strftime(
                        "%Y-%m-%d"
                    )
                ),
                "fullness_values": (
                    record[
                        "fullness_values"
                    ]
                ),
                "row_count": (
                    record[
                        "row_count"
                    ]
                ),
            }
            for record in conflict_records
        ],
    }

    return (
        deduplicated,
        duplicate_statistics,
    )


def build_profiles(
    frame: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    dict,
]:
    transition_frames = []

    for serial_number, group in (
        frame.groupby(
            "serial_number",
            sort=True,
        )
    ):
        group = (
            group.sort_values(
                "timestamp"
            )
            .reset_index(
                drop=True
            )
        )

        if len(
            group
        ) < 2:
            continue

        current = group.iloc[
            1:
        ].reset_index(
            drop=True
        )

        previous = group.iloc[
            :-1
        ].reset_index(
            drop=True
        )

        dt_days = (
            current[
                "timestamp"
            ]
            - previous[
                "timestamp"
            ]
        ).dt.total_seconds() / 86400.0

        previous_level = (
            previous[
                "latest_fullness"
            ].astype(
                float
            )
        )

        current_level = (
            current[
                "latest_fullness"
            ].astype(
                float
            )
        )

        delta_level = (
            current_level
            - previous_level
        )

        transition = pd.DataFrame(
            {
                "serial_number": (
                    serial_number
                ),
                "start_date": (
                    previous[
                        "timestamp"
                    ]
                ),
                "end_date": (
                    current[
                        "timestamp"
                    ]
                ),
                "dt_days": (
                    dt_days
                ),
                "source_start_level": (
                    previous_level
                ),
                "source_end_level": (
                    current_level
                ),
                "delta_source_level": (
                    delta_level
                ),
            }
        )

        transition_frames.append(
            transition
        )

    if not transition_frames:
        raise RuntimeError(
            "No temporal transitions generated"
        )

    transitions = pd.concat(
        transition_frames,
        ignore_index=True,
    )

    daily = transitions[
        np.isclose(
            transitions[
                "dt_days"
            ].to_numpy(
                dtype=float
            ),
            1.0,
            rtol=0.0,
            atol=1e-12,
        )
    ].copy()

    negative_count = int(
        (
            daily[
                "delta_source_level"
            ]
            < 0.0
        ).sum()
    )

    zero_count = int(
        (
            daily[
                "delta_source_level"
            ]
            == 0.0
        ).sum()
    )

    positive_count = int(
        (
            daily[
                "delta_source_level"
            ]
            > 0.0
        ).sum()
    )

    # --------------------------------------------------------
    # Benchmark rule:
    #
    # Source fullness level:
    #   0,2,4,6,8,10
    #
    # Benchmark-normalized fill:
    #   0,20,40,60,80,100 percent
    #
    # Negative daily deltas are interpreted as collection/reset
    # candidates and are excluded from accumulation calibration.
    #
    # Zero-growth periods are intentionally preserved.
    # --------------------------------------------------------

    profiles = daily[
        daily[
            "delta_source_level"
        ]
        >= 0.0
    ].copy()

    profiles[
        "initial_fill_percent"
    ] = (
        profiles[
            "source_start_level"
        ]
        * 10.0
    )

    profiles[
        "end_fill_percent"
    ] = (
        profiles[
            "source_end_level"
        ]
        * 10.0
    )

    profiles[
        "delta_fill_percent"
    ] = (
        profiles[
            "delta_source_level"
        ]
        * 10.0
    )

    profiles[
        "fill_rate_percent_per_hour"
    ] = (
        profiles[
            "delta_fill_percent"
        ]
        / 24.0
    )

    profiles = (
        profiles[
            [
                "serial_number",
                "start_date",
                "end_date",
                "source_start_level",
                "source_end_level",
                "initial_fill_percent",
                "end_fill_percent",
                "delta_fill_percent",
                "fill_rate_percent_per_hour",
            ]
        ]
        .sort_values(
            [
                "serial_number",
                "start_date",
                "end_date",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    profiles.insert(
        0,
        "profile_id",
        range(
            len(
                profiles
            )
        ),
    )

    statistics = {
        "daily_transition_count": (
            int(
                len(
                    daily
                )
            )
        ),
        "negative_collection_reset_count": (
            negative_count
        ),
        "zero_growth_count": (
            zero_count
        ),
        "positive_growth_count": (
            positive_count
        ),
        "retained_profile_count": (
            int(
                len(
                    profiles
                )
            )
        ),
        "zero_growth_fraction_retained": (
            float(
                (
                    profiles[
                        "delta_fill_percent"
                    ]
                    == 0.0
                ).mean()
            )
        ),
        "positive_growth_fraction_retained": (
            float(
                (
                    profiles[
                        "delta_fill_percent"
                    ]
                    > 0.0
                ).mean()
            )
        ),
        "initial_fill_percent_mean": (
            float(
                profiles[
                    "initial_fill_percent"
                ].mean()
            )
        ),
        "initial_fill_percent_median": (
            float(
                profiles[
                    "initial_fill_percent"
                ].median()
            )
        ),
        "fill_rate_percent_per_hour_mean": (
            float(
                profiles[
                    "fill_rate_percent_per_hour"
                ].mean()
            )
        ),
        "fill_rate_percent_per_hour_median": (
            float(
                profiles[
                    "fill_rate_percent_per_hour"
                ].median()
            )
        ),
        "fill_rate_percent_per_hour_max": (
            float(
                profiles[
                    "fill_rate_percent_per_hour"
                ].max()
            )
        ),
    }

    return (
        profiles,
        statistics,
    )


def write_profile_csv(
    profiles: pd.DataFrame,
) -> str:
    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    profiles.to_csv(
        OUTPUT_CSV,
        index=False,
        lineterminator="\n",
        float_format="%.12g",
        date_format="%Y-%m-%d",
    )

    return sha256_file(
        OUTPUT_CSV
    )


def write_manifest(
    *,
    frame: pd.DataFrame,
    deduplicated: pd.DataFrame,
    duplicate_statistics: dict,
    profiles: pd.DataFrame,
    profile_sha256: str,
    statistics: dict,
) -> str:
    manifest = {
        "schema_version": (
            PROFILE_SCHEMA_VERSION
        ),
        "source": {
            "dataset": (
                SOURCE_DATASET
            ),
            "url": (
                SOURCE_URL
            ),
            "raw_file": (
                str(
                    RAW_PATH
                )
            ),
            "raw_sha256": (
                EXPECTED_RAW_SHA256
            ),
            "raw_feature_count": (
                int(
                    len(
                        frame
                    )
                )
            ),
            "unique_serial_number_count": (
                int(
                    frame[
                        "serial_number"
                    ].nunique()
                )
            ),
            "date_min": (
                frame[
                    "timestamp"
                ]
                .min()
                .strftime(
                    "%Y-%m-%d"
                )
            ),
            "date_max": (
                frame[
                    "timestamp"
                ]
                .max()
                .strftime(
                    "%Y-%m-%d"
                )
            ),
        },
        "normalization": {
            "source_fullness_levels": [
                0,
                2,
                4,
                6,
                8,
                10,
            ],
            "fill_percent_rule": (
                "fill_percent = "
                "10 * source_fullness_level"
            ),
            "temporal_resolution": (
                "consecutive calendar-day intervals only"
            ),
            "rate_rule": (
                "fill_rate_percent_per_hour = "
                "(end_fill_percent - initial_fill_percent) / 24"
            ),
            "negative_delta_rule": (
                "exclude as collection/reset candidate"
            ),
            "zero_delta_rule": (
                "retain as empirical zero-growth profile"
            ),
            "profile_pairing_rule": (
                "initial fill and fill rate remain paired "
                "from the same empirical daily interval"
            ),
            "conflicting_duplicate_rule": (
                "exclude entire conflicting serial/date observation "
                "rather than selecting or averaging fullness values"
            ),
        },
        "duplicate_resolution": (
            duplicate_statistics
        ),
        "profile_library": {
            "file": (
                str(
                    OUTPUT_CSV
                )
            ),
            "sha256": (
                profile_sha256
            ),
            **statistics,
        },
    }

    payload = (
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    ).encode(
        "utf-8"
    )

    OUTPUT_MANIFEST.write_bytes(
        payload
    )

    return sha256_bytes(
        payload
    )


def main() -> None:
    frame = load_raw()

    (
        deduplicated,
        duplicate_statistics,
    ) = resolve_duplicates(
        frame
    )

    (
        profiles,
        statistics,
    ) = build_profiles(
        deduplicated
    )

    profile_sha256 = (
        write_profile_csv(
            profiles
        )
    )

    manifest_sha256 = (
        write_manifest(
            frame=frame,
            deduplicated=deduplicated,
            duplicate_statistics=(
                duplicate_statistics
            ),
            profiles=profiles,
            profile_sha256=(
                profile_sha256
            ),
            statistics=statistics,
        )
    )

    print(
        "=== WYNDHAM EMPIRICAL PROFILE LIBRARY ==="
    )

    print(
        "raw_records:",
        len(
            frame
        ),
    )

    print(
        "unique_serial_numbers:",
        frame[
            "serial_number"
        ].nunique(),
    )

    print()

    print(
        "duplicate_key_count_total:",
        duplicate_statistics[
            "duplicate_key_count_total"
        ],
    )

    print(
        "conflicting_duplicate_key_count:",
        duplicate_statistics[
            "conflicting_duplicate_key_count"
        ],
    )

    print(
        "conflicting_duplicate_rows_excluded:",
        duplicate_statistics[
            "conflicting_duplicate_rows_excluded"
        ],
    )

    print(
        "same_value_duplicate_rows_collapsed:",
        duplicate_statistics[
            "same_value_duplicate_rows_collapsed"
        ],
    )

    print(
        "total_rows_removed:",
        duplicate_statistics[
            "total_rows_removed_before_transition_building"
        ],
    )

    print(
        "deduplicated_records:",
        len(
            deduplicated
        ),
    )

    print()

    print(
        "conflicting_keys:"
    )

    for record in duplicate_statistics[
        "conflicting_keys"
    ]:
        print(
            " ",
            record,
        )

    print()

    for key, value in (
        statistics.items()
    ):
        print(
            f"{key}:",
            value,
        )

    print()

    print(
        "profile_csv:",
        OUTPUT_CSV,
    )

    print(
        "profile_sha256:",
        profile_sha256,
    )

    print(
        "manifest:",
        OUTPUT_MANIFEST,
    )

    print(
        "manifest_sha256:",
        manifest_sha256,
    )


if __name__ == "__main__":
    main()
