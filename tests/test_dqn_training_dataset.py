from __future__ import annotations

import json
from pathlib import Path

from smart_waste.experiments.dqn_training_dataset import (
    NUM_BINS,
    TRAINING_WORKLOAD_COUNT,
    VALIDATION_WORKLOAD_COUNT,
    build_dqn_training_dataset_manifest,
    workload_from_record,
    write_dqn_training_dataset_manifest,
)
from smart_waste.experiments.hazard_overlay import (
    CHALLENGE_ORDER,
)
from smart_waste.experiments.research_scenario_matrix import (
    DEFAULT_WORKLOAD_MANIFEST,
)


def frozen_confirmatory_profile_ids():
    manifest = json.loads(
        Path(
            DEFAULT_WORKLOAD_MANIFEST
        ).read_text(
            encoding="utf-8"
        )
    )

    result = set()

    for record in manifest[
        "workloads"
    ]:
        payload = json.loads(
            Path(
                record[
                    "file"
                ]
            ).read_text(
                encoding="utf-8"
            )
        )

        result.update(
            int(value)
            for value
            in payload[
                "profile_ids"
            ]
        )

    return result


def test_dataset_has_eight_training_and_two_validation_workloads():
    manifest = (
        build_dqn_training_dataset_manifest()
    )

    training = [
        record
        for record
        in manifest[
            "workloads"
        ]
        if (
            record[
                "split"
            ]
            == "training"
        )
    ]

    validation = [
        record
        for record
        in manifest[
            "workloads"
        ]
        if (
            record[
                "split"
            ]
            == "validation"
        )
    ]

    assert len(
        training
    ) == TRAINING_WORKLOAD_COUNT

    assert len(
        validation
    ) == VALIDATION_WORKLOAD_COUNT


def test_every_independent_workload_contains_1000_unique_profiles():
    manifest = (
        build_dqn_training_dataset_manifest()
    )

    for record in manifest[
        "workloads"
    ]:
        profile_ids = record[
            "profile_ids"
        ]

        assert len(
            profile_ids
        ) == NUM_BINS

        assert len(
            set(
                profile_ids
            )
        ) == NUM_BINS


def test_training_and_validation_are_pairwise_disjoint():
    manifest = (
        build_dqn_training_dataset_manifest()
    )

    profile_sets = [
        set(
            record[
                "profile_ids"
            ]
        )
        for record
        in manifest[
            "workloads"
        ]
    ]

    for first_index in range(
        len(
            profile_sets
        )
    ):
        for second_index in range(
            first_index + 1,
            len(
                profile_sets
            ),
        ):
            assert (
                profile_sets[
                    first_index
                ].isdisjoint(
                    profile_sets[
                        second_index
                    ]
                )
            )


def test_independent_dataset_excludes_all_confirmatory_profiles():
    manifest = (
        build_dqn_training_dataset_manifest()
    )

    confirmatory = (
        frozen_confirmatory_profile_ids()
    )

    independent = set()

    for record in manifest[
        "workloads"
    ]:
        independent.update(
            record[
                "profile_ids"
            ]
        )

    assert independent.isdisjoint(
        confirmatory
    )


def test_manifest_generation_is_bitwise_deterministic_logically():
    first = (
        build_dqn_training_dataset_manifest()
    )

    second = (
        build_dqn_training_dataset_manifest()
    )

    assert (
        first[
            "manifest_sha256"
        ]
        == second[
            "manifest_sha256"
        ]
    )

    assert (
        first[
            "workloads"
        ]
        == second[
            "workloads"
        ]
    )


def test_training_hazard_contract_is_balanced_and_distinct():
    manifest = (
        build_dqn_training_dataset_manifest()
    )

    for record in manifest[
        "workloads"
    ]:
        hazards = record[
            "hazard_challenges"
        ]

        assert tuple(
            hazards
        ) == CHALLENGE_ORDER

        assert hazards[
            "H0"
        ] == []

        assert len(
            hazards[
                "H1"
            ]
        ) == 1

        assert len(
            hazards[
                "H2"
            ]
        ) == 1

        assert len(
            hazards[
                "H3"
            ]
        ) == 3

        h1_bin = (
            hazards[
                "H1"
            ][
                0
            ][
                "bin_id"
            ]
        )

        h2_bin = (
            hazards[
                "H2"
            ][
                0
            ][
                "bin_id"
            ]
        )

        mixed = {
            item[
                "hazard_type"
            ]: item[
                "bin_id"
            ]
            for item in hazards[
                "H3"
            ]
        }

        assert (
            mixed[
                "predicted_combustion"
            ]
            == h1_bin
        )

        assert (
            mixed[
                "emergency_fire"
            ]
            == h2_bin
        )

        assert len(
            set(
                mixed.values()
            )
        ) == 3


def test_workload_materialization_recovers_empirical_values():
    manifest = (
        build_dqn_training_dataset_manifest()
    )

    record = manifest[
        "workloads"
    ][0]

    workload = workload_from_record(
        record
    )

    assert len(
        workload.profile_ids
    ) == NUM_BINS

    assert len(
        workload.initial_fill_percent
    ) == NUM_BINS

    assert len(
        workload.fill_rate_percent_per_hour
    ) == NUM_BINS

    assert all(
        0.0
        <= value
        <= 100.0
        for value
        in workload.initial_fill_percent
    )

    assert all(
        value >= 0.0
        for value
        in workload.fill_rate_percent_per_hour
    )


def test_manifest_can_be_written_and_reloaded(
    tmp_path,
):
    path = (
        tmp_path
        / "manifest.json"
    )

    manifest = (
        write_dqn_training_dataset_manifest(
            path
        )
    )

    loaded = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        loaded[
            "manifest_sha256"
        ]
        == manifest[
            "manifest_sha256"
        ]
    )
