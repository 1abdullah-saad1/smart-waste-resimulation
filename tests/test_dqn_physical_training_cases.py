from __future__ import annotations

from smart_waste.experiments.dqn_physical_training_cases import (
    build_dqn_physical_training_cases,
    build_physical_case_manifest,
    cases_for_topology,
)
from smart_waste.experiments.research_scenario_matrix import (
    FROZEN_TOPOLOGY_ORDER,
)


def test_physical_training_case_count():
    cases = (
        build_dqn_physical_training_cases()
    )

    assert len(
        cases
    ) == 40


def test_each_topology_has_eight_training_and_two_validation_cases():
    for topology_id in (
        FROZEN_TOPOLOGY_ORDER
    ):
        training = (
            cases_for_topology(
                topology_id,
                split="training",
            )
        )

        validation = (
            cases_for_topology(
                topology_id,
                split="validation",
            )
        )

        assert len(
            training
        ) == 8

        assert len(
            validation
        ) == 2

        assert tuple(
            case.slot_id
            for case in training
        ) == tuple(
            range(
                8
            )
        )

        assert tuple(
            case.slot_id
            for case in validation
        ) == (
            0,
            1,
        )


def test_physical_training_cases_are_deterministic():
    first = (
        build_dqn_physical_training_cases()
    )

    second = (
        build_dqn_physical_training_cases()
    )

    assert tuple(
        case.snapshot.sha256
        for case in first
    ) == tuple(
        case.snapshot.sha256
        for case in second
    )

    assert tuple(
        case.case_sha256
        for case in first
    ) == tuple(
        case.case_sha256
        for case in second
    )


def test_training_snapshots_preserve_1000_bin_action_space():
    cases = (
        build_dqn_physical_training_cases()
    )

    for case in cases:
        snapshot = case.snapshot

        assert len(
            snapshot.bins
        ) == 1000


def test_physical_case_manifest_is_deterministic():
    first = (
        build_physical_case_manifest()
    )

    second = (
        build_physical_case_manifest()
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
            "cases"
        ]
        == second[
            "cases"
        ]
    )


def test_training_hazard_maps_are_available_for_all_challenges():
    case = (
        cases_for_topology(
            "manhattan",
            split="training",
        )[
            0
        ]
    )

    assert (
        case.hazard_severity_by_bin(
            "H0"
        )
        == {}
    )

    assert len(
        case.hazard_severity_by_bin(
            "H1"
        )
    ) == 1

    assert len(
        case.hazard_severity_by_bin(
            "H2"
        )
    ) == 1

    assert len(
        case.hazard_severity_by_bin(
            "H3"
        )
    ) == 3
