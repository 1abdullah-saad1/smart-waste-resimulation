from __future__ import annotations

import json

import pytest

from smart_waste.experiments.hazard_overlay import (
    CHALLENGE_ORDER,
    HAZARD_SEVERITY,
    build_research_hazard_overlay,
    build_research_hazard_overlays,
    write_research_hazard_overlays,
)
from smart_waste.experiments.research_scenario_matrix import (
    FROZEN_TOPOLOGY_ORDER,
    build_research_scenario_matrix,
)


def test_overlay_count_and_challenge_coverage():
    overlays = (
        build_research_hazard_overlays()
    )

    assert len(
        overlays
    ) == 40

    for replicate_id in range(
        10
    ):
        selected = [
            overlay
            for overlay in overlays
            if (
                overlay.workload_replicate_id
                == replicate_id
            )
        ]

        assert tuple(
            overlay.challenge_id
            for overlay in selected
        ) == CHALLENGE_ORDER


def test_overlay_generation_is_deterministic():
    first = (
        build_research_hazard_overlays()
    )

    second = (
        build_research_hazard_overlays()
    )

    assert tuple(
        overlay.sha256
        for overlay in first
    ) == tuple(
        overlay.sha256
        for overlay in second
    )

    assert tuple(
        overlay.hazard_scenario_id
        for overlay in first
    ) == tuple(
        overlay.hazard_scenario_id
        for overlay in second
    )


def test_challenge_assignment_contract():
    for replicate_id in range(
        10
    ):
        h0 = (
            build_research_hazard_overlay(
                replicate_id=replicate_id,
                challenge_id="H0",
            )
        )

        h1 = (
            build_research_hazard_overlay(
                replicate_id=replicate_id,
                challenge_id="H1",
            )
        )

        h2 = (
            build_research_hazard_overlay(
                replicate_id=replicate_id,
                challenge_id="H2",
            )
        )

        h3 = (
            build_research_hazard_overlay(
                replicate_id=replicate_id,
                challenge_id="H3",
            )
        )

        assert h0.assignments == ()

        assert len(
            h1.assignments
        ) == 1

        assert (
            h1.assignments[
                0
            ].hazard_type
            == "predicted_combustion"
        )

        assert len(
            h2.assignments
        ) == 1

        assert (
            h2.assignments[
                0
            ].hazard_type
            == "emergency_fire"
        )

        assert len(
            h3.assignments
        ) == 3

        assert {
            assignment.hazard_type
            for assignment in h3.assignments
        } == {
            "emergency_fire",
            "predicted_combustion",
            "excess_moisture",
        }

        assert len(
            {
                assignment.bin_id
                for assignment in h3.assignments
            }
        ) == 3


def test_mixed_overlay_reuses_single_hazard_bins():
    for replicate_id in range(
        10
    ):
        h1 = (
            build_research_hazard_overlay(
                replicate_id=replicate_id,
                challenge_id="H1",
            )
        )

        h2 = (
            build_research_hazard_overlay(
                replicate_id=replicate_id,
                challenge_id="H2",
            )
        )

        h3 = (
            build_research_hazard_overlay(
                replicate_id=replicate_id,
                challenge_id="H3",
            )
        )

        predictive_h3 = next(
            assignment
            for assignment
            in h3.assignments
            if (
                assignment.hazard_type
                == "predicted_combustion"
            )
        )

        emergency_h3 = next(
            assignment
            for assignment
            in h3.assignments
            if (
                assignment.hazard_type
                == "emergency_fire"
            )
        )

        assert (
            h1.assignments[
                0
            ].bin_id
            == predictive_h3.bin_id
        )

        assert (
            h2.assignments[
                0
            ].bin_id
            == emergency_h3.bin_id
        )


def test_routing_severity_mapping_matches_frozen_encoding():
    overlay = (
        build_research_hazard_overlay(
            replicate_id=0,
            challenge_id="H3",
        )
    )

    severity = (
        overlay.routing_severity_by_bin
    )

    assert len(
        severity
    ) == 3

    for assignment in (
        overlay.assignments
    ):
        assert severity[
            assignment.bin_id
        ] == pytest.approx(
            HAZARD_SEVERITY[
                assignment.hazard_type
            ]
        )


def test_overlay_identity_is_independent_of_topology():
    cases = (
        build_research_scenario_matrix()
    )

    for replicate_id in range(
        10
    ):
        replicate_cases = [
            case
            for case in cases
            if (
                case.replicate_id
                == replicate_id
            )
        ]

        assert tuple(
            case.topology_id
            for case in replicate_cases
        ) == FROZEN_TOPOLOGY_ORDER

        workload_ids = {
            case.assembly.workload_id
            for case in replicate_cases
        }

        workload_shas = {
            case.assembly.workload_sha256
            for case in replicate_cases
        }

        assert len(
            workload_ids
        ) == 1

        assert len(
            workload_shas
        ) == 1

        for challenge_id in (
            CHALLENGE_ORDER
        ):
            overlay = (
                build_research_hazard_overlay(
                    replicate_id=replicate_id,
                    challenge_id=challenge_id,
                )
            )

            assert (
                overlay.workload_id
                in workload_ids
            )

            assert (
                overlay.workload_sha256
                in workload_shas
            )


def test_manifest_has_160_cross_topology_pairings(
    tmp_path,
):
    manifest = (
        write_research_hazard_overlays(
            output_dir=(
                tmp_path
                / "hazard"
            )
        )
    )

    assert (
        manifest[
            "overlay_count"
        ]
        == 40
    )

    assert (
        manifest[
            "composite_pairing_count"
        ]
        == 160
    )

    pairings = manifest[
        "composite_pairings"
    ]

    for replicate_id in range(
        10
    ):
        for challenge_id in (
            CHALLENGE_ORDER
        ):
            selected = [
                row
                for row in pairings
                if (
                    row[
                        "replicate_id"
                    ]
                    == replicate_id
                    and row[
                        "challenge_id"
                    ]
                    == challenge_id
                )
            ]

            assert len(
                selected
            ) == 4

            assert tuple(
                row[
                    "topology_id"
                ]
                for row in selected
            ) == FROZEN_TOPOLOGY_ORDER

            assert len(
                {
                    row[
                        "hazard_scenario_id"
                    ]
                    for row in selected
                }
            ) == 1

            assert len(
                {
                    row[
                        "hazard_overlay_sha256"
                    ]
                    for row in selected
                }
            ) == 1


def test_written_overlay_files_match_manifest_hashes(
    tmp_path,
):
    output_dir = (
        tmp_path
        / "hazard"
    )

    manifest = (
        write_research_hazard_overlays(
            output_dir=output_dir
        )
    )

    manifest_path = (
        output_dir
        / "research_hazard_overlays_manifest.json"
    )

    assert (
        manifest_path.is_file()
    )

    loaded_manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        loaded_manifest[
            "manifest_sha256"
        ]
        == manifest[
            "manifest_sha256"
        ]
    )

    for record in manifest[
        "overlays"
    ]:
        file_path = (
            output_dir
            / (
                f"rep-"
                f"{record['workload_replicate_id']:02d}"
                f"__"
                f"{record['challenge_id'].lower()}"
                f".json"
            )
        )

        assert (
            file_path.is_file()
        )

        payload = json.loads(
            file_path.read_text(
                encoding="utf-8"
            )
        )

        assert (
            payload[
                "hazard_scenario_id"
            ]
            == record[
                "hazard_scenario_id"
            ]
        )

        assert (
            payload[
                "overlay_sha256"
            ]
            == record[
                "overlay_sha256"
            ]
        )
