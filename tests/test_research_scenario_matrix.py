import pytest

from smart_waste.experiments.research_scenario_matrix import (
    EXPECTED_GRAPH_SCALE,
    FROZEN_TOPOLOGY_ORDER,
    build_research_scenario_matrix,
)


@pytest.fixture(scope="module")
def matrix():
    return (
        build_research_scenario_matrix()
    )


def test_research_matrix_has_exactly_forty_cases(
    matrix,
) -> None:
    assert len(
        matrix
    ) == 40


def test_research_matrix_order_is_replicate_major(
    matrix,
) -> None:
    expected = tuple(
        (
            replicate_id,
            topology_id,
        )
        for replicate_id in range(
            10
        )
        for topology_id in (
            FROZEN_TOPOLOGY_ORDER
        )
    )

    actual = tuple(
        (
            case.replicate_id,
            case.topology_id,
        )
        for case in matrix
    )

    assert actual == expected


def test_each_replicate_uses_one_exact_workload(
    matrix,
) -> None:
    for replicate_id in range(
        10
    ):
        cases = tuple(
            case
            for case in matrix
            if case.replicate_id
            == replicate_id
        )

        assert len(
            cases
        ) == 4

        assert len(
            {
                case.workload.sha256
                for case in cases
            }
        ) == 1

        assert len(
            {
                case.workload.workload_id
                for case in cases
            }
        ) == 1


def test_every_physical_scenario_identity_is_unique(
    matrix,
) -> None:
    assert len(
        {
            case.assembly.scenario_sha256
            for case in matrix
        }
    ) == 40

    assert len(
        {
            case.assembly.scenario_id
            for case in matrix
        }
    ) == 40


def test_all_cases_preserve_primary_scale(
    matrix,
) -> None:
    for case in matrix:
        snapshot = (
            case.assembly.snapshot
        )

        expected_nodes, expected_edges = (
            EXPECTED_GRAPH_SCALE[
                case.topology_id
            ]
        )

        assert len(
            snapshot.road_nodes
        ) == expected_nodes

        assert len(
            snapshot.road_edges
        ) == expected_edges

        assert len(
            snapshot.bins
        ) == 1000

        assert len(
            snapshot.trucks
        ) == 10

        assert (
            case.assembly.workload_sha256
            == case.workload.sha256
        )

        assert (
            case.assembly.workload_replicate_id
            == case.replicate_id
        )


def test_per_bin_empirical_workload_is_identical_across_topologies(
    matrix,
) -> None:
    for replicate_id in range(
        10
    ):
        cases = tuple(
            case
            for case in matrix
            if case.replicate_id
            == replicate_id
        )

        reference = (
            cases[
                0
            ].assembly.snapshot.bins
        )

        for case in cases[
            1:
        ]:
            candidate = (
                case.assembly.snapshot.bins
            )

            for bin_id in range(
                1000
            ):
                assert (
                    candidate[
                        bin_id
                    ].fill_percent
                    == reference[
                        bin_id
                    ].fill_percent
                )

                assert (
                    candidate[
                        bin_id
                    ].fill_rate_percent_per_hour
                    == reference[
                        bin_id
                    ].fill_rate_percent_per_hour
                )
