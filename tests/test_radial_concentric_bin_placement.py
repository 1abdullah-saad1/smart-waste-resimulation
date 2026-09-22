from collections import Counter

import pytest

from smart_waste.movement.topologies.radial_concentric import (
    RadialConcentricTopologySpec,
)
from smart_waste.movement.topologies.radial_concentric_layout import (
    PRIMARY_RADIAL_BIN_COUNT,
    allocate_weighted_ring_quotas,
    build_primary_radial_concentric_layout,
    radial_ring_weight,
)


EXPECTED_QUOTAS = (
    60,
    60,
    60,
    60,
    60,
    59,
    57,
    55,
    54,
    52,
    50,
    48,
    47,
    45,
    43,
    41,
    40,
    38,
    36,
    35,
)


@pytest.fixture(scope="module")
def primary_layout():
    return (
        build_primary_radial_concentric_layout()
    )


def test_primary_radial_layout_scale(
    primary_layout,
) -> None:
    placement = (
        primary_layout.bin_placement
    )

    assert (
        primary_layout.depot_node
        == 0
    )

    assert len(
        primary_layout.candidate_nodes
    ) == 1200

    assert (
        placement.bin_count
        == PRIMARY_RADIAL_BIN_COUNT
        == 1000
    )

    assert len(
        set(
            placement.road_nodes
        )
    ) == 1000

    assert (
        primary_layout.depot_node
        not in placement.road_nodes
    )


def test_radial_weight_contract() -> None:
    assert (
        radial_ring_weight(
            ring_index=1,
            ring_count=20,
        )
        == pytest.approx(
            1.95
        )
    )

    assert (
        radial_ring_weight(
            ring_index=10,
            ring_count=20,
        )
        == pytest.approx(
            1.5
        )
    )

    assert (
        radial_ring_weight(
            ring_index=20,
            ring_count=20,
        )
        == pytest.approx(
            1.0
        )
    )


def test_exact_primary_weighted_ring_quotas() -> None:
    quotas = (
        allocate_weighted_ring_quotas(
            spec=(
                RadialConcentricTopologySpec()
            ),
            bin_count=1000,
        )
    )

    assert quotas == EXPECTED_QUOTAS

    assert sum(
        quotas
    ) == 1000


def test_ring_quotas_are_nonincreasing(
    primary_layout,
) -> None:
    quotas = (
        primary_layout
        .bin_placement
        .ring_quotas
    )

    assert quotas == EXPECTED_QUOTAS

    assert all(
        first >= second
        for first, second in zip(
            quotas,
            quotas[1:],
        )
    )


def test_actual_selected_count_per_ring_matches_quota(
    primary_layout,
) -> None:
    graph = (
        primary_layout
        .topology
        .road_graph
        .graph
    )

    counts = Counter(
        int(
            graph.nodes[
                node_id
            ][
                "ring_index"
            ]
        )
        for node_id in (
            primary_layout
            .bin_placement
            .road_nodes
        )
    )

    actual = tuple(
        counts[
            ring_index
        ]
        for ring_index in range(
            1,
            21,
        )
    )

    assert actual == EXPECTED_QUOTAS


def test_all_radial_bins_use_candidate_nodes(
    primary_layout,
) -> None:
    candidates = set(
        primary_layout.candidate_nodes
    )

    assert all(
        node_id in candidates
        for node_id in (
            primary_layout
            .bin_placement
            .road_nodes
        )
    )


def test_radial_bin_ids_are_canonical_and_contiguous(
    primary_layout,
) -> None:
    assignments = (
        primary_layout
        .bin_placement
        .assignments
    )

    assert tuple(
        row.bin_id
        for row in assignments
    ) == tuple(
        range(
            1000
        )
    )

    road_nodes = tuple(
        row.road_node
        for row in assignments
    )

    assert road_nodes == tuple(
        sorted(
            road_nodes
        )
    )


def test_angular_selection_remains_distributed(
    primary_layout,
) -> None:
    graph = (
        primary_layout
        .topology
        .road_graph
        .graph
    )

    for ring_index in range(
        1,
        21,
    ):
        spokes = sorted(
            int(
                graph.nodes[
                    node_id
                ][
                    "spoke_index"
                ]
            )
            for node_id in (
                primary_layout
                .bin_placement
                .road_nodes
            )
            if int(
                graph.nodes[
                    node_id
                ][
                    "ring_index"
                ]
            ) == ring_index
        )

        assert spokes

        gaps = [
            (
                spokes[
                    (
                        index + 1
                    )
                    % len(
                        spokes
                    )
                ]
                - spokes[
                    index
                ]
            )
            % 60
            for index in range(
                len(
                    spokes
                )
            )
        ]

        # For the frozen 35..60 quotas, deterministic circular
        # farthest-point selection must not leave a large empty arc.
        assert max(
            gaps
        ) <= 3


def test_radial_layout_is_deterministic(
    primary_layout,
) -> None:
    second = (
        build_primary_radial_concentric_layout()
    )

    assert (
        second.depot_node
        == primary_layout.depot_node
    )

    assert (
        second.candidate_nodes
        == primary_layout.candidate_nodes
    )

    assert (
        second.bin_placement
        == primary_layout.bin_placement
    )
