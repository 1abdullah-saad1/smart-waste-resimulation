from math import pi, sqrt

import pytest

from smart_waste.movement.topologies.radial_concentric import (
    RADIAL_CONCENTRIC_TOPOLOGY_ID,
    RadialConcentricTopologySpec,
    generate_radial_concentric_topology,
)


@pytest.fixture(scope="module")
def topology():
    return (
        generate_radial_concentric_topology()
    )


def test_primary_radial_concentric_structure(
    topology,
) -> None:
    assert (
        topology.validation.topology_id
        == RADIAL_CONCENTRIC_TOPOLOGY_ID
    )

    assert (
        topology.validation.node_count
        == 1201
    )

    assert (
        topology.validation.edge_count
        == 2400
    )

    assert topology.road_graph.is_connected


def test_primary_radial_concentric_spec(
    topology,
) -> None:
    spec = topology.spec

    assert spec.ring_count == 20
    assert spec.spoke_count == 60

    assert (
        spec.radius_km
        == pytest.approx(
            sqrt(
                50.0 / pi
            )
        )
    )

    assert (
        spec.radius_km
        == pytest.approx(
            3.989422804014327
        )
    )


def test_primary_radial_concentric_length_and_density(
    topology,
) -> None:
    assert (
        topology.validation.total_road_length_km
        == pytest.approx(
            502.5613370771147
        )
    )

    assert (
        topology.validation.road_density_km_per_km2
        == pytest.approx(
            10.051226741542294
        )
    )

    assert (
        9.5
        <= topology.validation.road_density_km_per_km2
        <= 10.5
    )


def test_analytic_road_length_components(
    topology,
) -> None:
    spec = topology.spec

    assert (
        spec.expected_spoke_length_km
        == pytest.approx(
            239.36536824085962
        )
    )

    assert (
        spec.expected_ring_length_km
        == pytest.approx(
            263.1959688362551
        )
    )

    assert (
        spec.expected_total_road_length_km
        == pytest.approx(
            topology.validation.total_road_length_km
        )
    )


def test_radial_candidate_pool_is_all_noncentral_intersections(
    topology,
) -> None:
    graph = (
        topology.road_graph.graph
    )

    candidates = tuple(
        node_id
        for node_id, attributes
        in graph.nodes(
            data=True
        )
        if attributes.get(
            "radial_bin_candidate",
            False,
        )
    )

    assert len(
        candidates
    ) == 1200

    assert 0 not in candidates

    assert all(
        graph.nodes[
            node_id
        ][
            "ring_index"
        ] >= 1
        for node_id in candidates
    )


def test_center_has_one_connection_per_spoke(
    topology,
) -> None:
    graph = (
        topology.road_graph.graph
    )

    assert graph.degree[
        0
    ] == 60

    assert (
        graph.nodes[
            0
        ][
            "node_role"
        ]
        == "center"
    )


def test_expected_ring_degree_profile(
    topology,
) -> None:
    graph = (
        topology.road_graph.graph
    )

    for node_id, attributes in graph.nodes(
        data=True
    ):
        ring_index = (
            attributes[
                "ring_index"
            ]
        )

        if ring_index == 0:
            continue

        if ring_index < 20:
            assert (
                graph.degree[
                    node_id
                ]
                == 4
            )

        else:
            assert (
                graph.degree[
                    node_id
                ]
                == 3
            )


def test_edge_lengths_follow_radial_and_arc_physics(
    topology,
) -> None:
    graph = (
        topology.road_graph.graph
    )

    spec = topology.spec

    for _, _, attributes in graph.edges(
        data=True
    ):
        if (
            attributes[
                "street_class"
            ]
            == "spoke"
        ):
            assert (
                attributes[
                    "length_km"
                ]
                == pytest.approx(
                    spec.radial_spacing_km
                )
            )

        elif (
            attributes[
                "street_class"
            ]
            == "ring"
        ):
            ring_index = (
                attributes[
                    "ring_index"
                ]
            )

            radius = (
                ring_index
                * spec.radial_spacing_km
            )

            expected_arc = (
                radius
                * spec.angular_spacing_radians
            )

            assert (
                attributes[
                    "length_km"
                ]
                == pytest.approx(
                    expected_arc
                )
            )

        else:
            pytest.fail(
                "unknown Radial-Concentric street class"
            )


def test_radial_concentric_generation_is_deterministic() -> None:
    first = (
        generate_radial_concentric_topology()
    )

    second = (
        generate_radial_concentric_topology()
    )

    assert (
        first.road_graph.graph.graph
        == second.road_graph.graph.graph
    )

    assert (
        list(
            first.road_graph.graph.nodes(
                data=True
            )
        )
        == list(
            second.road_graph.graph.nodes(
                data=True
            )
        )
    )

    first_edges = sorted(
        (
            min(source, target),
            max(source, target),
            tuple(
                sorted(
                    attributes.items()
                )
            ),
        )
        for source, target, attributes
        in first.road_graph.graph.edges(
            data=True
        )
    )

    second_edges = sorted(
        (
            min(source, target),
            max(source, target),
            tuple(
                sorted(
                    attributes.items()
                )
            ),
        )
        for source, target, attributes
        in second.road_graph.graph.edges(
            data=True
        )
    )

    assert first_edges == second_edges


def test_radial_concentric_metadata_has_no_policy_identity(
    topology,
) -> None:
    metadata = (
        topology.road_graph.graph.graph
    )

    assert (
        metadata[
            "topology_id"
        ]
        == "radial_concentric"
    )

    assert (
        metadata[
            "topology_index"
        ]
        == 4
    )

    assert (
        metadata[
            "city_shape"
        ]
        == "disk"
    )

    assert (
        metadata[
            "ring_edge_metric"
        ]
        == "circular_arc_length"
    )

    forbidden = {
        "hdr",
        "tsr",
        "fdi",
        "poa",
        "reward",
        "route_performance",
        "fuel_performance",
    }

    assert not (
        forbidden
        & {
            str(key).lower()
            for key in metadata
        }
    )
