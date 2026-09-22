import networkx as nx
import pytest

from smart_waste.experiments.scenario_snapshot import (
    ScenarioSnapshotError,
    build_simulation_state,
    capture_physical_scenario,
)
from smart_waste.models.bin import WasteBin
from smart_waste.models.depot import Depot
from smart_waste.models.truck import (
    Truck,
    TruckStatus,
)
from smart_waste.movement.road_graph import (
    RoadGraph,
)
from smart_waste.simulation.state import (
    SimulationState,
)


def make_state(
    *,
    reverse_graph_insertion: bool = False,
) -> SimulationState:
    graph = nx.Graph()

    graph.graph["topology"] = "snapshot-test"
    graph.graph["metadata"] = {
        "replicate": 7,
        "tags": [
            "paired",
            "physical",
        ],
    }

    nodes = [
        (
            "depot",
            {
                "x": 0.0,
                "y": 0.0,
                "kind": "depot",
            },
        ),
        (
            "n101",
            {
                "x": 1.0,
                "y": 0.0,
            },
        ),
        (
            "n407",
            {
                "x": 2.0,
                "y": 0.0,
            },
        ),
    ]

    edges = [
        (
            "depot",
            "n101",
            {
                "length_km": 1.25,
                "road_class": "local",
            },
        ),
        (
            "n101",
            "n407",
            {
                "length_km": 2.5,
                "road_class": "collector",
            },
        ),
        (
            "depot",
            "n407",
            {
                "length_km": 4.0,
                "road_class": "arterial",
            },
        ),
    ]

    if reverse_graph_insertion:
        nodes = list(
            reversed(
                nodes
            )
        )

        edges = list(
            reversed(
                edges
            )
        )

    for node_id, attributes in nodes:
        graph.add_node(
            node_id,
            **attributes,
        )

    for source, target, attributes in edges:
        graph.add_edge(
            source,
            target,
            **attributes,
        )

    bins = {
        101: WasteBin(
            bin_id=101,
            road_node="n101",
            fill_percent=20.0,
            fill_rate_percent_per_hour=0.5,
            full_mass_kg=440.0,
            waste_age_hours=12.0,
        ),
        407: WasteBin(
            bin_id=407,
            road_node="n407",
            fill_percent=95.0,
            fill_rate_percent_per_hour=0.25,
            full_mass_kg=440.0,
            waste_age_hours=36.0,
        ),
    }

    trucks = {
        0: Truck(
            truck_id=0,
            current_node="depot",
            capacity_tonnes=10.0,
            speed_km_per_hour=30.0,
            fuel_capacity_litres=200.0,
            fuel_efficiency_km_per_litre=2.5,
        ),
        1: Truck(
            truck_id=1,
            current_node="depot",
            capacity_tonnes=10.0,
            speed_km_per_hour=30.0,
            fuel_capacity_litres=200.0,
            fuel_efficiency_km_per_litre=2.5,
        ),
    }

    return SimulationState(
        road_graph=RoadGraph(
            graph
        ),
        bins=bins,
        trucks=trucks,
        depot=Depot(
            road_node="depot",
            unloading_bays=1,
            unload_time_minutes=11.0,
            refuel_rate_litres_per_minute=60.0,
        ),
    )


def test_snapshot_round_trip_preserves_initial_physics() -> None:
    original = make_state()

    snapshot = capture_physical_scenario(
        original
    )

    rebuilt = build_simulation_state(
        snapshot
    )

    assert rebuilt.current_time_hours == pytest.approx(
        0.0
    )

    assert sorted(
        rebuilt.bins
    ) == [
        101,
        407,
    ]

    assert rebuilt.bins[
        101
    ].fill_percent == pytest.approx(
        20.0
    )

    assert rebuilt.bins[
        407
    ].fill_percent == pytest.approx(
        95.0
    )

    assert rebuilt.bins[
        101
    ].waste_age_hours == pytest.approx(
        12.0
    )

    assert rebuilt.trucks[
        0
    ].fuel_remaining_litres == pytest.approx(
        200.0
    )

    assert rebuilt.depot.unloading_bays == 1

    assert (
        rebuilt.road_graph.total_road_length_km
        == pytest.approx(
            7.75
        )
    )


def test_repeated_reconstruction_returns_independent_mutable_states() -> None:
    snapshot = capture_physical_scenario(
        make_state()
    )

    state_a = build_simulation_state(
        snapshot
    )

    state_b = build_simulation_state(
        snapshot
    )

    state_a.bins[
        101
    ].fill_percent = 0.0

    state_a.trucks[
        0
    ].current_load_tonnes = 3.0

    state_a.road_graph.graph.nodes[
        "n101"
    ][
        "x"
    ] = 999.0

    state_a.road_graph.graph.edges[
        "depot",
        "n101",
    ][
        "length_km"
    ] = 99.0

    assert (
        state_b.bins[
            101
        ].fill_percent
        == pytest.approx(
            20.0
        )
    )

    assert (
        state_b.trucks[
            0
        ].current_load_tonnes
        == pytest.approx(
            0.0
        )
    )

    assert (
        state_b.road_graph.graph.nodes[
            "n101"
        ][
            "x"
        ]
        == pytest.approx(
            1.0
        )
    )

    assert (
        state_b.road_graph.graph.edges[
            "depot",
            "n101",
        ][
            "length_km"
        ]
        == pytest.approx(
            1.25
        )
    )


def test_snapshot_hash_is_independent_of_graph_insertion_order() -> None:
    snapshot_a = capture_physical_scenario(
        make_state(
            reverse_graph_insertion=False
        )
    )

    snapshot_b = capture_physical_scenario(
        make_state(
            reverse_graph_insertion=True
        )
    )

    assert (
        snapshot_a.sha256
        == snapshot_b.sha256
    )

    assert (
        snapshot_a.scenario_id
        == snapshot_b.scenario_id
    )


def test_snapshot_hash_changes_when_physical_fill_changes() -> None:
    state_a = make_state()
    state_b = make_state()

    state_b.bins[
        101
    ].fill_percent = 21.0

    snapshot_a = capture_physical_scenario(
        state_a
    )

    snapshot_b = capture_physical_scenario(
        state_b
    )

    assert (
        snapshot_a.sha256
        != snapshot_b.sha256
    )


def test_snapshot_hash_changes_when_road_length_changes() -> None:
    state_a = make_state()
    state_b = make_state()

    state_b.road_graph.graph.edges[
        "depot",
        "n101",
    ][
        "length_km"
    ] = 1.5

    snapshot_a = capture_physical_scenario(
        state_a
    )

    snapshot_b = capture_physical_scenario(
        state_b
    )

    assert (
        snapshot_a.sha256
        != snapshot_b.sha256
    )


def test_snapshot_preserves_graph_and_node_metadata() -> None:
    snapshot = capture_physical_scenario(
        make_state()
    )

    rebuilt = build_simulation_state(
        snapshot
    )

    assert (
        rebuilt.road_graph.graph.graph[
            "topology"
        ]
        == "snapshot-test"
    )

    assert (
        rebuilt.road_graph.graph.graph[
            "metadata"
        ][
            "replicate"
        ]
        == 7
    )

    assert (
        rebuilt.road_graph.graph.graph[
            "metadata"
        ][
            "tags"
        ]
        == [
            "paired",
            "physical",
        ]
    )

    assert (
        rebuilt.road_graph.graph.nodes[
            "depot"
        ][
            "kind"
        ]
        == "depot"
    )

    assert (
        rebuilt.road_graph.graph.edges[
            "depot",
            "n101",
        ][
            "road_class"
        ]
        == "local"
    )


def test_snapshot_rejects_nonzero_simulation_time() -> None:
    state = make_state()

    state.current_time_hours = 0.25

    with pytest.raises(
        ScenarioSnapshotError,
        match="current_time_hours",
    ):
        capture_physical_scenario(
            state
        )


def test_snapshot_rejects_truck_runtime_state() -> None:
    state = make_state()

    state.trucks[
        0
    ].status = TruckStatus.SERVICING_BIN

    with pytest.raises(
        ScenarioSnapshotError,
        match="initial IDLE",
    ):
        capture_physical_scenario(
            state
        )


def test_snapshot_rejects_accumulated_truck_metrics() -> None:
    state = make_state()

    state.trucks[
        0
    ].cumulative_distance_km = 1.0

    with pytest.raises(
        ScenarioSnapshotError,
        match="cumulative_distance_km",
    ):
        capture_physical_scenario(
            state
        )


def test_snapshot_rejects_prior_bin_collection_history() -> None:
    state = make_state()

    state.bins[
        101
    ].collected_count = 1

    with pytest.raises(
        ScenarioSnapshotError,
        match="prior collection history",
    ):
        capture_physical_scenario(
            state
        )
