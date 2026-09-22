import networkx as nx
import pytest

from smart_waste.collection.hdr import (
    HDRCollectionPolicy,
)
from smart_waste.experiments.fdi_routing import (
    build_fdi_scenario_for_state,
    routing_telemetry_overrides,
    snapshot_physical_fill_percent,
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
from smart_waste.simulation.engine import (
    SimulationEngine,
)
from smart_waste.simulation.orchestrator import (
    SimulationOrchestrator,
)
from smart_waste.simulation.state import (
    SimulationState,
)


MASTER_SEED = 20261001
SERVICE_SECONDS = 36.0


def make_state(
    *,
    fills: dict[int, float],
) -> SimulationState:
    graph = nx.Graph()

    graph.add_node(
        "depot",
        x=0.0,
        y=0.0,
    )

    bins = {}

    for index, (
        bin_id,
        fill,
    ) in enumerate(
        sorted(
            fills.items()
        )
    ):
        node = f"bin-{bin_id}"

        graph.add_node(
            node,
            x=float(
                index + 1
            ),
            y=0.0,
        )

        graph.add_edge(
            "depot",
            node,
            length_km=float(
                index + 1
            ),
        )

        bins[
            bin_id
        ] = WasteBin(
            bin_id=bin_id,
            road_node=node,
            fill_percent=fill,
            fill_rate_percent_per_hour=0.0,
            full_mass_kg=440.0,
        )

    truck = Truck(
        truck_id=0,
        current_node="depot",
        capacity_tonnes=10.0,
        speed_km_per_hour=30.0,
        fuel_capacity_litres=200.0,
        fuel_efficiency_km_per_litre=2.5,
    )

    return SimulationState(
        road_graph=RoadGraph(
            graph
        ),
        bins=bins,
        trucks={
            0: truck,
        },
        depot=Depot(
            road_node="depot",
            unloading_bays=1,
            unload_time_minutes=11.0,
            refuel_rate_litres_per_minute=60.0,
        ),
    )


def run_hdr(
    state: SimulationState,
    *,
    reported_fill_percent: dict[int, float],
) -> SimulationOrchestrator:
    policy = HDRCollectionPolicy(
        road_graph=state.road_graph,
        threshold_percent=80.0,
    )

    orchestrator = SimulationOrchestrator(
        state=state,
        engine=SimulationEngine(
            state=state
        ),
        policy=policy,
        service_time_seconds=SERVICE_SECONDS,
        reported_fill_percent=(
            reported_fill_percent
        ),
    )

    orchestrator.start()

    orchestrator.engine.run(
        max_events=100
    )

    return orchestrator


def test_truth_snapshot_preserves_actual_noncontiguous_ids() -> None:
    state = make_state(
        fills={
            101: 20.0,
            407: 95.0,
            999: 55.0,
        }
    )

    snapshot = (
        snapshot_physical_fill_percent(
            state
        )
    )

    assert snapshot == {
        101: 20.0,
        407: 95.0,
        999: 55.0,
    }


def test_state_adapter_does_not_mutate_physical_truth() -> None:
    state = make_state(
        fills={
            101: 20.0,
            407: 95.0,
        }
    )

    before = (
        snapshot_physical_fill_percent(
            state
        )
    )

    scenario = (
        build_fdi_scenario_for_state(
            state,
            master_seed=MASTER_SEED,
            attack_rate=1.0,
            attack_type=(
                "external_unauthenticated"
            ),
            replicate_id=3,
        )
    )

    assert scenario.attacked_bin_ids == (
        101,
        407,
    )

    after = (
        snapshot_physical_fill_percent(
            state
        )
    )

    assert after == before


def test_external_attack_raw_and_poa_paths_differ_only_by_verification() -> None:
    state = make_state(
        fills={
            101: 20.0,
            407: 95.0,
        }
    )

    scenario = (
        build_fdi_scenario_for_state(
            state,
            master_seed=MASTER_SEED,
            attack_rate=1.0,
            attack_type=(
                "external_unauthenticated"
            ),
        )
    )

    raw = routing_telemetry_overrides(
        scenario,
        path="unprotected",
    )

    protected = routing_telemetry_overrides(
        scenario,
        path="poa_verified",
    )

    assert raw == {
        101: 100.0,
        407: 100.0,
    }

    assert protected == {}


def test_authenticated_compromise_survives_poa_path() -> None:
    state = make_state(
        fills={
            101: 20.0,
            407: 95.0,
        }
    )

    scenario = (
        build_fdi_scenario_for_state(
            state,
            master_seed=MASTER_SEED,
            attack_rate=1.0,
            attack_type=(
                "authenticated_compromise"
            ),
        )
    )

    raw = routing_telemetry_overrides(
        scenario,
        path="unprotected",
    )

    protected = routing_telemetry_overrides(
        scenario,
        path="poa_verified",
    )

    assert protected == raw

    assert protected == {
        101: 100.0,
        407: 100.0,
    }


def test_security_variants_use_same_state_attack_targets() -> None:
    state = make_state(
        fills={
            101: 20.0,
            205: 30.0,
            407: 40.0,
            611: 50.0,
            999: 60.0,
        }
    )

    external = (
        build_fdi_scenario_for_state(
            state,
            master_seed=MASTER_SEED,
            attack_rate=0.60,
            attack_type=(
                "external_unauthenticated"
            ),
            replicate_id=9,
        )
    )

    authenticated = (
        build_fdi_scenario_for_state(
            state,
            master_seed=MASTER_SEED,
            attack_rate=0.60,
            attack_type=(
                "authenticated_compromise"
            ),
            replicate_id=9,
        )
    )

    assert (
        external.selection_seed
        == authenticated.selection_seed
    )

    assert (
        external.attacked_bin_ids
        == authenticated.attacked_bin_ids
    )


def test_external_fdi_changes_unprotected_hdr_but_poa_preserves_truth() -> None:
    reference = make_state(
        fills={
            101: 20.0,
        }
    )

    scenario = (
        build_fdi_scenario_for_state(
            reference,
            master_seed=MASTER_SEED,
            attack_rate=1.0,
            attack_type=(
                "external_unauthenticated"
            ),
        )
    )

    raw = routing_telemetry_overrides(
        scenario,
        path="unprotected",
    )

    protected = routing_telemetry_overrides(
        scenario,
        path="poa_verified",
    )

    assert raw == {
        101: 100.0,
    }

    assert protected == {}

    # Identical physical initial conditions.
    unprotected_state = make_state(
        fills={
            101: 20.0,
        }
    )

    protected_state = make_state(
        fills={
            101: 20.0,
        }
    )

    unprotected_run = run_hdr(
        unprotected_state,
        reported_fill_percent=raw,
    )

    protected_run = run_hdr(
        protected_state,
        reported_fill_percent=protected,
    )

    # Unprotected routing accepts the forged 100%-full report.
    assert (
        unprotected_run.completed_services
        == [
            (
                0,
                101,
            )
        ]
    )

    # Physical service still collects true 20% mass only.
    assert (
        unprotected_state.bins[
            101
        ].fill_percent
        == pytest.approx(
            0.0
        )
    )

    # PoA rejects the invalid forged telemetry.
    # Physical truth = 20%, below HDR's 80% threshold.
    assert (
        protected_run.completed_services
        == []
    )

    assert (
        protected_state.bins[
            101
        ].fill_percent
        == pytest.approx(
            20.0
        )
    )

    assert (
        protected_state.trucks[
            0
        ].status
        == TruckStatus.FINISHED
    )


def test_authenticated_compromise_reaches_hdr_even_with_poa_verification() -> None:
    reference = make_state(
        fills={
            101: 20.0,
        }
    )

    scenario = (
        build_fdi_scenario_for_state(
            reference,
            master_seed=MASTER_SEED,
            attack_rate=1.0,
            attack_type=(
                "authenticated_compromise"
            ),
        )
    )

    protected = routing_telemetry_overrides(
        scenario,
        path="poa_verified",
    )

    assert protected == {
        101: 100.0,
    }

    state = make_state(
        fills={
            101: 20.0,
        }
    )

    run = run_hdr(
        state,
        reported_fill_percent=protected,
    )

    assert run.completed_services == [
        (
            0,
            101,
        )
    ]

    # Authentication does not magically recover physical truth
    # when a legitimately credentialed node itself is compromised.
    assert (
        state.bins[
            101
        ].fill_percent
        == pytest.approx(
            0.0
        )
    )
