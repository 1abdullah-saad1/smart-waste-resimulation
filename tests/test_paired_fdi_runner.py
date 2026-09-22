import networkx as nx
import pytest

from smart_waste.attacks.fdi import (
    generate_fdi_attack_from_truth,
)
from smart_waste.experiments.fdi_paired import (
    FDIPairedExperimentError,
    run_hdr_fdi_condition,
    run_paired_hdr_fdi_experiment,
)
from smart_waste.experiments.scenario_snapshot import (
    build_simulation_state,
    capture_physical_scenario,
)
from smart_waste.models.bin import WasteBin
from smart_waste.models.depot import Depot
from smart_waste.models.truck import Truck
from smart_waste.movement.road_graph import RoadGraph
from smart_waste.simulation.state import SimulationState


MASTER_SEED = 20261001


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
            x=float(index + 1),
            y=0.0,
        )

        graph.add_edge(
            "depot",
            node,
            length_km=float(
                index + 1
            ),
        )

        bins[bin_id] = WasteBin(
            bin_id=bin_id,
            road_node=node,
            fill_percent=fill,
            fill_rate_percent_per_hour=0.0,
            full_mass_kg=440.0,
        )

    trucks = {
        0: Truck(
            truck_id=0,
            current_node="depot",
            capacity_tonnes=10.0,
            speed_km_per_hour=30.0,
            fuel_capacity_litres=200.0,
            fuel_efficiency_km_per_litre=2.5,
        )
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


def test_external_fdi_pair_uses_same_physical_scenario_and_attack() -> None:
    snapshot = capture_physical_scenario(
        make_state(
            fills={
                101: 20.0,
            }
        )
    )

    result = (
        run_paired_hdr_fdi_experiment(
            snapshot,
            master_seed=MASTER_SEED,
            attack_rate=1.0,
            attack_type=(
                "external_unauthenticated"
            ),
        )
    )

    assert (
        result.unprotected.scenario_sha256
        == snapshot.sha256
    )

    assert (
        result.poa_verified.scenario_sha256
        == snapshot.sha256
    )

    assert (
        result.unprotected.attacked_bin_ids
        == result.poa_verified.attacked_bin_ids
        == (
            101,
        )
    )


def test_external_fdi_is_serviced_only_on_unprotected_path() -> None:
    snapshot = capture_physical_scenario(
        make_state(
            fills={
                101: 20.0,
            }
        )
    )

    result = (
        run_paired_hdr_fdi_experiment(
            snapshot,
            master_seed=MASTER_SEED,
            attack_rate=1.0,
            attack_type=(
                "external_unauthenticated"
            ),
        )
    )

    assert (
        result.unprotected.accepted_forged_bin_ids
        == (
            101,
        )
    )

    assert (
        result.poa_verified.accepted_forged_bin_ids
        == ()
    )

    assert (
        result.unprotected.serviced_bin_ids
        == (
            101,
        )
    )

    assert (
        result.poa_verified.serviced_bin_ids
        == ()
    )

    assert (
        result.unprotected.serviced_false_alert_bin_ids
        == (
            101,
        )
    )

    assert (
        result.poa_verified.serviced_false_alert_bin_ids
        == ()
    )


def test_authenticated_compromise_produces_identical_security_paths() -> None:
    snapshot = capture_physical_scenario(
        make_state(
            fills={
                101: 20.0,
                407: 95.0,
            }
        )
    )

    result = (
        run_paired_hdr_fdi_experiment(
            snapshot,
            master_seed=MASTER_SEED,
            attack_rate=1.0,
            attack_type=(
                "authenticated_compromise"
            ),
        )
    )

    assert (
        result.unprotected.accepted_forged_bin_ids
        == result.poa_verified.accepted_forged_bin_ids
        == (
            101,
            407,
        )
    )

    assert (
        result.unprotected.completed_services
        == result.poa_verified.completed_services
    )

    assert (
        result.unprotected.total_distance_km
        == pytest.approx(
            result.poa_verified.total_distance_km
        )
    )

    assert (
        result.unprotected.total_fuel_used_litres
        == pytest.approx(
            result.poa_verified.total_fuel_used_litres
        )
    )

    assert (
        result.unprotected.completion_time_hours
        == pytest.approx(
            result.poa_verified.completion_time_hours
        )
    )


def test_zero_attack_produces_identical_paired_runs() -> None:
    snapshot = capture_physical_scenario(
        make_state(
            fills={
                101: 20.0,
                407: 95.0,
            }
        )
    )

    result = (
        run_paired_hdr_fdi_experiment(
            snapshot,
            master_seed=MASTER_SEED,
            attack_rate=0.0,
            attack_type=(
                "external_unauthenticated"
            ),
        )
    )

    assert result.attacked_bin_ids == ()

    assert (
        result.unprotected.completed_services
        == result.poa_verified.completed_services
    )

    assert (
        result.unprotected.total_distance_km
        == pytest.approx(
            result.poa_verified.total_distance_km
        )
    )

    assert (
        result.unprotected.total_fuel_used_litres
        == pytest.approx(
            result.poa_verified.total_fuel_used_litres
        )
    )


def test_run_metrics_match_physical_truck_distance_and_fuel_relation() -> None:
    snapshot = capture_physical_scenario(
        make_state(
            fills={
                101: 90.0,
            }
        )
    )

    result = (
        run_paired_hdr_fdi_experiment(
            snapshot,
            master_seed=MASTER_SEED,
            attack_rate=0.0,
            attack_type=(
                "external_unauthenticated"
            ),
        )
    )

    run = result.unprotected

    assert run.total_distance_km > 0.0

    assert (
        run.total_fuel_used_litres
        == pytest.approx(
            run.total_distance_km
            / 2.5
        )
    )

    assert len(
        run.truck_metrics
    ) == 1

    truck = run.truck_metrics[
        0
    ]

    assert truck.final_node == "depot"
    assert truck.final_status == "finished"


def test_paired_runner_does_not_mutate_snapshot_initial_conditions() -> None:
    snapshot = capture_physical_scenario(
        make_state(
            fills={
                101: 20.0,
            }
        )
    )

    original_hash = (
        snapshot.sha256
    )

    run_paired_hdr_fdi_experiment(
        snapshot,
        master_seed=MASTER_SEED,
        attack_rate=1.0,
        attack_type=(
            "external_unauthenticated"
        ),
    )

    assert snapshot.sha256 == original_hash

    rebuilt = build_simulation_state(
        snapshot
    )

    assert (
        rebuilt.bins[
            101
        ].fill_percent
        == pytest.approx(
            20.0
        )
    )


def test_run_rejects_attack_scenario_from_different_physical_truth() -> None:
    snapshot = capture_physical_scenario(
        make_state(
            fills={
                101: 20.0,
            }
        )
    )

    stale_scenario = (
        generate_fdi_attack_from_truth(
            {
                101: 30.0,
            },
            master_seed=MASTER_SEED,
            attack_rate=1.0,
            attack_type=(
                "external_unauthenticated"
            ),
        )
    )

    with pytest.raises(
        FDIPairedExperimentError,
        match="true-fill snapshot mismatch",
    ):
        run_hdr_fdi_condition(
            snapshot,
            stale_scenario,
            path="unprotected",
        )


def test_run_rejects_event_limit_before_terminal_completion() -> None:
    snapshot = capture_physical_scenario(
        make_state(
            fills={
                101: 90.0,
            }
        )
    )

    scenario = (
        generate_fdi_attack_from_truth(
            {
                101: 90.0,
            },
            master_seed=MASTER_SEED,
            attack_rate=0.0,
            attack_type=(
                "external_unauthenticated"
            ),
        )
    )

    with pytest.raises(
        FDIPairedExperimentError,
        match="did not terminate",
    ):
        run_hdr_fdi_condition(
            snapshot,
            scenario,
            path="unprotected",
            max_events=1,
        )
