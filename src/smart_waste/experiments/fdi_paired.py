from __future__ import annotations

from dataclasses import dataclass
from math import isclose

from smart_waste.attacks.fdi import (
    AttackSelectionMode,
    AttackType,
    FDIAttackScenario,
    generate_fdi_attack_from_truth,
)
from smart_waste.collection.hdr import (
    HDRCollectionPolicy,
)
from smart_waste.experiments.fdi_routing import (
    FDIRoutingPath,
    routing_telemetry_overrides,
)
from smart_waste.experiments.scenario_snapshot import (
    NodeId,
    PhysicalScenarioSnapshot,
    build_simulation_state,
)
from smart_waste.models.truck import (
    TruckStatus,
)
from smart_waste.simulation.engine import (
    SimulationEngine,
)
from smart_waste.simulation.orchestrator import (
    SimulationOrchestrator,
)


class FDIPairedExperimentError(RuntimeError):
    """
    Raised when a paired FDI routing experiment violates its
    reproducibility or terminal-completion contract.
    """


@dataclass(frozen=True)
class TruckRunMetrics:
    truck_id: int

    distance_km: float
    fuel_used_litres: float
    refuelled_litres: float

    depot_returns: int
    capacity_returns: int
    fuel_returns: int
    combined_returns: int

    final_node: NodeId
    final_status: str


@dataclass(frozen=True)
class FDIRoutingRunResult:
    """
    One completed routing condition from one physical snapshot.
    """

    scenario_id: str
    scenario_sha256: str

    security_path: FDIRoutingPath

    attack_type: AttackType
    attack_rate: float
    replicate_id: int
    selection_seed: int

    attacked_bin_ids: tuple[int, ...]
    accepted_forged_bin_ids: tuple[int, ...]

    hdr_eligible_bin_ids: tuple[int, ...]

    completed_services: tuple[
        tuple[int, int],
        ...,
    ]

    serviced_bin_ids: tuple[int, ...]

    initial_false_service_alert_ids: tuple[int, ...]
    serviced_false_alert_bin_ids: tuple[int, ...]

    completion_time_hours: float
    processed_events: int

    total_distance_km: float
    total_fuel_used_litres: float
    total_refuelled_litres: float

    total_depot_returns: int
    total_capacity_returns: int
    total_fuel_returns: int
    total_combined_returns: int

    truck_metrics: tuple[
        TruckRunMetrics,
        ...,
    ]


@dataclass(frozen=True)
class PairedFDIRoutingResult:
    """
    Unprotected and PoA-verified runs sharing one physical scenario
    and one deterministic FDI attack realization.
    """

    scenario_id: str
    scenario_sha256: str

    attack_type: AttackType
    attack_rate: float
    replicate_id: int
    selection_seed: int

    attacked_bin_ids: tuple[int, ...]

    unprotected: FDIRoutingRunResult
    poa_verified: FDIRoutingRunResult


def _snapshot_truth(
    snapshot: PhysicalScenarioSnapshot,
) -> dict[int, float]:
    return {
        bin_.bin_id: bin_.fill_percent
        for bin_ in snapshot.bins
    }


def _validate_attack_matches_snapshot(
    *,
    snapshot: PhysicalScenarioSnapshot,
    scenario: FDIAttackScenario,
) -> None:
    truth = _snapshot_truth(
        snapshot
    )

    if (
        scenario.population_size
        != len(truth)
    ):
        raise FDIPairedExperimentError(
            "FDI scenario population size does not match "
            "physical scenario snapshot"
        )

    attacked_ids = set(
        scenario.attacked_bin_ids
    )

    unknown = (
        attacked_ids
        - set(truth)
    )

    if unknown:
        raise FDIPairedExperimentError(
            "FDI scenario references bins absent from physical "
            f"snapshot: {sorted(unknown)}"
        )

    for event in scenario.events:
        expected = truth[
            event.bin_id
        ]

        if not isclose(
            event.true_fill_percent,
            expected,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise FDIPairedExperimentError(
                f"FDI true-fill snapshot mismatch for bin "
                f"{event.bin_id}: attack={event.true_fill_percent}, "
                f"physical={expected}"
            )


def _extract_truck_metrics(
    orchestrator: SimulationOrchestrator,
) -> tuple[
    TruckRunMetrics,
    ...,
]:
    state = orchestrator.state

    return tuple(
        TruckRunMetrics(
            truck_id=truck.truck_id,
            distance_km=float(
                truck.cumulative_distance_km
            ),
            fuel_used_litres=float(
                truck.cumulative_fuel_used_litres
            ),
            refuelled_litres=float(
                truck.cumulative_refuelled_litres
            ),
            depot_returns=(
                truck.depot_returns
            ),
            capacity_returns=(
                truck.capacity_returns
            ),
            fuel_returns=(
                truck.fuel_returns
            ),
            combined_returns=(
                truck.combined_returns
            ),
            final_node=(
                truck.current_node
            ),
            final_status=(
                truck.status.value
            ),
        )
        for truck in sorted(
            state.trucks.values(),
            key=lambda item: item.truck_id,
        )
    )


def _validate_terminal_run(
    orchestrator: SimulationOrchestrator,
    *,
    max_events: int,
) -> None:
    if not orchestrator.engine.event_queue.is_empty:
        raise FDIPairedExperimentError(
            "simulation did not terminate before max_events="
            f"{max_events}"
        )

    if len(
        orchestrator.reservation_book
    ) != 0:
        raise FDIPairedExperimentError(
            "completed FDI run retains bin reservations"
        )

    completion_view = (
        orchestrator._build_policy_view()
    )

    if not orchestrator.policy.is_complete(
        completion_view
    ):
        raise FDIPairedExperimentError(
            "simulation event queue emptied before policy completion"
        )

    for truck in orchestrator.state.trucks.values():
        if truck.status != TruckStatus.FINISHED:
            raise FDIPairedExperimentError(
                f"truck {truck.truck_id} did not reach FINISHED state"
            )

        if (
            truck.current_node
            != orchestrator.state.depot.road_node
        ):
            raise FDIPairedExperimentError(
                f"truck {truck.truck_id} did not finish at depot"
            )

        if not isclose(
            truck.current_load_tonnes,
            0.0,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise FDIPairedExperimentError(
                f"truck {truck.truck_id} retained physical load"
            )

        if not isclose(
            truck.fuel_remaining_litres,
            truck.fuel_capacity_litres,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise FDIPairedExperimentError(
                f"truck {truck.truck_id} did not finish fully refuelled"
            )

    orchestrator.state.validate_physical_invariants()


def run_hdr_fdi_condition(
    snapshot: PhysicalScenarioSnapshot,
    scenario: FDIAttackScenario,
    *,
    path: FDIRoutingPath,
    threshold_percent: float = 80.0,
    service_time_seconds: float = 36.0,
    max_events: int = 1_000_000,
) -> FDIRoutingRunResult:
    """
    Run one fresh HDR condition from an immutable physical snapshot.

    No mutable state is reused between calls.
    """

    if max_events <= 0:
        raise ValueError(
            "max_events must be positive"
        )

    _validate_attack_matches_snapshot(
        snapshot=snapshot,
        scenario=scenario,
    )

    state = build_simulation_state(
        snapshot
    )

    overrides = routing_telemetry_overrides(
        scenario,
        path=path,
    )

    policy = HDRCollectionPolicy(
        road_graph=state.road_graph,
        threshold_percent=(
            threshold_percent
        ),
    )

    engine = SimulationEngine(
        state=state
    )

    orchestrator = SimulationOrchestrator(
        state=state,
        engine=engine,
        policy=policy,
        service_time_seconds=(
            service_time_seconds
        ),
        reported_fill_percent=(
            overrides
        ),
    )

    orchestrator.start()

    engine.run(
        max_events=max_events
    )

    _validate_terminal_run(
        orchestrator,
        max_events=max_events,
    )

    truck_metrics = (
        _extract_truck_metrics(
            orchestrator
        )
    )

    completed_services = tuple(
        orchestrator.completed_services
    )

    serviced_bin_ids = tuple(
        bin_id
        for _, bin_id in completed_services
    )

    initial_false_alert_ids = tuple(
        sorted(
            scenario.false_service_alert_ids(
                threshold_percent
            )
        )
    )

    initial_false_alert_set = set(
        initial_false_alert_ids
    )

    serviced_false_alert_bin_ids = tuple(
        sorted(
            {
                bin_id
                for bin_id in serviced_bin_ids
                if bin_id
                in initial_false_alert_set
            }
        )
    )

    return FDIRoutingRunResult(
        scenario_id=(
            snapshot.scenario_id
        ),
        scenario_sha256=(
            snapshot.sha256
        ),
        security_path=path,
        attack_type=(
            scenario.attack_type
        ),
        attack_rate=(
            scenario.attack_rate
        ),
        replicate_id=(
            scenario.replicate_id
        ),
        selection_seed=(
            scenario.selection_seed
        ),
        attacked_bin_ids=(
            scenario.attacked_bin_ids
        ),
        accepted_forged_bin_ids=tuple(
            sorted(
                overrides
            )
        ),
        hdr_eligible_bin_ids=(
            policy.eligibility_snapshot.eligible_bin_ids
        ),
        completed_services=(
            completed_services
        ),
        serviced_bin_ids=(
            serviced_bin_ids
        ),
        initial_false_service_alert_ids=(
            initial_false_alert_ids
        ),
        serviced_false_alert_bin_ids=(
            serviced_false_alert_bin_ids
        ),
        completion_time_hours=float(
            state.current_time_hours
        ),
        processed_events=(
            engine.processed_events
        ),
        total_distance_km=sum(
            row.distance_km
            for row in truck_metrics
        ),
        total_fuel_used_litres=sum(
            row.fuel_used_litres
            for row in truck_metrics
        ),
        total_refuelled_litres=sum(
            row.refuelled_litres
            for row in truck_metrics
        ),
        total_depot_returns=sum(
            row.depot_returns
            for row in truck_metrics
        ),
        total_capacity_returns=sum(
            row.capacity_returns
            for row in truck_metrics
        ),
        total_fuel_returns=sum(
            row.fuel_returns
            for row in truck_metrics
        ),
        total_combined_returns=sum(
            row.combined_returns
            for row in truck_metrics
        ),
        truck_metrics=(
            truck_metrics
        ),
    )


def run_paired_hdr_fdi_experiment(
    snapshot: PhysicalScenarioSnapshot,
    *,
    master_seed: int,
    attack_rate: float,
    attack_type: AttackType,
    replicate_id: int = 0,
    selection_mode: AttackSelectionMode = "paired_nested",
    forged_fill_percent: float = 100.0,
    threshold_percent: float = 80.0,
    service_time_seconds: float = 36.0,
    max_events: int = 1_000_000,
) -> PairedFDIRoutingResult:
    """
    Run a paired unprotected-versus-PoA comparison.

    Both conditions receive:
    - the exact same immutable physical scenario,
    - the exact same FDI attack realization,
    - independent freshly reconstructed mutable states.

    The only experimental difference is which forged telemetry
    survives the selected security path.
    """

    truth = _snapshot_truth(
        snapshot
    )

    scenario = generate_fdi_attack_from_truth(
        truth,
        master_seed=master_seed,
        attack_rate=attack_rate,
        attack_type=attack_type,
        selection_mode=selection_mode,
        forged_fill_percent=(
            forged_fill_percent
        ),
        replicate_id=replicate_id,
    )

    unprotected = run_hdr_fdi_condition(
        snapshot,
        scenario,
        path="unprotected",
        threshold_percent=(
            threshold_percent
        ),
        service_time_seconds=(
            service_time_seconds
        ),
        max_events=max_events,
    )

    poa_verified = run_hdr_fdi_condition(
        snapshot,
        scenario,
        path="poa_verified",
        threshold_percent=(
            threshold_percent
        ),
        service_time_seconds=(
            service_time_seconds
        ),
        max_events=max_events,
    )

    if (
        unprotected.scenario_sha256
        != poa_verified.scenario_sha256
        or unprotected.scenario_sha256
        != snapshot.sha256
    ):
        raise FDIPairedExperimentError(
            "paired conditions do not share one physical scenario"
        )

    if (
        unprotected.attacked_bin_ids
        != poa_verified.attacked_bin_ids
    ):
        raise FDIPairedExperimentError(
            "paired conditions do not share one attack realization"
        )

    return PairedFDIRoutingResult(
        scenario_id=(
            snapshot.scenario_id
        ),
        scenario_sha256=(
            snapshot.sha256
        ),
        attack_type=attack_type,
        attack_rate=(
            scenario.attack_rate
        ),
        replicate_id=(
            scenario.replicate_id
        ),
        selection_seed=(
            scenario.selection_seed
        ),
        attacked_bin_ids=(
            scenario.attacked_bin_ids
        ),
        unprotected=unprotected,
        poa_verified=poa_verified,
    )
