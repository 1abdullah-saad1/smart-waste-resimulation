from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite

from smart_waste.collection.dqn import (
    DQNCollectionPolicy,
)
from smart_waste.experiments.scenario_snapshot import (
    PhysicalScenarioSnapshot,
    build_simulation_state,
)
from smart_waste.models.fuel import (
    fuel_required_litres,
)
from smart_waste.models.truck import (
    TruckStatus,
)
from smart_waste.rl.dqn import (
    DQNAgent,
)
from smart_waste.rl.physical_training import (
    PhysicalDQNTrainingPolicy,
    PhysicalRewardWeights,
)
from smart_waste.simulation.engine import (
    SimulationEngine,
)
from smart_waste.simulation.orchestrator import (
    SimulationOrchestrator,
)
from smart_waste.simulation.policy_view import (
    build_policy_view,
)


class PhysicalDQNEpisodeError(RuntimeError):
    """Raised when a physical DQN episode is invalid."""


@dataclass(frozen=True)
class PhysicalDQNEpisodeResult:
    """
    Immutable summary of one physical DQN episode.

    action_distance_km
        Distance attributed to DQN transitions up to transition
        closure/service completion.

    fleet_distance_km
        Complete physical fleet distance, including terminal
        depot returns.

    terminal_overhead_distance_km
        Physical distance not included inside action-transition
        distance. Under the normal successful lifecycle this is
        principally final terminal-return travel.

    fuel_litres
        Total distance-derived fuel consumed by all trucks using
        each truck's physical fuel-efficiency configuration.
    """

    scenario_id: str
    scenario_sha256: str | None

    episode_index: int
    epsilon: float

    processed_events: int
    completed_services: int
    replay_transitions: int

    total_reward: float

    action_distance_km: float

    # Distance actually penalized by the DQN reward, including
    # terminal fleet return on the final transition.
    reward_distance_km: float

    # Terminal fleet-return distance assigned to the final
    # transition reward.
    terminal_return_distance_km: float

    fleet_distance_km: float

    # Independent physical accounting cross-check:
    # fleet distance not contained in pre-terminal action travel.
    terminal_overhead_distance_km: float

    fuel_litres: float

    completion_time_hours: float

    optimization_losses: tuple[
        float,
        ...,
    ]

    policy_complete: bool
    event_queue_empty: bool
    all_trucks_finished: bool
    reservations_remaining: int

    natural_completion: bool


def _validate_epsilon(
    epsilon: float,
) -> float:
    value = float(
        epsilon
    )

    if not (
        isfinite(
            value
        )
        and 0.0
        <= value
        <= 1.0
    ):
        raise PhysicalDQNEpisodeError(
            "epsilon must be finite and in [0, 1]"
        )

    return value


def _total_fleet_distance_km(
    state,
) -> float:
    total = sum(
        float(
            truck.cumulative_distance_km
        )
        for truck in state.trucks.values()
    )

    if (
        not isfinite(
            total
        )
        or total < -1e-10
    ):
        raise PhysicalDQNEpisodeError(
            "invalid total fleet distance"
        )

    return max(
        0.0,
        total,
    )


def _total_fuel_litres(
    state,
) -> float:
    """
    Derive total consumed fuel from the same physical movement
    equation used by the Core.

    This remains exact for the current benchmark because fuel
    consumption is distance-based and no idle-fuel term exists.
    """

    total = 0.0

    for truck in state.trucks.values():
        total += fuel_required_litres(
            distance_km=float(
                truck.cumulative_distance_km
            ),
            efficiency_km_per_litre=float(
                truck.fuel_efficiency_km_per_litre
            ),
        )

    if (
        not isfinite(
            total
        )
        or total < -1e-10
    ):
        raise PhysicalDQNEpisodeError(
            "invalid total fuel consumption"
        )

    return max(
        0.0,
        total,
    )


def run_physical_dqn_episode(
    *,
    snapshot: PhysicalScenarioSnapshot,
    scenario_id: str,
    agent: DQNAgent,
    episode_index: int,
    epsilon: float | None = None,
    scenario_sha256: str | None = None,
    service_time_seconds: float = 36.0,
    reward_weights: (
        PhysicalRewardWeights
        | None
    ) = None,
    verified_by_bin: (
        Mapping[int, bool]
        | None
    ) = None,
    hazard_severity_by_bin: (
        Mapping[int, float]
        | None
    ) = None,
    initial_full_elapsed_hours: (
        Mapping[int, float]
        | None
    ) = None,
    reported_fill_percent: (
        Mapping[int, float]
        | None
    ) = None,
    near_full_threshold_percent: float = 90.0,
    prediction_horizon_hours: float = 4.0,
    sla_limit_hours: float = 6.0,
    optimize_after_transition: bool = True,
    max_events: int = 1_000_000,
) -> PhysicalDQNEpisodeResult:
    """
    Run one complete event-driven DQN episode.

    A fresh mutable SimulationState is reconstructed from the
    immutable PhysicalScenarioSnapshot every time this function is
    called.

    The supplied DQNAgent is NOT recreated. Therefore its:
    - neural-network weights,
    - optimizer state,
    - replay buffer,
    - RNG state,
    - optimization-step counter

    persist across training episodes.

    Physical state does not persist across episodes.
    """

    if not isinstance(
        scenario_id,
        str,
    ) or not scenario_id:
        raise PhysicalDQNEpisodeError(
            "scenario_id must be a non-empty string"
        )

    if episode_index < 0:
        raise PhysicalDQNEpisodeError(
            "episode_index must be non-negative"
        )

    if max_events <= 0:
        raise PhysicalDQNEpisodeError(
            "max_events must be positive"
        )

    if service_time_seconds <= 0.0:
        raise PhysicalDQNEpisodeError(
            "service_time_seconds must be positive"
        )

    if epsilon is None:
        resolved_epsilon = (
            agent.epsilon(
                episode_index
            )
        )
    else:
        resolved_epsilon = (
            _validate_epsilon(
                epsilon
            )
        )

    resolved_epsilon = (
        _validate_epsilon(
            resolved_epsilon
        )
    )

    # Critical isolation rule:
    # every episode gets fresh physical mutable state.
    state = build_simulation_state(
        snapshot
    )

    num_bins = len(
        state.bins
    )

    required_input_dim = (
        2
        + 3 * num_bins
    )

    if (
        agent.input_dim
        != required_input_dim
    ):
        raise PhysicalDQNEpisodeError(
            "DQN input dimension does not match "
            "physical scenario: "
            f"agent={agent.input_dim}, "
            f"required={required_input_dim}"
        )

    if (
        agent.action_dim
        != num_bins
    ):
        raise PhysicalDQNEpisodeError(
            "DQN action dimension does not match "
            "physical scenario: "
            f"agent={agent.action_dim}, "
            f"required={num_bins}"
        )

    dqn_policy = DQNCollectionPolicy(
        road_graph=state.road_graph,
        agent=agent,
        near_full_threshold_percent=(
            near_full_threshold_percent
        ),
        prediction_horizon_hours=(
            prediction_horizon_hours
        ),
        sla_limit_hours=(
            sla_limit_hours
        ),
        epsilon=resolved_epsilon,
        verified_by_bin=(
            verified_by_bin
        ),
        hazard_severity_by_bin=(
            hazard_severity_by_bin
        ),
        initial_full_elapsed_hours=(
            initial_full_elapsed_hours
        ),
    )

    training_policy = (
        PhysicalDQNTrainingPolicy(
            policy=dqn_policy,
            reward_weights=(
                reward_weights
            ),
            optimize_after_transition=(
                optimize_after_transition
            ),
        )
    )

    engine = SimulationEngine(
        state=state
    )

    orchestrator = (
        SimulationOrchestrator(
            state=state,
            engine=engine,
            policy=training_policy,
            service_time_seconds=(
                service_time_seconds
            ),
            reported_fill_percent=(
                reported_fill_percent
            ),
        )
    )

    orchestrator.start()

    processed_events = (
        engine.run(
            max_events=max_events
        )
    )

    final_view = build_policy_view(
        state,
        reservations=(
            orchestrator
            .reservation_book
            .snapshot()
        ),
        reported_fill_percent=(
            reported_fill_percent
        ),
    )

    policy_complete = (
        training_policy.is_complete(
            final_view
        )
    )

    event_queue_empty = (
        engine.event_queue.is_empty
    )

    all_trucks_finished = all(
        truck.status
        == TruckStatus.FINISHED
        for truck in state.trucks.values()
    )

    reservations_remaining = len(
        orchestrator.reservation_book
    )

    natural_completion = (
        policy_complete
        and event_queue_empty
        and all_trucks_finished
        and reservations_remaining == 0
    )

    transitions = (
        training_policy.transitions
    )

    total_reward = sum(
        transition.reward
        for transition in transitions
    )

    action_distance_km = sum(
        transition.distance_km
        for transition in transitions
    )

    reward_distance_km = sum(
        transition.reward_distance_km
        for transition in transitions
    )

    terminal_return_distance_km = sum(
        transition.terminal_return_distance_km
        for transition in transitions
    )

    fleet_distance_km = (
        _total_fleet_distance_km(
            state
        )
    )

    if (
        action_distance_km
        > fleet_distance_km
        + 1e-8
    ):
        raise PhysicalDQNEpisodeError(
            "action-transition distance exceeds "
            "physical fleet distance"
        )

    terminal_overhead_distance_km = max(
        0.0,
        (
            fleet_distance_km
            - action_distance_km
        ),
    )

    if natural_completion:
        if abs(
            reward_distance_km
            - fleet_distance_km
        ) > 1e-8:
            raise PhysicalDQNEpisodeError(
                "reward-attributed distance does not match "
                "completed physical fleet distance"
            )

        if abs(
            terminal_return_distance_km
            - terminal_overhead_distance_km
        ) > 1e-8:
            raise PhysicalDQNEpisodeError(
                "terminal reward distance does not match "
                "physical terminal-return overhead"
            )

    fuel_litres = (
        _total_fuel_litres(
            state
        )
    )

    losses = tuple(
        float(
            transition.optimization_loss
        )
        for transition in transitions
        if (
            transition.optimization_loss
            is not None
        )
    )

    if natural_completion:
        state.validate_physical_invariants()

    return PhysicalDQNEpisodeResult(
        scenario_id=scenario_id,
        scenario_sha256=(
            scenario_sha256
        ),
        episode_index=int(
            episode_index
        ),
        epsilon=float(
            resolved_epsilon
        ),
        processed_events=int(
            processed_events
        ),
        completed_services=len(
            orchestrator.completed_services
        ),
        replay_transitions=len(
            transitions
        ),
        total_reward=float(
            total_reward
        ),
        action_distance_km=float(
            action_distance_km
        ),
        reward_distance_km=float(
            reward_distance_km
        ),
        terminal_return_distance_km=float(
            terminal_return_distance_km
        ),
        fleet_distance_km=float(
            fleet_distance_km
        ),
        terminal_overhead_distance_km=float(
            terminal_overhead_distance_km
        ),
        fuel_litres=float(
            fuel_litres
        ),
        completion_time_hours=float(
            state.current_time_hours
        ),
        optimization_losses=losses,
        policy_complete=bool(
            policy_complete
        ),
        event_queue_empty=bool(
            event_queue_empty
        ),
        all_trucks_finished=bool(
            all_trucks_finished
        ),
        reservations_remaining=int(
            reservations_remaining
        ),
        natural_completion=bool(
            natural_completion
        ),
    )
