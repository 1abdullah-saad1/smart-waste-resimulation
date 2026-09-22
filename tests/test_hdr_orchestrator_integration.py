import networkx as nx
import pytest

from smart_waste.collection.base import (
    CollectionPolicy,
    PolicyView,
)
from smart_waste.collection.hdr import (
    HDRCollectionPolicy,
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


SERVICE_SECONDS = 36.0


def make_state(
    *,
    fills: dict[int, float],
    edges: list[
        tuple[str, str, float]
    ] | None = None,
) -> SimulationState:
    graph = nx.Graph()

    graph.add_node(
        "depot",
        x=0.0,
        y=0.0,
    )

    for bin_id in fills:
        node = f"bin-{bin_id}"

        graph.add_node(
            node,
            x=float(bin_id + 1),
            y=0.0,
        )

    if edges is None:
        for bin_id in fills:
            graph.add_edge(
                "depot",
                f"bin-{bin_id}",
                length_km=float(
                    bin_id + 1
                ),
            )
    else:
        for first, second, distance in edges:
            graph.add_edge(
                first,
                second,
                length_km=distance,
            )

    bins = {
        bin_id: WasteBin(
            bin_id=bin_id,
            road_node=f"bin-{bin_id}",
            fill_percent=fill,
            fill_rate_percent_per_hour=0.0,
            full_mass_kg=440.0,
        )
        for bin_id, fill in fills.items()
    }

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


def make_hdr_orchestrator(
    state: SimulationState,
    *,
    reported_fill_percent: dict[int, float] | None = None,
) -> tuple[
    SimulationOrchestrator,
    HDRCollectionPolicy,
]:
    policy = HDRCollectionPolicy(
        road_graph=state.road_graph,
        threshold_percent=80.0,
    )

    engine = SimulationEngine(
        state=state
    )

    orchestrator = SimulationOrchestrator(
        state=state,
        engine=engine,
        policy=policy,
        service_time_seconds=SERVICE_SECONDS,
        reported_fill_percent=(
            reported_fill_percent
        ),
    )

    return (
        orchestrator,
        policy,
    )


def test_spoofed_full_bin_is_dispatched_but_true_mass_is_collected() -> None:
    state = make_state(
        fills={
            0: 20.0,
            1: 95.0,
        }
    )

    orchestrator, policy = (
        make_hdr_orchestrator(
            state,
            reported_fill_percent={
                # False positive:
                # physically only 20%, reported full.
                0: 100.0,

                # False negative:
                # physically 95%, reported below threshold.
                1: 20.0,
            },
        )
    )

    initial_results = (
        orchestrator.start()
    )

    assert (
        policy.eligibility_snapshot.eligible_bin_ids
        == (
            0,
        )
    )

    assert len(initial_results) == 1

    assert (
        initial_results[0].requested_bin_id
        == 0
    )

    # Arrival.
    orchestrator.engine.step()

    # Service completion.
    #
    # The bin is now physically collected. Because HDR has no
    # pending candidates left, terminal return has been scheduled,
    # but unloading has not happened yet.
    orchestrator.engine.step()

    assert orchestrator.completed_services == [
        (
            0,
            0,
        )
    ]

    # Full physical mass = 0.44 t.
    # True fill = 20%.
    # Collected physical mass = 0.44 * 0.20 = 0.088 t.
    assert (
        state.trucks[0].current_load_tonnes
        == pytest.approx(
            0.088
        )
    )

    assert (
        state.bins[0].fill_percent
        == pytest.approx(
            0.0
        )
    )

    # The reported 100% value must never have become physical truth.
    assert (
        state.trucks[0].current_load_tonnes
        != pytest.approx(
            0.44
        )
    )


def test_false_negative_physically_full_bin_remains_uncollected() -> None:
    state = make_state(
        fills={
            0: 20.0,
            1: 95.0,
        }
    )

    orchestrator, _ = (
        make_hdr_orchestrator(
            state,
            reported_fill_percent={
                0: 100.0,
                1: 20.0,
            },
        )
    )

    orchestrator.start()

    orchestrator.engine.run(
        max_events=100
    )

    assert sorted(
        orchestrator.completed_services
    ) == [
        (
            0,
            0,
        )
    ]

    # Bin 1 was physically 95% full but was hidden from the HDR
    # eligibility snapshot by its false reported value.
    assert (
        state.bins[1].fill_percent
        == pytest.approx(
            95.0
        )
    )

    assert (
        state.bins[1].waste_mass_tonnes
        == pytest.approx(
            0.44
            * 0.95
        )
    )


def test_nominal_hdr_integration_uses_true_fill_as_reported_fill() -> None:
    state = make_state(
        fills={
            0: 20.0,
            1: 95.0,
        }
    )

    orchestrator, policy = (
        make_hdr_orchestrator(
            state
        )
    )

    results = orchestrator.start()

    assert (
        policy.eligibility_snapshot.eligible_bin_ids
        == (
            1,
        )
    )

    assert results[0].requested_bin_id == 1

    orchestrator.engine.run(
        max_events=100
    )

    assert orchestrator.completed_services == [
        (
            0,
            1,
        )
    ]

    # Below-threshold physical bin remains untouched.
    assert (
        state.bins[0].fill_percent
        == pytest.approx(
            20.0
        )
    )


def test_hdr_end_to_end_finishes_at_depot_and_releases_reservations() -> None:
    state = make_state(
        fills={
            0: 90.0,
            1: 90.0,
        }
    )

    orchestrator, policy = (
        make_hdr_orchestrator(
            state
        )
    )

    orchestrator.start()

    orchestrator.engine.run(
        max_events=100
    )

    assert (
        policy.completed_bin_ids
        == (
            0,
            1,
        )
    )

    assert len(
        orchestrator.reservation_book
    ) == 0

    truck = state.trucks[
        0
    ]

    assert truck.status == TruckStatus.FINISHED
    assert truck.current_node == "depot"

    assert (
        truck.current_load_tonnes
        == pytest.approx(
            0.0
        )
    )

    assert (
        truck.fuel_remaining_litres
        == pytest.approx(
            truck.fuel_capacity_litres
        )
    )


def test_hdr_reselects_from_new_physical_location_after_service() -> None:
    state = make_state(
        fills={
            0: 90.0,
            1: 90.0,
            2: 90.0,
        },
        edges=[
            (
                "depot",
                "bin-0",
                1.0,
            ),
            (
                "depot",
                "bin-1",
                2.0,
            ),
            (
                "depot",
                "bin-2",
                4.0,
            ),
            (
                "bin-0",
                "bin-1",
                5.0,
            ),
            (
                "bin-0",
                "bin-2",
                1.0,
            ),
            (
                "bin-2",
                "bin-1",
                1.0,
            ),
        ],
    )

    orchestrator, _ = (
        make_hdr_orchestrator(
            state
        )
    )

    initial = orchestrator.start()

    # From depot bin 0 is nearest.
    assert initial[0].requested_bin_id == 0

    # Physical arrival at bin 0.
    orchestrator.engine.step()

    # Successful service at bin 0 triggers a fresh HDR decision.
    orchestrator.engine.step()

    assert (
        orchestrator.completed_services
        == [
            (
                0,
                0,
            )
        ]
    )

    # From the NEW physical position bin-0:
    #
    # bin 2 = 1 km
    # bin 1 = 2 km via bin-2
    #
    # Therefore dynamic HDR must select bin 2 next.
    assert (
        orchestrator.dispatch_history[-1].requested_bin_id
        == 2
    )


class TelemetryProbePolicy(
    CollectionPolicy
):
    """
    Regression probe proving every Orchestrator/Dispatcher
    PolicyView observes the same reported telemetry override.
    """

    def __init__(self) -> None:
        self.observed: list[
            tuple[str, float]
        ] = []

        self.completed = False

    def _record(
        self,
        stage: str,
        view: PolicyView,
    ) -> None:
        self.observed.append(
            (
                stage,
                view.bin_by_id(
                    0
                ).routing_fill_percent,
            )
        )

    def initialize(
        self,
        view: PolicyView,
    ) -> None:
        self._record(
            "initialize",
            view,
        )

    def select_next_bin(
        self,
        *,
        view: PolicyView,
        truck_id: int,
    ) -> int | None:
        self._record(
            "select",
            view,
        )

        if self.completed:
            return None

        return 0

    def on_service_complete(
        self,
        *,
        view: PolicyView,
        truck_id: int,
        bin_id: int,
    ) -> None:
        self._record(
            "service_complete",
            view,
        )

        self.completed = True

    def is_complete(
        self,
        view: PolicyView,
    ) -> bool:
        self._record(
            "is_complete",
            view,
        )

        return self.completed


def test_reported_telemetry_is_consistent_across_full_policy_lifecycle() -> None:
    state = make_state(
        fills={
            0: 20.0,
        }
    )

    policy = TelemetryProbePolicy()

    orchestrator = SimulationOrchestrator(
        state=state,
        engine=SimulationEngine(
            state=state
        ),
        policy=policy,
        service_time_seconds=SERVICE_SECONDS,
        reported_fill_percent={
            0: 100.0,
        },
    )

    orchestrator.start()

    orchestrator.engine.run(
        max_events=100
    )

    assert policy.observed

    assert {
        value
        for _, value in policy.observed
    } == {
        100.0,
    }

    observed_stages = {
        stage
        for stage, _ in policy.observed
    }

    assert "initialize" in observed_stages
    assert "select" in observed_stages
    assert "service_complete" in observed_stages
    assert "is_complete" in observed_stages

    # Despite all policy views observing reported=100%, physical
    # collection still used the true 20% fill.
    assert (
        state.bins[0].fill_percent
        == pytest.approx(
            0.0
        )
    )

    assert (
        orchestrator.completed_services
        == [
            (
                0,
                0,
            )
        ]
    )
