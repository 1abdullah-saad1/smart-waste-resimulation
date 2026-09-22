from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Mapping, Sequence

import networkx as nx

from smart_waste.config.loader import (
    load_yaml,
)
from smart_waste.experiments.scenario_snapshot import (
    PhysicalScenarioSnapshot,
    capture_physical_scenario,
)
from smart_waste.experiments.workload import (
    PhysicalWorkloadRealization,
)
from smart_waste.models.bin import WasteBin
from smart_waste.models.depot import Depot
from smart_waste.models.truck import Truck
from smart_waste.movement.bin_placement import (
    BinRoadAssignment,
)
from smart_waste.movement.road_graph import (
    RoadGraph,
)
from smart_waste.simulation.state import (
    SimulationState,
)


NodeId = int | str


class PhysicalScenarioFactoryError(ValueError):
    """Raised when a physical scenario cannot be assembled safely."""


@dataclass(frozen=True)
class PhysicalScenarioSpec:
    """
    Explicit initial-physics specification.

    Run-level policy/timing parameters such as bin service time,
    routing strategy, attack condition and queue scheduling policy
    are deliberately excluded.
    """

    bin_full_mass_kg: float

    truck_count: int
    truck_capacity_tonnes: float
    truck_speed_km_per_hour: float

    fuel_tank_capacity_litres: float
    fuel_efficiency_km_per_litre: float

    depot_unloading_bays: int
    depot_unload_time_minutes: float
    depot_refuel_rate_litres_per_minute: float

    def __post_init__(self) -> None:
        if self.truck_count <= 0:
            raise PhysicalScenarioFactoryError(
                "truck_count must be positive"
            )

        if self.depot_unloading_bays <= 0:
            raise PhysicalScenarioFactoryError(
                "depot_unloading_bays must be positive"
            )

        positive_values = {
            "bin_full_mass_kg": self.bin_full_mass_kg,
            "truck_capacity_tonnes": (
                self.truck_capacity_tonnes
            ),
            "truck_speed_km_per_hour": (
                self.truck_speed_km_per_hour
            ),
            "fuel_tank_capacity_litres": (
                self.fuel_tank_capacity_litres
            ),
            "fuel_efficiency_km_per_litre": (
                self.fuel_efficiency_km_per_litre
            ),
            "depot_refuel_rate_litres_per_minute": (
                self.depot_refuel_rate_litres_per_minute
            ),
        }

        for name, value in positive_values.items():
            if (
                not isfinite(value)
                or value <= 0.0
            ):
                raise PhysicalScenarioFactoryError(
                    f"{name} must be finite and positive"
                )

        if (
            not isfinite(
                self.depot_unload_time_minutes
            )
            or self.depot_unload_time_minutes < 0.0
        ):
            raise PhysicalScenarioFactoryError(
                "depot_unload_time_minutes must be finite "
                "and non-negative"
            )


@dataclass(frozen=True)
class PhysicalScenarioAssembly:
    """
    Immutable link between one physical scenario and the exact
    topology-independent workload used to construct it.
    """

    snapshot: PhysicalScenarioSnapshot

    workload_id: str
    workload_sha256: str
    workload_replicate_id: int

    @property
    def scenario_id(self) -> str:
        return self.snapshot.scenario_id

    @property
    def scenario_sha256(self) -> str:
        return self.snapshot.sha256


def _required_value(
    mapping: Mapping[str, object],
    *,
    key: str,
    source: str,
) -> object:
    if key not in mapping:
        raise PhysicalScenarioFactoryError(
            f"missing required configuration key "
            f"{source}.{key}"
        )

    return mapping[key]


def _required_float(
    mapping: Mapping[str, object],
    *,
    key: str,
    source: str,
) -> float:
    value = _required_value(
        mapping,
        key=key,
        source=source,
    )

    if isinstance(value, bool):
        raise PhysicalScenarioFactoryError(
            f"{source}.{key} must be numeric"
        )

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PhysicalScenarioFactoryError(
            f"{source}.{key} must be numeric"
        ) from exc

    if not isfinite(result):
        raise PhysicalScenarioFactoryError(
            f"{source}.{key} must be finite"
        )

    return result


def _required_int(
    mapping: Mapping[str, object],
    *,
    key: str,
    source: str,
) -> int:
    value = _required_value(
        mapping,
        key=key,
        source=source,
    )

    if isinstance(value, bool):
        raise PhysicalScenarioFactoryError(
            f"{source}.{key} must be an integer"
        )

    if not isinstance(value, int):
        raise PhysicalScenarioFactoryError(
            f"{source}.{key} must be an integer"
        )

    return value


def load_primary_physical_scenario_spec(
    config_root: str | Path = "configs",
) -> PhysicalScenarioSpec:
    """
    Load initial physical parameters from canonical YAML files.

    The loader deliberately does not read:
    - timing.yaml,
    - HDR/TSR configs,
    - FDI configs,
    - workload distribution bounds.
    """

    root = Path(
        config_root
    )

    bins = load_yaml(
        root / "core" / "bins.yaml"
    )

    trucks = load_yaml(
        root / "core" / "trucks.yaml"
    )

    depot = load_yaml(
        root / "core" / "depot.yaml"
    )

    fuel = load_yaml(
        root / "core" / "fuel.yaml"
    )

    location_policy = str(
        _required_value(
            depot,
            key="location_policy",
            source="depot",
        )
    ).strip().lower()

    if location_policy != "central":
        raise PhysicalScenarioFactoryError(
            "primary physical scenario currently requires "
            "central depot location policy"
        )

    return PhysicalScenarioSpec(
        bin_full_mass_kg=_required_float(
            bins,
            key="full_mass_kg",
            source="bins",
        ),
        truck_count=_required_int(
            trucks,
            key="count",
            source="trucks",
        ),
        truck_capacity_tonnes=_required_float(
            trucks,
            key="capacity_tonnes",
            source="trucks",
        ),
        truck_speed_km_per_hour=_required_float(
            trucks,
            key="speed_km_per_hour",
            source="trucks",
        ),
        fuel_tank_capacity_litres=_required_float(
            fuel,
            key="tank_capacity_litres",
            source="fuel",
        ),
        fuel_efficiency_km_per_litre=_required_float(
            fuel,
            key="efficiency_km_per_litre",
            source="fuel",
        ),
        depot_unloading_bays=_required_int(
            depot,
            key="unloading_bays",
            source="depot",
        ),
        depot_unload_time_minutes=_required_float(
            depot,
            key="unload_time_minutes",
            source="depot",
        ),
        depot_refuel_rate_litres_per_minute=(
            _required_float(
                fuel,
                key="refuel_rate_litres_per_minute",
                source="fuel",
            )
        ),
    )


def _clone_road_graph(
    road_graph: RoadGraph,
) -> RoadGraph:
    """
    Construct an independent physical graph.

    This prevents later simulation-state mutation from aliasing the
    topology/layout object used to construct the scenario.
    """

    source = (
        road_graph.graph
    )

    graph = nx.Graph()

    graph.graph.update(
        deepcopy(
            source.graph
        )
    )

    for node_id, attributes in source.nodes(
        data=True
    ):
        graph.add_node(
            node_id,
            **deepcopy(
                attributes
            ),
        )

    for source_node, target_node, attributes in (
        source.edges(
            data=True
        )
    ):
        graph.add_edge(
            source_node,
            target_node,
            **deepcopy(
                attributes
            ),
        )

    return RoadGraph(
        graph=graph,
        length_attribute=(
            road_graph.length_attribute
        ),
    )


def _validate_bin_assignments(
    *,
    road_graph: RoadGraph,
    depot_node: NodeId,
    assignments: Sequence[
        BinRoadAssignment
    ],
    workload: PhysicalWorkloadRealization,
) -> tuple[
    BinRoadAssignment,
    ...,
]:
    ordered = tuple(
        sorted(
            assignments,
            key=lambda row: row.bin_id,
        )
    )

    expected_count = (
        workload.spec.num_bins
    )

    if len(
        ordered
    ) != expected_count:
        raise PhysicalScenarioFactoryError(
            "bin-assignment count does not match workload"
        )

    if (
        len(
            workload.initial_fill_percent
        )
        != expected_count
        or len(
            workload.fill_rate_percent_per_hour
        )
        != expected_count
    ):
        raise PhysicalScenarioFactoryError(
            "workload realization length invariant violated"
        )

    expected_ids = tuple(
        range(
            expected_count
        )
    )

    actual_ids = tuple(
        row.bin_id
        for row in ordered
    )

    if actual_ids != expected_ids:
        raise PhysicalScenarioFactoryError(
            "bin IDs must be canonical contiguous IDs 0..N-1"
        )

    road_nodes = tuple(
        row.road_node
        for row in ordered
    )

    if len(
        set(
            road_nodes
        )
    ) != len(
        road_nodes
    ):
        raise PhysicalScenarioFactoryError(
            "bin assignments must use unique road nodes"
        )

    if depot_node in road_nodes:
        raise PhysicalScenarioFactoryError(
            "depot road node cannot also host a bin"
        )

    try:
        road_graph.validate_service_nodes(
            (
                depot_node,
                *road_nodes,
            )
        )
    except ValueError as exc:
        raise PhysicalScenarioFactoryError(
            str(exc)
        ) from exc

    return ordered


def build_pristine_simulation_state(
    *,
    road_graph: RoadGraph,
    depot_node: NodeId,
    assignments: Sequence[
        BinRoadAssignment
    ],
    workload: PhysicalWorkloadRealization,
    spec: PhysicalScenarioSpec,
) -> SimulationState:
    """
    Build one fresh revised-model SimulationState.

    The returned state is pristine and suitable for immediate
    capture by capture_physical_scenario().
    """

    fresh_road_graph = (
        _clone_road_graph(
            road_graph
        )
    )

    ordered_assignments = (
        _validate_bin_assignments(
            road_graph=fresh_road_graph,
            depot_node=depot_node,
            assignments=assignments,
            workload=workload,
        )
    )

    bins = {
        row.bin_id: WasteBin(
            bin_id=row.bin_id,
            road_node=row.road_node,
            fill_percent=(
                workload.initial_fill_percent[
                    row.bin_id
                ]
            ),
            fill_rate_percent_per_hour=(
                workload.fill_rate_percent_per_hour[
                    row.bin_id
                ]
            ),
            full_mass_kg=(
                spec.bin_full_mass_kg
            ),
            waste_age_hours=0.0,
        )
        for row in ordered_assignments
    }

    trucks = {
        truck_id: Truck(
            truck_id=truck_id,
            current_node=depot_node,
            capacity_tonnes=(
                spec.truck_capacity_tonnes
            ),
            speed_km_per_hour=(
                spec.truck_speed_km_per_hour
            ),
            fuel_capacity_litres=(
                spec.fuel_tank_capacity_litres
            ),
            fuel_efficiency_km_per_litre=(
                spec.fuel_efficiency_km_per_litre
            ),
        )
        for truck_id in range(
            spec.truck_count
        )
    }

    depot = Depot(
        road_node=depot_node,
        unloading_bays=(
            spec.depot_unloading_bays
        ),
        unload_time_minutes=(
            spec.depot_unload_time_minutes
        ),
        refuel_rate_litres_per_minute=(
            spec.depot_refuel_rate_litres_per_minute
        ),
    )

    state = SimulationState(
        road_graph=fresh_road_graph,
        bins=bins,
        trucks=trucks,
        depot=depot,
        current_time_hours=0.0,
    )

    state.validate_physical_invariants()

    return state


def assemble_physical_scenario(
    *,
    road_graph: RoadGraph,
    depot_node: NodeId,
    assignments: Sequence[
        BinRoadAssignment
    ],
    workload: PhysicalWorkloadRealization,
    spec: PhysicalScenarioSpec,
) -> PhysicalScenarioAssembly:
    """
    Assemble and freeze one physical initial condition.
    """

    state = build_pristine_simulation_state(
        road_graph=road_graph,
        depot_node=depot_node,
        assignments=assignments,
        workload=workload,
        spec=spec,
    )

    snapshot = capture_physical_scenario(
        state
    )

    return PhysicalScenarioAssembly(
        snapshot=snapshot,
        workload_id=(
            workload.workload_id
        ),
        workload_sha256=(
            workload.sha256
        ),
        workload_replicate_id=(
            workload.spec.replicate_id
        ),
    )
