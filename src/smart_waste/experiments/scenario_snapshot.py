from __future__ import annotations

import hashlib
import json

from collections.abc import Mapping
from dataclasses import dataclass
from math import isclose, isfinite
from numbers import Integral, Real

import networkx as nx

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


NodeId = int | str


class ScenarioSnapshotError(RuntimeError):
    """
    Raised when a mutable runtime state cannot safely become an
    immutable experimental initial-condition snapshot.
    """


@dataclass(frozen=True)
class RoadNodeSnapshot:
    node_id: NodeId
    attributes: tuple[
        tuple[str, object],
        ...,
    ]


@dataclass(frozen=True)
class RoadEdgeSnapshot:
    source: NodeId
    target: NodeId
    attributes: tuple[
        tuple[str, object],
        ...,
    ]


@dataclass(frozen=True)
class BinInitialState:
    bin_id: int
    road_node: NodeId

    fill_percent: float
    fill_rate_percent_per_hour: float
    full_mass_kg: float
    waste_age_hours: float


@dataclass(frozen=True)
class TruckInitialState:
    truck_id: int
    current_node: NodeId

    capacity_tonnes: float
    speed_km_per_hour: float

    fuel_capacity_litres: float
    fuel_efficiency_km_per_litre: float

    current_load_tonnes: float
    fuel_remaining_litres: float


@dataclass(frozen=True)
class DepotInitialState:
    road_node: NodeId

    unloading_bays: int
    unload_time_minutes: float
    refuel_rate_litres_per_minute: float


@dataclass(frozen=True)
class PhysicalScenarioSnapshot:
    """
    Immutable definition of one initial physical experiment.

    Deliberately excluded:
    - event queue,
    - processed-event counter,
    - reservations,
    - dispatch history,
    - completed-service history,
    - depot waiting/active queues,
    - pending truck travel,
    - accumulated run metrics.

    Every compared experimental condition must reconstruct its own
    fresh SimulationState from this same snapshot.
    """

    schema_version: str

    road_length_attribute: str

    graph_attributes: tuple[
        tuple[str, object],
        ...,
    ]

    road_nodes: tuple[
        RoadNodeSnapshot,
        ...,
    ]

    road_edges: tuple[
        RoadEdgeSnapshot,
        ...,
    ]

    bins: tuple[
        BinInitialState,
        ...,
    ]

    trucks: tuple[
        TruckInitialState,
        ...,
    ]

    depot: DepotInitialState

    @property
    def sha256(self) -> str:
        payload = _snapshot_payload(
            self
        )

        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
            ensure_ascii=False,
            allow_nan=False,
        ).encode(
            "utf-8"
        )

        return hashlib.sha256(
            encoded
        ).hexdigest()

    @property
    def scenario_id(self) -> str:
        return (
            "physical-"
            + self.sha256[:16]
        )


def _normalize_node_id(
    value: object,
) -> NodeId:
    if isinstance(
        value,
        bool,
    ):
        raise ScenarioSnapshotError(
            "boolean road-node IDs are not supported"
        )

    if isinstance(
        value,
        Integral,
    ):
        return int(
            value
        )

    if isinstance(
        value,
        str,
    ):
        return value

    raise ScenarioSnapshotError(
        "road-node IDs must be integer or string"
    )


def _node_sort_key(
    node_id: NodeId,
) -> tuple[
    int,
    int | str,
]:
    node_id = _normalize_node_id(
        node_id
    )

    if isinstance(
        node_id,
        int,
    ):
        return (
            0,
            node_id,
        )

    return (
        1,
        node_id,
    )


def _finite_float(
    value: object,
    *,
    name: str,
) -> float:
    try:
        numeric = float(
            value
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ScenarioSnapshotError(
            f"{name} must be numeric"
        ) from exc

    if not isfinite(
        numeric
    ):
        raise ScenarioSnapshotError(
            f"{name} must be finite"
        )

    return numeric


def _freeze_value(
    value: object,
) -> object:
    """
    Convert graph metadata into immutable deterministic values.

    Supported graph metadata:
    - None
    - bool
    - integer
    - finite real number
    - string
    - list
    - tuple
    - string-keyed mapping
    """

    if value is None:
        return None

    if isinstance(
        value,
        bool,
    ):
        return value

    if isinstance(
        value,
        Integral,
    ):
        return int(
            value
        )

    if isinstance(
        value,
        Real,
    ):
        numeric = float(
            value
        )

        if not isfinite(
            numeric
        ):
            raise ScenarioSnapshotError(
                "graph metadata numbers must be finite"
            )

        return numeric

    if isinstance(
        value,
        str,
    ):
        return value

    if isinstance(
        value,
        Mapping,
    ):
        frozen_items: list[
            tuple[str, object]
        ] = []

        for key, item in value.items():
            if not isinstance(
                key,
                str,
            ):
                raise ScenarioSnapshotError(
                    "graph metadata mapping keys "
                    "must be strings"
                )

            frozen_items.append(
                (
                    key,
                    _freeze_value(
                        item
                    ),
                )
            )

        return (
            "__mapping__",
            tuple(
                sorted(
                    frozen_items,
                    key=lambda pair: pair[0],
                )
            ),
        )

    if isinstance(
        value,
        list,
    ):
        return (
            "__list__",
            tuple(
                _freeze_value(
                    item
                )
                for item in value
            ),
        )

    if isinstance(
        value,
        tuple,
    ):
        return (
            "__tuple__",
            tuple(
                _freeze_value(
                    item
                )
                for item in value
            ),
        )

    raise ScenarioSnapshotError(
        "unsupported graph metadata value type: "
        f"{type(value).__name__}"
    )


def _thaw_value(
    value: object,
) -> object:
    if (
        isinstance(
            value,
            tuple,
        )
        and len(value) == 2
        and isinstance(
            value[0],
            str,
        )
    ):
        tag = value[0]
        payload = value[1]

        if tag == "__mapping__":
            return {
                key: _thaw_value(
                    item
                )
                for key, item in payload
            }

        if tag == "__list__":
            return [
                _thaw_value(
                    item
                )
                for item in payload
            ]

        if tag == "__tuple__":
            return tuple(
                _thaw_value(
                    item
                )
                for item in payload
            )

    return value


def _freeze_attributes(
    attributes: Mapping[str, object],
) -> tuple[
    tuple[str, object],
    ...,
]:
    frozen: list[
        tuple[str, object]
    ] = []

    for key, value in attributes.items():
        if not isinstance(
            key,
            str,
        ):
            raise ScenarioSnapshotError(
                "graph attribute names must be strings"
            )

        frozen.append(
            (
                key,
                _freeze_value(
                    value
                ),
            )
        )

    return tuple(
        sorted(
            frozen,
            key=lambda pair: pair[0],
        )
    )


def _thaw_attributes(
    attributes: tuple[
        tuple[str, object],
        ...,
    ],
) -> dict[str, object]:
    return {
        key: _thaw_value(
            value
        )
        for key, value in attributes
    }


def _node_id_payload(
    node_id: NodeId,
) -> list[object]:
    if isinstance(
        node_id,
        int,
    ):
        return [
            "int",
            node_id,
        ]

    return [
        "str",
        node_id,
    ]


def _frozen_value_payload(
    value: object,
) -> object:
    if (
        isinstance(
            value,
            tuple,
        )
        and len(value) == 2
        and value[0]
        in {
            "__mapping__",
            "__list__",
            "__tuple__",
        }
    ):
        tag = value[0]
        payload = value[1]

        if tag == "__mapping__":
            return {
                "kind": "mapping",
                "items": [
                    [
                        key,
                        _frozen_value_payload(
                            item
                        ),
                    ]
                    for key, item in payload
                ],
            }

        kind = (
            "list"
            if tag == "__list__"
            else "tuple"
        )

        return {
            "kind": kind,
            "items": [
                _frozen_value_payload(
                    item
                )
                for item in payload
            ],
        }

    return value


def _attributes_payload(
    attributes: tuple[
        tuple[str, object],
        ...,
    ],
) -> list[list[object]]:
    return [
        [
            key,
            _frozen_value_payload(
                value
            ),
        ]
        for key, value in attributes
    ]


def _snapshot_payload(
    snapshot: PhysicalScenarioSnapshot,
) -> dict[str, object]:
    return {
        "schema_version": (
            snapshot.schema_version
        ),
        "road_length_attribute": (
            snapshot.road_length_attribute
        ),
        "graph_attributes": (
            _attributes_payload(
                snapshot.graph_attributes
            )
        ),
        "road_nodes": [
            {
                "node_id": _node_id_payload(
                    node.node_id
                ),
                "attributes": (
                    _attributes_payload(
                        node.attributes
                    )
                ),
            }
            for node in snapshot.road_nodes
        ],
        "road_edges": [
            {
                "source": _node_id_payload(
                    edge.source
                ),
                "target": _node_id_payload(
                    edge.target
                ),
                "attributes": (
                    _attributes_payload(
                        edge.attributes
                    )
                ),
            }
            for edge in snapshot.road_edges
        ],
        "bins": [
            {
                "bin_id": bin_.bin_id,
                "road_node": _node_id_payload(
                    bin_.road_node
                ),
                "fill_percent": (
                    bin_.fill_percent
                ),
                "fill_rate_percent_per_hour": (
                    bin_.fill_rate_percent_per_hour
                ),
                "full_mass_kg": (
                    bin_.full_mass_kg
                ),
                "waste_age_hours": (
                    bin_.waste_age_hours
                ),
            }
            for bin_ in snapshot.bins
        ],
        "trucks": [
            {
                "truck_id": truck.truck_id,
                "current_node": _node_id_payload(
                    truck.current_node
                ),
                "capacity_tonnes": (
                    truck.capacity_tonnes
                ),
                "speed_km_per_hour": (
                    truck.speed_km_per_hour
                ),
                "fuel_capacity_litres": (
                    truck.fuel_capacity_litres
                ),
                "fuel_efficiency_km_per_litre": (
                    truck.fuel_efficiency_km_per_litre
                ),
                "current_load_tonnes": (
                    truck.current_load_tonnes
                ),
                "fuel_remaining_litres": (
                    truck.fuel_remaining_litres
                ),
            }
            for truck in snapshot.trucks
        ],
        "depot": {
            "road_node": _node_id_payload(
                snapshot.depot.road_node
            ),
            "unloading_bays": (
                snapshot.depot.unloading_bays
            ),
            "unload_time_minutes": (
                snapshot.depot.unload_time_minutes
            ),
            "refuel_rate_litres_per_minute": (
                snapshot.depot.refuel_rate_litres_per_minute
            ),
        },
    }


def _validate_pristine_runtime_state(
    state: SimulationState,
) -> None:
    if not isclose(
        state.current_time_hours,
        0.0,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ScenarioSnapshotError(
            "physical scenario snapshot requires "
            "current_time_hours == 0"
        )

    for bin_ in state.bins.values():
        if bin_.collected_count != 0:
            raise ScenarioSnapshotError(
                f"bin {bin_.bin_id} has prior collection history"
            )

    for truck in state.trucks.values():
        if truck.status != TruckStatus.IDLE:
            raise ScenarioSnapshotError(
                f"truck {truck.truck_id} is not in initial IDLE state"
            )

        if truck.next_event_time_hours is not None:
            raise ScenarioSnapshotError(
                f"truck {truck.truck_id} has a pending event"
            )

        if truck.travel_origin_node is not None:
            raise ScenarioSnapshotError(
                f"truck {truck.truck_id} has travel history/state"
            )

        if truck.destination_node is not None:
            raise ScenarioSnapshotError(
                f"truck {truck.truck_id} has a pending destination"
            )

        if not isclose(
            truck.pending_distance_km,
            0.0,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ScenarioSnapshotError(
                f"truck {truck.truck_id} has pending road distance"
            )

        if truck.route_nodes != [
            truck.current_node
        ]:
            raise ScenarioSnapshotError(
                f"truck {truck.truck_id} contains prior route history"
            )

        cumulative_values = {
            "cumulative_distance_km": (
                truck.cumulative_distance_km
            ),
            "cumulative_fuel_used_litres": (
                truck.cumulative_fuel_used_litres
            ),
            "cumulative_refuelled_litres": (
                truck.cumulative_refuelled_litres
            ),
        }

        for name, value in cumulative_values.items():
            if not isclose(
                value,
                0.0,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise ScenarioSnapshotError(
                    f"truck {truck.truck_id} has "
                    f"non-zero {name}"
                )

        if any(
            count != 0
            for count in (
                truck.depot_returns,
                truck.capacity_returns,
                truck.fuel_returns,
                truck.combined_returns,
            )
        ):
            raise ScenarioSnapshotError(
                f"truck {truck.truck_id} has prior depot-return history"
            )


def capture_physical_scenario(
    state: SimulationState,
) -> PhysicalScenarioSnapshot:
    """
    Capture one pristine mutable SimulationState as an immutable
    initial-condition definition.

    Runtime/event state is intentionally rejected rather than
    copied.
    """

    state.validate_physical_invariants()

    _validate_pristine_runtime_state(
        state
    )

    graph = state.road_graph.graph

    if graph.is_directed():
        raise ScenarioSnapshotError(
            "physical scenario road graph must be undirected"
        )

    if graph.is_multigraph():
        raise ScenarioSnapshotError(
            "physical scenario snapshot currently requires "
            "a simple nx.Graph"
        )

    road_nodes = tuple(
        RoadNodeSnapshot(
            node_id=_normalize_node_id(
                node_id
            ),
            attributes=_freeze_attributes(
                attributes
            ),
        )
        for node_id, attributes in sorted(
            graph.nodes(
                data=True
            ),
            key=lambda item: (
                _node_sort_key(
                    item[0]
                )
            ),
        )
    )

    edge_rows: list[
        RoadEdgeSnapshot
    ] = []

    for raw_source, raw_target, attributes in (
        graph.edges(
            data=True
        )
    ):
        source = _normalize_node_id(
            raw_source
        )

        target = _normalize_node_id(
            raw_target
        )

        if (
            _node_sort_key(
                target
            )
            < _node_sort_key(
                source
            )
        ):
            source, target = (
                target,
                source,
            )

        edge_rows.append(
            RoadEdgeSnapshot(
                source=source,
                target=target,
                attributes=(
                    _freeze_attributes(
                        attributes
                    )
                ),
            )
        )

    road_edges = tuple(
        sorted(
            edge_rows,
            key=lambda edge: (
                _node_sort_key(
                    edge.source
                ),
                _node_sort_key(
                    edge.target
                ),
            ),
        )
    )

    bins = tuple(
        BinInitialState(
            bin_id=bin_.bin_id,
            road_node=_normalize_node_id(
                bin_.road_node
            ),
            fill_percent=_finite_float(
                bin_.fill_percent,
                name=(
                    f"bin {bin_.bin_id} fill_percent"
                ),
            ),
            fill_rate_percent_per_hour=(
                _finite_float(
                    bin_.fill_rate_percent_per_hour,
                    name=(
                        f"bin {bin_.bin_id} fill rate"
                    ),
                )
            ),
            full_mass_kg=_finite_float(
                bin_.full_mass_kg,
                name=(
                    f"bin {bin_.bin_id} full_mass_kg"
                ),
            ),
            waste_age_hours=_finite_float(
                bin_.waste_age_hours,
                name=(
                    f"bin {bin_.bin_id} waste_age_hours"
                ),
            ),
        )
        for bin_ in sorted(
            state.bins.values(),
            key=lambda item: item.bin_id,
        )
    )

    trucks = tuple(
        TruckInitialState(
            truck_id=truck.truck_id,
            current_node=_normalize_node_id(
                truck.current_node
            ),
            capacity_tonnes=_finite_float(
                truck.capacity_tonnes,
                name=(
                    f"truck {truck.truck_id} capacity"
                ),
            ),
            speed_km_per_hour=_finite_float(
                truck.speed_km_per_hour,
                name=(
                    f"truck {truck.truck_id} speed"
                ),
            ),
            fuel_capacity_litres=_finite_float(
                truck.fuel_capacity_litres,
                name=(
                    f"truck {truck.truck_id} fuel capacity"
                ),
            ),
            fuel_efficiency_km_per_litre=(
                _finite_float(
                    truck.fuel_efficiency_km_per_litre,
                    name=(
                        f"truck {truck.truck_id} "
                        "fuel efficiency"
                    ),
                )
            ),
            current_load_tonnes=_finite_float(
                truck.current_load_tonnes,
                name=(
                    f"truck {truck.truck_id} current load"
                ),
            ),
            fuel_remaining_litres=_finite_float(
                truck.fuel_remaining_litres,
                name=(
                    f"truck {truck.truck_id} remaining fuel"
                ),
            ),
        )
        for truck in sorted(
            state.trucks.values(),
            key=lambda item: item.truck_id,
        )
    )

    depot = DepotInitialState(
        road_node=_normalize_node_id(
            state.depot.road_node
        ),
        unloading_bays=(
            state.depot.unloading_bays
        ),
        unload_time_minutes=_finite_float(
            state.depot.unload_time_minutes,
            name="depot unload time",
        ),
        refuel_rate_litres_per_minute=(
            _finite_float(
                state.depot.refuel_rate_litres_per_minute,
                name="depot refuel rate",
            )
        ),
    )

    return PhysicalScenarioSnapshot(
        schema_version=(
            "physical-scenario-v1"
        ),
        road_length_attribute=(
            state.road_graph.length_attribute
        ),
        graph_attributes=(
            _freeze_attributes(
                graph.graph
            )
        ),
        road_nodes=road_nodes,
        road_edges=road_edges,
        bins=bins,
        trucks=trucks,
        depot=depot,
    )


def build_simulation_state(
    snapshot: PhysicalScenarioSnapshot,
) -> SimulationState:
    """
    Reconstruct one fresh mutable physical state.

    Repeated calls return structurally equivalent but independent
    graphs, bins, trucks, and runtime state.
    """

    if (
        snapshot.schema_version
        != "physical-scenario-v1"
    ):
        raise ScenarioSnapshotError(
            "unsupported physical scenario schema: "
            f"{snapshot.schema_version}"
        )

    graph = nx.Graph()

    graph.graph.update(
        _thaw_attributes(
            snapshot.graph_attributes
        )
    )

    for node in snapshot.road_nodes:
        graph.add_node(
            node.node_id,
            **_thaw_attributes(
                node.attributes
            ),
        )

    for edge in snapshot.road_edges:
        graph.add_edge(
            edge.source,
            edge.target,
            **_thaw_attributes(
                edge.attributes
            ),
        )

    road_graph = RoadGraph(
        graph=graph,
        length_attribute=(
            snapshot.road_length_attribute
        ),
    )

    bins = {
        row.bin_id: WasteBin(
            bin_id=row.bin_id,
            road_node=row.road_node,
            fill_percent=(
                row.fill_percent
            ),
            fill_rate_percent_per_hour=(
                row.fill_rate_percent_per_hour
            ),
            full_mass_kg=(
                row.full_mass_kg
            ),
            waste_age_hours=(
                row.waste_age_hours
            ),
        )
        for row in snapshot.bins
    }

    trucks = {
        row.truck_id: Truck(
            truck_id=row.truck_id,
            current_node=(
                row.current_node
            ),
            capacity_tonnes=(
                row.capacity_tonnes
            ),
            speed_km_per_hour=(
                row.speed_km_per_hour
            ),
            fuel_capacity_litres=(
                row.fuel_capacity_litres
            ),
            fuel_efficiency_km_per_litre=(
                row.fuel_efficiency_km_per_litre
            ),
            current_load_tonnes=(
                row.current_load_tonnes
            ),
            fuel_remaining_litres=(
                row.fuel_remaining_litres
            ),
        )
        for row in snapshot.trucks
    }

    depot = Depot(
        road_node=(
            snapshot.depot.road_node
        ),
        unloading_bays=(
            snapshot.depot.unloading_bays
        ),
        unload_time_minutes=(
            snapshot.depot.unload_time_minutes
        ),
        refuel_rate_litres_per_minute=(
            snapshot.depot.refuel_rate_litres_per_minute
        ),
    )

    state = SimulationState(
        road_graph=road_graph,
        bins=bins,
        trucks=trucks,
        depot=depot,
        current_time_hours=0.0,
    )

    state.validate_physical_invariants()

    return state
