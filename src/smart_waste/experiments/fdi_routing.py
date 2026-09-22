from __future__ import annotations

from math import isfinite
from typing import Literal

from smart_waste.attacks.fdi import (
    AttackSelectionMode,
    AttackType,
    FDIAttackScenario,
    generate_fdi_attack_from_truth,
)
from smart_waste.simulation.state import (
    SimulationState,
)


FDIRoutingPath = Literal[
    "unprotected",
    "poa_verified",
]


class FDIRoutingAdapterError(RuntimeError):
    """
    Raised when a physical simulation state cannot be converted
    into a valid FDI routing experiment input.
    """


def snapshot_physical_fill_percent(
    state: SimulationState,
) -> dict[int, float]:
    """
    Snapshot latent physical fill from the current scenario.

    The returned mapping is a detached copy:

        actual bin_id -> true physical fill percent

    No telemetry value is read here and no state is mutated.
    """

    if not state.bins:
        raise FDIRoutingAdapterError(
            "FDI routing experiment requires at least one bin"
        )

    truth: dict[
        int,
        float,
    ] = {}

    for mapping_bin_id, bin_ in sorted(
        state.bins.items(),
        key=lambda item: item[0],
    ):
        if mapping_bin_id != bin_.bin_id:
            raise FDIRoutingAdapterError(
                "SimulationState bin mapping key does not match "
                f"WasteBin.bin_id: key={mapping_bin_id}, "
                f"bin_id={bin_.bin_id}"
            )

        fill = float(
            bin_.fill_percent
        )

        if not isfinite(
            fill
        ):
            raise FDIRoutingAdapterError(
                f"physical fill for bin {bin_.bin_id} "
                "must be finite"
            )

        if not (
            0.0
            <= fill
            <= 100.0
        ):
            raise FDIRoutingAdapterError(
                f"physical fill for bin {bin_.bin_id} "
                "must be between 0 and 100"
            )

        truth[
            bin_.bin_id
        ] = fill

    return truth


def build_fdi_scenario_for_state(
    state: SimulationState,
    *,
    master_seed: int,
    attack_rate: float,
    attack_type: AttackType,
    selection_mode: AttackSelectionMode = "paired_nested",
    forged_fill_percent: float = 100.0,
    replicate_id: int = 0,
) -> FDIAttackScenario:
    """
    Generate one deterministic FDI scenario from physical truth.

    Attack membership is generated from a snapshot of the current
    physical state. Subsequent telemetry processing cannot mutate
    that truth snapshot or the SimulationState.
    """

    truth = snapshot_physical_fill_percent(
        state
    )

    return generate_fdi_attack_from_truth(
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


def routing_telemetry_overrides(
    scenario: FDIAttackScenario,
    *,
    path: FDIRoutingPath,
) -> dict[int, float]:
    """
    Resolve the routing telemetry visible under one security path.

    unprotected
        Every forged reading reaches routing.

    poa_verified
        Only forged readings carrying valid credentials/signatures
        survive verification.

    A fresh dictionary is returned on every call.
    """

    if path == "unprotected":
        return (
            scenario.reported_fill_overrides()
        )

    if path == "poa_verified":
        return (
            scenario.poa_accepted_overrides()
        )

    raise ValueError(
        f"unsupported FDI routing path: {path}"
    )
