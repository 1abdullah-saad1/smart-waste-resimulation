from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import floor, isfinite
from numbers import Integral
from typing import Literal, Protocol

import numpy as np

from smart_waste.core.seed_manager import (
    create_rng,
    derive_seed,
)


AttackType = Literal[
    "external_unauthenticated",
    "authenticated_compromise",
]

AttackSelectionMode = Literal[
    "paired_nested",
    "independent_by_rate",
]

AttackCountRule = Literal[
    "round_half_up",
]


class _LegacyBinLike(Protocol):
    bin_id: int
    fill_percent: float


class _LegacyCityLike(Protocol):
    bins: object


@dataclass(frozen=True)
class FDIEvent:
    """
    One false-data-injection event against one physical bin.

    true_fill_percent is an immutable snapshot of latent physical
    truth at attack-generation time.

    forged_fill_percent is routing/security telemetry only.
    """

    bin_id: int
    true_fill_percent: float
    forged_fill_percent: float
    signature_valid: bool

    def creates_false_service_alert(
        self,
        threshold_percent: float,
    ) -> bool:
        threshold = _validated_percentage(
            threshold_percent,
            name="threshold_percent",
        )

        return (
            self.true_fill_percent < threshold
            and self.forged_fill_percent >= threshold
        )


@dataclass(frozen=True)
class FDIAttackScenario:
    """
    Immutable deterministic FDI scenario.

    Attack membership is independent from attack credential type.
    Therefore external-unauthenticated and authenticated-compromise
    variants can target exactly the same physical bins for paired
    comparison.

    attack_type changes security acceptance semantics only.
    """

    attack_rate: float
    attack_type: AttackType
    selection_mode: AttackSelectionMode
    count_rule: AttackCountRule

    replicate_id: int
    population_size: int
    selection_seed: int

    events: tuple[FDIEvent, ...]

    @property
    def attacked_bin_ids(self) -> tuple[int, ...]:
        return tuple(
            event.bin_id
            for event in self.events
        )

    @property
    def num_attacked_nodes(self) -> int:
        return len(
            self.events
        )

    @property
    def realized_attack_rate(self) -> float:
        if self.population_size == 0:
            return 0.0

        return (
            self.num_attacked_nodes
            / self.population_size
        )

    def reported_fill_overrides(
        self,
    ) -> dict[int, float]:
        """
        Raw forged telemetry seen by an unprotected routing path.
        """

        return {
            event.bin_id: event.forged_fill_percent
            for event in self.events
        }

    def poa_accepted_overrides(
        self,
    ) -> dict[int, float]:
        """
        Forged telemetry surviving signature verification.

        external_unauthenticated:
            rejected

        authenticated_compromise:
            survives authentication because credentials are valid
        """

        return {
            event.bin_id: event.forged_fill_percent
            for event in self.events
            if event.signature_valid
        }

    def false_service_alert_ids(
        self,
        threshold_percent: float,
    ) -> tuple[int, ...]:
        return tuple(
            event.bin_id
            for event in self.events
            if event.creates_false_service_alert(
                threshold_percent
            )
        )


def _validated_percentage(
    value: float,
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
        raise ValueError(
            f"{name} must be numeric"
        ) from exc

    if not isfinite(
        numeric
    ):
        raise ValueError(
            f"{name} must be finite"
        )

    if not (
        0.0
        <= numeric
        <= 100.0
    ):
        raise ValueError(
            f"{name} must be between 0 and 100"
        )

    return numeric


def _validated_attack_rate(
    attack_rate: float,
) -> float:
    try:
        rate = float(
            attack_rate
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            "attack_rate must be numeric"
        ) from exc

    if not isfinite(
        rate
    ):
        raise ValueError(
            "attack_rate must be finite"
        )

    if not (
        0.0
        <= rate
        <= 1.0
    ):
        raise ValueError(
            "attack_rate must be between 0 and 1"
        )

    return rate


def attack_count_from_rate(
    *,
    population_size: int,
    attack_rate: float,
) -> int:
    """
    Convert attack severity to an exact attacked-node count.

    The revised benchmark uses round-half-up rather than Python's
    built-in round(), whose tie handling is round-to-even.

    Examples:
        1000 * 0.15 = 150   -> 150
        10   * 0.25 = 2.5   -> 3
    """

    if not isinstance(
        population_size,
        Integral,
    ):
        raise ValueError(
            "population_size must be an integer"
        )

    population_size = int(
        population_size
    )

    if population_size < 0:
        raise ValueError(
            "population_size must be non-negative"
        )

    rate = _validated_attack_rate(
        attack_rate
    )

    count = int(
        floor(
            population_size
            * rate
            + 0.5
        )
    )

    return min(
        population_size,
        count,
    )


def _validated_truth_population(
    true_fill_percent_by_bin: Mapping[int, float],
) -> dict[int, float]:
    if not isinstance(
        true_fill_percent_by_bin,
        Mapping,
    ):
        raise TypeError(
            "true_fill_percent_by_bin must be a mapping"
        )

    if not true_fill_percent_by_bin:
        raise ValueError(
            "FDI population cannot be empty"
        )

    validated: dict[
        int,
        float,
    ] = {}

    for raw_bin_id, raw_fill in (
        true_fill_percent_by_bin.items()
    ):
        if (
            not isinstance(
                raw_bin_id,
                Integral,
            )
            or isinstance(
                raw_bin_id,
                bool,
            )
        ):
            raise ValueError(
                "FDI bin IDs must be integers"
            )

        bin_id = int(
            raw_bin_id
        )

        if bin_id in validated:
            raise ValueError(
                f"duplicate FDI bin_id: {bin_id}"
            )

        validated[
            bin_id
        ] = _validated_percentage(
            raw_fill,
            name=(
                f"true fill for bin {bin_id}"
            ),
        )

    return validated


def _validated_attack_type(
    attack_type: AttackType,
) -> AttackType:
    if attack_type not in (
        "external_unauthenticated",
        "authenticated_compromise",
    ):
        raise ValueError(
            f"unsupported attack_type: {attack_type}"
        )

    return attack_type


def _selection_namespace(
    *,
    selection_mode: AttackSelectionMode,
    attack_rate: float,
    replicate_id: int,
) -> str:
    """
    Build the node-selection namespace.

    attack_type is deliberately excluded.

    This pairs security variants on the exact same compromised-bin
    population so that signature validity is the only threat-model
    difference.

    For paired_nested mode, attack_rate is also excluded. All
    severities use prefixes of one common random permutation.
    """

    if selection_mode == "paired_nested":
        return (
            "fdi:"
            "paired_nested:"
            f"replicate:{replicate_id}"
        )

    if selection_mode == "independent_by_rate":
        return (
            "fdi:"
            "independent_by_rate:"
            f"rate:{attack_rate:.12f}:"
            f"replicate:{replicate_id}"
        )

    raise ValueError(
        f"unsupported selection_mode: {selection_mode}"
    )


def generate_fdi_attack_from_truth(
    true_fill_percent_by_bin: Mapping[int, float],
    *,
    master_seed: int,
    attack_rate: float,
    attack_type: AttackType,
    selection_mode: AttackSelectionMode = "paired_nested",
    forged_fill_percent: float = 100.0,
    replicate_id: int = 0,
) -> FDIAttackScenario:
    """
    Generate a deterministic FDI scenario from actual bin IDs.

    Revised experimental contract
    -----------------------------
    1. The attack population is the exact set of keys supplied in
       true_fill_percent_by_bin. IDs need not be contiguous.

    2. Sampling is uniform without replacement.

    3. paired_nested mode creates one deterministic permutation per
       replicate. Severity levels use prefixes of that permutation:

           10% subset 15% subset 20% subset 30%

    4. Attack selection is independent of attack_type. Therefore
       external and authenticated threat variants target identical
       bins when all other parameters are equal.

    5. FDI generation never mutates the supplied physical truth.

    6. Forged telemetry defaults to 100%, matching the routing
       attack scenario used by the benchmark.

    7. Severity-to-count conversion uses explicit round-half-up.
    """

    rate = _validated_attack_rate(
        attack_rate
    )

    forged_fill = _validated_percentage(
        forged_fill_percent,
        name="forged_fill_percent",
    )

    if not isinstance(
        replicate_id,
        Integral,
    ):
        raise ValueError(
            "replicate_id must be an integer"
        )

    replicate_id = int(
        replicate_id
    )

    if replicate_id < 0:
        raise ValueError(
            "replicate_id must be non-negative"
        )

    attack_type = _validated_attack_type(
        attack_type
    )

    truth = _validated_truth_population(
        true_fill_percent_by_bin
    )

    population_ids = tuple(
        sorted(
            truth
        )
    )

    num_attacked = attack_count_from_rate(
        population_size=len(
            population_ids
        ),
        attack_rate=rate,
    )

    namespace = _selection_namespace(
        selection_mode=selection_mode,
        attack_rate=rate,
        replicate_id=replicate_id,
    )

    selection_seed = derive_seed(
        master_seed,
        namespace,
    )

    rng = create_rng(
        selection_seed
    )

    # Permute population positions, then map back to the real bin
    # IDs. No 0..N-1 bin-ID assumption is made.
    ordered_indices = rng.permutation(
        len(
            population_ids
        )
    )

    selected_ids = tuple(
        sorted(
            population_ids[
                int(index)
            ]
            for index in (
                ordered_indices[
                    :num_attacked
                ]
            )
        )
    )

    signature_valid = (
        attack_type
        == "authenticated_compromise"
    )

    events = tuple(
        FDIEvent(
            bin_id=bin_id,
            true_fill_percent=(
                truth[
                    bin_id
                ]
            ),
            forged_fill_percent=(
                forged_fill
            ),
            signature_valid=(
                signature_valid
            ),
        )
        for bin_id in selected_ids
    )

    return FDIAttackScenario(
        attack_rate=rate,
        attack_type=attack_type,
        selection_mode=selection_mode,
        count_rule="round_half_up",
        replicate_id=replicate_id,
        population_size=len(
            population_ids
        ),
        selection_seed=selection_seed,
        events=events,
    )


def generate_fdi_attack(
    city: _LegacyCityLike,
    *,
    master_seed: int,
    attack_rate: float,
    attack_type: AttackType,
    selection_mode: AttackSelectionMode = "paired_nested",
    forged_fill_percent: float = 100.0,
    threshold_percent: float = 80.0,
    replicate_id: int = 0,
) -> FDIAttackScenario:
    """
    Compatibility wrapper for the legacy City-based API.

    New modular simulation code should call
    generate_fdi_attack_from_truth() instead.

    threshold_percent is retained only for backwards compatibility;
    attack generation itself does not depend on the service
    threshold. False-alert classification is performed by
    FDIAttackScenario.false_service_alert_ids().
    """

    # Preserve validation behavior of the original API.
    _validated_percentage(
        threshold_percent,
        name="threshold_percent",
    )

    try:
        legacy_bins = city.bins
    except AttributeError as exc:
        raise TypeError(
            "legacy city object must provide bins"
        ) from exc

    truth: dict[
        int,
        float,
    ] = {}

    for bin_ in legacy_bins:
        try:
            bin_id = int(
                bin_.bin_id
            )

            fill = float(
                bin_.fill_percent
            )
        except AttributeError as exc:
            raise TypeError(
                "legacy city bins must expose "
                "bin_id and fill_percent"
            ) from exc

        if bin_id in truth:
            raise ValueError(
                f"duplicate legacy bin_id: {bin_id}"
            )

        truth[
            bin_id
        ] = fill

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
