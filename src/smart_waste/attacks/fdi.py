from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from smart_waste.core.city import City
from smart_waste.core.seed_manager import (
    create_rng,
    derive_seed,
)


AttackType = Literal[
    "external_unauthenticated",
    "authenticated_compromise",
]


@dataclass(frozen=True)
class FDIEvent:
    bin_id: int
    true_fill_percent: float
    forged_fill_percent: float
    signature_valid: bool

    def creates_false_service_alert(
        self,
        threshold_percent: float,
    ) -> bool:
        return (
            self.true_fill_percent < threshold_percent
            and self.forged_fill_percent >= threshold_percent
        )


@dataclass(frozen=True)
class FDIAttackScenario:
    attack_rate: float
    attack_type: AttackType
    replicate_id: int
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
        return len(self.events)

    def reported_fill_overrides(
        self,
    ) -> dict[int, float]:
        return {
            event.bin_id: event.forged_fill_percent
            for event in self.events
        }

    def poa_accepted_overrides(
        self,
    ) -> dict[int, float]:
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
def generate_fdi_attack(
    city: City,
    *,
    master_seed: int,
    attack_rate: float,
    attack_type: AttackType,
    forged_fill_percent: float = 100.0,
    threshold_percent: float = 80.0,
    replicate_id: int = 0,
) -> FDIAttackScenario:
    """
    Generate a deterministic False Data Injection scenario.

    Experimental design:
    - each replicate gets one deterministic random node ordering;
    - attack severities within the same replicate are paired/nested;
    - e.g. the 10% attacked set is a subset of the 15% set.

    Nodes are sampled uniformly without replacement.
    """

    if not 0.0 <= attack_rate <= 1.0:
        raise ValueError(
            "attack_rate must be between 0 and 1"
        )

    if not 0.0 <= forged_fill_percent <= 100.0:
        raise ValueError(
            "forged_fill_percent must be between 0 and 100"
        )

    if not 0.0 <= threshold_percent <= 100.0:
        raise ValueError(
            "threshold_percent must be between 0 and 100"
        )

    if replicate_id < 0:
        raise ValueError(
            "replicate_id must be non-negative"
        )

    if attack_type not in (
        "external_unauthenticated",
        "authenticated_compromise",
    ):
        raise ValueError(
            f"unsupported attack_type: {attack_type}"
        )

    num_nodes = len(city.bins)

    num_attacked = int(
        round(num_nodes * attack_rate)
    )

    # IMPORTANT:
    # attack_rate is deliberately NOT part of this namespace.
    # Therefore all attack severities within one replicate use
    # the same random node ordering.
    namespace = (
        f"fdi:"
        f"{attack_type}:"
        f"replicate:{replicate_id}"
    )

    selection_seed = derive_seed(
        master_seed,
        namespace,
    )

    rng = create_rng(selection_seed)

    # Generate one complete random ordering of nodes.
    node_order = rng.permutation(num_nodes)

    # Each severity takes a prefix of the same ordering.
    selected_ids = np.sort(
        node_order[:num_attacked]
    )

    signature_valid = (
        attack_type
        == "authenticated_compromise"
    )

    events = tuple(
        FDIEvent(
            bin_id=int(bin_id),
            true_fill_percent=float(
                city.bins[int(bin_id)].fill_percent
            ),
            forged_fill_percent=float(
                forged_fill_percent
            ),
            signature_valid=signature_valid,
        )
        for bin_id in selected_ids
    )

    return FDIAttackScenario(
        attack_rate=attack_rate,
        attack_type=attack_type,
        replicate_id=replicate_id,
        selection_seed=selection_seed,
        events=events,
    )