from __future__ import annotations

from dataclasses import dataclass


EMERGENCY_FIRE_THRESHOLD_C = 70.0


@dataclass(frozen=True)
class HazardState:
    """
    Hazard state supplied to the routing environment.

    The manuscript explicitly contains:
    - emergency fire at temperature >= 70 C,
    - predicted high combustion risk,
    - excess-moisture hazard.

    The numerical severity encoding is a reconstruction
    convention required for the DQN state H_t.
    """

    bin_id: int

    temperature_c: float
    humidity_percent: float

    emergency_fire: bool
    predicted_combustion_risk: bool
    excess_moisture: bool

    @property
    def event_required(self) -> bool:
        return (
            self.emergency_fire
            or self.predicted_combustion_risk
            or self.excess_moisture
        )

    @property
    def preemptive_fire_collection(self) -> bool:
        return (
            self.emergency_fire
            or self.predicted_combustion_risk
        )

    @property
    def severity_score(self) -> float:
        """
        Reconstruction encoding for H_t.

        0 = no hazard
        1 = excess moisture
        2 = predicted combustion risk
        3 = emergency fire

        For simultaneous hazards, the most severe state wins.
        """

        if self.emergency_fire:
            return 3.0

        if self.predicted_combustion_risk:
            return 2.0

        if self.excess_moisture:
            return 1.0

        return 0.0

    @property
    def normalized_severity(self) -> float:
        """
        Normalize H_t to [0, 1] for neural-network input.
        """

        return self.severity_score / 3.0


def evaluate_hazard(
    *,
    bin_id: int,
    temperature_c: float,
    humidity_percent: float,
    predicted_fire_risk_high: bool = False,
    excess_moisture_detected: bool = False,
    emergency_fire_threshold_c: float = (
        EMERGENCY_FIRE_THRESHOLD_C
    ),
) -> HazardState:
    """
    Evaluate the manuscript hazard logic.

    Manuscript-defined rule:
        temperature >= 70 C -> EMERGENCY_FIRE

    Predicted fire risk and excess-moisture flags are supplied
    by their respective predictive/sensing components.

    Their exact predictive equations are not specified by the
    manuscript and therefore are not invented here.
    """

    if bin_id < 0:
        raise ValueError(
            "bin_id must be non-negative"
        )

    if not 0.0 <= humidity_percent <= 100.0:
        raise ValueError(
            "humidity_percent must be between 0 and 100"
        )

    emergency_fire = (
        temperature_c
        >= emergency_fire_threshold_c
    )

    return HazardState(
        bin_id=bin_id,
        temperature_c=float(temperature_c),
        humidity_percent=float(humidity_percent),
        emergency_fire=emergency_fire,
        predicted_combustion_risk=bool(
            predicted_fire_risk_high
        ),
        excess_moisture=bool(
            excess_moisture_detected
        ),
    )