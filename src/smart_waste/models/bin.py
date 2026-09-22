from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WasteBin:
    """Physical state of one road-accessible waste bin."""

    bin_id: int
    road_node: int | str
    fill_percent: float
    fill_rate_percent_per_hour: float
    full_mass_kg: float

    waste_age_hours: float = 0.0
    collected_count: int = 0

    def __post_init__(self) -> None:
        if self.bin_id < 0:
            raise ValueError("bin_id must be non-negative")

        if not 0.0 <= self.fill_percent <= 100.0:
            raise ValueError(
                "fill_percent must be between 0 and 100"
            )

        if self.fill_rate_percent_per_hour < 0.0:
            raise ValueError(
                "fill_rate_percent_per_hour cannot be negative"
            )

        if self.full_mass_kg <= 0.0:
            raise ValueError(
                "full_mass_kg must be greater than zero"
            )

        if self.waste_age_hours < 0.0:
            raise ValueError(
                "waste_age_hours cannot be negative"
            )

    @property
    def waste_mass_kg(self) -> float:
        return (
            self.full_mass_kg
            * self.fill_percent
            / 100.0
        )

    @property
    def waste_mass_tonnes(self) -> float:
        return self.waste_mass_kg / 1000.0

    @property
    def is_full(self) -> bool:
        return self.fill_percent >= 100.0

    def advance(self, elapsed_hours: float) -> None:
        """Advance fill and waste age by physical elapsed time."""

        if elapsed_hours < 0.0:
            raise ValueError(
                "elapsed_hours cannot be negative"
            )

        self.fill_percent = min(
            100.0,
            self.fill_percent
            + self.fill_rate_percent_per_hour
            * elapsed_hours,
        )

        self.waste_age_hours += elapsed_hours

    def collect(self) -> float:
        """
        Empty the bin and return the collected mass in tonnes.
        """

        collected_mass = self.waste_mass_tonnes

        self.fill_percent = 0.0
        self.waste_age_hours = 0.0
        self.collected_count += 1

        return collected_mass
