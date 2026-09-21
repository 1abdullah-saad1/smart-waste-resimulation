from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WasteBin:
    """
    Representation of one simulated municipal waste bin.
    """

    bin_id: int
    x_km: float
    y_km: float
    fill_percent: float

    def __post_init__(self) -> None:
        if self.bin_id < 0:
            raise ValueError("bin_id must be non-negative")

        if self.x_km < 0:
            raise ValueError("x_km must be non-negative")

        if self.y_km < 0:
            raise ValueError("y_km must be non-negative")

        if not 0.0 <= self.fill_percent <= 100.0:
            raise ValueError(
                "fill_percent must be between 0 and 100"
            )

    def requires_service(
        self,
        threshold_percent: float = 80.0,
    ) -> bool:
        return self.fill_percent >= threshold_percent

    def is_overflowing(self) -> bool:
        return self.fill_percent >= 100.0

    @property
    def location(self) -> tuple[float, float]:
        return self.x_km, self.y_km

    def waste_mass_kg(
        self,
        *,
        bin_volume_m3: float,
        waste_density_kg_per_m3: float,
    ) -> float:
        """
        Convert fill percentage into estimated waste mass.

        mass = volume × density × fill_fraction
        """

        if bin_volume_m3 <= 0:
            raise ValueError(
                "bin_volume_m3 must be greater than zero"
            )

        if waste_density_kg_per_m3 <= 0:
            raise ValueError(
                "waste_density_kg_per_m3 must be greater than zero"
            )

        fill_fraction = self.fill_percent / 100.0

        return (
            bin_volume_m3
            * waste_density_kg_per_m3
            * fill_fraction
        )

    def waste_mass_tonnes(
        self,
        *,
        bin_volume_m3: float,
        waste_density_kg_per_m3: float,
    ) -> float:
        return self.waste_mass_kg(
            bin_volume_m3=bin_volume_m3,
            waste_density_kg_per_m3=(
                waste_density_kg_per_m3
            ),
        ) / 1000.0