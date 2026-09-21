from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np

from smart_waste.core.bin import WasteBin
from smart_waste.core.seed_manager import (
    create_rng,
    derive_seed,
)


@dataclass(frozen=True)
class City:
    """
    Deterministically generated synthetic city used by the
    smart-waste resimulation.

    The manuscript specifies total area but does not specify
    its geometry. The resimulation therefore uses a square
    region whose side length is sqrt(area_km2).
    """

    area_km2: float
    bins: tuple[WasteBin, ...]
    depot_x_km: float
    depot_y_km: float
    master_seed: int

    @property
    def side_length_km(self) -> float:
        return sqrt(self.area_km2)

    @property
    def depot_location(self) -> tuple[float, float]:
        return self.depot_x_km, self.depot_y_km

    @classmethod
    def generate(
        cls,
        *,
        area_km2: float,
        num_bins: int,
        master_seed: int,
        minimum_fill_percent: float = 0.0,
        maximum_fill_percent: float = 100.0,
    ) -> "City":
        if area_km2 <= 0:
            raise ValueError("area_km2 must be greater than zero")

        if num_bins <= 0:
            raise ValueError("num_bins must be greater than zero")

        if not (
            0.0
            <= minimum_fill_percent
            <= maximum_fill_percent
            <= 100.0
        ):
            raise ValueError(
                "fill range must satisfy "
                "0 <= minimum <= maximum <= 100"
            )

        side_length = sqrt(area_km2)

        location_seed = derive_seed(
            master_seed,
            "city-bin-locations",
        )

        fill_seed = derive_seed(
            master_seed,
            "city-bin-fill-levels",
        )

        location_rng = create_rng(location_seed)
        fill_rng = create_rng(fill_seed)

        coordinates = location_rng.uniform(
            low=0.0,
            high=side_length,
            size=(num_bins, 2),
        )

        fill_levels = fill_rng.uniform(
            low=minimum_fill_percent,
            high=maximum_fill_percent,
            size=num_bins,
        )

        bins = tuple(
            WasteBin(
                bin_id=index,
                x_km=float(coordinates[index, 0]),
                y_km=float(coordinates[index, 1]),
                fill_percent=float(fill_levels[index]),
            )
            for index in range(num_bins)
        )

        depot_coordinate = side_length / 2.0

        return cls(
            area_km2=area_km2,
            bins=bins,
            depot_x_km=depot_coordinate,
            depot_y_km=depot_coordinate,
            master_seed=master_seed,
        )

    def coordinate_array(self) -> np.ndarray:
        """
        Return bin coordinates as an N x 2 NumPy array.
        """

        return np.array(
            [
                [bin_.x_km, bin_.y_km]
                for bin_ in self.bins
            ],
            dtype=np.float64,
        )

    def fill_array(self) -> np.ndarray:
        """
        Return bin fill levels in bin-ID order.
        """

        return np.array(
            [
                bin_.fill_percent
                for bin_ in self.bins
            ],
            dtype=np.float64,
        )

    def service_required_bins(
        self,
        threshold_percent: float = 80.0,
    ) -> tuple[WasteBin, ...]:
        return tuple(
            bin_
            for bin_ in self.bins
            if bin_.requires_service(threshold_percent)
        )