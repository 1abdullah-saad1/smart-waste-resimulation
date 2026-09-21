from __future__ import annotations

import numpy as np

from smart_waste.core.city import City


def euclidean_distance(
    point_a: tuple[float, float],
    point_b: tuple[float, float],
) -> float:
    ax, ay = point_a
    bx, by = point_b

    return float(np.hypot(ax - bx, ay - by))


def build_distance_matrix(city: City) -> np.ndarray:
    """
    Build a symmetric Euclidean distance matrix.

    Index 0 represents the depot.
    Indices 1..N represent bins 0..N-1.

    For 1000 bins the matrix has shape (1001, 1001).
    """

    bin_coordinates = city.coordinate_array()

    depot = np.array(
        [[city.depot_x_km, city.depot_y_km]],
        dtype=np.float64,
    )

    locations = np.vstack((depot, bin_coordinates))

    delta = locations[:, np.newaxis, :] - locations[np.newaxis, :, :]

    distances = np.sqrt(
        np.sum(delta**2, axis=2)
    )

    return distances