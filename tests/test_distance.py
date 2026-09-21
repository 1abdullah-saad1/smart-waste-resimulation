import numpy as np
import pytest

from smart_waste.core.city import City
from smart_waste.core.distance import (
    build_distance_matrix,
    euclidean_distance,
)


def make_city():
    return City.generate(
        area_km2=50.0,
        num_bins=1000,
        master_seed=20260922,
    )


def test_simple_euclidean_distance():
    distance = euclidean_distance(
        (0.0, 0.0),
        (3.0, 4.0),
    )

    assert distance == pytest.approx(5.0)


def test_distance_matrix_shape():
    city = make_city()

    matrix = build_distance_matrix(city)

    assert matrix.shape == (1001, 1001)


def test_distance_matrix_diagonal_is_zero():
    city = make_city()

    matrix = build_distance_matrix(city)

    assert np.allclose(
        np.diag(matrix),
        0.0,
    )


def test_distance_matrix_is_symmetric():
    city = make_city()

    matrix = build_distance_matrix(city)

    assert np.allclose(
        matrix,
        matrix.T,
    )


def test_distance_matrix_nonnegative():
    city = make_city()

    matrix = build_distance_matrix(city)

    assert np.all(matrix >= 0.0)


def test_depot_to_first_bin_distance():
    city = make_city()

    matrix = build_distance_matrix(city)

    expected = euclidean_distance(
        city.depot_location,
        city.bins[0].location,
    )

    assert matrix[0, 1] == pytest.approx(expected)