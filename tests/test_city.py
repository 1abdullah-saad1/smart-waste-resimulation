import numpy as np
import pytest

from smart_waste.core.city import City


MASTER_SEED = 20260922


def make_city(seed: int = MASTER_SEED) -> City:
    return City.generate(
        area_km2=50.0,
        num_bins=1000,
        master_seed=seed,
        minimum_fill_percent=0.0,
        maximum_fill_percent=100.0,
    )


def test_city_contains_exactly_1000_bins():
    city = make_city()

    assert len(city.bins) == 1000


def test_city_area_is_50_km2():
    city = make_city()

    assert city.area_km2 == pytest.approx(50.0)

    reconstructed_area = city.side_length_km ** 2

    assert reconstructed_area == pytest.approx(50.0)


def test_depot_is_at_city_center():
    city = make_city()

    expected = city.side_length_km / 2.0

    assert city.depot_x_km == pytest.approx(expected)
    assert city.depot_y_km == pytest.approx(expected)


def test_all_bins_are_inside_city_boundary():
    city = make_city()
    coordinates = city.coordinate_array()

    assert np.all(coordinates >= 0.0)
    assert np.all(
        coordinates <= city.side_length_km
    )


def test_all_fill_levels_are_valid():
    city = make_city()
    fills = city.fill_array()

    assert np.all(fills >= 0.0)
    assert np.all(fills <= 100.0)


def test_same_seed_reproduces_identical_city():
    city_a = make_city(20260922)
    city_b = make_city(20260922)

    assert np.array_equal(
        city_a.coordinate_array(),
        city_b.coordinate_array(),
    )

    assert np.array_equal(
        city_a.fill_array(),
        city_b.fill_array(),
    )


def test_different_seed_changes_city():
    city_a = make_city(20260922)
    city_b = make_city(20260923)

    assert not np.array_equal(
        city_a.coordinate_array(),
        city_b.coordinate_array(),
    )


def test_bin_ids_are_unique_and_sequential():
    city = make_city()

    ids = [bin_.bin_id for bin_ in city.bins]

    assert ids == list(range(1000))