import numpy as np

from smart_waste.core.seed_manager import create_rng


def test_same_seed_produces_same_values():
    rng1 = create_rng(12345)
    rng2 = create_rng(12345)

    values1 = rng1.random(100)
    values2 = rng2.random(100)

    assert np.array_equal(values1, values2)


def test_different_seeds_produce_different_values():
    rng1 = create_rng(12345)
    rng2 = create_rng(54321)

    values1 = rng1.random(100)
    values2 = rng2.random(100)

    assert not np.array_equal(values1, values2)