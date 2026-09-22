import numpy as np

from smart_waste.experiments.reproducibility import (
    create_rng,
    derive_seed,
)


def test_same_namespace_reproduces_seed() -> None:
    assert (
        derive_seed(20261001, "network")
        == derive_seed(20261001, "network")
    )


def test_namespaces_produce_different_seeds() -> None:
    assert (
        derive_seed(20261001, "network")
        != derive_seed(20261001, "workload")
    )


def test_rng_stream_is_reproducible() -> None:
    first = create_rng(
        20261001,
        "bin-placement",
    ).random(10)

    second = create_rng(
        20261001,
        "bin-placement",
    ).random(10)

    assert np.array_equal(
        first,
        second,
    )


def test_rng_streams_are_independent() -> None:
    network = create_rng(
        20261001,
        "network",
    ).random(10)

    sensors = create_rng(
        20261001,
        "sensors",
    ).random(10)

    assert not np.array_equal(
        network,
        sensors,
    )
