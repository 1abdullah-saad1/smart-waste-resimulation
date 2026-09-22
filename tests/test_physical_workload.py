import pytest

from smart_waste.core.fill_dynamics import (
    generate_fill_rates,
)
from smart_waste.experiments.workload import (
    PhysicalWorkloadSpec,
    WorkloadError,
    generate_physical_workload,
)


def make_spec(
    *,
    replicate_id: int = 0,
) -> PhysicalWorkloadSpec:
    return PhysicalWorkloadSpec(
        num_bins=1000,
        master_seed=20261001,
        replicate_id=replicate_id,
        minimum_initial_fill_percent=0.0,
        maximum_initial_fill_percent=100.0,
        minimum_fill_rate_percent_per_hour=0.1,
        maximum_fill_rate_percent_per_hour=1.0,
    )


def test_workload_is_deterministic() -> None:
    first = generate_physical_workload(
        make_spec()
    )

    second = generate_physical_workload(
        make_spec()
    )

    assert first == second
    assert first.sha256 == second.sha256
    assert first.workload_id == second.workload_id


def test_workload_has_exact_bin_count() -> None:
    workload = generate_physical_workload(
        make_spec()
    )

    assert len(
        workload.initial_fill_percent
    ) == 1000

    assert len(
        workload.fill_rate_percent_per_hour
    ) == 1000


def test_initial_fill_respects_explicit_bounds() -> None:
    workload = generate_physical_workload(
        make_spec()
    )

    assert all(
        0.0 <= value <= 100.0
        for value in workload.initial_fill_percent
    )


def test_fill_rate_respects_explicit_bounds() -> None:
    workload = generate_physical_workload(
        make_spec()
    )

    assert all(
        0.1 <= value <= 1.0
        for value
        in workload.fill_rate_percent_per_hour
    )


def test_fill_rate_stream_matches_existing_generator() -> None:
    spec = make_spec()

    workload = generate_physical_workload(
        spec
    )

    expected = generate_fill_rates(
        num_bins=spec.num_bins,
        master_seed=spec.master_seed,
        minimum_rate_percent_per_hour=(
            spec.minimum_fill_rate_percent_per_hour
        ),
        maximum_rate_percent_per_hour=(
            spec.maximum_fill_rate_percent_per_hour
        ),
        replicate_id=spec.replicate_id,
    )

    assert (
        workload.fill_rate_percent_per_hour
        == tuple(
            float(value)
            for value in expected
        )
    )


def test_replicate_changes_workload_realization() -> None:
    first = generate_physical_workload(
        make_spec(
            replicate_id=0
        )
    )

    second = generate_physical_workload(
        make_spec(
            replicate_id=1
        )
    )

    assert (
        first.initial_fill_percent
        != second.initial_fill_percent
    )

    assert (
        first.fill_rate_percent_per_hour
        != second.fill_rate_percent_per_hour
    )

    assert first.sha256 != second.sha256


def test_workload_identity_contains_no_topology_dependency() -> None:
    workload = generate_physical_workload(
        make_spec()
    )

    assert not hasattr(
        workload.spec,
        "topology_id",
    )

    assert not hasattr(
        workload,
        "topology_id",
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "num_bins": 0,
        },
        {
            "master_seed": -1,
        },
        {
            "replicate_id": -1,
        },
        {
            "minimum_initial_fill_percent": -1.0,
        },
        {
            "maximum_initial_fill_percent": 101.0,
        },
        {
            "minimum_fill_rate_percent_per_hour": -0.1,
        },
        {
            "minimum_fill_rate_percent_per_hour": 2.0,
            "maximum_fill_rate_percent_per_hour": 1.0,
        },
    ],
)
def test_workload_rejects_invalid_spec(
    kwargs,
) -> None:
    values = {
        "num_bins": 1000,
        "master_seed": 20261001,
        "replicate_id": 0,
        "minimum_initial_fill_percent": 0.0,
        "maximum_initial_fill_percent": 100.0,
        "minimum_fill_rate_percent_per_hour": 0.1,
        "maximum_fill_rate_percent_per_hour": 1.0,
    }

    values.update(
        kwargs
    )

    with pytest.raises(
        WorkloadError,
    ):
        PhysicalWorkloadSpec(
            **values
        )
