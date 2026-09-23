from smart_waste.experiments.empirical_workload import (
    EXPECTED_PROFILE_LIBRARY_SHA256,
    EmpiricalWorkloadSpec,
    generate_empirical_workload,
    load_empirical_profile_library,
)


def _spec(
    replicate_id: int,
) -> EmpiricalWorkloadSpec:
    return EmpiricalWorkloadSpec(
        num_bins=1000,
        master_seed=20261002,
        replicate_id=replicate_id,
    )


def test_empirical_profile_library_identity() -> None:
    spec = _spec(
        0
    )

    frame = (
        load_empirical_profile_library(
            spec
        )
    )

    assert (
        spec.profile_library_sha256
        == EXPECTED_PROFILE_LIBRARY_SHA256
    )

    assert len(
        frame
    ) == 25441


def test_empirical_workload_has_primary_scale() -> None:
    workload = (
        generate_empirical_workload(
            _spec(
                0
            )
        )
    )

    assert (
        workload.num_bins
        == 1000
    )

    assert len(
        workload.profile_ids
    ) == 1000

    assert len(
        workload.initial_fill_percent
    ) == 1000

    assert len(
        workload.fill_rate_percent_per_hour
    ) == 1000


def test_sampling_is_without_replacement() -> None:
    workload = (
        generate_empirical_workload(
            _spec(
                0
            )
        )
    )

    assert len(
        set(
            workload.profile_ids
        )
    ) == 1000


def test_same_replicate_is_exactly_deterministic() -> None:
    first = (
        generate_empirical_workload(
            _spec(
                0
            )
        )
    )

    second = (
        generate_empirical_workload(
            _spec(
                0
            )
        )
    )

    assert (
        first
        == second
    )


def test_different_replicates_are_distinct() -> None:
    first = (
        generate_empirical_workload(
            _spec(
                0
            )
        )
    )

    second = (
        generate_empirical_workload(
            _spec(
                1
            )
        )
    )

    assert (
        first.sha256
        != second.sha256
    )

    assert (
        first.profile_ids
        != second.profile_ids
    )


def test_empirical_values_are_physically_valid() -> None:
    workload = (
        generate_empirical_workload(
            _spec(
                0
            )
        )
    )

    allowed_initial_fill = {
        0.0,
        20.0,
        40.0,
        60.0,
        80.0,
        100.0,
    }

    assert set(
        workload.initial_fill_percent
    ).issubset(
        allowed_initial_fill
    )

    assert all(
        rate >= 0.0
        for rate in (
            workload.fill_rate_percent_per_hour
        )
    )

    assert all(
        rate <= (
            100.0 / 24.0
        )
        + 1e-9
        for rate in (
            workload.fill_rate_percent_per_hour
        )
    )


def test_empirical_workload_preserves_profile_pairing() -> None:
    spec = _spec(
        0
    )

    library = (
        load_empirical_profile_library(
            spec
        )
        .set_index(
            "profile_id"
        )
    )

    workload = (
        generate_empirical_workload(
            spec
        )
    )

    for bin_id, profile_id in enumerate(
        workload.profile_ids
    ):
        row = library.loc[
            profile_id
        ]

        assert (
            workload.initial_fill_percent[
                bin_id
            ]
            == float(
                row[
                    "initial_fill_percent"
                ]
            )
        )

        assert (
            workload.fill_rate_percent_per_hour[
                bin_id
            ]
            == float(
                row[
                    "fill_rate_percent_per_hour"
                ]
            )
        )
