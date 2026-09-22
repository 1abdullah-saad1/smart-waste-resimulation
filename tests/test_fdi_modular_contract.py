import pytest

from smart_waste.attacks.fdi import (
    attack_count_from_rate,
    generate_fdi_attack_from_truth,
)


MASTER_SEED = 20261001


def make_noncontiguous_truth(
    count: int = 20,
) -> dict[int, float]:
    return {
        1000
        + index * 7: float(
            (index * 11) % 101
        )
        for index in range(
            count
        )
    }


def test_modular_fdi_uses_actual_noncontiguous_bin_ids() -> None:
    truth = make_noncontiguous_truth(
        20
    )

    scenario = generate_fdi_attack_from_truth(
        truth,
        master_seed=MASTER_SEED,
        attack_rate=0.25,
        attack_type=(
            "external_unauthenticated"
        ),
        replicate_id=3,
    )

    assert scenario.num_attacked_nodes == 5

    assert set(
        scenario.attacked_bin_ids
    ).issubset(
        truth
    )

    assert all(
        event.true_fill_percent
        == pytest.approx(
            truth[event.bin_id]
        )
        for event in scenario.events
    )


def test_attack_count_uses_round_half_up() -> None:
    # Explicitly distinguishes the revised contract from Python's:
    #
    # round(2.5) == 2
    #
    # Revised FDI contract:
    #
    # 2.5 -> 3
    assert (
        attack_count_from_rate(
            population_size=10,
            attack_rate=0.25,
        )
        == 3
    )

    assert (
        attack_count_from_rate(
            population_size=1000,
            attack_rate=0.15,
        )
        == 150
    )


def test_nested_severities_work_with_noncontiguous_ids() -> None:
    truth = make_noncontiguous_truth(
        100
    )

    scenarios = {
        rate: generate_fdi_attack_from_truth(
            truth,
            master_seed=MASTER_SEED,
            attack_rate=rate,
            attack_type=(
                "external_unauthenticated"
            ),
            replicate_id=7,
        )
        for rate in (
            0.10,
            0.15,
            0.20,
            0.30,
        )
    }

    ids_10 = set(
        scenarios[0.10].attacked_bin_ids
    )

    ids_15 = set(
        scenarios[0.15].attacked_bin_ids
    )

    ids_20 = set(
        scenarios[0.20].attacked_bin_ids
    )

    ids_30 = set(
        scenarios[0.30].attacked_bin_ids
    )

    assert ids_10 < ids_15
    assert ids_15 < ids_20
    assert ids_20 < ids_30

    # Same permutation seed across paired severities.
    assert len({
        scenario.selection_seed
        for scenario in scenarios.values()
    }) == 1


def test_attack_types_target_identical_bins_for_paired_comparison() -> None:
    truth = make_noncontiguous_truth(
        100
    )

    external = generate_fdi_attack_from_truth(
        truth,
        master_seed=MASTER_SEED,
        attack_rate=0.30,
        attack_type=(
            "external_unauthenticated"
        ),
        replicate_id=4,
    )

    authenticated = generate_fdi_attack_from_truth(
        truth,
        master_seed=MASTER_SEED,
        attack_rate=0.30,
        attack_type=(
            "authenticated_compromise"
        ),
        replicate_id=4,
    )

    assert (
        external.selection_seed
        == authenticated.selection_seed
    )

    assert (
        external.attacked_bin_ids
        == authenticated.attacked_bin_ids
    )


def test_attack_type_changes_security_semantics_not_targets() -> None:
    truth = make_noncontiguous_truth(
        50
    )

    external = generate_fdi_attack_from_truth(
        truth,
        master_seed=MASTER_SEED,
        attack_rate=0.20,
        attack_type=(
            "external_unauthenticated"
        ),
    )

    authenticated = generate_fdi_attack_from_truth(
        truth,
        master_seed=MASTER_SEED,
        attack_rate=0.20,
        attack_type=(
            "authenticated_compromise"
        ),
    )

    assert all(
        not event.signature_valid
        for event in external.events
    )

    assert all(
        event.signature_valid
        for event in authenticated.events
    )

    assert (
        external.poa_accepted_overrides()
        == {}
    )

    assert (
        authenticated.poa_accepted_overrides()
        == authenticated.reported_fill_overrides()
    )


def test_generator_does_not_mutate_physical_truth_mapping() -> None:
    truth = make_noncontiguous_truth(
        20
    )

    before = dict(
        truth
    )

    scenario = generate_fdi_attack_from_truth(
        truth,
        master_seed=MASTER_SEED,
        attack_rate=0.50,
        attack_type=(
            "external_unauthenticated"
        ),
    )

    assert scenario.num_attacked_nodes == 10
    assert truth == before


def test_event_truth_is_snapshotted_not_aliased() -> None:
    truth = make_noncontiguous_truth(
        20
    )

    scenario = generate_fdi_attack_from_truth(
        truth,
        master_seed=MASTER_SEED,
        attack_rate=1.0,
        attack_type=(
            "external_unauthenticated"
        ),
    )

    attacked_id = (
        scenario.attacked_bin_ids[0]
    )

    captured_truth = next(
        event.true_fill_percent
        for event in scenario.events
        if event.bin_id == attacked_id
    )

    truth[attacked_id] = 100.0

    assert (
        next(
            event.true_fill_percent
            for event in scenario.events
            if event.bin_id == attacked_id
        )
        == pytest.approx(
            captured_truth
        )
    )


def test_invalid_physical_truth_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="between 0 and 100",
    ):
        generate_fdi_attack_from_truth(
            {
                17: 101.0,
            },
            master_seed=MASTER_SEED,
            attack_rate=0.10,
            attack_type=(
                "external_unauthenticated"
            ),
        )


def test_attack_events_are_deterministically_sorted_by_real_bin_id() -> None:
    truth = make_noncontiguous_truth(
        40
    )

    scenario = generate_fdi_attack_from_truth(
        truth,
        master_seed=MASTER_SEED,
        attack_rate=0.50,
        attack_type=(
            "external_unauthenticated"
        ),
        replicate_id=9,
    )

    assert scenario.attacked_bin_ids == tuple(
        sorted(
            scenario.attacked_bin_ids
        )
    )


def test_zero_attack_preserves_population_metadata() -> None:
    truth = make_noncontiguous_truth(
        40
    )

    scenario = generate_fdi_attack_from_truth(
        truth,
        master_seed=MASTER_SEED,
        attack_rate=0.0,
        attack_type=(
            "external_unauthenticated"
        ),
    )

    assert scenario.population_size == 40
    assert scenario.num_attacked_nodes == 0
    assert scenario.realized_attack_rate == pytest.approx(
        0.0
    )
    assert scenario.reported_fill_overrides() == {}
