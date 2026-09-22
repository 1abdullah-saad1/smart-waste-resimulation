import pytest

from smart_waste.attacks.fdi import (
    generate_fdi_attack,
)
from smart_waste.core.city import City


MASTER_SEED = 20260922


def make_city():
    return City.generate(
        area_km2=50.0,
        num_bins=1000,
        master_seed=MASTER_SEED,
    )


def test_zero_attack_selects_no_nodes():
    scenario = generate_fdi_attack(
        make_city(),
        master_seed=MASTER_SEED,
        attack_rate=0.0,
        attack_type="external_unauthenticated",
    )

    assert scenario.num_attacked_nodes == 0


def test_fifteen_percent_attacks_exactly_150_nodes():
    scenario = generate_fdi_attack(
        make_city(),
        master_seed=MASTER_SEED,
        attack_rate=0.15,
        attack_type="external_unauthenticated",
    )

    assert scenario.num_attacked_nodes == 150
    assert len(set(scenario.attacked_bin_ids)) == 150


def test_attack_is_reproducible():
    city = make_city()

    a = generate_fdi_attack(
        city,
        master_seed=MASTER_SEED,
        attack_rate=0.15,
        attack_type="external_unauthenticated",
    )

    b = generate_fdi_attack(
        city,
        master_seed=MASTER_SEED,
        attack_rate=0.15,
        attack_type="external_unauthenticated",
    )

    assert a.attacked_bin_ids == b.attacked_bin_ids


def test_different_master_seed_changes_attack():
    city = make_city()

    a = generate_fdi_attack(
        city,
        master_seed=20260922,
        attack_rate=0.15,
        attack_type="external_unauthenticated",
    )

    b = generate_fdi_attack(
        city,
        master_seed=20260923,
        attack_rate=0.15,
        attack_type="external_unauthenticated",
    )

    assert a.attacked_bin_ids != b.attacked_bin_ids


def test_all_forged_values_are_100_percent():
    scenario = generate_fdi_attack(
        make_city(),
        master_seed=MASTER_SEED,
        attack_rate=0.15,
        attack_type="external_unauthenticated",
    )

    assert all(
        value == 100.0
        for value
        in scenario.reported_fill_overrides().values()
    )


def test_external_forgery_has_invalid_signatures():
    scenario = generate_fdi_attack(
        make_city(),
        master_seed=MASTER_SEED,
        attack_rate=0.15,
        attack_type="external_unauthenticated",
    )

    assert all(
        not event.signature_valid
        for event in scenario.events
    )


def test_poa_rejects_all_external_forged_overrides():
    scenario = generate_fdi_attack(
        make_city(),
        master_seed=MASTER_SEED,
        attack_rate=0.15,
        attack_type="external_unauthenticated",
    )

    assert scenario.poa_accepted_overrides() == {}


def test_authenticated_compromise_survives_poa():
    scenario = generate_fdi_attack(
        make_city(),
        master_seed=MASTER_SEED,
        attack_rate=0.15,
        attack_type="authenticated_compromise",
    )

    assert len(
        scenario.poa_accepted_overrides()
    ) == 150


def test_false_service_alert_count_is_correct():
    city = make_city()

    scenario = generate_fdi_attack(
        city,
        master_seed=MASTER_SEED,
        attack_rate=0.15,
        attack_type="external_unauthenticated",
        threshold_percent=80.0,
    )

    expected = {
        event.bin_id
        for event in scenario.events
        if event.true_fill_percent < 80.0
    }

    actual = set(
        scenario.false_service_alert_ids(
            80.0
        )
    )

    assert actual == expected


def test_invalid_attack_rate_rejected():
    with pytest.raises(ValueError):
        generate_fdi_attack(
            make_city(),
            master_seed=MASTER_SEED,
            attack_rate=1.5,
            attack_type="external_unauthenticated",
        )


def test_attack_severities_are_nested_within_replicate():
    city = make_city()

    attack_10 = generate_fdi_attack(
        city,
        master_seed=MASTER_SEED,
        attack_rate=0.10,
        attack_type="external_unauthenticated",
        replicate_id=7,
    )

    attack_15 = generate_fdi_attack(
        city,
        master_seed=MASTER_SEED,
        attack_rate=0.15,
        attack_type="external_unauthenticated",
        replicate_id=7,
    )

    attack_20 = generate_fdi_attack(
        city,
        master_seed=MASTER_SEED,
        attack_rate=0.20,
        attack_type="external_unauthenticated",
        replicate_id=7,
    )

    attack_30 = generate_fdi_attack(
        city,
        master_seed=MASTER_SEED,
        attack_rate=0.30,
        attack_type="external_unauthenticated",
        replicate_id=7,
    )

    ids_10 = set(attack_10.attacked_bin_ids)
    ids_15 = set(attack_15.attacked_bin_ids)
    ids_20 = set(attack_20.attacked_bin_ids)
    ids_30 = set(attack_30.attacked_bin_ids)

    assert ids_10 < ids_15
    assert ids_15 < ids_20
    assert ids_20 < ids_30


def test_different_replicates_change_attack_selection():
    city = make_city()

    a = generate_fdi_attack(
        city,
        master_seed=MASTER_SEED,
        attack_rate=0.15,
        attack_type="external_unauthenticated",
        replicate_id=1,
    )

    b = generate_fdi_attack(
        city,
        master_seed=MASTER_SEED,
        attack_rate=0.15,
        attack_type="external_unauthenticated",
        replicate_id=2,
    )

    assert a.selection_seed != b.selection_seed
    assert a.attacked_bin_ids != b.attacked_bin_ids

def test_independent_by_rate_uses_different_seeds():
    city = make_city()

    attack_10 = generate_fdi_attack(
        city,
        master_seed=MASTER_SEED,
        attack_rate=0.10,
        attack_type="external_unauthenticated",
        replicate_id=5,
        selection_mode="independent_by_rate",
    )

    attack_15 = generate_fdi_attack(
        city,
        master_seed=MASTER_SEED,
        attack_rate=0.15,
        attack_type="external_unauthenticated",
        replicate_id=5,
        selection_mode="independent_by_rate",
    )

    assert (
        attack_10.selection_seed
        != attack_15.selection_seed
    )