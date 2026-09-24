from smart_waste.rl.physical_convergence_protocol import (
    load_physical_convergence_protocol,
)


def test_frozen_physical_convergence_protocol():
    protocol = (
        load_physical_convergence_protocol()
    )

    assert (
        protocol.minimum_episodes
        == 192
    )

    assert (
        protocol.maximum_episodes
        == 640
    )

    assert (
        protocol.curriculum_period_episodes
        == 32
    )

    assert (
        protocol.validation_interval_episodes
        == 32
    )

    assert (
        protocol.training_window_episodes
        == 64
    )

    assert (
        protocol.patience_consecutive_checks
        == 3
    )

    assert (
        protocol.dataset_manifest_sha256
        == (
            "18bbf059305886970d3bcfc363a2140"
            "a4fe32d837901bd3ccd67af8d24384133"
        )
    )


def test_convergence_window_is_curriculum_balanced():
    protocol = (
        load_physical_convergence_protocol()
    )

    assert (
        protocol.training_window_episodes
        % protocol.curriculum_period_episodes
        == 0
    )


def test_validation_occurs_after_complete_curriculum_cycles():
    protocol = (
        load_physical_convergence_protocol()
    )

    assert (
        protocol.validation_interval_episodes
        == protocol.curriculum_period_episodes
    )
