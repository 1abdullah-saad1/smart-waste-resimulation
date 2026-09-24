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


def test_protocol_v2_removes_validation_metric_stability_from_gate():
    from pathlib import Path

    protocol = (
        load_physical_convergence_protocol(
            Path(
                "configs/rl/"
                "dqn_physical_convergence_v2.yaml"
            )
        )
    )

    assert (
        protocol.schema_version
        == "dqn-physical-convergence-v2"
    )

    assert (
        protocol.protocol_version
        == "2.0.0"
    )

    assert (
        protocol.validation_metric_stability_required
        is False
    )

    # Core training convergence thresholds remain unchanged.
    assert (
        protocol.normalized_reward_slope_threshold
        == 0.0025
    )

    assert (
        protocol.normalized_loss_slope_threshold
        == 0.005
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
        protocol.patience_consecutive_checks
        == 3
    )


def test_protocol_v1_still_preserves_original_validation_gate():
    protocol = (
        load_physical_convergence_protocol()
    )

    assert (
        protocol.schema_version
        == "dqn-physical-convergence-v1"
    )

    assert (
        protocol.validation_metric_stability_required
        is True
    )
