from pathlib import Path

import yaml


def test_dqn_frozen_configuration():
    path = Path(
        "configs/collection/dqn.yaml"
    )

    config = yaml.safe_load(
        path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        config[
            "state"
        ][
            "num_bins"
        ]
        == 1000
    )

    assert (
        config[
            "state"
        ][
            "dimension"
        ]
        == 3002
    )

    assert config[
        "network"
    ][
        "hidden_layers"
    ] == [
        256,
        128,
        64,
    ]

    training = config[
        "training"
    ]

    assert (
        training[
            "learning_rate"
        ]
        == 0.0001
    )

    assert (
        training[
            "gamma"
        ]
        == 0.95
    )

    assert (
        training[
            "replay_capacity"
        ]
        == 100000
    )

    assert (
        training[
            "batch_size"
        ]
        == 64
    )

    assert (
        training[
            "epsilon_start"
        ]
        == 1.0
    )

    assert (
        training[
            "epsilon_end"
        ]
        == 0.01
    )

    reward = config[
        "reward"
    ]

    assert reward == {
        "collection": 1.0,
        "distance": 0.5,
        "hazard_delay": 5.0,
        "sla_violation": 3.0,
    }
