import json

from smart_waste.experiments.manifest import (
    ConfigBundleIdentity,
    ConfigFileIdentity,
    GitIdentity,
)
from smart_waste.experiments.smoke_matrix import (
    DEVELOPMENT_CLASSIFICATION,
    EXPECTED_SMOKE_PAIR_COUNT,
    build_development_smoke_snapshot,
    development_smoke_spec,
    run_development_smoke_matrix,
)


def fake_git() -> GitIdentity:
    return GitIdentity(
        commit_sha="a" * 40,
        is_dirty=False,
    )


def fake_config(
) -> ConfigBundleIdentity:
    return ConfigBundleIdentity(
        files=(
            ConfigFileIdentity(
                path="configs/test.yaml",
                sha256="b" * 64,
                size_bytes=10,
            ),
        ),
        sha256="c" * 64,
    )


def test_development_smoke_snapshot_is_deterministic() -> None:
    first = (
        build_development_smoke_snapshot()
    )

    second = (
        build_development_smoke_snapshot()
    )

    assert first.sha256 == second.sha256
    assert first.scenario_id == second.scenario_id

    assert len(
        first.bins
    ) == 10

    assert len(
        first.trucks
    ) == 2


def test_development_smoke_spec_has_expected_pair_count() -> None:
    spec = development_smoke_spec()

    actual = (
        len(
            spec.attack_rates
        )
        * len(
            spec.attack_types
        )
        * len(
            spec.replicate_ids
        )
    )

    assert actual == EXPECTED_SMOKE_PAIR_COUNT
    assert actual == 12


def test_development_smoke_writes_explicit_nonconfirmatory_evidence(
    tmp_path,
) -> None:
    manifest_dir = (
        tmp_path
        / "manifests"
    )

    raw_dir = (
        tmp_path
        / "raw"
    )

    result = (
        run_development_smoke_matrix(
            project_root=tmp_path,
            manifest_directory=(
                manifest_dir
            ),
            raw_directory=raw_dir,
            git_identity=fake_git(),
            config_bundle=fake_config(),
        )
    )

    assert (
        result.classification
        == DEVELOPMENT_CLASSIFICATION
    )

    assert (
        result.matrix.completed_pair_count
        == 12
    )

    assert (
        result.matrix.expected_pair_count
        == 12
    )

    assert len(
        list(
            manifest_dir.glob(
                "*__manifest.json"
            )
        )
    ) == 12

    assert len(
        list(
            raw_dir.glob(
                "*__results.json"
            )
        )
    ) == 12

    marker = (
        manifest_dir
        / "DEVELOPMENT_NON_CONFIRMATORY.txt"
    )

    assert marker.is_file()

    assert (
        "NON-CONFIRMATORY"
        in marker.read_text(
            encoding="utf-8"
        )
    )

    summary = json.loads(
        (
            manifest_dir
            / "DEVELOPMENT_SMOKE_SUMMARY.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert (
        summary[
            "evidence_classification"
        ]
        == DEVELOPMENT_CLASSIFICATION
    )

    assert not summary[
        "confirmatory"
    ]

    assert (
        summary[
            "matrix"
        ][
            "completed_pair_count"
        ]
        == 12
    )


def test_development_smoke_is_idempotent(
    tmp_path,
) -> None:
    kwargs = dict(
        project_root=tmp_path,
        manifest_directory=(
            tmp_path
            / "manifests"
        ),
        raw_directory=(
            tmp_path
            / "raw"
        ),
        git_identity=fake_git(),
        config_bundle=fake_config(),
    )

    first = (
        run_development_smoke_matrix(
            **kwargs
        )
    )

    second = (
        run_development_smoke_matrix(
            **kwargs
        )
    )

    assert (
        first.scenario_sha256
        == second.scenario_sha256
    )

    assert (
        first.matrix.pair_ids
        == second.matrix.pair_ids
    )

    assert (
        first.matrix.records
        == second.matrix.records
    )
