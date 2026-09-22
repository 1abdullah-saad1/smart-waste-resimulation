from __future__ import annotations

import hashlib
import json
import subprocess

from dataclasses import dataclass
from math import isclose, isfinite
from pathlib import Path
from typing import Literal

from smart_waste.attacks.fdi import (
    FDIAttackScenario,
)
from smart_waste.experiments.scenario_snapshot import (
    PhysicalScenarioSnapshot,
)


SecurityPath = Literal[
    "unprotected",
    "poa_verified",
]


class ExperimentManifestError(RuntimeError):
    """
    Raised when experiment provenance cannot be represented
    deterministically and safely.
    """


@dataclass(frozen=True)
class GitIdentity:
    commit_sha: str
    is_dirty: bool


@dataclass(frozen=True)
class ConfigFileIdentity:
    path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class ConfigBundleIdentity:
    files: tuple[
        ConfigFileIdentity,
        ...,
    ]
    sha256: str


@dataclass(frozen=True)
class FDIExperimentManifest:
    """
    Canonical identity of one paired FDI experiment.

    This object describes experimental inputs/provenance only.
    It deliberately contains no measured results.
    """

    schema_version: str
    experiment_name: str

    scenario_id: str
    scenario_sha256: str

    git_commit_sha: str
    git_dirty: bool

    config_sha256: str
    config_files: tuple[
        ConfigFileIdentity,
        ...,
    ]

    master_seed: int
    replicate_id: int

    attack_rate: float
    attack_type: str
    selection_mode: str
    selection_seed: int
    attacked_bin_ids: tuple[int, ...]

    forged_fill_percent: float
    threshold_percent: float

    service_time_seconds: float
    max_events: int

    @property
    def pair_id(self) -> str:
        return (
            "fdi"
            f"__{self.scenario_id}"
            f"__rep-{self.replicate_id:04d}"
            f"__rate-{_rate_token(self.attack_rate)}"
            f"__{self.attack_type.replace('_', '-')}"
        )

    def run_id(
        self,
        path: SecurityPath,
    ) -> str:
        _validate_security_path(
            path
        )

        return (
            f"{self.pair_id}"
            f"__{path.replace('_', '-')}"
        )

    @property
    def manifest_sha256(self) -> str:
        encoded = _canonical_json_bytes(
            self.to_dict()
        )

        return hashlib.sha256(
            encoded
        ).hexdigest()

    def to_dict(
        self,
    ) -> dict[str, object]:
        return {
            "schema_version": (
                self.schema_version
            ),
            "experiment_name": (
                self.experiment_name
            ),
            "pair_id": (
                self.pair_id
            ),
            "run_ids": {
                "unprotected": (
                    self.run_id(
                        "unprotected"
                    )
                ),
                "poa_verified": (
                    self.run_id(
                        "poa_verified"
                    )
                ),
            },
            "physical_scenario": {
                "scenario_id": (
                    self.scenario_id
                ),
                "sha256": (
                    self.scenario_sha256
                ),
            },
            "code": {
                "git_commit_sha": (
                    self.git_commit_sha
                ),
                "git_dirty": (
                    self.git_dirty
                ),
            },
            "configuration": {
                "sha256": (
                    self.config_sha256
                ),
                "files": [
                    {
                        "path": row.path,
                        "sha256": (
                            row.sha256
                        ),
                        "size_bytes": (
                            row.size_bytes
                        ),
                    }
                    for row in self.config_files
                ],
            },
            "randomness": {
                "master_seed": (
                    self.master_seed
                ),
                "replicate_id": (
                    self.replicate_id
                ),
                "selection_seed": (
                    self.selection_seed
                ),
            },
            "attack": {
                "attack_rate": (
                    self.attack_rate
                ),
                "attack_type": (
                    self.attack_type
                ),
                "selection_mode": (
                    self.selection_mode
                ),
                "forged_fill_percent": (
                    self.forged_fill_percent
                ),
                "attacked_bin_ids": list(
                    self.attacked_bin_ids
                ),
            },
            "routing": {
                "policy": (
                    "heuristic_dynamic_routing"
                ),
                "threshold_percent": (
                    self.threshold_percent
                ),
                "security_paths": [
                    "unprotected",
                    "poa_verified",
                ],
            },
            "simulation": {
                "service_time_seconds": (
                    self.service_time_seconds
                ),
                "max_events": (
                    self.max_events
                ),
            },
        }


def default_fdi_config_paths() -> tuple[
    str,
    ...,
]:
    """
    Configuration files capable of affecting the current HDR/FDI
    physical-routing experiment.

    TSR configuration is intentionally excluded because TSR is not
    executed by this experiment.
    """

    return (
        "configs/core/bins.yaml",
        "configs/core/city.yaml",
        "configs/core/depot.yaml",
        "configs/core/fuel.yaml",
        "configs/core/timing.yaml",
        "configs/core/trucks.yaml",
        "configs/movement/road_network.yaml",
        "configs/collection/hdr.yaml",
        "configs/experiments/fdi.yaml",
    )


def _sha256_bytes(
    content: bytes,
) -> str:
    return hashlib.sha256(
        content
    ).hexdigest()


def _canonical_json_bytes(
    value: object,
) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        ensure_ascii=False,
        allow_nan=False,
    ).encode(
        "utf-8"
    )


def _rate_token(
    attack_rate: float,
) -> str:
    value = _validated_rate(
        attack_rate
    )

    text = (
        f"{value:.6f}"
        .rstrip("0")
        .rstrip(".")
    )

    if "." not in text:
        text = (
            text
            + ".0"
        )

    return text.replace(
        ".",
        "p",
    )


def _validated_rate(
    attack_rate: float,
) -> float:
    try:
        value = float(
            attack_rate
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ExperimentManifestError(
            "attack_rate must be numeric"
        ) from exc

    if not isfinite(
        value
    ):
        raise ExperimentManifestError(
            "attack_rate must be finite"
        )

    if not (
        0.0
        <= value
        <= 1.0
    ):
        raise ExperimentManifestError(
            "attack_rate must be between 0 and 1"
        )

    return value


def _validated_percentage(
    value: float,
    *,
    name: str,
) -> float:
    try:
        numeric = float(
            value
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ExperimentManifestError(
            f"{name} must be numeric"
        ) from exc

    if not isfinite(
        numeric
    ):
        raise ExperimentManifestError(
            f"{name} must be finite"
        )

    if not (
        0.0
        <= numeric
        <= 100.0
    ):
        raise ExperimentManifestError(
            f"{name} must be between 0 and 100"
        )

    return numeric


def _validate_security_path(
    path: str,
) -> None:
    if path not in (
        "unprotected",
        "poa_verified",
    ):
        raise ExperimentManifestError(
            f"unsupported security path: {path}"
        )


def discover_git_identity(
    project_root: str | Path,
) -> GitIdentity:
    """
    Resolve exact Git commit and working-tree cleanliness.

    Final experiment execution should require is_dirty == False.
    """

    root = Path(
        project_root
    ).resolve()

    try:
        commit_process = subprocess.run(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )

        status_process = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (
        OSError,
        subprocess.CalledProcessError,
    ) as exc:
        raise ExperimentManifestError(
            "unable to resolve Git provenance"
        ) from exc

    commit_sha = (
        commit_process.stdout.strip()
    )

    if (
        len(commit_sha) != 40
        or any(
            character
            not in "0123456789abcdef"
            for character in commit_sha.lower()
        )
    ):
        raise ExperimentManifestError(
            "git rev-parse returned an invalid commit SHA"
        )

    return GitIdentity(
        commit_sha=commit_sha,
        is_dirty=bool(
            status_process.stdout.strip()
        ),
    )


def build_config_bundle_identity(
    project_root: str | Path,
    *,
    relative_paths: tuple[
        str,
        ...,
    ] | None = None,
) -> ConfigBundleIdentity:
    """
    Hash exact configuration bytes plus stable relative paths.

    The aggregate hash changes when:
    - any included file contents change,
    - an included path changes,
    - the set of included config files changes.
    """

    root = Path(
        project_root
    ).resolve()

    requested = (
        default_fdi_config_paths()
        if relative_paths is None
        else relative_paths
    )

    normalized_paths = tuple(
        sorted(
            set(
                requested
            )
        )
    )

    if not normalized_paths:
        raise ExperimentManifestError(
            "configuration bundle cannot be empty"
        )

    identities: list[
        ConfigFileIdentity
    ] = []

    aggregate = hashlib.sha256()

    for relative_path in normalized_paths:
        candidate = Path(
            relative_path
        )

        if candidate.is_absolute():
            raise ExperimentManifestError(
                "configuration paths must be relative"
            )

        resolved = (
            root
            / candidate
        ).resolve()

        try:
            resolved.relative_to(
                root
            )
        except ValueError as exc:
            raise ExperimentManifestError(
                "configuration path escapes project root: "
                f"{relative_path}"
            ) from exc

        if not resolved.is_file():
            raise ExperimentManifestError(
                "configuration file not found: "
                f"{relative_path}"
            )

        content = resolved.read_bytes()

        file_sha256 = (
            _sha256_bytes(
                content
            )
        )

        canonical_path = (
            candidate.as_posix()
        )

        identities.append(
            ConfigFileIdentity(
                path=canonical_path,
                sha256=file_sha256,
                size_bytes=len(
                    content
                ),
            )
        )

        aggregate.update(
            canonical_path.encode(
                "utf-8"
            )
        )
        aggregate.update(
            b"\0"
        )
        aggregate.update(
            file_sha256.encode(
                "ascii"
            )
        )
        aggregate.update(
            b"\n"
        )

    return ConfigBundleIdentity(
        files=tuple(
            identities
        ),
        sha256=aggregate.hexdigest(),
    )


def build_fdi_experiment_manifest(
    *,
    snapshot: PhysicalScenarioSnapshot,
    scenario: FDIAttackScenario,
    git_identity: GitIdentity,
    config_bundle: ConfigBundleIdentity,
    master_seed: int,
    forged_fill_percent: float = 100.0,
    threshold_percent: float = 80.0,
    service_time_seconds: float = 36.0,
    max_events: int = 1_000_000,
    require_clean_git: bool = True,
) -> FDIExperimentManifest:
    """
    Build canonical provenance for one paired FDI experiment.
    """

    if master_seed < 0:
        raise ExperimentManifestError(
            "master_seed must be non-negative"
        )

    if require_clean_git and git_identity.is_dirty:
        raise ExperimentManifestError(
            "experiment manifest requires a clean Git working tree"
        )

    if service_time_seconds <= 0.0:
        raise ExperimentManifestError(
            "service_time_seconds must be positive"
        )

    if max_events <= 0:
        raise ExperimentManifestError(
            "max_events must be positive"
        )

    rate = _validated_rate(
        scenario.attack_rate
    )

    forged_fill = (
        _validated_percentage(
            forged_fill_percent,
            name="forged_fill_percent",
        )
    )

    threshold = (
        _validated_percentage(
            threshold_percent,
            name="threshold_percent",
        )
    )

    snapshot_truth = {
        row.bin_id: row.fill_percent
        for row in snapshot.bins
    }

    if (
        scenario.population_size
        != len(
            snapshot_truth
        )
    ):
        raise ExperimentManifestError(
            "attack population does not match physical scenario"
        )

    attacked_ids = (
        scenario.attacked_bin_ids
    )

    if len(
        attacked_ids
    ) != len(
        set(
            attacked_ids
        )
    ):
        raise ExperimentManifestError(
            "attack scenario contains duplicate bin IDs"
        )

    unknown = (
        set(
            attacked_ids
        )
        - set(
            snapshot_truth
        )
    )

    if unknown:
        raise ExperimentManifestError(
            "attack scenario references unknown physical bins: "
            f"{sorted(unknown)}"
        )

    for event in scenario.events:
        true_fill = (
            snapshot_truth[
                event.bin_id
            ]
        )

        if not isclose(
            event.true_fill_percent,
            true_fill,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ExperimentManifestError(
                "attack true-fill snapshot does not match "
                f"physical scenario for bin {event.bin_id}"
            )

        if not isclose(
            event.forged_fill_percent,
            forged_fill,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ExperimentManifestError(
                "attack forged-fill value does not match "
                "manifest configuration"
            )

    return FDIExperimentManifest(
        schema_version=(
            "fdi-experiment-manifest-v1"
        ),
        experiment_name=(
            "paired_hdr_false_data_injection"
        ),
        scenario_id=(
            snapshot.scenario_id
        ),
        scenario_sha256=(
            snapshot.sha256
        ),
        git_commit_sha=(
            git_identity.commit_sha
        ),
        git_dirty=(
            git_identity.is_dirty
        ),
        config_sha256=(
            config_bundle.sha256
        ),
        config_files=(
            config_bundle.files
        ),
        master_seed=master_seed,
        replicate_id=(
            scenario.replicate_id
        ),
        attack_rate=rate,
        attack_type=(
            scenario.attack_type
        ),
        selection_mode=(
            scenario.selection_mode
        ),
        selection_seed=(
            scenario.selection_seed
        ),
        attacked_bin_ids=tuple(
            scenario.attacked_bin_ids
        ),
        forged_fill_percent=(
            forged_fill
        ),
        threshold_percent=(
            threshold
        ),
        service_time_seconds=float(
            service_time_seconds
        ),
        max_events=int(
            max_events
        ),
    )


def write_fdi_experiment_manifest(
    manifest: FDIExperimentManifest,
    output_directory: str | Path,
) -> Path:
    """
    Write a deterministic JSON manifest without silent overwrite.

    Existing identical content is accepted idempotently.
    Existing conflicting content raises an error.
    """

    directory = Path(
        output_directory
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        directory
        / (
            manifest.pair_id
            + "__manifest.json"
        )
    )

    payload = dict(
        manifest.to_dict()
    )

    payload[
        "manifest_sha256"
    ] = (
        manifest.manifest_sha256
    )

    content = (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode(
        "utf-8"
    )

    if output_path.exists():
        existing = (
            output_path.read_bytes()
        )

        if existing == content:
            return output_path

        raise ExperimentManifestError(
            "refusing to overwrite conflicting manifest: "
            f"{output_path}"
        )

    temporary_path = (
        output_path.with_suffix(
            output_path.suffix
            + ".tmp"
        )
    )

    temporary_path.write_bytes(
        content
    )

    temporary_path.replace(
        output_path
    )

    return output_path
