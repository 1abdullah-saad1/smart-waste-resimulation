from __future__ import annotations

import json

from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from smart_waste.attacks.fdi import (
    AttackSelectionMode,
    AttackType,
    generate_fdi_attack_from_truth,
)
from smart_waste.experiments.fdi_paired import (
    FDIRoutingRunResult,
    PairedFDIRoutingResult,
    run_paired_hdr_fdi_scenario,
)
from smart_waste.experiments.manifest import (
    ConfigBundleIdentity,
    ExperimentManifestError,
    FDIExperimentManifest,
    GitIdentity,
    build_fdi_experiment_manifest,
    write_fdi_experiment_manifest,
)
from smart_waste.experiments.scenario_snapshot import (
    PhysicalScenarioSnapshot,
)


class FDIMatrixError(RuntimeError):
    """Raised when an FDI experiment matrix is incomplete or invalid."""


@dataclass(frozen=True)
class FDIMatrixSpec:
    master_seed: int

    attack_rates: tuple[
        float,
        ...,
    ]

    attack_types: tuple[
        AttackType,
        ...,
    ]

    replicate_ids: tuple[
        int,
        ...,
    ]

    selection_mode: AttackSelectionMode = (
        "paired_nested"
    )

    forged_fill_percent: float = 100.0
    threshold_percent: float = 80.0

    service_time_seconds: float = 36.0
    max_events: int = 1_000_000

    def __post_init__(self) -> None:
        if self.master_seed < 0:
            raise ValueError(
                "master_seed must be non-negative"
            )

        rates = tuple(
            sorted(
                set(
                    float(rate)
                    for rate in self.attack_rates
                )
            )
        )

        if not rates:
            raise ValueError(
                "attack_rates cannot be empty"
            )

        for rate in rates:
            if (
                not isfinite(rate)
                or not 0.0 <= rate <= 1.0
            ):
                raise ValueError(
                    "attack rates must be finite and in [0, 1]"
                )

        attack_types = tuple(
            sorted(
                set(
                    self.attack_types
                )
            )
        )

        if not attack_types:
            raise ValueError(
                "attack_types cannot be empty"
            )

        allowed_types = {
            "external_unauthenticated",
            "authenticated_compromise",
        }

        if not set(
            attack_types
        ).issubset(
            allowed_types
        ):
            raise ValueError(
                "unsupported attack type in matrix"
            )

        replicates = tuple(
            sorted(
                set(
                    self.replicate_ids
                )
            )
        )

        if not replicates:
            raise ValueError(
                "replicate_ids cannot be empty"
            )

        if any(
            replicate < 0
            for replicate in replicates
        ):
            raise ValueError(
                "replicate IDs must be non-negative"
            )

        if self.selection_mode not in (
            "paired_nested",
            "independent_by_rate",
        ):
            raise ValueError(
                "unsupported attack selection mode"
            )

        if not (
            0.0
            <= self.forged_fill_percent
            <= 100.0
        ):
            raise ValueError(
                "forged_fill_percent must be in [0, 100]"
            )

        if not (
            0.0
            <= self.threshold_percent
            <= 100.0
        ):
            raise ValueError(
                "threshold_percent must be in [0, 100]"
            )

        if self.service_time_seconds <= 0.0:
            raise ValueError(
                "service_time_seconds must be positive"
            )

        if self.max_events <= 0:
            raise ValueError(
                "max_events must be positive"
            )

        object.__setattr__(
            self,
            "attack_rates",
            rates,
        )

        object.__setattr__(
            self,
            "attack_types",
            attack_types,
        )

        object.__setattr__(
            self,
            "replicate_ids",
            replicates,
        )


@dataclass(frozen=True)
class FDIMatrixRecord:
    pair_id: str
    manifest_sha256: str

    manifest_path: str
    raw_result_path: str


@dataclass(frozen=True)
class FDIMatrixExecutionResult:
    expected_pair_count: int
    completed_pair_count: int

    records: tuple[
        FDIMatrixRecord,
        ...,
    ]

    @property
    def is_complete(self) -> bool:
        return (
            self.completed_pair_count
            == self.expected_pair_count
        )

    @property
    def pair_ids(self) -> tuple[str, ...]:
        return tuple(
            record.pair_id
            for record in self.records
        )


def _run_result_to_dict(
    result: FDIRoutingRunResult,
    *,
    run_id: str,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "security_path": (
            result.security_path
        ),
        "scenario_id": (
            result.scenario_id
        ),
        "scenario_sha256": (
            result.scenario_sha256
        ),
        "attack_type": (
            result.attack_type
        ),
        "attack_rate": (
            result.attack_rate
        ),
        "replicate_id": (
            result.replicate_id
        ),
        "selection_seed": (
            result.selection_seed
        ),
        "attacked_bin_ids": list(
            result.attacked_bin_ids
        ),
        "accepted_forged_bin_ids": list(
            result.accepted_forged_bin_ids
        ),
        "hdr_eligible_bin_ids": list(
            result.hdr_eligible_bin_ids
        ),
        "completed_services": [
            [
                truck_id,
                bin_id,
            ]
            for truck_id, bin_id
            in result.completed_services
        ],
        "serviced_bin_ids": list(
            result.serviced_bin_ids
        ),
        "initial_false_service_alert_ids": list(
            result.initial_false_service_alert_ids
        ),
        "serviced_false_alert_bin_ids": list(
            result.serviced_false_alert_bin_ids
        ),
        "metrics": {
            "completion_time_hours": (
                result.completion_time_hours
            ),
            "processed_events": (
                result.processed_events
            ),
            "total_distance_km": (
                result.total_distance_km
            ),
            "total_fuel_used_litres": (
                result.total_fuel_used_litres
            ),
            "total_refuelled_litres": (
                result.total_refuelled_litres
            ),
            "total_depot_returns": (
                result.total_depot_returns
            ),
            "total_capacity_returns": (
                result.total_capacity_returns
            ),
            "total_fuel_returns": (
                result.total_fuel_returns
            ),
            "total_combined_returns": (
                result.total_combined_returns
            ),
        },
        "trucks": [
            {
                "truck_id": row.truck_id,
                "distance_km": (
                    row.distance_km
                ),
                "fuel_used_litres": (
                    row.fuel_used_litres
                ),
                "refuelled_litres": (
                    row.refuelled_litres
                ),
                "depot_returns": (
                    row.depot_returns
                ),
                "capacity_returns": (
                    row.capacity_returns
                ),
                "fuel_returns": (
                    row.fuel_returns
                ),
                "combined_returns": (
                    row.combined_returns
                ),
                "final_node": (
                    row.final_node
                ),
                "final_status": (
                    row.final_status
                ),
            }
            for row in result.truck_metrics
        ],
    }


def paired_result_document(
    *,
    manifest: FDIExperimentManifest,
    result: PairedFDIRoutingResult,
) -> dict[str, object]:
    """
    Build deterministic raw JSON content for one completed pair.
    """

    if (
        result.scenario_sha256
        != manifest.scenario_sha256
    ):
        raise FDIMatrixError(
            "paired result physical scenario does not match manifest"
        )

    if (
        result.selection_seed
        != manifest.selection_seed
    ):
        raise FDIMatrixError(
            "paired result attack seed does not match manifest"
        )

    if (
        result.attacked_bin_ids
        != manifest.attacked_bin_ids
    ):
        raise FDIMatrixError(
            "paired result attacked bins do not match manifest"
        )

    return {
        "schema_version": (
            "paired-fdi-raw-result-v1"
        ),
        "pair_id": (
            manifest.pair_id
        ),
        "manifest_sha256": (
            manifest.manifest_sha256
        ),
        "scenario_id": (
            manifest.scenario_id
        ),
        "scenario_sha256": (
            manifest.scenario_sha256
        ),
        "unprotected": (
            _run_result_to_dict(
                result.unprotected,
                run_id=manifest.run_id(
                    "unprotected"
                ),
            )
        ),
        "poa_verified": (
            _run_result_to_dict(
                result.poa_verified,
                run_id=manifest.run_id(
                    "poa_verified"
                ),
            )
        ),
    }


def write_paired_fdi_raw_result(
    *,
    manifest: FDIExperimentManifest,
    result: PairedFDIRoutingResult,
    output_directory: str | Path,
) -> Path:
    """
    Atomically write deterministic raw paired output.

    Existing byte-identical output is accepted.
    Conflicting output is never overwritten silently.
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
            + "__results.json"
        )
    )

    document = paired_result_document(
        manifest=manifest,
        result=result,
    )

    content = (
        json.dumps(
            document,
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

        raise FDIMatrixError(
            "refusing to overwrite conflicting raw result: "
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


def execute_fdi_matrix(
    snapshots: tuple[
        PhysicalScenarioSnapshot,
        ...,
    ],
    *,
    spec: FDIMatrixSpec,
    git_identity: GitIdentity,
    config_bundle: ConfigBundleIdentity,
    manifest_directory: str | Path,
    raw_directory: str | Path,
    require_clean_git: bool = True,
) -> FDIMatrixExecutionResult:
    """
    Execute a deterministic matrix of paired HDR/FDI experiments.

    Every pair:
    - reconstructs fresh mutable states internally,
    - receives one deterministic FDI realization,
    - writes one provenance manifest,
    - writes one raw paired-result record.

    This function does not perform statistical analysis or freeze
    results.
    """

    if not snapshots:
        raise FDIMatrixError(
            "experiment matrix requires at least one scenario"
        )

    ordered_snapshots = tuple(
        sorted(
            snapshots,
            key=lambda snapshot: (
                snapshot.scenario_id
            ),
        )
    )

    scenario_hashes = [
        snapshot.sha256
        for snapshot in ordered_snapshots
    ]

    if len(
        scenario_hashes
    ) != len(
        set(
            scenario_hashes
        )
    ):
        raise FDIMatrixError(
            "experiment matrix contains duplicate physical scenarios"
        )

    expected_pair_count = (
        len(
            ordered_snapshots
        )
        * len(
            spec.attack_rates
        )
        * len(
            spec.attack_types
        )
        * len(
            spec.replicate_ids
        )
    )

    records: list[
        FDIMatrixRecord
    ] = []

    seen_pair_ids: set[
        str
    ] = set()

    for snapshot in ordered_snapshots:
        truth = {
            row.bin_id: row.fill_percent
            for row in snapshot.bins
        }

        for replicate_id in spec.replicate_ids:
            for attack_rate in spec.attack_rates:
                for attack_type in spec.attack_types:
                    scenario = (
                        generate_fdi_attack_from_truth(
                            truth,
                            master_seed=(
                                spec.master_seed
                            ),
                            attack_rate=(
                                attack_rate
                            ),
                            attack_type=(
                                attack_type
                            ),
                            selection_mode=(
                                spec.selection_mode
                            ),
                            forged_fill_percent=(
                                spec.forged_fill_percent
                            ),
                            replicate_id=(
                                replicate_id
                            ),
                        )
                    )

                    manifest = (
                        build_fdi_experiment_manifest(
                            snapshot=snapshot,
                            scenario=scenario,
                            git_identity=(
                                git_identity
                            ),
                            config_bundle=(
                                config_bundle
                            ),
                            master_seed=(
                                spec.master_seed
                            ),
                            forged_fill_percent=(
                                spec.forged_fill_percent
                            ),
                            threshold_percent=(
                                spec.threshold_percent
                            ),
                            service_time_seconds=(
                                spec.service_time_seconds
                            ),
                            max_events=(
                                spec.max_events
                            ),
                            require_clean_git=(
                                require_clean_git
                            ),
                        )
                    )

                    if (
                        manifest.pair_id
                        in seen_pair_ids
                    ):
                        raise FDIMatrixError(
                            "duplicate matrix pair_id: "
                            f"{manifest.pair_id}"
                        )

                    result = (
                        run_paired_hdr_fdi_scenario(
                            snapshot,
                            scenario,
                            threshold_percent=(
                                spec.threshold_percent
                            ),
                            service_time_seconds=(
                                spec.service_time_seconds
                            ),
                            max_events=(
                                spec.max_events
                            ),
                        )
                    )

                    manifest_path = (
                        write_fdi_experiment_manifest(
                            manifest,
                            manifest_directory,
                        )
                    )

                    raw_path = (
                        write_paired_fdi_raw_result(
                            manifest=manifest,
                            result=result,
                            output_directory=(
                                raw_directory
                            ),
                        )
                    )

                    seen_pair_ids.add(
                        manifest.pair_id
                    )

                    records.append(
                        FDIMatrixRecord(
                            pair_id=(
                                manifest.pair_id
                            ),
                            manifest_sha256=(
                                manifest.manifest_sha256
                            ),
                            manifest_path=str(
                                manifest_path
                            ),
                            raw_result_path=str(
                                raw_path
                            ),
                        )
                    )

    if (
        len(records)
        != expected_pair_count
    ):
        raise FDIMatrixError(
            "matrix execution completed an unexpected "
            f"number of pairs: expected={expected_pair_count}, "
            f"actual={len(records)}"
        )

    if len(
        seen_pair_ids
    ) != expected_pair_count:
        raise FDIMatrixError(
            "matrix pair IDs are not unique and complete"
        )

    for record in records:
        if not Path(
            record.manifest_path
        ).is_file():
            raise FDIMatrixError(
                "matrix manifest output is missing: "
                f"{record.manifest_path}"
            )

        if not Path(
            record.raw_result_path
        ).is_file():
            raise FDIMatrixError(
                "matrix raw output is missing: "
                f"{record.raw_result_path}"
            )

    return FDIMatrixExecutionResult(
        expected_pair_count=(
            expected_pair_count
        ),
        completed_pair_count=len(
            records
        ),
        records=tuple(
            records
        ),
    )
