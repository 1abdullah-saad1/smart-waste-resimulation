#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

from smart_waste.experiments.smoke_matrix import (
    run_development_smoke_matrix,
)


def main() -> None:
    project_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    result = (
        run_development_smoke_matrix(
            project_root=project_root
        )
    )

    print(
        "DEVELOPMENT / NON-CONFIRMATORY "
        "FDI SMOKE MATRIX"
    )

    print(
        f"scenario_id: {result.scenario_id}"
    )

    print(
        "scenario_sha256: "
        f"{result.scenario_sha256}"
    )

    print(
        "git_commit_sha: "
        f"{result.git_commit_sha}"
    )

    print(
        "config_sha256: "
        f"{result.config_sha256}"
    )

    print(
        "pairs: "
        f"{result.matrix.completed_pair_count}"
        "/"
        f"{result.matrix.expected_pair_count}"
    )

    print(
        f"summary: {result.summary_path}"
    )

    print(
        f"marker: {result.marker_path}"
    )


if __name__ == "__main__":
    main()
