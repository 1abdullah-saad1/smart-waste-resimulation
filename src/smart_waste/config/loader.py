from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a configuration file is invalid."""


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load one YAML configuration file as a mapping."""

    config_path = Path(path)

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as stream:
        data = yaml.safe_load(stream)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ConfigError(
            f"Top-level YAML value must be a mapping: {config_path}"
        )

    return data
