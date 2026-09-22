from pathlib import Path

import pytest

from smart_waste.config.loader import (
    ConfigError,
    load_yaml,
)


def test_load_yaml_mapping(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "value: 42\n",
        encoding="utf-8",
    )

    assert load_yaml(path) == {"value": 42}


def test_load_yaml_rejects_non_mapping(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "- one\n- two\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError):
        load_yaml(path)


def test_load_yaml_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_yaml("does-not-exist.yaml")
