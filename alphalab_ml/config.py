"""Load and expose the factor_ml.yaml configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


_DEFAULT_CONFIG = Path(__file__).parent.parent / "configs" / "factor_ml.yaml"


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Return the parsed YAML config as a plain dict.

    Parameters
    ----------
    path:
        Path to a YAML config file.  Defaults to ``configs/factor_ml.yaml``
        relative to the project root.
    """
    config_path = Path(path) if path else _DEFAULT_CONFIG
    with config_path.open("r") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)
    return cfg
