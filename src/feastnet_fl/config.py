from pathlib import Path
from typing import Any, Dict

import yaml


ConfigDict = Dict[str, Any]


def load_config(config_path: str) -> ConfigDict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    project_root = path.resolve().parents[1]
    config["project_root"] = str(project_root)
    config["config_path"] = str(path.resolve())
    return config


def resolve_path(config: ConfigDict, *parts: str) -> Path:
    return Path(config["project_root"], *parts)
