from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def find_project_root(start: Path | None = None) -> Path:
    """
    Find the repository root from the current path.
    A valid root is identified by README.md plus the ga_sh directory.
    """
    start = Path.cwd() if start is None else Path(start).resolve()

    for path in [start, *start.parents]:
        if (path / "README.md").exists() and (path / "ga_sh").is_dir():
            return path

    raise RuntimeError("Could not find project root.")


def load_config(config_path: str | Path) -> dict[str, Any]:
    """
    Load YAML configuration and resolve relative paths against project root.
    """
    config_path = Path(config_path).resolve()

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError(f"Empty config file: {config_path}")

    root = find_project_root(config_path.parent)
    cfg["_project_root"] = str(root)

    return resolve_paths(cfg, root)


def resolve_paths(cfg: dict[str, Any], root: Path) -> dict[str, Any]:
    """
    Convert configured relative paths into absolute paths.
    """
    path_keys = [
        ("project", "output_dir"),
        ("project", "cache_dir"),
        ("thermo", "tdb_path"),
        ("thermo", "surface_json_path"),
        ("thermo", "omega_table_path"),
        ("ml", "bulk_model_path"),
        ("ml", "surface_model_path"),
    ]

    for section, key in path_keys:
        if section not in cfg or key not in cfg[section]:
            continue

        p = Path(cfg[section][key])
        if not p.is_absolute():
            p = root / p

        cfg[section][key] = str(p.resolve())

    return cfg
