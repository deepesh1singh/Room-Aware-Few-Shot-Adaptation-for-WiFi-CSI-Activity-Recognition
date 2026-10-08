"""YAML configuration loading, dotted overrides, and workspace path resolution."""
from __future__ import annotations

from pathlib import Path

import yaml


def cget(cfg: dict, dotted: str, default=None):
    """Read a nested config value with a dotted key, e.g. cget(cfg, 'data.window_length', 150)."""
    cur = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _set_dotted(cfg: dict, dotted: str, value) -> None:
    parts = dotted.split(".")
    cur = cfg
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


def parse_overrides(items) -> dict:
    """['ssl.epochs=5', 'device=cpu'] -> {'ssl.epochs': 5, 'device': 'cpu'} (values parsed as YAML)."""
    out = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got: {item!r}")
        key, value = item.split("=", 1)
        out[key.strip()] = yaml.safe_load(value)
    return out


def load_config(path, overrides: dict | None = None) -> dict:
    path = Path(path).expanduser().resolve()
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    for key, value in (overrides or {}).items():
        _set_dotted(cfg, key, value)
    repo_root = path.parent.parent if path.parent.name == "configs" else path.parent
    cfg["_repo_root"] = str(repo_root)
    cfg["_config_path"] = str(path)

    def _resolve(p) -> str:
        p = Path(str(p)).expanduser()
        return str((p if p.is_absolute() else repo_root / p).resolve())

    cfg["_raw_root"] = _resolve(cget(cfg, "paths.raw_root", "data/raw"))
    cfg["_work_dir"] = _resolve(cget(cfg, "paths.work_dir", "workspace"))
    return cfg


def workspace_paths(cfg: dict, create: bool = True) -> dict:
    """All derived-data locations. Raw data is never written to."""
    root = Path(cfg["_work_dir"])
    ws = {
        "root": root,
        "intermediate": root / "intermediate_csv",
        "windows_dev": root / "windows" / "development",
        "windows_val": root / "windows" / "local_validation",
        "manifests": root / "manifests",
        "artifacts": root / "artifacts",
        "results": root / "results",
        "figures": root / "figures",
        "logs": root / "logs",
    }
    if create:
        for p in ws.values():
            p.mkdir(parents=True, exist_ok=True)
    return ws


def preprocessing_signature(cfg: dict) -> dict:
    """The config sections that determine the content of the derived windows (hashed for provenance)."""
    keys = ["data", "room_a", "room_b", "qc", "validation_split", "normalization"]
    return {k: cfg.get(k) for k in keys}
