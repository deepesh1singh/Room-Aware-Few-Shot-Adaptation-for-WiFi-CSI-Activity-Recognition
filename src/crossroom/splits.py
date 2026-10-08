"""Development / validation structure.

Room A  : the supplied csidata/validation/<class>/ captures (25 per class) are the validation set.
Room B  : Experiment-3 ships no validation folder, so 25 captures per activity are drawn at capture
          level with the public seed (default 42) from the *sorted* file names of each activity.
          The result is written once to manifests/room_B_validation_split.csv and re-used on every
          later run (the validation set must never be re-sampled during the project).

The challenge text does not say which random generator the teaching team used, so the backend is
configurable (validation_split.rng_backend): python_random (default) | numpy_randomstate | numpy_generator.
If a reference split is announced, only _sample_names() needs to change.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

from .config import cget


def _make_rng(backend: str, seed: int):
    if backend == "python_random":
        return random.Random(seed)
    if backend == "numpy_randomstate":
        return np.random.RandomState(seed)
    if backend == "numpy_generator":
        return np.random.default_rng(seed)
    raise ValueError(f"Unknown validation_split.rng_backend: {backend!r}")


def _sample_names(names: list, n: int, backend: str, rng) -> list:
    if backend == "python_random":
        return list(rng.sample(names, n))
    idx = rng.choice(len(names), size=n, replace=False)
    return [names[int(i)] for i in idx]


def assign_room_b_split(caps_b: list, cfg: dict, manifest_path, logger=None) -> dict:
    """Return {file_name: 'dev'|'val'} for every Room-B capture (creating/reading the split manifest)."""
    manifest_path = Path(manifest_path)
    n_val = int(cget(cfg, "validation_split.per_activity", 25))
    seed = int(cget(cfg, "validation_split.seed", 42))
    backend = str(cget(cfg, "validation_split.rng_backend", "python_random"))
    reseed = bool(cget(cfg, "validation_split.reseed_per_activity", True))

    names_now = {c["file_name"] for c in caps_b}
    if manifest_path.exists():
        df = pd.read_csv(manifest_path)
        mapping = dict(zip(df["file_name"], df["split"]))
        if set(mapping) != names_now:
            raise RuntimeError(
                f"{manifest_path} does not match the Room-B captures found on disk. The validation split is "
                "created once and must not change; if the raw data really changed, delete that manifest "
                "(and everything derived from it) to rebuild.")
        per_label = df[df["split"] == "val"].groupby("label").size()
        if not (per_label == n_val).all():
            raise RuntimeError(f"{manifest_path} does not hold {n_val} validation captures per activity.")
        if logger:
            logger.info("Re-using existing Room-B validation split: %s", manifest_path)
        return mapping

    by_label: dict = {}
    for c in caps_b:
        by_label.setdefault(c["label"], []).append(c["file_name"])

    shared_rng = None if reseed else _make_rng(backend, seed)
    mapping, records = {}, []
    for label in sorted(by_label):
        names = sorted(by_label[label])  # deterministic lexicographic order before sampling
        if len(names) <= n_val:
            raise ValueError(f"Activity {label!r} has only {len(names)} captures; need > {n_val}")
        rng = _make_rng(backend, seed) if reseed else shared_rng
        val = set(_sample_names(names, n_val, backend, rng))
        for name in names:
            split = "val" if name in val else "dev"
            mapping[name] = split
            records.append({"file_name": name, "label": label, "split": split})
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(manifest_path, index=False)
    if logger:
        logger.info("Created Room-B validation split (%d per activity, seed=%d, backend=%s) -> %s",
                    n_val, seed, backend, manifest_path)
    return mapping
