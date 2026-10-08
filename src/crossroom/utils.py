"""Small shared helpers: seeding, logging, hashing, natural sorting, device selection."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import sys
from pathlib import Path

import numpy as np


def set_seed(seed: int) -> None:
    """Seed python, numpy and (if installed) torch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


def get_logger(name: str = "crossroom", log_file=None) -> logging.Logger:
    """Console logger, optionally mirrored to a file (configured only once per target)."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in logger.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
            fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    return logger


def ensure_dir(path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def stable_hash(obj, length: int = 12) -> str:
    """Deterministic short hash of a JSON-serialisable object (used for provenance)."""
    blob = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:length]


def stable_int(*parts) -> int:
    """Deterministic 32-bit integer from arbitrary parts (python's hash() is salted per process)."""
    text = "|".join(str(p) for p in parts)
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)


def natural_key(text: str):
    """Sort key so that 'walk2' < 'walk10'."""
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", str(text))]


def pick_device(preference: str = "auto") -> str:
    import torch

    if preference == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return preference
