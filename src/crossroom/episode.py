"""Inference on a challenge *episode*: a folder (or a single file) of raw captures.

Supported inputs, all brought into the same (150, 256) amplitude-window contract by the same code that
produced the development/validation windows:
  * .pcap  - raw Nexmon captures (decoded in Python, QC'd, windowed);
  * .csv   - 256-column amplitude captures (header optional);
  * .npy   - either ready-made windows (150, 256) / (N, 150, 256) or a (T, 256) amplitude matrix.
The room is NOT given: it is predicted from the window content (routing) and selects the adapted model.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .cleaning import flag_partial_band
from .config import cget, preprocessing_signature
from .dataset import is_junk
from .encoder import encode, load_encoder
from .fewshot import route_and_predict
from .ingest import ingest_capture_file, write_intermediate_csv
from .normalization import WindowNormalizer
from .room_classifier import RoomClassifier
from .utils import natural_key, stable_hash
from .windowing import window_starts

EPISODE_SUFFIXES = (".pcap", ".csv", ".npy")


def collect_episode_files(path) -> list:
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Episode path not found: {path}")
    files = [p for p in path.rglob("*")
             if p.is_file() and p.suffix.lower() in EPISODE_SUFFIXES and not is_junk(p)]
    return sorted(files, key=lambda p: natural_key(str(p.relative_to(path))))


def episode_windows(files: list, base, cfg: dict, logger, max_windows=None, export_dir=None):
    """Raw files -> (windows (N, 150, 256) float32, provenance DataFrame, skipped [(source, reason)])."""
    base = Path(base)
    length = int(cget(cfg, "data.window_length", 150))
    stride = int(cget(cfg, "data.window_stride", 35))
    n_sub = int(cget(cfg, "data.n_subcarriers", 256))
    tau = float(cget(cfg, "qc.partial_band_tau", 0.05))
    max_w = int(max_windows or cget(cfg, "episode.max_windows_per_capture",
                                    cget(cfg, "data.max_windows_per_capture", 3)))
    windows, rows, skipped = [], [], []
    for f in files:
        rel = f.name if base.is_file() else str(f.relative_to(base))
        suffix = f.suffix.lower()
        try:
            if suffix == ".npy":
                arr = np.load(f)
                if arr.ndim == 3:
                    items = [(arr[i].astype(np.float32), f"{rel}#{i}") for i in range(arr.shape[0])]
                elif arr.ndim == 2:
                    items = [(arr.astype(np.float32), rel)]
                else:
                    raise ValueError(f"unsupported array shape {arr.shape}")
            else:
                amp, _qc = ingest_capture_file(f, cfg)
                items = [(amp, rel)]
                if export_dir is not None and amp.shape[0] > 0:
                    write_intermediate_csv(amp, Path(export_dir) / (Path(rel).with_suffix("").as_posix().replace("/", "__") + ".csv"))
            for amp, tag in items:
                if amp.ndim != 2 or amp.shape[1] != n_sub:
                    skipped.append((tag, f"bad shape {amp.shape}"))
                    continue
                starts = window_starts(amp.shape[0], length, stride, max_w)
                if not starts:
                    skipped.append((tag, f"only {amp.shape[0]} valid observations (< {length})"))
                    continue
                for s in starts:
                    w = np.ascontiguousarray(amp[s:s + length], dtype=np.float32)
                    windows.append(w)
                    rows.append({"source_file": tag, "start_row": int(s), "n_valid_rows": int(amp.shape[0]),
                                 "partial_band_fraction": float(flag_partial_band(w, tau).mean())})
        except Exception as exc:  # one unreadable file must not abort the episode
            skipped.append((rel, f"{type(exc).__name__}: {exc}"))
    for source, reason in skipped:
        logger.warning("skipped %s -> %s", source, reason)
    prov = pd.DataFrame(rows)
    if len(prov):
        prov.insert(0, "window_id", [f"window_{i:05d}" for i in range(len(prov))])
        X = np.stack(windows).astype(np.float32)
    else:
        prov = pd.DataFrame(columns=["window_id", "source_file", "start_row", "n_valid_rows", "partial_band_fraction"])
        X = np.zeros((0, length, n_sub), dtype=np.float32)
    return X, prov, skipped


def load_artifacts(art_dir, device: str = "cpu") -> dict:
    art_dir = Path(art_dir)
    need = ["normalizer.npz", "encoder_ssl.pt", "room_classifier.joblib", "episode_models.joblib"]
    missing = [n for n in need if not (art_dir / n).exists()]
    if missing:
        raise FileNotFoundError(f"Missing artifacts in {art_dir}: {missing}. Run `python run_pipeline.py --stage all` first.")
    return {
        "normalizer": WindowNormalizer.load(art_dir / "normalizer.npz"),
        "encoder": load_encoder(art_dir / "encoder_ssl.pt", device),
        "room_clf": RoomClassifier.load(art_dir / "room_classifier.joblib"),
        "pack": joblib.load(art_dir / "episode_models.joblib"),
    }


def predict_episode(X_raw: np.ndarray, art: dict, device: str = "cpu"):
    """Raw windows -> (predicted room, activity label, confidence) using routing + the adapted room models."""
    Xn = art["normalizer"].transform(X_raw)
    Z = encode(art["encoder"], Xn, device)
    rooms = art["room_clf"].predict(Xn).astype(str)
    labels, conf = route_and_predict(Z, rooms, art["pack"]["room_models"])
    return rooms, labels, conf


def config_matches_training(cfg: dict, art: dict) -> bool:
    return art["pack"].get("preprocessing_hash") == stable_hash(preprocessing_signature(cfg))
