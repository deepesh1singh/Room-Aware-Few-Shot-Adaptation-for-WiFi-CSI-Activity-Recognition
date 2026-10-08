"""Deterministic windowing of captures into the official (150, 256) model input.

Rule (documented, label-free, validation-agnostic):
  * the observation sequence of a capture is the QC-cleaned amplitude matrix in capture order;
  * windows start at row 0 and advance by `window_stride`; at most `max_windows_per_capture` windows
    are taken (the earliest ones); a capture with fewer than `window_length` valid rows is ineligible;
  * a window never crosses capture boundaries, and every window of a capture stays in that capture's split.
Each window is saved as window_XXXXX.npy (float32, shape (150, 256), linear amplitude) and recorded in
manifests/windows.csv together with its originating capture and the preprocessing-config hash.
"""
from __future__ import annotations


import numpy as np
import pandas as pd
from tqdm import tqdm

from .cleaning import flag_partial_band
from .config import cget, preprocessing_signature
from .ingest import read_intermediate_csv
from .utils import natural_key, stable_hash


def window_starts(n_obs: int, length: int, stride: int, max_windows: int) -> list:
    if n_obs < length:
        return []
    return list(range(0, n_obs - length + 1, stride))[: int(max_windows)]


def build_windows(cfg: dict, ws: dict, logger) -> pd.DataFrame:
    length = int(cget(cfg, "data.window_length", 150))
    stride = int(cget(cfg, "data.window_stride", 35))
    max_w = int(cget(cfg, "data.max_windows_per_capture", 3))
    n_sub = int(cget(cfg, "data.n_subcarriers", 256))
    tau = float(cget(cfg, "qc.partial_band_tau", 0.05))
    pre_hash = stable_hash(preprocessing_signature(cfg))

    caps = pd.read_csv(ws["manifests"] / "captures.csv")
    caps = caps[caps["eligible"].astype(bool)].copy()
    caps["_nk"] = caps["stem"].map(natural_key)
    order = sorted(range(len(caps)), key=lambda i: (caps.iloc[i]["room"], caps.iloc[i]["split"],
                                                    caps.iloc[i]["label"], caps.iloc[i]["_nk"]))
    caps = caps.iloc[order]

    rows, wid = [], 0
    for _, cap in tqdm(caps.iterrows(), total=len(caps), desc="windows", unit="capture"):
        amp = read_intermediate_csv(ws["root"] / cap["intermediate_csv"])
        for start in window_starts(amp.shape[0], length, stride, max_w):
            win = np.ascontiguousarray(amp[start:start + length], dtype=np.float32)
            assert win.shape == (length, n_sub), win.shape
            out_dir = ws["windows_dev"] if cap["split"] == "dev" else ws["windows_val"]
            window_id = f"window_{wid:05d}"
            np.save(out_dir / f"{window_id}.npy", win)
            rows.append({
                "window_id": window_id, "split": cap["split"], "room": cap["room"], "label": cap["label"],
                "capture_id": cap["capture_id"], "start_row": int(start), "end_row": int(start + length),
                "n_valid_rows_in_capture": int(amp.shape[0]),
                "partial_band_fraction": float(flag_partial_band(win, tau).mean()),
                "npy_path": str((out_dir / f"{window_id}.npy").relative_to(ws["root"])),
                "preprocessing_hash": pre_hash,
            })
            wid += 1
    df = pd.DataFrame(rows)
    df.to_csv(ws["manifests"] / "windows.csv", index=False)
    summ = df.groupby(["split", "room"]).size().to_dict()
    logger.info("Built %d windows %s; preprocessing hash %s", len(df), summ, pre_hash)
    return df


def load_window_manifest(ws: dict):
    df = pd.read_csv(ws["manifests"] / "windows.csv")
    dev = df[df["split"] == "dev"].reset_index(drop=True)
    val = df[df["split"] == "val"].reset_index(drop=True)
    return dev, val


def load_windows(meta: pd.DataFrame, ws: dict) -> np.ndarray:
    """Load the raw (linear-amplitude) windows listed in a manifest into one float32 array (N, T, F)."""
    n = len(meta)
    if n == 0:
        return np.zeros((0, 0, 0), dtype=np.float32)
    first = np.load(ws["root"] / meta.iloc[0]["npy_path"])
    X = np.empty((n,) + first.shape, dtype=np.float32)
    X[0] = first
    for i in range(1, n):
        X[i] = np.load(ws["root"] / meta.iloc[i]["npy_path"])
    return X
