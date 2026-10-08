"""Loading derived windows + normalisation, and construction of the compared representations."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import cget
from .encoder import build_random_encoder, encode, load_encoder
from .features import handcrafted_features
from .normalization import WindowNormalizer, load_or_fit_normalizer
from .windowing import load_window_manifest, load_windows


@dataclass
class Bundle:
    dev_meta: pd.DataFrame
    val_meta: pd.DataFrame
    Xn_dev: np.ndarray      # normalised development windows (N_dev, 150, 256)
    Xn_val: np.ndarray      # normalised validation windows  (N_val, 150, 256)
    X_val_raw: np.ndarray   # raw validation windows (needed to crop BEFORE normalising in the length study)
    normalizer: WindowNormalizer


def build_bundle(cfg: dict, ws: dict, logger) -> Bundle:
    dev_meta, val_meta = load_window_manifest(ws)
    if len(dev_meta) == 0 or len(val_meta) == 0:
        raise RuntimeError("No windows found - run the 'ingest' and 'windows' stages first.")
    logger.info("Loading %d development and %d validation windows", len(dev_meta), len(val_meta))
    X_dev_raw = load_windows(dev_meta, ws)
    X_val_raw = load_windows(val_meta, ws)
    normalizer = load_or_fit_normalizer(cfg, ws, X_dev_raw, logger)  # fitted on DEVELOPMENT windows only
    Xn_dev = normalizer.transform(X_dev_raw)
    del X_dev_raw
    Xn_val = normalizer.transform(X_val_raw)
    return Bundle(dev_meta, val_meta, Xn_dev, Xn_val, X_val_raw, normalizer)


def build_representations(bundle: Bundle, cfg: dict, ws: dict, device: str, logger) -> dict:
    """{'ssl': (Z_dev, Z_val), 'random': (...), 'handcrafted': (...)} - all frozen, no labels involved."""
    ckpt = ws["artifacts"] / "encoder_ssl.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"{ckpt} not found - run the 'ssl' stage first.")
    seed = int(cget(cfg, "seed", 42))
    reps = {}
    enc = load_encoder(ckpt, device)
    reps["ssl"] = (encode(enc, bundle.Xn_dev, device), encode(enc, bundle.Xn_val, device))
    rnd = build_random_encoder(cfg, seed)
    reps["random"] = (encode(rnd, bundle.Xn_dev, device), encode(rnd, bundle.Xn_val, device))
    reps["handcrafted"] = (handcrafted_features(bundle.Xn_dev), handcrafted_features(bundle.Xn_val))
    for name, (zd, zv) in reps.items():
        logger.info("representation %-12s dev %s  val %s", name, zd.shape, zv.shape)
    return reps
