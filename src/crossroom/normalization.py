"""Model-facing normalisation of raw amplitude windows.

Steps (identical for development, validation and episode windows; only step 5 is *fitted*, on
development windows only, pooled over both rooms - no labels, no validation statistics):
  1. partial-band rows are detected from the row itself (cleaning.flag_partial_band);
  2. log1p(amplitude)                                  - compresses the dynamic range of fading;
  3. per-row median subtraction                        - removes per-packet gain / AGC differences
                                                         (both rooms use a different amplitude scale);
  4. optional repair: rows flagged in step 1 are replaced by linear interpolation (in time) between the
     nearest full-band rows of the same window - the out-of-band part of a 20/40 MHz frame is noise;
  5. per-subcarrier standardisation with development mean/std, clipped to +-clip.
Per-room alignment is deliberately NOT done here (it needs the room, which is the routing stage's job).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .cleaning import flag_partial_band
from .config import cget


def repair_flagged_rows(L: np.ndarray, flags: np.ndarray) -> np.ndarray:
    """Replace flagged rows by time-interpolation of the neighbouring unflagged rows (per window)."""
    out = L.copy()
    n, T, _F = L.shape
    for i in range(n):
        f = flags[i]
        n_bad = int(f.sum())
        if n_bad == 0 or n_bad >= T - 1:  # nothing to repair / not enough good rows to interpolate from
            continue
        good = np.flatnonzero(~f)
        bad = np.flatnonzero(f)
        pos = np.searchsorted(good, bad)
        lo = good[np.clip(pos - 1, 0, len(good) - 1)]
        hi = good[np.clip(pos, 0, len(good) - 1)]
        same = hi <= lo
        denom = np.where(same, 1, hi - lo).astype(np.float32)
        w = np.where(same, 0.0, (bad - lo) / denom).astype(np.float32)
        out[i, bad] = (1.0 - w)[:, None] * L[i, lo] + w[:, None] * L[i, hi]
    return out


class WindowNormalizer:
    def __init__(self, tau: float = 0.05, repair: bool = True, clip: float = 8.0, std_floor: float = 0.1):
        self.tau = float(tau)
        self.repair = bool(repair)
        self.clip = float(clip)
        self.std_floor = float(std_floor)
        self.mean = None
        self.std = None

    @classmethod
    def from_config(cls, cfg: dict) -> "WindowNormalizer":
        return cls(tau=float(cget(cfg, "qc.partial_band_tau", 0.05)),
                   repair=bool(cget(cfg, "normalization.repair_partial_band_rows", True)),
                   clip=float(cget(cfg, "normalization.clip", 8.0)),
                   std_floor=float(cget(cfg, "normalization.std_floor", 0.1)))

    def params(self) -> dict:
        return {"tau": self.tau, "repair": self.repair, "clip": self.clip, "std_floor": self.std_floor}

    def _base(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        flags = flag_partial_band(X, self.tau)
        L = np.log1p(np.maximum(X, 0.0))
        L -= np.median(L, axis=2, keepdims=True)
        if self.repair:
            L = repair_flagged_rows(L, flags)
        return L, flags

    def fit(self, X: np.ndarray, batch_size: int = 256) -> "WindowNormalizer":
        """Fit per-subcarrier mean/std on DEVELOPMENT windows (raw amplitude, shape (N, T, F))."""
        n_f = X.shape[2]
        s = np.zeros(n_f, dtype=np.float64)
        ss = np.zeros(n_f, dtype=np.float64)
        n_rows = 0
        for i in range(0, X.shape[0], batch_size):
            L, _ = self._base(X[i:i + batch_size])
            s += L.sum(axis=(0, 1), dtype=np.float64)
            ss += np.square(L, dtype=np.float64).sum(axis=(0, 1))
            n_rows += L.shape[0] * L.shape[1]
        mean = s / max(n_rows, 1)
        var = np.maximum(ss / max(n_rows, 1) - mean ** 2, 0.0)
        self.mean = mean.astype(np.float32)
        self.std = np.maximum(np.sqrt(var), self.std_floor).astype(np.float32)
        return self

    def transform(self, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """Normalise raw windows (N, T, F) (any T) -> float32 (N, T, F)."""
        if self.mean is None:
            raise RuntimeError("WindowNormalizer must be fitted or loaded before transform()")
        out = np.empty(X.shape, dtype=np.float32)
        for i in range(0, X.shape[0], batch_size):
            L, _ = self._base(X[i:i + batch_size])
            Z = (L - self.mean) / self.std
            np.clip(Z, -self.clip, self.clip, out=Z)
            out[i:i + batch_size] = Z
        return out

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, std=self.std, params=json.dumps(self.params()))

    @classmethod
    def load(cls, path) -> "WindowNormalizer":
        d = np.load(path, allow_pickle=False)
        obj = cls(**json.loads(str(d["params"])))
        obj.mean = d["mean"].astype(np.float32)
        obj.std = d["std"].astype(np.float32)
        return obj


def load_or_fit_normalizer(cfg: dict, ws: dict, X_dev_raw: np.ndarray, logger) -> WindowNormalizer:
    """Fit on development windows only (never on validation) and cache the parameters."""
    path = ws["artifacts"] / "normalizer.npz"
    wanted = WindowNormalizer.from_config(cfg)
    if path.exists():
        norm = WindowNormalizer.load(path)
        if norm.params() == wanted.params():
            return norm
        logger.warning("normalizer.npz was fitted with different settings %s -> refitting", norm.params())
    logger.info("Fitting normaliser on %d DEVELOPMENT windows (both rooms pooled, no labels)", X_dev_raw.shape[0])
    wanted.fit(X_dev_raw)
    wanted.save(path)
    return wanted
