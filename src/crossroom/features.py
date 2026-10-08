"""Handcrafted CSI features (baseline representation for Task 2 / Task 3).

Input: normalised windows (N, T, F). Output: (N, 81) float32 with
  * 16 sub-bands x {mean level, temporal std, mean |first difference|, p95-p5 range}   (64)
  * dynamics of the band-averaged series: std, mean |diff|, skewness, kurtosis, autocorr lag 1-3     (7)
  * spectral shape of that series: energy fraction in 6 frequency bands + spectral centroid          (7)
  * subcarrier-wise dynamics: mean / std of temporal std, mean adjacent-subcarrier correlation       (3)
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-8


def _chunk_features(X: np.ndarray, n_bands: int) -> np.ndarray:
    n, T, F = X.shape
    bw = F // n_bands
    B = X[:, :, : bw * n_bands].reshape(n, T, n_bands, bw).mean(axis=3)  # (n, T, n_bands)

    feats = [
        B.mean(axis=1),
        B.std(axis=1),
        np.abs(np.diff(B, axis=1)).mean(axis=1),
    ]
    p = np.percentile(B, [5, 95], axis=1)
    feats.append(p[1] - p[0])

    g = B.mean(axis=2)  # (n, T) band-averaged series
    mu = g.mean(axis=1)
    d = g - mu[:, None]
    m2 = (d ** 2).mean(axis=1)
    skew = (d ** 3).mean(axis=1) / (m2 ** 1.5 + _EPS)
    kurt = (d ** 4).mean(axis=1) / (m2 ** 2 + _EPS)
    g_feats = [np.sqrt(m2), np.abs(np.diff(g, axis=1)).mean(axis=1), skew, kurt]
    for lag in (1, 2, 3):
        if T > lag:
            g_feats.append((d[:, :-lag] * d[:, lag:]).mean(axis=1) / (m2 + _EPS))
        else:
            g_feats.append(np.zeros(n))
    feats.append(np.stack(g_feats, axis=1))

    P = np.abs(np.fft.rfft(d, axis=1)) ** 2
    P = P[:, 1:]  # drop DC
    if P.shape[1] >= 6:
        groups = np.array_split(np.arange(P.shape[1]), 6)
        tot = P.sum(axis=1) + _EPS
        bands = np.stack([P[:, idx].sum(axis=1) / tot for idx in groups], axis=1)
        centroid = (P * np.arange(1, P.shape[1] + 1)[None, :]).sum(axis=1) / (tot * P.shape[1])
    else:
        bands = np.zeros((n, 6))
        centroid = np.zeros(n)
    feats.append(np.concatenate([bands, centroid[:, None]], axis=1))

    X0 = X - X.mean(axis=1, keepdims=True)
    tstd = X0.std(axis=1)  # (n, F)
    num = (X0[:, :, :-1] * X0[:, :, 1:]).sum(axis=1)
    den = np.sqrt((X0[:, :, :-1] ** 2).sum(axis=1) * (X0[:, :, 1:] ** 2).sum(axis=1)) + _EPS
    adj = (num / den).mean(axis=1)
    feats.append(np.stack([tstd.mean(axis=1), tstd.std(axis=1), adj], axis=1))

    return np.concatenate(feats, axis=1)


def handcrafted_features(Xn: np.ndarray, n_bands: int = 16, chunk: int = 256) -> np.ndarray:
    parts = [_chunk_features(Xn[i:i + chunk].astype(np.float32), n_bands) for i in range(0, Xn.shape[0], chunk)]
    out = np.concatenate(parts, axis=0).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
