"""Conversion of complex CSI to a real-valued amplitude representation."""
from __future__ import annotations

import numpy as np


def csi_to_amplitude(csi: np.ndarray) -> np.ndarray:
    """|CSI| as float32, shape preserved."""
    return np.abs(csi).astype(np.float32)


def apply_subcarrier_order(amp: np.ndarray, fftshift: bool = False) -> np.ndarray:
    """Optionally re-centre the subcarrier axis. Both supplied rooms are already in the same centred
    order (DC / guard nulls at the same indices), so the default is to leave the order untouched."""
    return np.fft.fftshift(amp, axes=-1) if fftshift else amp


def null_signature(amp_matrix: np.ndarray, k: int = 8):
    """Indices of the k subcarriers with the lowest median amplitude (guard / DC nulls).
    Used as a cheap check that two sources share the same subcarrier ordering."""
    med = np.median(np.asarray(amp_matrix), axis=0)
    return sorted(int(i) for i in np.argsort(med)[:k])
