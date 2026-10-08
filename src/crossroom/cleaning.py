"""Quality control (QC) for CSI amplitude observations.

Every rejected observation is counted by reason and written to the capture manifest / QC summary.

Rejection rules (applied identically to PCAP- and CSV-derived rows):
  * malformed packet (wrong UDP payload length, wrong Nexmon magic, not a CSI datagram)  [PCAP only]
  * observations from a secondary core/spatial stream (only the dominant stream is kept)  [PCAP only]
  * wrong number of columns (capture must expose exactly n_subcarriers values)            [CSV only]
  * non-finite / missing values, negative amplitudes
  * zero-information rows (largest amplitude below `min_peak`)
  * exact duplicate observations (identical 256-value rows; first occurrence is kept)

Partial-band observations are *flagged, not rejected*: roughly two thirds of the Room-A records are
20/40 MHz frames (ACK-type frames) whose out-of-band subcarriers only hold receiver noise. No Room-A
capture contains 150 full-band observations, so these rows must stay in the observation sequence for the
official (150, 256) window to exist. The model-facing normaliser repairs them (see normalization.py).
"""
from __future__ import annotations

import numpy as np

QC_KEYS = [
    "n_raw_rows", "rej_not_csi_packet", "rej_bad_length", "rej_bad_magic", "rej_other_stream",
    "rej_bad_width", "rej_nonfinite", "rej_negative", "rej_zero_information", "rej_duplicate",
    "n_valid", "n_partial_band",
]


def new_qc() -> dict:
    return {k: 0 for k in QC_KEYS}


def row_quality_mask(amp: np.ndarray, min_peak: float = 1.0, drop_duplicates: bool = True):
    """Row-level QC on an (n_obs, n_subcarriers) amplitude matrix.

    Returns (keep_mask, counts) with counts keys: nonfinite, negative, zero_information, duplicate.
    """
    amp = np.asarray(amp, dtype=np.float64)
    n = amp.shape[0]
    keep = np.ones(n, dtype=bool)
    counts = {}

    finite = np.isfinite(amp).all(axis=1)
    counts["nonfinite"] = int((~finite).sum())
    keep &= finite
    safe = np.where(np.isfinite(amp), amp, 0.0)

    negative = (safe < 0).any(axis=1) & keep
    counts["negative"] = int(negative.sum())
    keep &= ~negative

    zero_info = (safe.max(axis=1) < float(min_peak)) & keep
    counts["zero_information"] = int(zero_info.sum())
    keep &= ~zero_info

    duplicate = np.zeros(n, dtype=bool)
    if drop_duplicates:
        seen = set()
        for i in np.flatnonzero(keep):
            key = safe[i].tobytes()
            if key in seen:
                duplicate[i] = True
            else:
                seen.add(key)
    counts["duplicate"] = int(duplicate.sum())
    keep &= ~duplicate
    return keep, counts


def partial_band_ratio(amp: np.ndarray, n_blocks: int = 4) -> np.ndarray:
    """min/max over the median amplitude of `n_blocks` equal sub-bands (4 x 64 subcarriers = 4 x 20 MHz).

    A full-band 80 MHz observation has comparable energy in all four sub-bands (ratio ~0.1-1);
    a 20/40 MHz frame leaves at least one sub-band at the noise floor (ratio << 0.05).
    Works on any (..., n_subcarriers) array and returns (...,).
    """
    a = np.asarray(amp, dtype=np.float32)
    f = a.shape[-1]
    if f % n_blocks != 0:
        raise ValueError(f"{f} subcarriers cannot be split into {n_blocks} equal sub-bands")
    blocks = a.reshape(a.shape[:-1] + (n_blocks, f // n_blocks))
    med = np.median(blocks, axis=-1)
    return med.min(axis=-1) / np.maximum(med.max(axis=-1), 1e-6)


def flag_partial_band(amp: np.ndarray, tau: float = 0.05, n_blocks: int = 4) -> np.ndarray:
    """Boolean flag (...,) for observations that do not occupy the full 80 MHz band."""
    return partial_band_ratio(amp, n_blocks) < float(tau)
