"""Raw captures -> clean common intermediate form (one 256-column amplitude CSV per capture).

Room A (Experiment-2): PCAP -> UDP payloads -> Nexmon header check -> packed-float CSI decode -> amplitude.
Room B (Experiment-3): already amplitude CSV -> schema check -> QC.
Both paths end in the same representation: an (n_valid_observations, 256) float matrix whose rows keep
the temporal order of the capture, written as intermediate_csv/room_<R>/<split>/<activity>/<stem>.csv.
Source file names / folders are used for labelling and provenance only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .amplitude import apply_subcarrier_order, csi_to_amplitude, null_signature
from .cleaning import flag_partial_band, new_qc, row_quality_mask
from .config import cget
from .csi_decode import parse_payloads, unpack_packed_float
from .dataset import (find_room_a_root, find_room_b_root, list_room_a_captures,
                      list_room_b_captures, make_capture_id)
from .pcap_reader import read_pcap_udp
from .splits import assign_room_b_split


# --------------------------------------------------------------------------------------- CSV helpers
def _first_line_is_header(line: str) -> bool:
    toks = [t.strip().strip('"') for t in line.strip().split(",")]
    toks = [t for t in toks if t != ""]
    if not toks:
        return False
    vals = []
    for t in toks:
        try:
            vals.append(float(t))
        except ValueError:
            return True  # a non-numeric token -> named header such as Subcarrier_1
    # purely numeric header such as 0,1,...,255 (or 1,...,256)
    return vals == [float(i) for i in range(len(vals))] or vals == [float(i) for i in range(1, len(vals) + 1)]


def read_amplitude_csv(path, n_subcarriers: int = 256):
    """Read an amplitude CSV (optional header row, optional leading index column).

    Returns (matrix float64 (n_rows, n_subcarriers) or None, info). None means the width is wrong.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        first = fh.readline()
    has_header = _first_line_is_header(first)
    df = pd.read_csv(path, header=0 if has_header else None)
    df = df.apply(pd.to_numeric, errors="coerce")
    info = {"n_rows": int(df.shape[0]), "n_cols": int(df.shape[1]), "had_header": bool(has_header)}
    if df.shape[1] == n_subcarriers + 1:  # drop a leading running-index column if present
        col0 = df.iloc[:, 0].to_numpy()
        n = len(col0)
        if np.array_equal(col0, np.arange(n)) or np.array_equal(col0, np.arange(1, n + 1)):
            df = df.iloc[:, 1:]
            info["n_cols"] = int(df.shape[1])
    if df.shape[1] != n_subcarriers:
        return None, info
    return df.to_numpy(dtype=np.float64), info


def write_intermediate_csv(amp: np.ndarray, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [f"subcarrier_{i:03d}" for i in range(1, amp.shape[1] + 1)]
    pd.DataFrame(amp, columns=cols).to_csv(path, index=False, float_format="%.4f")


def read_intermediate_csv(path) -> np.ndarray:
    return pd.read_csv(path).to_numpy(dtype=np.float32)


# ----------------------------------------------------------------------------------- single captures
def _finish_qc(amp: np.ndarray, qc: dict, cfg: dict) -> np.ndarray:
    keep, rej = row_quality_mask(
        amp,
        min_peak=float(cget(cfg, "qc.min_row_peak", 1.0)),
        drop_duplicates=bool(cget(cfg, "qc.drop_duplicates", True)),
    )
    for key, val in rej.items():
        qc["rej_" + key] = int(val)
    amp = amp[keep].astype(np.float32)
    qc["n_valid"] = int(amp.shape[0])
    tau = float(cget(cfg, "qc.partial_band_tau", 0.05))
    qc["n_partial_band"] = int(flag_partial_band(amp, tau).sum()) if amp.shape[0] else 0
    return amp


def ingest_pcap(path, cfg: dict):
    """One Experiment-2 style PCAP -> (amplitude (n_valid, 256) float32, qc counts)."""
    ra = cfg.get("room_a", {}) or {}
    nfft = int(ra.get("nfft", 256))
    qc = new_qc()
    records, pstats = read_pcap_udp(path, udp_port=ra.get("udp_port", 5500))
    qc["n_raw_rows"] = int(pstats["n_records"])
    qc["rej_not_csi_packet"] = int(pstats["n_records"] - len(records))
    parsed = parse_payloads(records, nfft=nfft, header_bytes=int(ra.get("nexmon_header_bytes", 18)),
                            magic=int(ra.get("expected_magic", 0x1111)))
    qc["rej_bad_length"] = int(parsed["rejected"]["bad_length"])
    qc["rej_bad_magic"] = int(parsed["rejected"]["bad_magic"])
    words = parsed["words"]
    if words.shape[0] == 0:
        return np.zeros((0, nfft), dtype=np.float32), qc

    css = parsed["css"]
    if len(np.unique(css)) > 1:  # several core/spatial streams interleaved -> keep the dominant one
        vals, counts = np.unique(css, return_counts=True)
        keep_stream = css == vals[int(np.argmax(counts))]
        qc["rej_other_stream"] = int((~keep_stream).sum())
        words = words[keep_stream]

    csi = unpack_packed_float(words, nman=int(ra.get("nman", 12)), nexp=int(ra.get("nexp", 6)),
                              nbits=int(ra.get("nbits", 10)))
    amp = csi_to_amplitude(csi)
    amp = apply_subcarrier_order(amp, bool(ra.get("apply_fftshift", False)))
    return _finish_qc(amp, qc, cfg), qc


def ingest_csv(path, cfg: dict):
    """One Experiment-3 style amplitude CSV -> (amplitude (n_valid, 256) float32, qc counts)."""
    n_sub = int(cget(cfg, "data.n_subcarriers", 256))
    qc = new_qc()
    mat, info = read_amplitude_csv(path, n_sub)
    qc["n_raw_rows"] = int(info["n_rows"])
    if mat is None:
        qc["rej_bad_width"] = int(info["n_rows"])
        return np.zeros((0, n_sub), dtype=np.float32), qc
    mat = apply_subcarrier_order(mat, bool(cget(cfg, "room_b.apply_fftshift", False)))
    return _finish_qc(mat, qc, cfg), qc


def ingest_capture_file(path, cfg: dict):
    """Dispatch on the file *format* (needed to parse it) - the room is never an input here."""
    suffix = Path(path).suffix.lower()
    if suffix == ".pcap":
        return ingest_pcap(path, cfg)
    if suffix == ".csv":
        return ingest_csv(path, cfg)
    raise ValueError(f"Unsupported capture format: {path}")


# ----------------------------------------------------------------------------------------- full run
def run_ingest(cfg: dict, ws: dict, logger):
    raw_root = Path(cfg["_raw_root"])
    a_root = find_room_a_root(raw_root)
    b_root = find_room_b_root(raw_root)
    logger.info("Room A (Experiment-2) csidata : %s", a_root)
    logger.info("Room B (Experiment-3) Data    : %s", b_root)

    caps_a = list_room_a_captures(a_root)
    caps_b = list_room_b_captures(b_root)
    n_a_val = sum(c["split"] == "val" for c in caps_a)
    logger.info("Room A: %d pcaps (%d development + %d supplied validation) | Room B: %d csv captures",
                len(caps_a), len(caps_a) - n_a_val, n_a_val, len(caps_b))

    split_map = assign_room_b_split(caps_b, cfg, ws["manifests"] / "room_B_validation_split.csv", logger)
    for cap in caps_b:
        cap["split"] = split_map[cap["file_name"]]

    window_length = int(cget(cfg, "data.window_length", 150))
    n_sub = int(cget(cfg, "data.n_subcarriers", 256))
    rows, profiles = [], {"A": [], "B": []}
    for cap in tqdm(caps_a + caps_b, desc="ingest", unit="capture"):
        room, split, label, stem = cap["room"], cap["split"], cap["label"], cap["stem"]
        status = "ok"
        try:
            amp, qc = ingest_pcap(cap["path"], cfg) if room == "A" else ingest_csv(cap["path"], cfg)
        except Exception as exc:  # a corrupt capture is recorded, not fatal
            amp, qc = np.zeros((0, n_sub), dtype=np.float32), new_qc()
            status = f"error:{type(exc).__name__}"
        csv_rel = ""
        if amp.shape[0] > 0:
            out = ws["intermediate"] / f"room_{room}" / split / label / f"{stem}.csv"
            write_intermediate_csv(amp, out)
            csv_rel = str(out.relative_to(ws["root"]))
            if split == "dev" and len(profiles[room]) < 300:
                profiles[room].append(np.median(amp, axis=0))
        elif status == "ok":
            status = "empty"
        row = {
            "capture_id": make_capture_id(room, split, label, stem),
            "room": room, "split": split, "label": label, "stem": stem,
            "source_file": str(Path(cap["path"]).relative_to(raw_root)),
            "intermediate_csv": csv_rel, "status": status,
            "eligible": bool(amp.shape[0] >= window_length),
        }
        row.update(qc)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(ws["manifests"] / "captures.csv", index=False)

    summary = {"window_length": window_length, "rooms": {}}
    for room in ("A", "B"):
        d = df[df["room"] == room]
        qc_cols = [c for c in d.columns if c.startswith("rej_")] + ["n_raw_rows", "n_valid", "n_partial_band"]
        summary["rooms"][room] = {
            "captures": int(len(d)),
            "captures_by_split": {k: int(v) for k, v in d["split"].value_counts().items()},
            "eligible_captures_by_split": {k: int(v) for k, v in d[d["eligible"]]["split"].value_counts().items()},
            "ineligible_captures": int((~d["eligible"]).sum()),
            "qc_totals": {c: int(d[c].sum()) for c in qc_cols},
            "partial_band_fraction": float(d["n_partial_band"].sum() / max(d["n_valid"].sum(), 1)),
            "median_valid_observations": float(d["n_valid"].median()) if len(d) else 0.0,
            "lowest_amplitude_subcarriers": null_signature(np.stack(profiles[room]), 10) if profiles[room] else [],
        }
    with open(ws["manifests"] / "qc_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    for room in ("A", "B"):
        s = summary["rooms"][room]
        logger.info("Room %s: %d captures, %d ineligible (<%d valid rows), partial-band rows %.1f%%, "
                    "lowest-amplitude subcarriers %s", room, s["captures"], s["ineligible_captures"],
                    window_length, 100 * s["partial_band_fraction"], s["lowest_amplitude_subcarriers"])
    logger.info("Wrote %s and %s", ws["manifests"] / "captures.csv", ws["manifests"] / "qc_summary.json")
    return df
