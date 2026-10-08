#!/usr/bin/env python
"""Hint-1 style inspection of the raw data (read-only): where the data was found, how many captures per
class, packet/row counts, Nexmon header fields and a few decoded-amplitude sanity numbers.

    python scripts/inspect_data.py [--config configs/experiment.yaml] [--n 5]
"""
import argparse
import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from crossroom.amplitude import null_signature  # noqa: E402
from crossroom.config import load_config, parse_overrides  # noqa: E402
from crossroom.dataset import (find_room_a_root, find_room_b_root, list_room_a_captures,  # noqa: E402
                               list_room_b_captures)
from crossroom.ingest import ingest_csv, ingest_pcap  # noqa: E402
from crossroom.pcap_reader import read_pcap_udp  # noqa: E402
from crossroom.csi_decode import parse_payloads  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "experiment.yaml"))
    ap.add_argument("--n", type=int, default=5, help="captures per room to decode for the sanity statistics")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, parse_overrides(args.set))
    raw = Path(cfg["_raw_root"])
    print(f"raw_root = {raw}")

    a_root, b_root = find_room_a_root(raw), find_room_b_root(raw)
    print(f"Room A csidata : {a_root}\nRoom B Data    : {b_root}\n")
    caps_a, caps_b = list_room_a_captures(a_root), list_room_b_captures(b_root)
    cnt = collections.Counter((c["split"], c["label"]) for c in caps_a)
    print("Room A pcaps (dev/val per class):")
    for label in sorted({c["label"] for c in caps_a}):
        print(f"  {label:8s} dev {cnt[('dev', label)]:4d}   val {cnt[('val', label)]:3d}")
    print(f"  total {len(caps_a)}  (expected 1084 = 934 dev + 150 val)")
    cb = collections.Counter(c["label"] for c in caps_b)
    print("Room B csv captures per activity:", dict(sorted(cb.items())), f"total {len(caps_b)} (expected 1201)\n")

    for cap in caps_a[: args.n]:
        recs, st = read_pcap_udp(cap["path"], udp_port=cfg["room_a"]["udp_port"])
        parsed = parse_payloads(recs)
        seq = parsed["seq"]
        print(f"[A] {cap['label']}/{cap['stem']}: records {st['n_records']}, csi payloads {len(recs)}, "
              f"seq==0xFFFF {(seq == 65535).sum()}, core/stream ids {sorted(set(parsed['css'].tolist()))}, "
              f"chanspec {sorted(hex(x) for x in set(parsed['chanspec'].tolist()))}")
    amps = []
    for cap in caps_a[: args.n]:
        amp, qc = ingest_pcap(cap["path"], cfg)
        amps.append(amp)
        print(f"[A] {cap['stem']}: valid obs {qc['n_valid']}, partial-band {qc['n_partial_band']}, rejected "
              f"{ {k: v for k, v in qc.items() if k.startswith('rej_') and v} }")
    for cap in caps_b[: args.n]:
        amp, qc = ingest_csv(cap["path"], cfg)
        print(f"[B] {cap['stem']}: valid obs {qc['n_valid']}, partial-band {qc['n_partial_band']}, "
              f"amp min/median/max {amp.min():.0f}/{np.median(amp):.0f}/{amp.max():.0f}")
    if amps:
        print("\nLowest-amplitude subcarriers, Room A sample:", null_signature(np.concatenate(amps), 10))
    print("(Both rooms should show nulls around index 128 and at the band edges - same subcarrier order.)")


if __name__ == "__main__":
    main()
