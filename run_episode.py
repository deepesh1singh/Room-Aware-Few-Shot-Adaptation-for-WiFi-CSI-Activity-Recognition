#!/usr/bin/env python
"""Process a challenge episode (folder or single file of .pcap / .csv / .npy captures) and write predictions.

Output (CSV): window_id, room, label, confidence      (+ <out>_provenance.csv with source file / start row)

    python run_episode.py --episode path/to/episode --out predictions.csv
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from crossroom.config import load_config, parse_overrides, workspace_paths  # noqa: E402
from crossroom.episode import (collect_episode_files, config_matches_training, episode_windows,  # noqa: E402
                               load_artifacts, predict_episode)
from crossroom.utils import get_logger, pick_device  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", required=True, help="folder (searched recursively) or single capture file")
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--config", default=str(ROOT / "configs" / "experiment.yaml"))
    ap.add_argument("--artifacts", default=None, help="directory with the trained artifacts (default: <work_dir>/artifacts)")
    ap.add_argument("--max-windows", type=int, default=None, help="max windows taken from each capture")
    ap.add_argument("--export-csv-dir", default=None, help="optionally write the common 256-column CSV of every capture here")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()

    cfg = load_config(args.config, parse_overrides(args.set))
    ws = workspace_paths(cfg, create=False)
    logger = get_logger("crossroom")
    device = pick_device(str(cfg.get("device", "auto")))

    episode = Path(args.episode)
    files = collect_episode_files(episode)
    logger.info("episode %s: %d candidate capture file(s)", episode, len(files))
    X, prov, skipped = episode_windows(files, episode, cfg, logger, max_windows=args.max_windows,
                                       export_dir=args.export_csv_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if len(X) == 0:
        pd.DataFrame(columns=["window_id", "room", "label", "confidence"]).to_csv(out, index=False)
        logger.error("no valid windows could be built from the episode (%d file(s) skipped)", len(skipped))
        return 1

    art = load_artifacts(Path(args.artifacts) if args.artifacts else ws["artifacts"], device)
    if not config_matches_training(cfg, art):
        logger.warning("preprocessing config differs from the one used for training - predictions may be unreliable")
    rooms, labels, conf = predict_episode(X, art, device)
    pred = pd.DataFrame({"window_id": prov["window_id"], "room": rooms, "label": labels,
                         "confidence": conf.round(6)})
    pred.to_csv(out, index=False)
    prov.assign(room=rooms, label=labels, confidence=conf.round(6)).to_csv(
        out.with_name(out.stem + "_provenance.csv"), index=False)
    logger.info("wrote %d window predictions -> %s", len(pred), out)
    logger.info("room counts: %s | label counts: %s", pred["room"].value_counts().to_dict(),
                pred["label"].value_counts().to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
