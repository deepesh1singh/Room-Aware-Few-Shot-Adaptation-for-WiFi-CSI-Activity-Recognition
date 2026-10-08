#!/usr/bin/env python
"""End-to-end benchmark runner: raw captures -> common CSV -> windows -> SSL -> Tasks 1-3 -> figures.

Examples
    python run_pipeline.py --stage all
    python run_pipeline.py --stage ingest            # one stage at a time: ingest windows ssl task1 task2 task3 export
    python run_pipeline.py --stage ssl --set ssl.epochs=5 device=cpu     # quick smoke test
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from crossroom import stages  # noqa: E402
from crossroom.config import (load_config, parse_overrides, preprocessing_signature,  # noqa: E402
                              workspace_paths)
from crossroom.utils import get_logger, set_seed, stable_hash  # noqa: E402

ORDER = ["ingest", "windows", "ssl", "task1", "task2", "task3"]
CHOICES = ORDER + ["export", "all"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "experiment.yaml"))
    ap.add_argument("--stage", default="all", choices=CHOICES)
    ap.add_argument("--force", action="store_true", help="recompute stages even if their outputs already exist")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values, e.g. ssl.epochs=5")
    args = ap.parse_args()

    cfg = load_config(args.config, parse_overrides(args.set))
    ws = workspace_paths(cfg)
    logger = get_logger("crossroom", ws["logs"] / "pipeline.log")
    set_seed(int(cfg.get("seed", 42)))
    sig = preprocessing_signature(cfg)  # provenance: maps the hash stored in windows.csv to the actual settings
    (ws["manifests"] / "preprocessing_config.json").write_text(
        json.dumps({"hash": stable_hash(sig), "config": sig}, indent=2, default=str), encoding="utf-8")
    logger.info("config=%s | raw_root=%s | work_dir=%s", cfg["_config_path"], cfg["_raw_root"], cfg["_work_dir"])

    todo = ORDER if args.stage == "all" else [args.stage]
    for name in todo:
        logger.info("=== stage: %s ===", name)
        if name == "ingest":
            stages.stage_ingest(cfg, ws, logger, force=args.force)
        elif name == "windows":
            stages.stage_windows(cfg, ws, logger, force=args.force)
        elif name == "ssl":
            stages.stage_ssl(cfg, ws, logger, force=args.force)
        elif name == "task1":
            stages.stage_task1(cfg, ws, logger)
        elif name == "task2":
            stages.stage_task2(cfg, ws, logger)
        elif name == "task3":
            stages.stage_task3(cfg, ws, logger)
        elif name == "export":
            stages.stage_export(cfg, ws, logger)
    logger.info("done. Results: %s | Figures: %s", ws["results"], ws["figures"])


if __name__ == "__main__":
    main()
