"""Pipeline stages. Every stage reads/writes only the derived workspace; raw data stays read-only."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import plots
from .bundle import build_bundle, build_representations
from .config import cget, preprocessing_signature
from .dataset import ROOM_ACTIVITIES, ROOMS
from .encoder import encode, load_encoder, save_encoder
from .evaluation import (ResultsLog, activity_metrics, confusion_with_other, merge_results,
                         per_class_report)
from .fewshot import (build_room_models, make_rep_space, meta_arrays, route_and_predict, run_task3_rep)
from .ingest import run_ingest
from .probes import run_task2
from .room_classifier import RoomClassifier, accuracy_vs_window_length
from .ssl_train import train_ssl
from .utils import pick_device, stable_hash
from .windowing import build_windows

_CACHE: dict = {}


def get_bundle(cfg, ws, logger):
    key = cfg["_work_dir"]
    if key not in _CACHE:
        _CACHE[key] = build_bundle(cfg, ws, logger)
    return _CACHE[key]


def _device(cfg) -> str:
    return pick_device(str(cfg.get("device", "auto")))


# ------------------------------------------------------------------------------------ data stages
def stage_ingest(cfg, ws, logger, force=False):
    if (ws["manifests"] / "captures.csv").exists() and not force:
        logger.info("manifests/captures.csv exists -> skipping ingest (use --force to redo)")
        return
    run_ingest(cfg, ws, logger)


def stage_windows(cfg, ws, logger, force=False):
    if (ws["manifests"] / "windows.csv").exists() and not force:
        logger.info("manifests/windows.csv exists -> skipping window export (use --force to redo)")
        return
    for d in (ws["windows_dev"], ws["windows_val"]):
        for f in d.glob("window_*.npy"):
            f.unlink()
    stale = ws["artifacts"] / "normalizer.npz"
    if stale.exists():
        stale.unlink()  # windows changed -> the development-fitted normaliser must be refitted
    _CACHE.pop(cfg["_work_dir"], None)
    build_windows(cfg, ws, logger)


# --------------------------------------------------------------------------------------- SSL stage
def stage_ssl(cfg, ws, logger, force=False):
    ckpt = ws["artifacts"] / "encoder_ssl.pt"
    if ckpt.exists() and not force:
        logger.info("%s exists -> skipping SSL pre-training (use --force to retrain)", ckpt.name)
        return
    bundle = get_bundle(cfg, ws, logger)
    encoder, history = train_ssl(bundle.Xn_dev, cfg, _device(cfg), logger)
    save_encoder(encoder, ckpt, history)
    plots.plot_ssl_loss(history, ws["figures"] / "ssl_training_loss.png")
    logger.info("Saved %s", ckpt)


# ------------------------------------------------------------------------------------------ Task 1
def get_room_classifier(cfg, ws, bundle, logger, refit=False) -> RoomClassifier:
    t1 = cfg.get("task1", {}) or {}
    features = str(t1.get("features", "mean"))
    path = ws["artifacts"] / "room_classifier.joblib"
    if path.exists() and not refit:
        clf = RoomClassifier.load(path)
        if clf.features == features:
            return clf
    clf = RoomClassifier(features=features, C=float(t1.get("C", 1.0)), seed=int(cget(cfg, "seed", 42)))
    clf.fit(bundle.Xn_dev, bundle.dev_meta["room"].to_numpy().astype(str))  # development windows only
    clf.save(path)
    return clf


def stage_task1(cfg, ws, logger):
    bundle = get_bundle(cfg, ws, logger)
    clf = get_room_classifier(cfg, ws, bundle, logger, refit=True)
    rooms_val = bundle.val_meta["room"].to_numpy().astype(str)
    pred = clf.predict(bundle.Xn_val)
    log = ResultsLog()
    acc = float((pred == rooms_val).mean())
    log.add("task1_room_id", "ALL", None, "room_accuracy", acc, n_val_windows=int(len(rooms_val)))
    for room in ROOMS:
        m = rooms_val == room
        log.add("task1_room_id", room, None, "room_accuracy", float((pred[m] == room).mean()),
                n_val_windows=int(m.sum()))
    logger.info("Task 1 room accuracy on %d official 150-observation validation windows: %.4f", len(rooms_val), acc)

    lengths = [int(x) for x in (cfg.get("task1", {}) or {}).get("window_lengths", [5, 10, 20, 30, 50, 75, 100, 150])]
    rows = accuracy_vs_window_length(clf, bundle.normalizer, bundle.X_val_raw, rooms_val, lengths)
    for r in rows:
        log.add("task1_fingerprint_vs_length", "ALL", None, f"room_accuracy_len{r['length']}", r["accuracy"],
                window_length=r["length"])
    log.save(ws["results"] / "task1_results.csv")
    merge_results(ws["results"])
    plots.plot_fingerprint_curve(rows, ws["figures"] / "fingerprint_accuracy_vs_window_length.png")
    return acc


# ------------------------------------------------------------------------------------------ Task 2
def embed_2d(Z: np.ndarray, ref: np.ndarray, method: str = "auto", seed: int = 42):
    """2-D map of embeddings Z (purely descriptive). Standardised with DEVELOPMENT statistics `ref`."""
    Zs = (Z - ref.mean(axis=0)) / (ref.std(axis=0) + 1e-8)
    if method in ("auto", "umap"):
        try:
            import umap  # optional dependency

            return umap.UMAP(n_components=2, random_state=seed).fit_transform(Zs), "UMAP"
        except ImportError:
            if method == "umap":
                raise
    from sklearn.manifold import TSNE

    perplexity = float(min(30, max(5, (len(Zs) - 1) // 3)))
    emb = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=seed).fit_transform(Zs)
    return emb, "t-SNE"


def stage_task2(cfg, ws, logger):
    bundle = get_bundle(cfg, ws, logger)
    reps = build_representations(bundle, cfg, ws, _device(cfg), logger)
    dev, val = meta_arrays(bundle.dev_meta), meta_arrays(bundle.val_meta)
    seed = int(cget(cfg, "seed", 42))
    log = ResultsLog()
    run_task2(reps, dev.room, dev.label, val.room, val.label,
              C=float(cget(cfg, "task2.probe_C", 1.0)), seed=seed, log=log)
    df = log.save(ws["results"] / "task2_results.csv")
    merge_results(ws["results"])
    head = df[(df["experiment"].str.startswith("task2_probe_joint")) & (df["target_room"] == "ALL")
              & (df["metric"] == "macro_f1")]
    for r in head.itertuples():
        logger.info("Task 2 joint linear probe  %-12s macro-F1 %.4f", r.representation, r.value)

    Z_val = reps["ssl"][1]
    idx = np.arange(len(Z_val))
    max_pts = int(cget(cfg, "task2.embedding_plot_max_points", 2000))
    if len(idx) > max_pts:
        idx = np.sort(np.random.RandomState(seed).choice(len(idx), max_pts, replace=False))
    emb, method = embed_2d(Z_val[idx], reps["ssl"][0], str(cget(cfg, "task2.embedding_method", "auto")), seed)
    plots.plot_encoder_diagnosis(emb, val.room[idx], val.label[idx], method,
                                 ws["figures"] / "encoder_embedding_2d.png")
    np.savez(ws["artifacts"] / "embeddings_validation_ssl.npz", Z=Z_val, window_id=val.window_id)


# ------------------------------------------------------------------------------------------ Task 3
def _export_episode_models(cfg, ws, rep, dev, logger):
    ec = cfg.get("episode", {}) or {}
    K, seed, head = int(ec.get("k", 20)), int(ec.get("seed", 0)), str(ec.get("head", "logreg"))
    models = build_room_models(rep, dev, K, seed, head, cfg.get("task3", {}) or {})
    joblib.dump({"K": K, "seed": seed, "head": head, "room_models": models,
                 "window_length": int(cget(cfg, "data.window_length", 150)),
                 "preprocessing_hash": stable_hash(preprocessing_signature(cfg))},
                ws["artifacts"] / "episode_models.joblib")
    logger.info("Saved bidirectional episode models (K=%d, seed=%d, head=%s) -> episode_models.joblib", K, seed, head)
    return models


def stage_task3(cfg, ws, logger):
    bundle = get_bundle(cfg, ws, logger)
    reps = build_representations(bundle, cfg, ws, _device(cfg), logger)
    dev, val = meta_arrays(bundle.dev_meta), meta_arrays(bundle.val_meta)
    clf = get_room_classifier(cfg, ws, bundle, logger)
    room_pred_val = clf.predict(bundle.Xn_val).astype(str)
    logger.info("Predicted-room routing accuracy on validation windows: %.4f", float((room_pred_val == val.room).mean()))

    t3 = cfg.get("task3", {}) or {}
    primary = t3.get("primary", {"rep": "ssl", "head": "logreg", "variant": "predicted"})
    confusion_k = int(t3.get("confusion_k", 5))
    spec = {"rep": primary["rep"], "head": primary["head"], "variant": primary["variant"], "ks": [confusion_k]}
    routing_reps = list(t3.get("routing_reps", ["ssl"]))

    log, collector, spaces = ResultsLog(), {}, {}
    for name, (Zd, Zv) in reps.items():
        spaces[name] = make_rep_space(name, Zd, Zv, dev.room)
        variants = ["oracle", "predicted", "none"] if name in routing_reps else ["oracle"]
        run_task3_rep(spaces[name], dev, val, room_pred_val, t3, variants, log, collector, spec, logger)

    # end-to-end routed system (both rooms deployed together) at the episode K - required prediction format
    ec = cfg.get("episode", {}) or {}
    models = _export_episode_models(cfg, ws, spaces["ssl"], dev, logger)
    labels, conf = route_and_predict(spaces["ssl"].Z_val, room_pred_val, models)
    pred_df = pd.DataFrame({"window_id": val.window_id, "room": room_pred_val, "label": labels,
                            "confidence": np.round(conf, 6)})
    pred_df.to_csv(ws["results"] / "predictions_validation.csv", index=False)
    pred_df.assign(true_room=val.room, true_label=val.label, capture_id=val.cap).to_csv(
        ws["results"] / "predictions_validation_with_truth.csv", index=False)
    e2e = activity_metrics(val.label, labels, sorted(set(val.label.tolist())))
    for metric, value in e2e.items():
        log.add("task3_end_to_end_routed", "ALL", int(ec.get("k", 20)), metric, value,
                representation="ssl", head=str(ec.get("head", "logreg")), routing="predicted")
    logger.info("End-to-end routed system (K=%s) on all validation windows: macro-F1 %.4f", ec.get("k", 20), e2e["macro_f1"])

    df = log.save(ws["results"] / "task3_results.csv")
    merged = merge_results(ws["results"])

    # ---- required figures
    exp = lambda rep, head, var: f"task3_{rep}_{head}_{var}"  # noqa: E731
    plots.plot_label_efficiency(df, exp(primary["rep"], primary["head"], primary["variant"]),
                                ws["figures"] / "label_efficiency_curve.png")
    plots.plot_routing_contribution(
        df, {"predicted-room routing": exp("ssl", primary["head"], "predicted"),
             "oracle room": exp("ssl", primary["head"], "oracle"),
             "no routing": exp("ssl", primary["head"], "none")},
        ws["figures"] / "routing_contribution.png")
    plots.plot_representation_ablation(
        df, {name: exp(name, primary["head"], "oracle") for name in reps},
        ws["figures"] / "representation_ablation.png")
    n_seeds = len(t3.get("seeds", [0, 1, 2, 3, 4]))
    for target in ROOMS:
        if (target, confusion_k) not in collector:
            continue
        y_true = np.concatenate(collector[(target, confusion_k)][0])
        y_pred = np.concatenate(collector[(target, confusion_k)][1])
        labels_t = ROOM_ACTIVITIES[target]
        cm, cols = confusion_with_other(y_true, y_pred, labels_t)
        prf = per_class_report(y_true, y_pred, labels_t)
        prf["support"] = prf["support"] / n_seeds
        prf.to_csv(ws["results"] / f"per_class_metrics_target{target}_K{confusion_k}.csv", index=False)
        plots.plot_confusion(cm / n_seeds, labels_t, cols, prf,
                             f"Target room {target}, K={confusion_k} (mean over {n_seeds} support draws)",
                             ws["figures"] / f"confusion_target{target}_K{confusion_k}.png")
    write_summary(ws["results"])
    return merged


def stage_export(cfg, ws, logger):
    """(Re)build the deployable bidirectional models used by run_episode.py (uses episode.k / episode.seed)."""
    bundle = get_bundle(cfg, ws, logger)
    device = _device(cfg)
    enc = load_encoder(ws["artifacts"] / "encoder_ssl.pt", device)
    rep = make_rep_space("ssl", encode(enc, bundle.Xn_dev, device), encode(enc, bundle.Xn_val, device),
                         bundle.dev_meta["room"].to_numpy().astype(str))
    get_room_classifier(cfg, ws, bundle, logger)
    _export_episode_models(cfg, ws, rep, meta_arrays(bundle.dev_meta), logger)


# ------------------------------------------------------------------------------------------ summary
def write_summary(results_dir) -> None:
    results_dir = Path(results_dir)
    p = results_dir / "results.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    out = ["VALIDATION RESULTS (fixed validation windows; development data only used for fitting)", ""]
    t1 = df[df["experiment"] == "task1_room_id"]
    out += ["== Task 1: room identification accuracy (official 150-observation windows) ==",
            t1[["target_room", "metric", "value"]].round(4).to_string(index=False), ""]
    t2 = df[df["experiment"].str.startswith("task2_probe_joint") & (df["target_room"] == "ALL")]
    if len(t2):
        out += ["== Task 2: frozen representation, joint linear probe (both rooms) ==",
                t2.pivot_table(index="representation", columns="metric", values="value").round(4).to_string(), ""]
    t3 = df[df["experiment"].str.startswith("task3_") & df["metric"].isin(["macro_f1", "exclusive_macro_f1"])
            & df["K"].notna() & df["target_room"].isin(list(ROOMS))]
    if len(t3):
        out += ["== Task 3: few-label cross-room adaptation (mean over support draws) ==",
                t3.pivot_table(index=["experiment", "target_room"], columns=["metric", "K"], values="value")
                .round(3).to_string(), ""]
    (results_dir / "summary.txt").write_text("\n".join(out), encoding="utf-8")
