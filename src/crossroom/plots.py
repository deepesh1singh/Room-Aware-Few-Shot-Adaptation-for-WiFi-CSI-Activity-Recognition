"""Required figures (matplotlib, headless)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOM_COLORS = {"A": "#1f77b4", "B": "#d62728"}


def _save(fig, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _series(df, experiment, target, metric):
    d = df[(df["experiment"] == experiment) & (df["target_room"] == target) & (df["metric"] == metric)]
    d = d.sort_values("K")
    return d["K"].to_numpy(dtype=float), d["value"].to_numpy(dtype=float), d["std"].to_numpy(dtype=float)


def plot_ssl_loss(history, path) -> None:
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(np.arange(1, len(history) + 1), history)
    ax.set_xlabel("epoch")
    ax.set_ylabel("NT-Xent loss")
    ax.set_title("Self-supervised pre-training")
    _save(fig, path)


def plot_label_efficiency(df, experiment: str, path) -> None:
    """macro-F1 against K per target room; separate curves for target-labelled (exclusive) and transferred
    (shared) activities, plus the overall primary macro-F1."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    specs = [("macro_f1", "All activities (primary macro-F1)", "k", "-"),
             ("exclusive_macro_f1", "Target-labelled (target-exclusive) activities", "#2ca02c", "--"),
             ("shared_macro_f1", "Transferred shared activities (SIT/STAND/WALK)", "#ff7f0e", ":")]
    for ax, target in zip(axes, ("A", "B")):
        for metric, label, color, ls in specs:
            k, v, sd = _series(df, experiment, target, metric)
            if len(k):
                ax.errorbar(k, v, yerr=sd, label=label, color=color, linestyle=ls, marker="o", capsize=3)
        ax.set_xscale("log")
        ax.set_xticks([1, 5, 10, 20])
        ax.set_xticklabels(["1", "5", "10", "20"])
        ax.set_xlabel("K labelled target captures per target-exclusive activity")
        src = "B" if target == "A" else "A"
        ax.set_title(f"Target room {target} (source {src})")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("macro-F1 (validation)")
    axes[0].legend(fontsize=7, loc="lower right")
    _save(fig, path)


def plot_representation_ablation(df, experiments: dict, path) -> None:
    """macro-F1 against K for the compared representations (SSL / random weights / handcrafted)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    colors = {"ssl": "#1f77b4", "random": "#7f7f7f", "handcrafted": "#9467bd"}
    for ax, target in zip(axes, ("A", "B")):
        for name, exp in experiments.items():
            k, v, sd = _series(df, exp, target, "macro_f1")
            if len(k):
                ax.errorbar(k, v, yerr=sd, label=name, color=colors.get(name), marker="o", capsize=3)
        ax.set_xscale("log")
        ax.set_xticks([1, 5, 10, 20])
        ax.set_xticklabels(["1", "5", "10", "20"])
        ax.set_xlabel("K")
        ax.set_title(f"Target room {target}")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("macro-F1 (validation)")
    axes[0].legend(title="representation")
    _save(fig, path)


def plot_routing_contribution(df, experiments: dict, path) -> None:
    """Predicted-room routing vs oracle-room vs the same adaptation model without the routing stage."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    colors = {"predicted-room routing": "#1f77b4", "oracle room": "#2ca02c", "no routing": "#d62728"}
    for ax, target in zip(axes, ("A", "B")):
        ks, width = None, 0.26
        for i, (name, exp) in enumerate(experiments.items()):
            k, v, sd = _series(df, exp, target, "macro_f1")
            if not len(k):
                continue
            ks = k
            x = np.arange(len(k)) + (i - 1) * width
            ax.bar(x, v, width=width, yerr=sd, label=name, color=colors.get(name), capsize=2)
        if ks is not None:
            ax.set_xticks(np.arange(len(ks)))
            ax.set_xticklabels([str(int(k)) for k in ks])
        ax.set_xlabel("K")
        ax.set_title(f"Target room {target}")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("macro-F1 (validation)")
    axes[0].legend(fontsize=8, loc="lower right")
    _save(fig, path)


def plot_fingerprint_curve(rows: list, path) -> None:
    """Room accuracy against window length on the fixed validation capture pool."""
    L = [r["length"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(L, [r["accuracy"] for r in rows], "ko-", label="overall")
    ax.plot(L, [r["accuracy_A"] for r in rows], "-s", color=ROOM_COLORS["A"], label="Room A windows", alpha=0.8)
    ax.plot(L, [r["accuracy_B"] for r in rows], "-^", color=ROOM_COLORS["B"], label="Room B windows", alpha=0.8)
    ax.set_xlabel("window length (observations)")
    ax.set_ylabel("room accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.3)
    ax.set_title("Environment fingerprint vs window length")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_encoder_diagnosis(emb2d: np.ndarray, rooms, labels, method: str, path) -> None:
    """2-D embedding of validation windows coloured once by room and once by activity (qualitative)."""
    rooms, labels = np.asarray(rooms), np.asarray(labels)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for r, c in ROOM_COLORS.items():
        m = rooms == r
        axes[0].scatter(emb2d[m, 0], emb2d[m, 1], s=9, alpha=0.7, color=c, label=f"Room {r}")
    axes[0].set_title(f"{method} of frozen SSL embeddings - coloured by room")
    axes[0].legend(markerscale=2)
    cmap = plt.get_cmap("tab10")
    for i, lab in enumerate(sorted(set(labels.tolist()))):
        m = labels == lab
        axes[1].scatter(emb2d[m, 0], emb2d[m, 1], s=9, alpha=0.7, color=cmap(i % 10), label=lab)
    axes[1].set_title(f"{method} - coloured by activity")
    axes[1].legend(markerscale=2, fontsize=8, ncol=2)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    _save(fig, path)


def plot_confusion(cm: np.ndarray, row_labels, col_labels, prf, title: str, path) -> None:
    """Confusion matrix (row-normalised colours, counts annotated) + per-activity precision / recall / F1 table."""
    fig, (ax, ax_t) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
    row_sum = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, np.maximum(row_sum, 1), dtype=float)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    fmt = "{:.0f}" if np.allclose(cm, np.round(cm)) else "{:.1f}"
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, fmt.format(cm[i, j]), ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax_t.axis("off")
    cell = [[r.activity, f"{r.precision:.2f}", f"{r.recall:.2f}", f"{r.f1:.2f}", f"{r.support:.0f}"]
            for r in prf.itertuples()]
    table = ax_t.table(cellText=cell, colLabels=["activity", "precision", "recall", "F1", "support"],
                       loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.4)
    _save(fig, path)
