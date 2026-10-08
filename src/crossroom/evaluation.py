"""Metrics and the experiment-results log.

Results rows follow the required schema `experiment, target_room, K, metric, value`; extra columns
(representation, head, routing, direction, std, ...) are appended after the required ones.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

REQUIRED_COLUMNS = ["experiment", "target_room", "K", "metric", "value"]


def balanced_accuracy(y_true, y_pred) -> float:
    """Mean per-class recall over the classes present in y_true (predictions outside count as errors)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    recalls = [float((y_pred[y_true == c] == c).mean()) for c in np.unique(y_true)]
    return float(np.mean(recalls)) if recalls else float("nan")


def macro_f1(y_true, y_pred, labels) -> float:
    return float(f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0))


def activity_metrics(y_true, y_pred, labels) -> dict:
    return {
        "macro_f1": macro_f1(y_true, y_pred, labels),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def per_class_report(y_true, y_pred, labels) -> pd.DataFrame:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=list(labels), zero_division=0)
    return pd.DataFrame({"activity": list(labels), "precision": p, "recall": r, "f1": f, "support": s})


def confusion_with_other(y_true, y_pred, labels):
    """Confusion matrix over `labels` plus an 'other' column that collects predictions outside `labels`.
    Returns (matrix with len(labels) rows and len(labels)+1 columns, column names)."""
    labels = list(labels)
    y_pred_m = np.where(np.isin(y_pred, labels), y_pred, "other")
    cm = confusion_matrix(y_true, y_pred_m, labels=labels + ["other"])
    return cm[: len(labels)], labels + ["other"]


class ResultsLog:
    def __init__(self):
        self.rows = []

    def add(self, experiment: str, target_room, K, metric: str, value, **extra) -> None:
        row = {"experiment": experiment, "target_room": target_room, "K": K, "metric": metric,
               "value": float(value)}
        row.update(extra)
        self.rows.append(row)

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.rows)
        if df.empty:
            return pd.DataFrame(columns=REQUIRED_COLUMNS)
        rest = [c for c in df.columns if c not in REQUIRED_COLUMNS]
        return df[REQUIRED_COLUMNS + rest]

    def save(self, path) -> pd.DataFrame:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_frame()
        df.to_csv(path, index=False)
        return df


def merge_results(results_dir) -> pd.DataFrame:
    """Concatenate the per-task result tables into results/results.csv."""
    results_dir = Path(results_dir)
    parts = []
    for name in ("task1_results.csv", "task2_results.csv", "task3_results.csv"):
        p = results_dir / name
        if p.exists():
            parts.append(pd.read_csv(p))
    if not parts:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    df = pd.concat(parts, ignore_index=True)
    rest = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    df = df[REQUIRED_COLUMNS + rest]
    df.to_csv(results_dir / "results.csv", index=False)
    return df
