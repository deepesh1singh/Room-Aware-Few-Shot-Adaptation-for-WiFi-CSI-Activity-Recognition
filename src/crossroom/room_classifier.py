"""Task 1 - recorded-environment fingerprint (Room A vs Room B).

Operates only on the numerical content of a normalised window. The default features are the temporal
mean of the normalised log-amplitude per subcarrier: the static multipath / frequency-selective-fading
profile of the environment. It is length-robust (any number of rows works), which the window-length
study relies on. No file names, directories, capture lengths or packet counts are ever used.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def room_features(Xn: np.ndarray, kind: str = "mean") -> np.ndarray:
    mean = Xn.mean(axis=1)
    if kind == "mean":
        return mean
    if kind == "mean_std":
        return np.concatenate([mean, Xn.std(axis=1)], axis=1)
    raise ValueError(f"unknown room-classifier feature set: {kind!r}")


class RoomClassifier:
    def __init__(self, features: str = "mean", C: float = 1.0, seed: int = 42):
        self.features = features
        self.C = float(C)
        self.seed = int(seed)
        self.pipe = None
        self.classes_ = None

    def fit(self, Xn: np.ndarray, rooms) -> "RoomClassifier":
        self.pipe = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=self.C, max_iter=3000, random_state=self.seed)),
        ])
        self.pipe.fit(room_features(Xn, self.features), np.asarray(rooms))
        self.classes_ = self.pipe.named_steps["clf"].classes_
        return self

    def predict_proba(self, Xn: np.ndarray) -> np.ndarray:
        return self.pipe.predict_proba(room_features(Xn, self.features))

    def predict(self, Xn: np.ndarray) -> np.ndarray:
        return self.pipe.predict(room_features(Xn, self.features))

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "RoomClassifier":
        return joblib.load(path)


def accuracy_vs_window_length(clf: RoomClassifier, normalizer, X_val_raw: np.ndarray, rooms_val, lengths) -> list:
    """Room accuracy on the fixed validation windows cropped to their first L rows (raw rows are cropped
    first, then normalised, so a short window never sees the rows that were cut away)."""
    rooms_val = np.asarray(rooms_val)
    out = []
    for L in lengths:
        L = int(min(L, X_val_raw.shape[1]))
        pred = clf.predict(normalizer.transform(X_val_raw[:, :L, :]))
        out.append({
            "length": L,
            "accuracy": float((pred == rooms_val).mean()),
            "accuracy_A": float((pred[rooms_val == "A"] == "A").mean()) if (rooms_val == "A").any() else float("nan"),
            "accuracy_B": float((pred[rooms_val == "B"] == "B").mean()) if (rooms_val == "B").any() else float("nan"),
        })
    return out
