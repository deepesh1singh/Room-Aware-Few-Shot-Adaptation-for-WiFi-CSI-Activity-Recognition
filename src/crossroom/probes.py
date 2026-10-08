"""Task 2 - evaluation of frozen representations with linear probes.

For every representation (self-supervised encoder, same-architecture random-weight encoder,
handcrafted features) a multinomial logistic-regression probe is fitted on LABELLED DEVELOPMENT windows
only and scored on the fixed validation windows:
  joint      : both rooms pooled, union label space (headline), plus the per-room breakdown;
  in_room    : fitted and scored inside one room (that room's own label space);
  cross_room : fitted on the shared activities (SIT/STAND/WALK) of one room, scored on the other room.
No per-room alignment is applied here - the probe sees the representation as it is.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .dataset import ROOM_ACTIVITIES, ROOMS, SHARED_ACTIVITIES, other_room
from .evaluation import ResultsLog, activity_metrics


def fit_predict_probe(Z_train, y_train, Z_test, C: float = 1.0, seed: int = 42):
    scaler = StandardScaler().fit(Z_train)
    clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced", random_state=seed)
    clf.fit(scaler.transform(Z_train), y_train)
    return clf.predict(scaler.transform(Z_test))


def run_task2(reps: dict, room_dev, label_dev, room_val, label_val, C: float, seed: int, log: ResultsLog) -> None:
    """reps: {name: (Z_dev, Z_val)}; label / room arrays are numpy string arrays."""
    for name, (Zd, Zv) in reps.items():
        def add(experiment, room, metrics, **extra):
            for metric, value in metrics.items():
                log.add(experiment, room, None, metric, value, representation=name, **extra)

        # joint probe (both rooms, union label space)
        pred = fit_predict_probe(Zd, label_dev, Zv, C, seed)
        add(f"task2_probe_joint_{name}", "ALL", activity_metrics(label_val, pred, sorted(set(label_val))),
            probe="joint")
        for room in ROOMS:
            m = room_val == room
            add(f"task2_probe_joint_{name}", room, activity_metrics(label_val[m], pred[m], ROOM_ACTIVITIES[room]),
                probe="joint")

        # in-room probes
        for room in ROOMS:
            tr, te = room_dev == room, room_val == room
            pred_r = fit_predict_probe(Zd[tr], label_dev[tr], Zv[te], C, seed)
            add(f"task2_probe_inroom_{name}", room, activity_metrics(label_val[te], pred_r, ROOM_ACTIVITIES[room]),
                probe="in_room")

        # cross-room probes on the shared activities only
        for target in ROOMS:
            source = other_room(target)
            tr = (room_dev == source) & np.isin(label_dev, SHARED_ACTIVITIES)
            te = (room_val == target) & np.isin(label_val, SHARED_ACTIVITIES)
            pred_c = fit_predict_probe(Zd[tr], label_dev[tr], Zv[te], C, seed)
            add(f"task2_probe_cross_{name}", target, activity_metrics(label_val[te], pred_c, SHARED_ACTIVITIES),
                probe="cross_room", direction=f"{source}->{target}")
