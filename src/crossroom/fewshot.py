"""Task 3 - cross-room few-label adaptation with environment-aware routing.

Setting for a transfer direction S -> T (source room S, target room T):
  * the source room contributes ALL labelled development windows (its whole label space);
  * the target room contributes ONLY K labelled development *captures* per target-exclusive activity
    (A: fall / sitdown / standup,  B: empty / lying) - the shared SIT/STAND/WALK of the target room are
    never labelled; all windows of a chosen capture are used but count as ONE shot;
  * validation windows are never used for support, fitting, alignment statistics or anything else.

Adaptation = frozen representation  ->  per-room standardisation of the embedding (statistics from
UNLABELLED development windows of that room, i.e. the same data the encoder was pre-trained on)  ->
small head trained on [aligned source windows + aligned K-shot support] (logistic regression with balanced
class weights, or cosine prototypes).

Routing variants evaluated on the target room's validation windows:
  oracle    : the true room selects the target-adapted model (T statistics, T label space);
  predicted : the Task-1 room classifier selects it; windows routed to the other room are answered by the
              source-only model (S statistics, S label space);
  none      : one room-agnostic model - global statistics, union label space, no room information.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from .dataset import EXCLUSIVE_ACTIVITIES, ROOM_ACTIVITIES, ROOMS, SHARED_ACTIVITIES, other_room
from .evaluation import ResultsLog, balanced_accuracy, macro_f1
from .utils import stable_int

T3_METRICS = ("macro_f1", "balanced_accuracy", "exclusive_macro_f1", "shared_macro_f1")


@dataclass
class MetaArrays:
    window_id: np.ndarray
    room: np.ndarray
    label: np.ndarray
    cap: np.ndarray


def meta_arrays(df) -> MetaArrays:
    return MetaArrays(
        window_id=df["window_id"].to_numpy().astype(str),
        room=df["room"].to_numpy().astype(str),
        label=df["label"].to_numpy().astype(str),
        cap=df["capture_id"].to_numpy().astype(str),
    )


# ------------------------------------------------------------------------------------------ alignment
class Alignment:
    """Per-room (and global) diagonal standardisation fitted on unlabelled development embeddings."""

    def __init__(self, clip: float = 20.0):
        self.clip = float(clip)
        self.stats = {}

    def fit(self, Z_dev: np.ndarray, rooms_dev: np.ndarray) -> "Alignment":
        def _stats(Z):
            mu, sd = Z.mean(axis=0), Z.std(axis=0)
            floor = 0.05 * float(np.median(sd)) + 1e-8
            return mu.astype(np.float32), np.maximum(sd, floor).astype(np.float32)

        self.stats = {r: _stats(Z_dev[rooms_dev == r]) for r in ROOMS}
        self.stats["ALL"] = _stats(Z_dev)
        return self

    def apply(self, Z: np.ndarray, key: str) -> np.ndarray:
        mu, sd = self.stats[key]
        return np.clip((Z - mu) / sd, -self.clip, self.clip).astype(np.float32)


@dataclass
class RepSpace:
    name: str
    Z_dev: np.ndarray
    Z_val: np.ndarray
    align: Alignment
    dev_al: dict  # key ('A', 'B', 'ALL') -> aligned development embeddings


def make_rep_space(name: str, Z_dev: np.ndarray, Z_val: np.ndarray, rooms_dev: np.ndarray) -> RepSpace:
    align = Alignment().fit(Z_dev, rooms_dev)
    dev_al = {k: align.apply(Z_dev, k) for k in ("A", "B", "ALL")}
    return RepSpace(name, Z_dev, Z_val, align, dev_al)


# ----------------------------------------------------------------------------------------------- heads
class LogRegHead:
    """Multinomial logistic regression with balanced class weights (support classes count as much as source classes)."""

    def __init__(self, C: float = 0.5, max_iter: int = 2000, balance_power: float = 1.0, seed: int = 42):
        self.C, self.max_iter, self.balance_power, self.seed = float(C), int(max_iter), float(balance_power), int(seed)
        self.clf = None

    def fit(self, X, y) -> "LogRegHead":
        y = np.asarray(y)
        classes, inv, counts = np.unique(y, return_inverse=True, return_counts=True)
        w_class = (len(y) / (len(classes) * counts)) ** self.balance_power
        self.clf = LogisticRegression(C=self.C, max_iter=self.max_iter, random_state=self.seed)
        self.clf.fit(X, y, sample_weight=w_class[inv])
        return self

    @property
    def classes_(self):
        return self.clf.classes_

    def predict_proba(self, X):
        return self.clf.predict_proba(X)


class ProtoHead:
    """Nearest-class-mean classifier on L2-normalised features (prototypical-network style, no training)."""

    def __init__(self, temperature: float = 10.0):
        self.temperature = float(temperature)
        self.classes_ = None
        self.protos = None

    @staticmethod
    def _l2(X):
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    def fit(self, X, y) -> "ProtoHead":
        y = np.asarray(y)
        Xn = self._l2(X)
        self.classes_ = np.unique(y)
        self.protos = self._l2(np.stack([Xn[y == c].mean(axis=0) for c in self.classes_]))
        return self

    def predict_proba(self, X):
        logits = self.temperature * (self._l2(X) @ self.protos.T)
        logits = logits - logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(axis=1, keepdims=True)


def make_head(kind: str, t3: dict, seed: int):
    if kind == "logreg":
        return LogRegHead(C=float(t3.get("logreg_C", 0.5)), max_iter=int(t3.get("logreg_max_iter", 2000)),
                          balance_power=float(t3.get("balance_power", 1.0)), seed=seed)
    if kind == "proto":
        return ProtoHead(temperature=float(t3.get("proto_temperature", 10.0)))
    raise ValueError(f"unknown head: {kind!r}")


@dataclass
class RoomModel:
    """Alignment statistics + fitted head + the label set the model may answer with (None = all classes)."""
    head: object
    stats: tuple
    allowed: object = None
    clip: float = 20.0

    def predict_proba(self, Z: np.ndarray):
        classes = np.asarray(self.head.classes_)
        if len(Z) == 0:
            return np.zeros((0, len(classes))), classes
        mu, sd = self.stats
        A = np.clip((Z - mu) / sd, -self.clip, self.clip).astype(np.float32)
        P = self.head.predict_proba(A)
        if self.allowed is not None:
            mask = np.isin(classes, list(self.allowed))
            P = P * mask[None, :]
            s = P.sum(axis=1, keepdims=True)
            uniform = mask / max(int(mask.sum()), 1)
            P = np.where(s > 1e-12, P / np.maximum(s, 1e-12), uniform[None, :])
        return P, classes

    def predict(self, Z: np.ndarray):
        P, classes = self.predict_proba(Z)
        if len(P) == 0:
            return np.array([], dtype=str), np.array([], dtype=float)
        j = P.argmax(axis=1)
        return classes[j], P[np.arange(len(j)), j]


# ----------------------------------------------------------------------------------- support selection
def select_support_captures(dev: MetaArrays, room: str, labels, K: int, seed: int) -> list:
    """K distinct development captures of `room` for every activity in `labels`.
    One random permutation per (seed, room, activity) is drawn and the first K are taken, so the
    supports are nested across K = 1 < 5 < 10 < 20 (the curves differ only by the added shots)."""
    chosen = []
    for lab in labels:
        caps = sorted(set(dev.cap[(dev.room == room) & (dev.label == lab)].tolist()))
        rng = np.random.RandomState(stable_int(seed, room, lab))
        perm = rng.permutation(len(caps))
        chosen.extend(caps[int(i)] for i in perm[:K])
    return chosen


def _t3_metrics(y_true, y_pred, target: str) -> dict:
    return {
        "macro_f1": macro_f1(y_true, y_pred, ROOM_ACTIVITIES[target]),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred),
        "exclusive_macro_f1": macro_f1(y_true, y_pred, EXCLUSIVE_ACTIVITIES[target]),
        "shared_macro_f1": macro_f1(y_true, y_pred, SHARED_ACTIVITIES),
    }


def _training_set(rep: RepSpace, dev: MetaArrays, source: str, target: str, sup_caps: list, key_source: str, key_target: str):
    src = dev.room == source
    sup = (dev.room == target) & np.isin(dev.cap, sup_caps)
    X = np.vstack([rep.dev_al[key_source][src], rep.dev_al[key_target][sup]])
    y = np.concatenate([dev.label[src], dev.label[sup]])
    return X, y


# --------------------------------------------------------------------------------------- experiments
def run_task3_rep(rep: RepSpace, dev: MetaArrays, val: MetaArrays, room_pred_val: np.ndarray, t3: dict,
                  variants: list, log: ResultsLog, collector=None, collect_spec=None, logger=None) -> None:
    """Run all (direction, head, K, seed) settings for one representation.

    collector (optional) receives pooled predictions {(target, K): ([y_true...], [y_pred...])} for the
    setting described by collect_spec = {'rep', 'head', 'variant', 'ks'}.
    """
    seeds = [int(s) for s in t3.get("seeds", [0, 1, 2, 3, 4])]
    ks = [int(k) for k in t3.get("ks", [1, 5, 10, 20])]
    heads = list(t3.get("heads", ["logreg", "proto"]))

    for target in ROOMS:
        source = other_room(target)
        direction = f"{source}->{target}"
        src_mask = dev.room == source
        tv = val.room == target
        y_true = val.label[tv]
        rhat = room_pred_val[tv]
        Zt = rep.Z_val[tv]

        for head_kind in heads:
            model_src = None
            if "predicted" in variants:
                head_s = make_head(head_kind, t3, 0).fit(rep.dev_al[source][src_mask], dev.label[src_mask])
                model_src = RoomModel(head_s, rep.align.stats[source], list(ROOM_ACTIVITIES[source]))

            for K in ks:
                per_variant = {v: [] for v in variants}
                for seed in seeds:
                    sup_caps = select_support_captures(dev, target, EXCLUSIVE_ACTIVITIES[target], K, seed)
                    X_t, y_t = _training_set(rep, dev, source, target, sup_caps, source, target)
                    head_t = make_head(head_kind, t3, seed).fit(X_t, y_t)
                    model_t = RoomModel(head_t, rep.align.stats[target], list(ROOM_ACTIVITIES[target]))
                    pred_oracle, _ = model_t.predict(Zt)

                    preds = {"oracle": pred_oracle}
                    if "predicted" in variants:
                        pred_pred = pred_oracle.astype("<U16")  # wide dtype: routed labels may be longer
                        mis = rhat != target
                        if mis.any():
                            pred_pred[mis] = model_src.predict(Zt[mis])[0]
                        preds["predicted"] = pred_pred
                    if "none" in variants:
                        X_g, y_g = _training_set(rep, dev, source, target, sup_caps, "ALL", "ALL")
                        head_g = make_head(head_kind, t3, seed).fit(X_g, y_g)
                        preds["none"] = RoomModel(head_g, rep.align.stats["ALL"], None).predict(Zt)[0]

                    for v in variants:
                        per_variant[v].append(_t3_metrics(y_true, preds[v], target))
                        if (collector is not None and collect_spec is not None and rep.name == collect_spec["rep"]
                                and head_kind == collect_spec["head"] and v == collect_spec["variant"]
                                and K in collect_spec["ks"]):
                            ys = collector.setdefault((target, K), ([], []))
                            ys[0].append(y_true)
                            ys[1].append(preds[v])

                for v in variants:
                    for metric in T3_METRICS:
                        vals = np.array([m[metric] for m in per_variant[v]], dtype=float)
                        log.add(f"task3_{rep.name}_{head_kind}_{v}", target, K, metric, vals.mean(),
                                direction=direction, representation=rep.name, head=head_kind, routing=v,
                                std=float(vals.std()), n_seeds=len(seeds))
                if logger:
                    m0 = {v: float(np.mean([m['macro_f1'] for m in per_variant[v]])) for v in variants}
                    logger.info("task3 rep=%s head=%s %s K=%d macro-F1 %s", rep.name, head_kind, direction, K,
                                {k: round(x, 3) for k, x in m0.items()})


# ------------------------------------------------------------------------------ end-to-end prediction
def build_room_models(rep: RepSpace, dev: MetaArrays, K: int, seed: int, head_kind: str, t3: dict) -> dict:
    """One adapted model per room (that room as target, the other as source) - the deployed bidirectional system."""
    models = {}
    for target in ROOMS:
        source = other_room(target)
        sup_caps = select_support_captures(dev, target, EXCLUSIVE_ACTIVITIES[target], K, seed)
        X, y = _training_set(rep, dev, source, target, sup_caps, source, target)
        head = make_head(head_kind, t3, seed).fit(X, y)
        models[target] = RoomModel(head, rep.align.stats[target], list(ROOM_ACTIVITIES[target]))
    return models


def route_and_predict(Z: np.ndarray, rooms_pred: np.ndarray, models: dict):
    """Environment-aware routing: each window is answered by the model of its predicted room."""
    labels = np.empty(len(Z), dtype=object)
    conf = np.zeros(len(Z), dtype=float)
    for room in ROOMS:
        m = rooms_pred == room
        if m.any():
            lab, c = models[room].predict(Z[m])
            labels[m] = lab
            conf[m] = c
    return labels.astype(str), conf
