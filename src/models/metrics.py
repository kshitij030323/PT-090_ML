"""Metrics used for model comparison (candidate level + observation level + route continuity)."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, log_loss, accuracy_score


def top1_accuracy(df, proba):
    """Observation-level Top-1 accuracy: highest-scoring candidate per observation == reference arc.
    Denominator = all observations (an observation whose reference arc is not in the candidate set counts as wrong)."""
    t = df[["trip", "obs", "label"]].copy()
    t["p"] = proba
    idx = t.groupby(["trip", "obs"])["p"].idxmax()
    picked = t.loc[idx]
    n_obs_with_cand = len(picked)
    correct = int(picked["label"].sum())
    return correct, n_obs_with_cand


def candidate_metrics(y, proba, thr=0.5):
    pred = (proba >= thr).astype(int)
    return {
        "macro_f1": f1_score(y, pred, average="macro"),
        "pos_f1": f1_score(y, pred, pos_label=1),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "accuracy": accuracy_score(y, pred),
        "log_loss": log_loss(y, np.clip(proba, 1e-7, 1 - 1e-7), labels=[0, 1]),
    }


def route_continuity(df, proba, arcs_by_trip):
    """Fraction of consecutive observations (in time order) whose predicted arcs are the same arc or directly
    connected (end node of previous arc == start node of next arc). Computed per trip, pooled over all pairs."""
    t = df[["trip", "obs", "arc"]].copy()
    t["p"] = proba
    best = t.loc[t.groupby(["trip", "obs"])["p"].idxmax()].sort_values(["trip", "obs"])
    ok = tot = 0
    for trip, g in best.groupby("trip"):
        arcs = arcs_by_trip[trip]
        a = g["arc"].to_numpy()
        if len(a) < 2:
            continue
        prev, nxt = a[:-1], a[1:]
        conn = (prev == nxt) | (arcs[prev, 1] == arcs[nxt, 0])
        ok += int(conn.sum()); tot += len(conn)
    return ok / max(tot, 1)
