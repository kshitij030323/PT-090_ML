"""Feature engineering for (GPS observation, candidate arc) pairs.

Every row is one candidate arc for one GPS observation; the classifier predicts P(arc is the reference arc).
Only information available at inference time is used (GPS track + road network), never the .route file.
"""
import numpy as np
import pandas as pd
from src.data.candidates import arc_endpoints, point_segment

FEATURE_COLS = [
    "d", "d_gap", "d_ratio", "rank", "n_cand", "frac", "lat_signed", "arc_len",
    "cos_ang", "sin_ang", "abs_ang", "speed", "heading_ok", "dt_prev",
    "d_m2", "d_m1", "d_p1", "d_p2", "d_win_mean", "d_win_max", "rank_win",
    "out_deg_to", "in_deg_from", "has_reverse",
]


def build_features(trip, obs_idx, cand, window_offsets):
    n = len(trip.x)
    a, b = arc_endpoints(trip)
    oi = cand["obs_pos"]
    obs = obs_idx[oi]                 # absolute observation index in the trip
    arc = cand["arc"]
    df = pd.DataFrame({"obs": obs, "arc": arc, "d": cand["d"], "frac": cand["frac"],
                       "lat_signed": cand["lat"], "rank": cand["rank"]})
    df["n_cand"] = df.groupby("obs")["d"].transform("size")
    dmin = df.groupby("obs")["d"].transform("min")
    df["d_gap"] = df["d"] - dmin
    df["d_ratio"] = df["d"] / (dmin + 1.0)

    ax, ay, bx, by = a[arc, 0], a[arc, 1], b[arc, 0], b[arc, 1]
    vx, vy = bx - ax, by - ay
    L = np.hypot(vx, vy)
    df["arc_len"] = L

    # observation heading from the neighbouring GPS points (central difference) and speed
    lo, hi = np.clip(obs - 1, 0, n - 1), np.clip(obs + 1, 0, n - 1)
    hx, hy = trip.x[hi] - trip.x[lo], trip.y[hi] - trip.y[lo]
    hn = np.hypot(hx, hy)
    dtt = np.maximum(trip.t[hi] - trip.t[lo], 1e-3)
    df["speed"] = np.minimum(hn / dtt, 80.0)
    ok = (hn > 2.0) & (L > 0)
    df["heading_ok"] = ok.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        cosv = np.where(ok, (hx * vx + hy * vy) / (hn * L), 0.0)
        sinv = np.where(ok, (hx * vy - hy * vx) / (hn * L), 0.0)
    df["cos_ang"], df["sin_ang"] = cosv, sinv
    df["abs_ang"] = np.where(ok, np.abs(np.arctan2(sinv, cosv)), 0.0)
    df["dt_prev"] = trip.t[obs] - trip.t[np.clip(obs - 1, 0, n - 1)]

    # context: distance of neighbouring GPS points to the SAME candidate arc
    ds = []
    for name, off in zip(["d_m2", "d_m1", "d_p1", "d_p2"], window_offsets):
        j = np.clip(obs + off, 0, n - 1)
        dj, _, _ = point_segment(trip.x[j], trip.y[j], ax, ay, bx, by)
        df[name] = dj
        ds.append(dj)
    allw = np.vstack([df["d"].to_numpy()] + ds)
    df["d_win_mean"] = allw.mean(0)
    df["d_win_max"] = allw.max(0)
    df["rank_win"] = df.groupby("obs")["d_win_mean"].rank(method="first") - 1

    # network topology
    out_deg = np.bincount(trip.arcs[:, 0], minlength=len(trip.nodes_xy))
    in_deg = np.bincount(trip.arcs[:, 1], minlength=len(trip.nodes_xy))
    df["out_deg_to"] = out_deg[trip.arcs[arc, 1]]
    df["in_deg_from"] = in_deg[trip.arcs[arc, 0]]
    N = len(trip.nodes_xy)
    keys = np.unique(trip.arcs[:, 0].astype(np.int64) * N + trip.arcs[:, 1])
    rev = trip.arcs[arc, 1].astype(np.int64) * N + trip.arcs[arc, 0]
    df["has_reverse"] = np.isin(rev, keys).astype(float)
    return df
