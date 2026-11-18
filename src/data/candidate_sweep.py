"""Justify the candidate-generation settings (radius, K) with measured evidence.

For each (radius, K) it reports, on the TRAINING trips only:
  - candidate miss rate: reference arc not among the generated candidates (this is the ceiling on Top-1)
  - mean candidates per observation (cost: table size, training and inference time)
Run: python -m src.data.candidate_sweep
Writes experiments/results/candidate_sweep.csv
"""
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from src import config as C
from src.data.trip_loader import load_trip, list_trips
from src.data.candidates import build_arc_tree, generate_candidates, reference_arcs
from src.data.build_tables import select_obs

RADII = [50.0, 100.0, 150.0, 200.0, 300.0]
KS = [5, 8, 12]


def one_trip(trip_id):
    trip = load_trip(C.DATASET_DIR, "train", trip_id)
    obs_idx = select_obs(len(trip.x), "train")
    tree, a, b = build_arc_tree(trip)
    ref, rdist = reference_arcs(trip, obs_idx)
    out = []
    for r in RADII:
        cand = generate_candidates(trip, obs_idx, r, max(KS), tree, (a, b))
        for k in KS:
            m = cand["rank"] < k
            oi, arc = cand["obs_pos"][m], cand["arc"][m]
            hit = np.zeros(len(obs_idx), dtype=bool)
            ok = (arc == ref[oi]) & (rdist[oi] <= r)
            hit[oi[ok]] = True
            out.append(dict(radius=r, k=k, n_obs=len(obs_idx), n_hit=int(hit.sum()), n_cand=int(m.sum())))
    return out


if __name__ == "__main__":
    trips = list_trips(C.DATASET_DIR, "train")
    res = Parallel(n_jobs=2, verbose=1)(delayed(one_trip)(t) for t in trips)
    df = pd.DataFrame([r for sub in res for r in sub])
    g = df.groupby(["radius", "k"]).sum(numeric_only=True)
    g["miss_rate"] = 1 - g["n_hit"] / g["n_obs"]
    g["mean_candidates_per_obs"] = g["n_cand"] / g["n_obs"]
    g = g[["n_obs", "miss_rate", "mean_candidates_per_obs"]].reset_index()
    g.to_csv(C.RESULTS_DIR / "candidate_sweep.csv", index=False)
    print(g.round(4).to_string(index=False))
