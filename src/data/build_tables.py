"""Build the candidate/feature tables for all trips (run once; results cached in data/processed/).

Usage: python -m src.data.build_tables [--n-jobs 2]
"""
import argparse, time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from src import config as C
from src.data.trip_loader import load_trip, list_trips
from src.data.candidates import build_arc_tree, generate_candidates, reference_arcs
from src.features.features import build_features


def select_obs(n_obs, partition):
    if partition == "train" and n_obs > C.TRAIN_MAX_OBS_PER_TRIP:
        return np.unique(np.linspace(0, n_obs - 1, C.TRAIN_MAX_OBS_PER_TRIP).astype(int))
    return np.arange(n_obs)


def process_trip(partition, trip_id):
    t0 = time.time()
    trip = load_trip(C.DATASET_DIR, partition, trip_id)
    obs_idx = select_obs(len(trip.x), partition)
    tree, a, b = build_arc_tree(trip)
    cand = generate_candidates(trip, obs_idx, C.CAND_RADIUS_M, C.CAND_K, tree, (a, b))
    df = build_features(trip, obs_idx, cand, C.WINDOW_OFFSETS)
    ref, rdist = reference_arcs(trip, obs_idx)
    pos = {o: i for i, o in enumerate(obs_idx)}
    ref_of_obs = ref[np.searchsorted(obs_idx, df["obs"].to_numpy())]
    df["label"] = ((df["arc"].to_numpy() == ref_of_obs) & (rdist[np.searchsorted(obs_idx, df["obs"].to_numpy())] <= C.CAND_RADIUS_M)).astype(np.int8)
    df.insert(0, "trip", trip_id)
    df.insert(1, "partition", partition)
    # per-observation summary (candidate miss rate etc.)
    has_cand = np.isin(obs_idx, df["obs"].unique())
    hit = np.isin(obs_idx, df.loc[df["label"] == 1, "obs"].unique())
    summ = pd.DataFrame({"trip": trip_id, "partition": partition, "obs": obs_idx, "ref_arc": ref,
                         "ref_dist": rdist, "has_candidates": has_cand, "ref_in_candidates": hit})
    info = {"trip": trip_id, "partition": partition, "n_obs_total": len(trip.x), "n_obs_used": len(obs_idx),
            "n_nodes": len(trip.nodes_xy), "n_arcs": len(trip.arcs), "n_route_arcs": len(trip.route),
            "epsg": trip.epsg, "seconds": round(time.time() - t0, 1)}
    return df, summ, info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N trips per partition")
    args = ap.parse_args()
    jobs = []
    for part in ["train", "student_test"]:
        ids = list_trips(C.DATASET_DIR, part)
        jobs += [(part, t) for t in (ids[: args.limit] if args.limit else ids)]
    res = Parallel(n_jobs=args.n_jobs, verbose=5)(delayed(process_trip)(p, t) for p, t in jobs)
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    tab = pd.concat([r[0] for r in res], ignore_index=True)
    summ = pd.concat([r[1] for r in res], ignore_index=True)
    info = pd.DataFrame([r[2] for r in res])
    tab.to_parquet(C.PROCESSED_DIR / "candidates_features.parquet")
    summ.to_parquet(C.PROCESSED_DIR / "obs_summary.parquet")
    info.to_csv(C.RESULTS_DIR / "trip_inventory.csv", index=False)
    # Candidate Miss Rate is a required reported metric, so keep it as a committed CSV
    miss = summ.groupby("partition").agg(
        n_observations=("obs", "size"),
        obs_without_any_candidate=("has_candidates", lambda x: int((~x).sum())),
        candidate_miss_rate=("ref_in_candidates", lambda x: 1 - x.mean()),
        mean_ref_dist_m=("ref_dist", lambda x: x[np.isfinite(x)].mean()),
    ).reset_index()
    miss.insert(1, "radius_m", C.CAND_RADIUS_M); miss.insert(2, "k", C.CAND_K)
    miss.to_csv(C.RESULTS_DIR / "candidate_miss_rate.csv", index=False)
    summ.groupby(["partition", "trip"])["ref_in_candidates"].agg(
        n_observations="size", candidate_miss_rate=lambda x: 1 - x.mean()
    ).reset_index().to_csv(C.RESULTS_DIR / "candidate_miss_rate_per_trip.csv", index=False)
    print(miss.round(4).to_string(index=False))
    print(tab.shape, tab["label"].mean())
    print(summ.groupby("partition")[["has_candidates", "ref_in_candidates"]].mean())
