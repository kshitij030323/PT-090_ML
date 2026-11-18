"""Structured EDA over all trips. Writes CSVs to experiments/results/eda_*.csv"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import networkx as nx
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data import loader as L

OUT = Path(__file__).resolve().parents[2] / "experiments" / "results"
OUT.mkdir(parents=True, exist_ok=True)


def haversine(lon1, lat1, lon2, lat2):
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


rows = []
for part in L.PARTITIONS:
    for trip in L.list_trips(part):
        tr, nd, ar, rt = (L.load_track(part, trip), L.load_nodes(part, trip),
                          L.load_arcs(part, trip), L.load_route(part, trip))
        dt = np.diff(tr.t.to_numpy())
        step = haversine(tr.lon[:-1].to_numpy(), tr.lat[:-1].to_numpy(),
                         tr.lon[1:].to_numpy(), tr.lat[1:].to_numpy())
        G = nx.Graph(); G.add_nodes_from(range(len(nd))); G.add_edges_from(ar.to_numpy())
        comps = list(nx.connected_components(G))
        deg = np.bincount(np.concatenate([ar.from_node, ar.to_node]), minlength=len(nd))
        # route validity: arc ids in range, and consecutive arcs connected (to_node == next from_node)
        valid = rt.max() < len(ar) and rt.min() >= 0
        if valid:
            fr, to = ar.from_node.to_numpy(), ar.to_node.to_numpy()
            conn = float(np.mean(to[rt[:-1]] == fr[rt[1:]])) if len(rt) > 1 else np.nan
        else:
            conn = np.nan
        rows.append(dict(
            partition=part, trip=trip, n_obs=len(tr), n_nodes=len(nd), n_arcs=len(ar), n_route=len(rt),
            n_missing=int(tr.isna().sum().sum()), t_sorted=bool((dt > 0).all()), n_dup_t=int((dt == 0).sum()),
            n_neg_dt=int((dt < 0).sum()), dt_median=float(np.median(dt)), dt_max=float(dt.max()),
            frac_dt_1=float((dt == 1).mean()), n_gaps_gt5=int((dt > 5).sum()),
            duration_s=float(tr.t.iloc[-1] - tr.t.iloc[0]), path_len_m=float(step.sum()),
            max_step_m=float(step.max()),
            lon_min=tr.lon.min(), lon_max=tr.lon.max(), lat_min=tr.lat.min(), lat_max=tr.lat.max(),
            extent_km=float(haversine(tr.lon.min(), tr.lat.min(), tr.lon.max(), tr.lat.max()) / 1000),
            net_components=len(comps), net_largest_comp=max(map(len, comps)),
            isolated_nodes=int((deg == 0).sum()),
            route_arc_ids_valid=bool(valid), route_consecutive_connected_frac=conn,
            route_unique_arcs=len(set(rt.tolist())),
            net_lon_min=nd.lon.min(), net_lon_max=nd.lon.max(), net_lat_min=nd.lat.min(), net_lat_max=nd.lat.max(),
        ))
        print(part, trip, "done", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT / "eda_trip_inventory.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 50)
print("\n== counts per partition\n", df.groupby("partition").size())
print("\n== describe (train)\n", df[df.partition == "train"].drop(columns=["trip"]).describe().T)
print("\n== describe (student_test)\n", df[df.partition == "student_test"].drop(columns=["trip"]).describe().T)
