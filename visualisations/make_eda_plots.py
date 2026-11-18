"""Real EDA plots + Folium maps from the provided dataset. Run: python3 -I visualisations/make_eda_plots.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import folium
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import loader as L

OUT = ROOT / "visualisations"; RES = ROOT / "experiments" / "results"
inv = pd.read_csv(RES / "eda_trip_inventory.csv", dtype={"trip": str})
col = {"train": "#1f77b4", "student_test": "#d62728"}

def local_xy(lon, lat, lat0, lon0):
    return (np.asarray(lon) - lon0) * 111320 * np.cos(np.radians(lat0)), (np.asarray(lat) - lat0) * 110540

# 1. missing-value table
mv = inv.groupby("partition")[["n_missing", "n_dup_t", "n_neg_dt", "isolated_nodes"]].sum()
mv["trips"] = inv.groupby("partition").size(); mv.to_csv(RES / "eda_missing_values.csv"); print(mv)

# 2. distributions
fig, ax = plt.subplots(2, 3, figsize=(16, 8))
for p, g in inv.groupby("partition"):
    ax[0,0].hist(g.n_obs, bins=np.logspace(2, 4.4, 25), alpha=.6, color=col[p], label=f"{p} (n={len(g)})")
    ax[0,1].hist(g.frac_dt_1, bins=20, alpha=.6, color=col[p], label=p)
    ax[0,2].hist(g.dt_max.clip(lower=1), bins=np.logspace(0, 5.3, 25), alpha=.6, color=col[p], label=p)
    ax[1,0].scatter(g.n_obs, g.n_arcs, s=18, alpha=.7, color=col[p], label=p)
    ax[1,1].hist(g.net_components, bins=np.logspace(0, 2.7, 25), alpha=.6, color=col[p], label=p)
    ax[1,2].hist(g.extent_km, bins=np.logspace(0, 2, 25), alpha=.6, color=col[p], label=p)
for a, t, xl in [(ax[0,0],"GPS observations per trip","observations"),(ax[0,1],"Fraction of steps exactly 1 s","fraction"),
                 (ax[0,2],"Largest time gap per trip","seconds"),(ax[1,0],"Road-network arcs vs observations","observations"),
                 (ax[1,1],"Connected components per network","components"),(ax[1,2],"Trajectory extent (diagonal)","km")]:
    a.set_title(t); a.set_xlabel(xl); a.legend()
for a in (ax[0,0], ax[0,2], ax[1,1], ax[1,2]): a.set_xscale("log")
ax[1,0].set_xscale("log"); ax[1,0].set_yscale("log"); ax[1,0].set_ylabel("arcs")
for a in (ax[0,0],ax[0,1],ax[0,2],ax[1,1],ax[1,2]): a.set_ylabel("trips")
fig.suptitle("PathSnap EDA – train (80 trips) vs student_test (10 trips)"); fig.tight_layout()
fig.savefig(OUT / "eda_distributions.png", dpi=130); plt.close(fig)

# 3. where trips are
fig, a = plt.subplots(figsize=(11, 5))
for p, g in inv.groupby("partition"):
    a.scatter((g.lon_min+g.lon_max)/2, (g.lat_min+g.lat_max)/2, s=30, color=col[p], label=p, alpha=.8)
a.set_xlabel("longitude"); a.set_ylabel("latitude"); a.set_title("Trip locations (centroid of each trajectory)"); a.legend(); a.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT / "eda_trip_locations.png", dpi=130); plt.close(fig)

# 4. GPS offset from reference route (point -> nearest route arc segment), all trips
def pt_seg_dist(P, A, B):
    d = B - A; l2 = (d**2).sum(1); l2[l2 == 0] = 1e-12
    out = np.empty(len(P))
    for i in range(0, len(P), 500):
        p = P[i:i+500, None, :]
        t = np.clip(((p - A[None]) * d[None]).sum(2) / l2[None], 0, 1)
        proj = A[None] + t[..., None] * d[None]
        out[i:i+500] = np.sqrt(((p - proj)**2).sum(2)).min(1)
    return out
offs = []
for _, r in inv.iterrows():
    part, trip = r.partition, r.trip
    tr, nd, ar, rt = L.load_track(part, trip), L.load_nodes(part, trip), L.load_arcs(part, trip), L.load_route(part, trip)
    lat0, lon0 = tr.lat.mean(), tr.lon.mean()
    nx_, ny_ = local_xy(nd.lon, nd.lat, lat0, lon0)
    X = np.c_[nx_, ny_]; fr, to = ar.from_node.to_numpy()[rt], ar.to_node.to_numpy()[rt]
    px, py = local_xy(tr.lon, tr.lat, lat0, lon0)
    d = pt_seg_dist(np.c_[px, py], X[fr], X[to])
    offs.append(pd.DataFrame({"partition": part, "trip": trip, "offset_m": d}))
off = pd.concat(offs); off.to_csv(RES / "eda_gps_offset_from_route.csv", index=False)
summ = off.groupby("partition").offset_m.describe(percentiles=[.5,.9,.95,.99]); summ.to_csv(RES / "eda_gps_offset_summary.csv"); print(summ.T)
per_trip = off.groupby("trip").offset_m.agg(["median", lambda s: s.quantile(.95), "max"]); per_trip.columns = ["median","p95","max"]
per_trip.to_csv(RES / "eda_gps_offset_per_trip.csv")
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for p, g in off.groupby("partition"):
    ax[0].hist(g.offset_m.clip(upper=100), bins=100, alpha=.6, color=col[p], density=True, label=p)
ax[0].set_xlabel("distance GPS point → nearest reference-route arc (m, clipped at 100)"); ax[0].set_ylabel("density"); ax[0].legend()
ax[0].set_title("GPS noise relative to reference route")
ax[1].hist(per_trip["p95"].clip(upper=500), bins=40, color="gray"); ax[1].set_title("Per-trip 95th percentile offset (m, clipped at 500)"); ax[1].set_xlabel("m"); ax[1].set_ylabel("trips")
fig.tight_layout(); fig.savefig(OUT / "eda_gps_offset.png", dpi=130); plt.close(fig)

# 5. trajectory + network + route static plots, plus Folium maps
def sample_trips():
    s = []
    for part in ("train", "student_test"):
        g = inv[(inv.partition == part) & (inv.dt_max < 60)].sort_values("n_obs")
        s.append((part, g.iloc[len(g)//2].trip))
    return s
fig, axs = plt.subplots(1, 2, figsize=(15, 7))
for a, (part, trip) in zip(axs, sample_trips()):
    tr, nd, ar, rt = L.load_track(part, trip), L.load_nodes(part, trip), L.load_arcs(part, trip), L.load_route(part, trip)
    pad = 0.003; m = ((ar.from_node.map(nd.lon) > tr.lon.min()-pad) & (ar.from_node.map(nd.lon) < tr.lon.max()+pad) &
                      (ar.from_node.map(nd.lat) > tr.lat.min()-pad) & (ar.from_node.map(nd.lat) < tr.lat.max()+pad))
    sub = ar[m]
    for f, t in zip(sub.from_node, sub.to_node):
        a.plot([nd.lon[f], nd.lon[t]], [nd.lat[f], nd.lat[t]], color="lightgray", lw=.6, zorder=1)
    for i, k in enumerate(rt):
        f, t = ar.from_node[k], ar.to_node[k]
        a.plot([nd.lon[f], nd.lon[t]], [nd.lat[f], nd.lat[t]], color="green", lw=2, zorder=2, label="reference route" if i == 0 else None)
    a.scatter(tr.lon, tr.lat, s=3, color="blue", zorder=3, label="raw GPS")
    a.set_title(f"{part} trip {trip}: {len(tr)} GPS pts, {len(rt)} route arcs"); a.legend(); a.set_xlabel("lon"); a.set_ylabel("lat")
fig.tight_layout(); fig.savefig(OUT / "eda_trajectory_network_route.png", dpi=130); plt.close(fig)

for part, trip in sample_trips():
    tr, nd, ar, rt = L.load_track(part, trip), L.load_nodes(part, trip), L.load_arcs(part, trip), L.load_route(part, trip)
    m = folium.Map(location=[tr.lat.mean(), tr.lon.mean()], zoom_start=14, tiles="OpenStreetMap")
    pad = 0.003; inb = ((nd.lon > tr.lon.min()-pad) & (nd.lon < tr.lon.max()+pad) & (nd.lat > tr.lat.min()-pad) & (nd.lat < tr.lat.max()+pad)).to_numpy()
    sub = ar[inb[ar.from_node.to_numpy()]]
    net = folium.FeatureGroup("Road network (arcs)", show=True)
    for f, t in zip(sub.from_node, sub.to_node):
        folium.PolyLine([(nd.lat[f], nd.lon[f]), (nd.lat[t], nd.lon[t])], color="gray", weight=1, opacity=.5).add_to(net)
    net.add_to(m)
    rg = folium.FeatureGroup("Reference route (.route)", show=True)
    for k in rt:
        f, t = ar.from_node[k], ar.to_node[k]
        folium.PolyLine([(nd.lat[f], nd.lon[f]), (nd.lat[t], nd.lon[t])], color="green", weight=4).add_to(rg)
    rg.add_to(m)
    gg = folium.FeatureGroup("Raw GPS points", show=True)
    for i in range(0, len(tr), max(1, len(tr)//1500)):
        folium.CircleMarker((tr.lat[i], tr.lon[i]), radius=2, color="blue", fill=True, tooltip=f"t={tr.t[i]:.0f}s").add_to(gg)
    gg.add_to(m); folium.LayerControl().add_to(m)
    m.save(OUT / f"folium_{part}_{trip}.html"); print("saved folium", part, trip)
