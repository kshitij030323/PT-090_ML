"""Interactive Folium map of one trip: raw GPS, nearby road network, reference route and (optionally) the
classifier-only predicted route.

Usage:
  python -m visualisations.folium_map --partition student_test --trip 00000052 --model LightGBM
  python -m visualisations.folium_map --partition train --trip 00000005          (no prediction layer)
Output: visualisations/maps/<partition>_<trip>[_<model>].html
"""
import argparse
import numpy as np
import pandas as pd
import folium
from src import config as C
from src.data.trip_loader import load_trip


def arc_latlon(trip, arc_ids):
    ll = trip.lonlat_nodes
    return [[(ll[trip.arcs[a, 0], 1], ll[trip.arcs[a, 0], 0]), (ll[trip.arcs[a, 1], 1], ll[trip.arcs[a, 1], 0])] for a in arc_ids]


def make_map(partition, trip_id, model=None, out=None):
    trip = load_trip(C.DATASET_DIR, partition, trip_id)
    lat, lon = trip.lonlat_track[:, 1], trip.lonlat_track[:, 0]
    m = folium.Map(location=[lat.mean(), lon.mean()], zoom_start=15, tiles="OpenStreetMap")
    m.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]])

    # road network: only arcs with an endpoint inside the (padded) GPS bounding box
    pad = 0.003
    nl = trip.lonlat_nodes
    inbox = (nl[:, 0] > lon.min() - pad) & (nl[:, 0] < lon.max() + pad) & (nl[:, 1] > lat.min() - pad) & (nl[:, 1] < lat.max() + pad)
    sel = np.flatnonzero(inbox[trip.arcs[:, 0]] & inbox[trip.arcs[:, 1]])
    if len(sel) > 6000:
        sel = sel[:: int(np.ceil(len(sel) / 6000))]
    net = folium.FeatureGroup(name=f"Road network ({len(sel)} arcs shown)", show=True)
    for seg in arc_latlon(trip, sel):
        folium.PolyLine(seg, color="#9aa0a6", weight=1.5, opacity=0.7).add_to(net)
    net.add_to(m)

    ref = folium.FeatureGroup(name="Reference route (green)", show=True)
    for seg in arc_latlon(trip, trip.route):
        folium.PolyLine(seg, color="#1a9850", weight=5, opacity=0.8).add_to(ref)
    ref.add_to(m)

    gps = folium.FeatureGroup(name="Raw GPS observations (blue)", show=True)
    step = max(1, len(lat) // 1500)
    for i in range(0, len(lat), step):
        folium.CircleMarker((lat[i], lon[i]), radius=2.5, color="#2166ac", fill=True, fill_opacity=0.8,
                            popup=f"obs {i}, t={trip.t[i]:.0f}s").add_to(gps)
    gps.add_to(m)

    if model is not None:
        tab = pd.read_parquet(C.PROCESSED_DIR / "candidates_features.parquet")
        test = tab[tab.partition == "student_test"].reset_index(drop=True)
        test["p"] = np.load(C.PROCESSED_DIR / f"student_proba_{model}.npy")
        t = test[test.trip == trip_id]
        best = t.loc[t.groupby("obs")["p"].idxmax()].sort_values("obs")
        pred = folium.FeatureGroup(name=f"Predicted route, {model} classifier only (red)", show=True)
        seen = []
        for a in best["arc"]:
            if not seen or seen[-1] != a:
                seen.append(a)
        for seg in arc_latlon(trip, seen):
            folium.PolyLine(seg, color="#d73027", weight=3, opacity=0.9).add_to(pred)
        pred.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    legend = ("<div style='position:fixed;bottom:25px;left:25px;z-index:9999;background:white;padding:8px 10px;"
              "border:1px solid #888;font-size:12px'><b>Legend</b><br><span style='color:#2166ac'>&#9679;</span> raw GPS<br>"
              "<span style='color:#1a9850'>&#9644;</span> reference route<br><span style='color:#d73027'>&#9644;</span> predicted route<br>"
              "<span style='color:#9aa0a6'>&#9644;</span> road network</div>")
    m.get_root().html.add_child(folium.Element(legend))
    out = out or (C.ROOT / "visualisations" / "maps" / f"{partition}_{trip_id}{'_' + model if model else ''}.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="student_test"); ap.add_argument("--trip", default="00000052"); ap.add_argument("--model", default=None)
    a = ap.parse_args()
    print(make_map(a.partition, a.trip, a.model))
