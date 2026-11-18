"""Reading raw PathSnap trips and projecting them to a metric CRS.

Raw files (whitespace separated, no header):
  .track : lon lat timestamp_seconds
  .nodes : lon lat                      (row index = node_id)
  .arcs  : from_node_id to_node_id      (row index = arc_id)
  .route : arc_id                       (reference route, chronological)
Every trip has its OWN road network, so nodes/arcs are loaded per trip.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from pyproj import Transformer


def _read(path, dtype):
    return pd.read_csv(path, sep=r"\s+", header=None, dtype=dtype).to_numpy()


def utm_epsg(lon: float, lat: float) -> int:
    """EPSG code of the UTM zone containing (lon, lat). Metric distances are only valid locally,
    so each trip is projected into its own zone (the dataset spans several cities)."""
    zone = int((lon + 180.0) // 6.0) + 1
    zone = min(max(zone, 1), 60)
    return (32600 if lat >= 0 else 32700) + zone


@dataclass
class Trip:
    trip_id: str
    partition: str
    epsg: int
    x: np.ndarray        # projected GPS x (m)
    y: np.ndarray
    t: np.ndarray        # timestamp (s)
    nodes_xy: np.ndarray # (n_nodes, 2) projected
    arcs: np.ndarray     # (n_arcs, 2) int  from_node, to_node
    route: np.ndarray    # reference arc ids (int)
    lonlat_track: np.ndarray
    lonlat_nodes: np.ndarray


def load_trip(dataset_dir, partition: str, trip_id: str) -> Trip:
    d = Path(dataset_dir) / partition / trip_id
    track = _read(d / f"{trip_id}.track", float)
    nodes = _read(d / f"{trip_id}.nodes", float)
    arcs = _read(d / f"{trip_id}.arcs", np.int64)
    route = _read(d / f"{trip_id}.route", np.int64).reshape(-1)
    epsg = utm_epsg(float(np.median(track[:, 0])), float(np.median(track[:, 1])))
    tf = Transformer.from_crs(4326, epsg, always_xy=True)
    gx, gy = tf.transform(track[:, 0], track[:, 1])
    nx, ny = tf.transform(nodes[:, 0], nodes[:, 1])
    return Trip(trip_id, partition, epsg, np.asarray(gx), np.asarray(gy), track[:, 2].copy(),
                np.column_stack([nx, ny]), arcs, route, track[:, :2].copy(), nodes[:, :2].copy())


def list_trips(dataset_dir, partition: str):
    return sorted(p.name for p in (Path(dataset_dir) / partition).iterdir() if p.is_dir())
