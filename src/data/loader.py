"""Raw data loaders for PathSnap trips (.track/.nodes/.arcs/.route)."""
from pathlib import Path
import numpy as np
import pandas as pd

DATASET = Path(__file__).resolve().parents[2] / "dataset"
PARTITIONS = ("train", "student_test")


def list_trips(partition):
    return sorted(p.name for p in (DATASET / partition).iterdir() if p.is_dir())


def trip_dir(partition, trip):
    return DATASET / partition / trip


def load_track(partition, trip):
    return pd.read_csv(trip_dir(partition, trip) / f"{trip}.track", sep=r"\s+",
                       header=None, names=["lon", "lat", "t"])


def load_nodes(partition, trip):
    return pd.read_csv(trip_dir(partition, trip) / f"{trip}.nodes", sep=r"\s+",
                       header=None, names=["lon", "lat"])


def load_arcs(partition, trip):
    return pd.read_csv(trip_dir(partition, trip) / f"{trip}.arcs", sep=r"\s+",
                       header=None, names=["from_node", "to_node"])


def load_route(partition, trip):
    return pd.read_csv(trip_dir(partition, trip) / f"{trip}.route", sep=r"\s+",
                       header=None, names=["arc_id"])["arc_id"].to_numpy()
