"""Load an unlabeled trip for inference without opening its reference ``.route`` file."""
from pathlib import Path

import numpy as np


def load_trip_for_inference(dataset_dir, partition: str, trip_id: str):
    """Read and project a trip's ``.track``, ``.nodes`` and ``.arcs`` files.

    The returned object is the same ``Trip`` dataclass used by training. Its ``route``
    field is an empty integer array because reference labels are not read at inference.
    """
    # Import here so importing the API does not require loading projection libraries
    # until a request actually needs trip inference.
    from src.data.trip_loader import Trip, _read, utm_epsg
    from pyproj import Transformer

    trip_dir = Path(dataset_dir) / partition / trip_id
    track = _read(trip_dir / f"{trip_id}.track", float)
    nodes = _read(trip_dir / f"{trip_id}.nodes", float)
    arcs = _read(trip_dir / f"{trip_id}.arcs", np.int64)

    if track.ndim != 2 or track.shape[1] != 3:
        raise ValueError(f"{trip_id}.track must contain lon, lat, and timestamp columns")
    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise ValueError(f"{trip_id}.nodes must contain lon and lat columns")
    if arcs.ndim != 2 or arcs.shape[1] != 2:
        raise ValueError(f"{trip_id}.arcs must contain from_node and to_node columns")

    epsg = utm_epsg(float(np.median(track[:, 0])), float(np.median(track[:, 1])))
    transformer = Transformer.from_crs(4326, epsg, always_xy=True)
    gps_x, gps_y = transformer.transform(track[:, 0], track[:, 1])
    node_x, node_y = transformer.transform(nodes[:, 0], nodes[:, 1])

    return Trip(
        trip_id=trip_id,
        partition=partition,
        epsg=epsg,
        x=np.asarray(gps_x),
        y=np.asarray(gps_y),
        t=track[:, 2].copy(),
        nodes_xy=np.column_stack([node_x, node_y]),
        arcs=arcs,
        route=np.empty(0, dtype=np.int64),
        lonlat_track=track[:, :2].copy(),
        lonlat_nodes=nodes[:, :2].copy(),
    )
