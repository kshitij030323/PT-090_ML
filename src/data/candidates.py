"""Candidate generation: for every GPS observation, find nearby road arcs, and label the correct one.

Reference label rule (documented assumption): the .route file only lists the arcs of the route, not which arc
each GPS point belongs to. For each observation the reference arc is the ROUTE arc geometrically nearest to the
point (if within CAND_RADIUS_M). A candidate is positive iff it equals that reference arc. Observations whose
nearest route arc is farther than the radius have no positive (counted in the candidate miss rate).
"""
import numpy as np
import shapely
from shapely import STRtree


def arc_endpoints(trip):
    a = trip.nodes_xy[trip.arcs[:, 0]]
    b = trip.nodes_xy[trip.arcs[:, 1]]
    return a, b


def point_segment(px, py, ax, ay, bx, by):
    """Distance from points to segments, projected fraction (clamped to [0,1]) and signed lateral offset."""
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy
    wx, wy = px - ax, py - ay
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(L2 > 0, (wx * vx + wy * vy) / L2, 0.0)
    f = np.clip(f, 0.0, 1.0)
    cx, cy = ax + f * vx, ay + f * vy
    d = np.hypot(px - cx, py - cy)
    L = np.sqrt(L2)
    with np.errstate(divide="ignore", invalid="ignore"):
        lat = np.where(L > 0, (vx * wy - vy * wx) / L, 0.0)   # + : point is left of the arc direction
    return d, f, lat


def build_arc_tree(trip):
    a, b = arc_endpoints(trip)
    geoms = shapely.linestrings(np.stack([a, b], axis=1))
    return STRtree(geoms), a, b


def generate_candidates(trip, obs_idx, radius, k, tree=None, ends=None):
    """Return DataFrame(obs, arc, d, frac, lat, rank) of the <=k nearest arcs within `radius` of each obs."""
    if tree is None:
        tree, a, b = build_arc_tree(trip)
    else:
        a, b = ends
    px, py = trip.x[obs_idx], trip.y[obs_idx]
    pts = shapely.points(px, py)
    pairs = tree.query(pts, predicate="dwithin", distance=radius)     # (2, n): input idx, arc idx
    oi, ai = pairs[0], pairs[1]
    d, f, lat = point_segment(px[oi], py[oi], a[ai, 0], a[ai, 1], b[ai, 0], b[ai, 1])
    order = np.lexsort((d, oi))
    oi, ai, d, f, lat = oi[order], ai[order], d[order], f[order], lat[order]
    # rank within each observation
    starts = np.r_[0, np.flatnonzero(np.diff(oi)) + 1] if len(oi) else np.array([], dtype=int)
    first = np.zeros(len(oi), dtype=np.int64)
    if len(oi):
        grp = np.cumsum(np.r_[1, np.diff(oi) != 0]) - 1
        first = starts[grp]
    rank = np.arange(len(oi)) - first
    keep = rank < k
    return {"obs_pos": oi[keep], "arc": ai[keep], "d": d[keep], "frac": f[keep], "lat": lat[keep],
            "rank": rank[keep]}


def reference_arcs(trip, obs_idx, arc_tree_geoms_fn=None):
    """Nearest ROUTE arc (id, distance) for every observation in obs_idx."""
    a, b = arc_endpoints(trip)
    route_ids = np.unique(trip.route)
    geoms = shapely.linestrings(np.stack([a[route_ids], b[route_ids]], axis=1))
    rt = STRtree(geoms)
    pts = shapely.points(trip.x[obs_idx], trip.y[obs_idx])
    idx, dist = rt.query_nearest(pts, return_distance=True, all_matches=False)
    ref = np.full(len(obs_idx), -1, dtype=np.int64)
    rd = np.full(len(obs_idx), np.inf)
    ref[idx[0]] = route_ids[idx[1]]
    rd[idx[0]] = dist
    return ref, rd
