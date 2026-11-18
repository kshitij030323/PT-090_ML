"""Feature engineering for PathSnap map-matching."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data import loader as L

R = 6371000.0


def haversine(lon1, lat1, lon2, lat2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def local_xy(lon, lat, lat0, lon0):
    x = (np.asarray(lon) - lon0) * 111320 * np.cos(np.radians(lat0))
    y = (np.asarray(lat) - lat0) * 110540
    return x, y


def project_point_to_segment(px, py, ax, ay, bx, by):
    """Project point P onto segment AB. Returns (dist, proj_x, proj_y, t) where t in [0,1]."""
    apx, apy = px - ax, py - ay
    abx, aby = bx - ax, by - ay
    ab2 = abx * abx + aby * aby
    if ab2 == 0:
        return np.hypot(apx, apy), ax, ay, 0.0
    t = (apx * abx + apy * aby) / ab2
    t = np.clip(t, 0.0, 1.0)
    proj_x = ax + t * abx
    proj_y = ay + t * aby
    dist = np.hypot(px - proj_x, py - proj_y)
    return dist, proj_x, proj_y, t


def bearing(ax, ay, bx, by):
    return np.arctan2(bx - ax, by - ay)


def build_arc_kdtree(nodes, arcs, lat0, lon0):
    """Build KDTree for arc midpoints."""
    fr, to = arcs.from_node.to_numpy(), arcs.to_node.to_numpy()
    nx, ny = local_xy(nodes.lon, nodes.lat, lat0, lon0)
    mid_x = (nx[fr] + nx[to]) / 2
    mid_y = (ny[fr] + ny[to]) / 2
    tree = cKDTree(np.c_[mid_x, mid_y])
    return tree, mid_x, mid_y, nx, ny


def generate_candidates(track, nodes, arcs, route_arcs, radius_m=200.0, max_candidates=20):
    """
    For each GPS point, find candidate arcs within radius_m.
    Returns list of candidate dicts per GPS point.
    Uses KD-tree on arc midpoints with 2x radius, then filters by actual segment distance.
    """
    lat0, lon0 = track.lat.mean(), track.lon.mean()
    tree, mid_x, mid_y, nx, ny = build_arc_kdtree(nodes, arcs, lat0, lon0)
    px, py = local_xy(track.lon, track.lat, lat0, lon0)
    
    all_candidates = []
    route_set = set(route_arcs.tolist())
    search_radius = radius_m * 2.0  # Search wider to catch arcs whose midpoints are far but segments are close
    
    for i in range(len(track)):
        idxs = tree.query_ball_point([px[i], py[i]], r=search_radius)
        if not idxs:
            all_candidates.append([])
            continue
        
        cands = []
        for arc_idx in idxs:
            fr, to = arcs.from_node[arc_idx], arcs.to_node[arc_idx]
            ax, ay = nx[fr], ny[fr]
            bx, by = nx[to], ny[to]
            
            dist, proj_x, proj_y, t = project_point_to_segment(px[i], py[i], ax, ay, bx, by)
            
            if dist <= radius_m:
                arc_bearing = bearing(ax, ay, bx, by)
                cands.append({
                    'arc_id': int(arc_idx),
                    'dist_to_arc': float(dist),
                    'proj_t': float(t),
                    'arc_bearing': float(arc_bearing),
                    'arc_length': float(np.hypot(bx - ax, by - ay)),
                    'is_route': 1 if arc_idx in route_set else 0
                })
        
        cands.sort(key=lambda x: x['dist_to_arc'])
        all_candidates.append(cands[:max_candidates])
    
    return all_candidates


def extract_features(track, nodes, arcs, candidates, lat0, lon0, route_arcs=None):
    """
    Extract features for each (GPS point, candidate arc) pair.
    Returns DataFrame with one row per candidate.
    """
    rows = []
    nx, ny = local_xy(nodes.lon, nodes.lat, lat0, lon0)
    px, py = local_xy(track.lon, track.lat, lat0, lon0)
    
    # Node degree
    deg = np.bincount(np.concatenate([arcs.from_node, arcs.to_node]), minlength=len(nodes))
    
    # GPS speed and heading
    dt = np.diff(track.t.to_numpy(), prepend=track.t.iloc[0])
    dt[dt == 0] = 1.0
    dx = np.diff(px, prepend=px[0])
    dy = np.diff(py, prepend=py[0])
    gps_speed = np.hypot(dx, dy) / dt
    gps_bearing = np.arctan2(dx, dy)
    
    # Route arc set for connectivity features
    route_set = set(route_arcs.tolist()) if route_arcs is not None else set()
    
    # Build adjacency for route connectivity
    fr_all, to_all = arcs.from_node.to_numpy(), arcs.to_node.to_numpy()
    adj_from = {}
    adj_to = {}
    for i in range(len(arcs)):
        f, t = fr_all[i], to_all[i]
        adj_from.setdefault(f, []).append(i)
        adj_to.setdefault(t, []).append(i)
    
    for i, cands in enumerate(candidates):
        if not cands:
            continue
        for c in cands:
            arc_idx = c['arc_id']
            fr, to = arcs.from_node[arc_idx], arcs.to_node[arc_idx]
            ax, ay = nx[fr], ny[fr]
            bx, by = nx[to], ny[to]
            
            # Distance features
            dist = c['dist_to_arc']
            t_proj = c['proj_t']
            arc_len = c['arc_length']
            arc_bear = c['arc_bearing']
            
            # Heading difference
            hdg_diff = abs(gps_bearing[i] - arc_bear)
            hdg_diff = min(hdg_diff, 2 * np.pi - hdg_diff)
            
            # Along-track distance from arc start
            along_dist = t_proj * arc_len
            
            # Node degree features
            fr_deg = deg[fr]
            to_deg = deg[to]
            
            # Arc connectivity: is this arc connected to route arcs?
            # Check if fr or to connects to any route arc
            connected_to_route = 0
            if route_set:
                for rf in route_set:
                    rfr, rto = fr_all[rf], to_all[rf]
                    if fr == rfr or fr == rto or to == rfr or to == rto:
                        connected_to_route = 1
                        break
            
            # Distance to nearest route arc (if available)
            dist_to_nearest_route = -1
            if route_set and len(route_set) > 0:
                # Compute min distance to any route arc segment
                min_dist = float('inf')
                for rf in route_set:
                    rfr, rto = fr_all[rf], to_all[rf]
                    rax, ray = nx[rfr], ny[rfr]
                    rbx, rby = nx[rto], ny[rto]
                    d, _, _, _ = project_point_to_segment(px[i], py[i], rax, ray, rbx, rby)
                    if d < min_dist:
                        min_dist = d
                dist_to_nearest_route = min_dist
            
            # Is candidate the route arc at this point (label)
            is_route = c['is_route']
            
            rows.append({
                'obs_idx': i,
                'arc_id': arc_idx,
                'dist_to_arc': dist,
                'proj_t': t_proj,
                'along_dist': along_dist,
                'arc_length': arc_len,
                'arc_bearing': arc_bear,
                'gps_speed': gps_speed[i],
                'gps_bearing': gps_bearing[i],
                'hdg_diff': hdg_diff,
                'dt': dt[i],
                'fr_deg': fr_deg,
                'to_deg': to_deg,
                'connected_to_route': connected_to_route,
                'dist_to_nearest_route': dist_to_nearest_route,
                'is_route': is_route
            })
    
    return pd.DataFrame(rows)


def process_trip(partition, trip, radius_m=200.0, max_candidates=20):
    """Process a single trip: generate candidates and extract features."""
    track = L.load_track(partition, trip)
    nodes = L.load_nodes(partition, trip)
    arcs = L.load_arcs(partition, trip)
    route = L.load_route(partition, trip)
    
    lat0, lon0 = track.lat.mean(), track.lon.mean()
    candidates = generate_candidates(track, nodes, arcs, route, radius_m, max_candidates)
    features = extract_features(track, nodes, arcs, candidates, lat0, lon0, route)
    
    return features, candidates, track, nodes, arcs, route


def build_dataset(partition, radius_m=200.0, max_candidates=30):
    """Build feature dataset for all trips in a partition."""
    all_features = []
    for trip in L.list_trips(partition):
        feats, _, _, _, _, _ = process_trip(partition, trip, radius_m, max_candidates)
        if len(feats) > 0:
            feats['trip'] = trip
            feats['partition'] = partition
            all_features.append(feats)
            print(f"  {partition}/{trip}: {len(feats)} candidate rows")
    
    if all_features:
        return pd.concat(all_features, ignore_index=True)
    return pd.DataFrame()