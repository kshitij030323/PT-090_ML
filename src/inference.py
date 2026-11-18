"""Classifier-only PathSnap inference using the existing candidate and feature pipeline."""
import numpy as np

from src import config as C


def predict_trip(trip, model):
    """Predict one candidate arc per GPS observation, without consulting route labels."""
    from src.data.candidates import build_arc_tree, generate_candidates
    from src.features.features import FEATURE_COLS, build_features

    obs_idx = np.arange(len(trip.x), dtype=np.int64)
    tree, start_xy, end_xy = build_arc_tree(trip)
    candidates = generate_candidates(
        trip,
        obs_idx,
        C.CAND_RADIUS_M,
        C.CAND_K,
        tree,
        (start_xy, end_xy),
    )
    feature_rows = build_features(trip, obs_idx, candidates, C.WINDOW_OFFSETS)

    predicted_arc_ids = [None] * len(obs_idx)
    confidence_scores = [None] * len(obs_idx)

    # An empty candidate set is valid (for example, a GPS point farther than the
    # configured radius from every road segment). Avoid calling predict_proba on it.
    if not feature_rows.empty:
        X = feature_rows.loc[:, FEATURE_COLS]
        probabilities = model.predict_proba(X)[:, 1]
        feature_rows = feature_rows.assign(_probability=probabilities)

        for obs, rows in feature_rows.groupby("obs", sort=False):
            best = rows.loc[rows["_probability"].idxmax()]
            predicted_arc_ids[int(obs)] = int(best["arc"])
            confidence_scores[int(obs)] = float(best["_probability"])

    route_segments = []
    for arc_id in predicted_arc_ids:
        if arc_id is None:
            continue
        from_node, to_node = trip.arcs[arc_id]
        route_segments.append([
            trip.lonlat_nodes[from_node].astype(float).tolist(),
            trip.lonlat_nodes[to_node].astype(float).tolist(),
        ])

    return {
        "trip_id": trip.trip_id,
        "prediction_method": "classifier_only",
        "decoder_status": "not_implemented",
        "gps_coordinates": trip.lonlat_track.astype(float).tolist(),
        "predicted_arc_ids": predicted_arc_ids,
        "confidence_scores": confidence_scores,
        "route_geometry": {"type": "MultiLineString", "coordinates": route_segments},
    }
