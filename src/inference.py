
"""PathSnap inference using tuned LightGBM + HMM/Viterbi."""

import numpy as np

from src import config as C
from src.hmm.viterbi import viterbi_decode


def predict_trip(trip, model):
    """Predict road segments and decode the most likely route using Viterbi."""

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

    feature_rows = build_features(
        trip,
        obs_idx,
        candidates,
        C.WINDOW_OFFSETS,
    )

    predicted_arc_ids = [None] * len(obs_idx)
    confidence_scores = [None] * len(obs_idx)

    if not feature_rows.empty:
        # Step 1: Get LightGBM probability for every candidate.
        X = feature_rows.loc[:, FEATURE_COLS]
        probabilities = model.predict_proba(X)[:, 1]

        # Viterbi expects columns: obs, arc, frac, prob.
        hmm_candidates = feature_rows[
            ["obs", "arc", "frac"]
        ].copy()

        hmm_candidates["prob"] = probabilities

        # Step 2: Decode the most likely sequence of road segments.
        decoded = viterbi_decode(trip, hmm_candidates)

        # Step 3: Store the decoded arc for each observation.
        for row in decoded.itertuples(index=False):
            obs = int(row.obs)
            arc = int(row.predicted_arc)

            predicted_arc_ids[obs] = arc

            # This is the LightGBM probability of the selected HMM arc.
            selected = hmm_candidates[
                (hmm_candidates["obs"] == obs)
                & (hmm_candidates["arc"] == arc)
            ]

            if not selected.empty:
                confidence_scores[obs] = float(
                    selected.iloc[0]["prob"]
                )

    # Step 4: Construct route geometry using HMM-decoded arcs.
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
        "prediction_method": "LightGBM + HMM (Viterbi)",
        "decoder_status": "completed",
        "gps_coordinates": trip.lonlat_track.astype(float).tolist(),
        "predicted_arc_ids": predicted_arc_ids,
        "confidence_scores": confidence_scores,
        "route_geometry": {
            "type": "MultiLineString",
            "coordinates": route_segments,
        },
    }
