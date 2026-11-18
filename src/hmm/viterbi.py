"""Viterbi decoding for road-segment sequence reconstruction."""

import numpy as np
import pandas as pd

from src.hmm.transition_probability import TransitionModel


def viterbi_decode(trip, candidates):
    """
    candidates must contain:
        obs, arc, frac, prob

    prob is the LightGBM probability for each candidate.
    Returns one predicted arc for each observation with candidates.
    """

    if candidates.empty:
        return pd.DataFrame(columns=["obs", "predicted_arc"])

    candidates = candidates.sort_values(
        ["obs", "arc"]
    ).copy()

    # One list of candidate states per observation
    observations = sorted(candidates["obs"].unique())

    states = []

    for obs in observations:
        group = candidates[candidates["obs"] == obs]

        states.append({
            "arcs": group["arc"].to_numpy(dtype=int),
            "fractions": group["frac"].to_numpy(dtype=float),
            "emissions": group["prob"].to_numpy(dtype=float)
        })

    transition_model = TransitionModel(trip)

    # Store the best log probability and predecessor for each state.
    dp = []
    backpointers = []

    # Initial observation: use classifier probabilities as emissions.
    first_emission = np.clip(
        states[0]["emissions"], 1e-12, 1.0
    )

    dp.append(np.log(first_emission))
    backpointers.append(
        np.full(len(first_emission), -1, dtype=int)
    )

    for t in range(1, len(states)):
        previous = states[t - 1]
        current = states[t]

        previous_obs = observations[t - 1]
        current_obs = observations[t]

        gps_distance = np.hypot(
            trip.x[current_obs] - trip.x[previous_obs],
            trip.y[current_obs] - trip.y[previous_obs]
        )

        time_gap = (
            trip.t[current_obs] - trip.t[previous_obs]
        )

        emission = np.clip(
            current["emissions"], 1e-12, 1.0
        )

        current_scores = np.full(
            len(current["arcs"]), -np.inf
        )

        current_backpointer = np.zeros(
            len(current["arcs"]), dtype=int
        )

        for j, current_arc in enumerate(current["arcs"]):

            transition_scores = []

            for i, previous_arc in enumerate(previous["arcs"]):

                score = transition_model.transition_score(
                    previous_arc=previous_arc,
                    current_arc=current_arc,
                    previous_fraction=previous["fractions"][i],
                    current_fraction=current["fractions"][j],
                    gps_distance=gps_distance,
                    time_gap=time_gap
                )

                transition_scores.append(
                    dp[t - 1][i] + np.log(max(score, 1e-12))
                )

            transition_scores = np.asarray(
                transition_scores
            )

            best_previous = int(
                np.argmax(transition_scores)
            )

            current_scores[j] = (
                transition_scores[best_previous]
                + np.log(emission[j])
            )

            current_backpointer[j] = best_previous

        dp.append(current_scores)
        backpointers.append(current_backpointer)

    # Backtrack from the most likely final state.
    final_state = int(np.argmax(dp[-1]))

    best_path = [final_state]

    for t in range(len(states) - 1, 0, -1):
        final_state = backpointers[t][final_state]
        best_path.append(final_state)

    best_path.reverse()

    predicted_arcs = [
        states[t]["arcs"][best_path[t]]
        for t in range(len(states))
    ]

    return pd.DataFrame({
        "obs": observations,
        "predicted_arc": predicted_arcs
    })