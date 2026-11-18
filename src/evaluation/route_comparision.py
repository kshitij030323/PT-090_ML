
"""Compare tuned/untuned LightGBM with and without HMM + Viterbi."""

import joblib
import numpy as np
import pandas as pd

from src import config as C
from src.data.trip_loader import load_trip
from src.features.features import FEATURE_COLS
from src.hmm.viterbi import viterbi_decode


def calculate_continuity(predicted_arcs, road_arcs):
    """Count consecutive predictions that are identical or directly connected."""

    predicted_arcs = np.asarray(predicted_arcs, dtype=int)

    if len(predicted_arcs) < 2:
        return 0, 0

    previous = predicted_arcs[:-1]
    current = predicted_arcs[1:]

    connected = (
        (previous == current)
        | (road_arcs[previous, 1] == road_arcs[current, 0])
    )

    return int(connected.sum()), len(connected)


def calculate_metrics(true_arcs, predicted_arcs, road_arcs):
    """Calculate accuracy and route continuity counts."""

    true_arcs = np.asarray(true_arcs, dtype=int)
    predicted_arcs = np.asarray(predicted_arcs, dtype=int)

    correct = int(np.sum(true_arcs == predicted_arcs))
    observations = len(true_arcs)

    connected, transitions = calculate_continuity(
        predicted_arcs, road_arcs
    )

    return {
        "correct_predictions": correct,
        "observations": observations,
        "accuracy": correct / observations if observations else 0.0,
        "connected_transitions": connected,
        "total_transitions": transitions,
        "continuity": (
            connected / transitions if transitions else 0.0
        ),
    }


def main():
    data = pd.read_parquet(
        C.PROCESSED_DIR / "candidates_features.parquet"
    )

    required = {
        "trip", "partition", "obs", "arc", "label", "frac"
    }
    missing = required - set(data.columns)

    if missing:
        raise ValueError(f"Missing required dataset columns: {sorted(missing)}")

    missing_features = set(FEATURE_COLS) - set(data.columns)
    if missing_features:
        raise ValueError(
            f"Missing model features: {sorted(missing_features)}"
        )

    test = data[data["partition"] == "student_test"].copy()

    if test.empty:
        raise ValueError("No student_test rows found.")

    model_paths = {
        "Untuned_LightGBM": C.ARTIFACTS_DIR / "LightGBM.joblib",
        "Tuned_LightGBM": C.ARTIFACTS_DIR / "LightGBM_tuned.joblib",
    }

    models = {}

    for name, path in model_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found: {path}")

        model = joblib.load(path)

        if getattr(model, "n_features_in_", len(FEATURE_COLS)) != len(FEATURE_COLS):
            raise ValueError(
                f"{name} expects {model.n_features_in_} features, "
                f"but the current dataset provides {len(FEATURE_COLS)}."
            )

        models[name] = model

    all_results = []

    print("Starting four-way route comparison...")
    print("Evaluation partition: student_test only")
    print("Models: untuned/tuned LightGBM, each with/without HMM\n")

    for trip_id, trip_df in test.groupby("trip", sort=True):

        print(f"Processing trip {trip_id}...")

        trip = load_trip(
            C.DATASET_DIR,
            "student_test",
            trip_id
        )

        # Ground-truth reference arcs for student-test observations.
        truth_rows = (
            trip_df[trip_df["label"] == 1]
            .drop_duplicates("obs")
            .set_index("obs")["arc"]
        )

        for model_name, model in models.items():

            current_df = trip_df.copy()

            # LightGBM candidate probabilities.
            current_df["prob"] = model.predict_proba(
                current_df[FEATURE_COLS]
            )[:, 1]

            # Independent classifier prediction.
            classifier_predictions = (
                current_df.loc[
                    current_df.groupby("obs")["prob"].idxmax()
                ]
                .set_index("obs")["arc"]
            )

            # Normalize probabilities within each observation for HMM emissions.
            totals = current_df.groupby("obs")["prob"].transform("sum")

            current_df["prob"] = (
                current_df["prob"] / totals.clip(lower=1e-12)
            )

            # HMM + Viterbi route prediction.
            hmm_predictions = (
                viterbi_decode(
                    trip,
                    current_df[["obs", "arc", "frac", "prob"]]
                )
                .set_index("obs")["predicted_arc"]
            )

            prediction_methods = {
                model_name: classifier_predictions,
                f"{model_name}_HMM": hmm_predictions,
            }

            for variant_name, predictions in prediction_methods.items():

                common_obs = sorted(
                    set(truth_rows.index) & set(predictions.index)
                )

                if not common_obs:
                    continue

                true_arcs = truth_rows.loc[common_obs].to_numpy()
                predicted_arcs = predictions.loc[common_obs].to_numpy()

                metrics = calculate_metrics(
                    true_arcs,
                    predicted_arcs,
                    trip.arcs
                )

                result = {
                    "trip": trip_id,
                    "model": variant_name,
                    **metrics,
                }

                all_results.append(result)

                print(
                    f"  {variant_name}: "
                    f"Accuracy={metrics['accuracy']:.4f}, "
                    f"Continuity={metrics['continuity']:.4f}"
                )

    if not all_results:
        raise RuntimeError("No evaluation results were generated.")

    results = pd.DataFrame(all_results)

    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save individual trip results.
    per_trip_path = (
        C.RESULTS_DIR / "four_model_route_comparison.csv"
    )
    results.to_csv(per_trip_path, index=False)

    # Aggregate across observations and transitions.
    summary_rows = []

    for model_name, group in results.groupby("model", sort=False):

        total_correct = group["correct_predictions"].sum()
        total_observations = group["observations"].sum()

        total_connected = group["connected_transitions"].sum()
        total_transitions = group["total_transitions"].sum()

        summary_rows.append({
            "model": model_name,
            "observation_weighted_accuracy": (
                total_correct / total_observations
                if total_observations else 0.0
            ),
            "observation_weighted_continuity": (
                total_connected / total_transitions
                if total_transitions else 0.0
            ),
            "mean_trip_accuracy": group["accuracy"].mean(),
            "mean_trip_continuity": group["continuity"].mean(),
            "total_correct_predictions": total_correct,
            "total_observations": total_observations,
            "total_connected_transitions": total_connected,
            "total_transitions": total_transitions,
            "trips_evaluated": group["trip"].nunique(),
        })

    summary = pd.DataFrame(summary_rows)

    summary_path = (
        C.RESULTS_DIR / "four_model_route_summary.csv"
    )
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 75)
    print("FOUR-WAY COMPARISON: OBSERVATION-WEIGHTED RESULTS")
    print("=" * 75)

    print(
        summary[
            [
                "model",
                "observation_weighted_accuracy",
                "observation_weighted_continuity",
                "trips_evaluated",
            ]
        ].to_string(index=False)
    )

    print("\nSaved results:")
    print(per_trip_path)
    print(summary_path)


if __name__ == "__main__":
    main()
