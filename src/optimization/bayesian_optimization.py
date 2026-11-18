
"""Bayesian hyperparameter optimisation for the selected LightGBM model.

Uses:
- candidates_features.parquet
- FEATURE_COLS from src.features.features
- label as target
- GroupKFold grouped by trip
- student_test for final development comparison

Evaluator-test is never loaded.

Run:
python -m src.optimization.bayesian_optimization
"""

import json
import time

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from sklearn.model_selection import GroupKFold

from src import config as C
from src.data.trip_loader import load_trip
from src.features.features import FEATURE_COLS
from src.models.metrics import (
    candidate_metrics,
    top1_accuracy,
    route_continuity,
)


SEED = C.SEED
N_TRIALS = 25


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

def load_data():

    path = C.PROCESSED_DIR / "candidates_features.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist.\n"
            "First run: python -m src.data.build_tables"
        )

    data = pd.read_parquet(path)

    required = {
        "trip",
        "partition",
        "obs",
        "arc",
        "label",
        *FEATURE_COLS,
    }

    missing = required.difference(data.columns)

    if missing:
        raise ValueError(
            f"Dataset is not from the current feature pipeline. "
            f"Missing columns: {sorted(missing)}\n"
            f"Found columns: {data.columns.tolist()}"
        )

    train = data[
        data["partition"] == "train"
    ].copy().reset_index(drop=True)

    test = data[
        data["partition"] == "student_test"
    ].copy().reset_index(drop=True)

    if train.empty or test.empty:
        raise ValueError(
            "Training or student_test partition is empty."
        )

    return train, test


# --------------------------------------------------
# 2. LIGHTGBM MODEL
# --------------------------------------------------

def make_model(params):

    return lgb.LGBMClassifier(
        **params,
        objective="binary",
        random_state=SEED,
        n_jobs=2,
        verbose=-1,
    )


# --------------------------------------------------
# 3. BAYESIAN OPTIMISATION OBJECTIVE
# --------------------------------------------------

def objective_factory(train, folds):

    X = train[FEATURE_COLS]
    y = train["label"].astype(int)

    def objective(trial):

        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators", 200, 700
            ),

            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.15, log=True
            ),

            "num_leaves": trial.suggest_int(
                "num_leaves", 15, 63
            ),

            "max_depth": trial.suggest_categorical(
                "max_depth", [-1, 5, 7, 9, 12]
            ),

            "min_child_samples": trial.suggest_int(
                "min_child_samples", 10, 80
            ),

            "subsample": trial.suggest_float(
                "subsample", 0.7, 1.0
            ),

            "subsample_freq": 1,

            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.7, 1.0
            ),

            "reg_alpha": trial.suggest_float(
                "reg_alpha", 1e-4, 10.0, log=True
            ),

            "reg_lambda": trial.suggest_float(
                "reg_lambda", 1e-4, 10.0, log=True
            ),
        }

        scores = []

        for fold_no, (train_idx, valid_idx) in enumerate(folds):

            model = make_model(params)

            model.fit(
                X.iloc[train_idx],
                y.iloc[train_idx],
            )

            probabilities = model.predict_proba(
                X.iloc[valid_idx]
            )[:, 1]

            metrics = candidate_metrics(
                y.iloc[valid_idx],
                probabilities,
            )

            score = metrics["macro_f1"]
            scores.append(score)

            trial.report(
                float(np.mean(scores)),
                step=fold_no,
            )

            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(scores))

    return objective


# --------------------------------------------------
# 4. EVALUATION
# --------------------------------------------------

def evaluate_model(name, model, test, arcs_by_trip):

    probabilities = model.predict_proba(
        test[FEATURE_COLS]
    )[:, 1]

    metrics = candidate_metrics(
        test["label"].astype(int),
        probabilities,
    )

    correct, observations_with_candidates = top1_accuracy(
        test,
        probabilities,
    )

    total_observations = test[
        ["trip", "obs"]
    ].drop_duplicates().shape[0]

    continuity = route_continuity(
        test,
        probabilities,
        arcs_by_trip,
    )

    return {
        "model": name,

        "student_top1_accuracy":
            correct / max(total_observations, 1),

        "student_top1_correct": correct,

        "student_observations": total_observations,

        "observations_with_candidates":
            observations_with_candidates,

        "student_macro_f1": metrics["macro_f1"],

        "student_pos_f1": metrics["pos_f1"],

        "student_precision": metrics["precision"],

        "student_recall": metrics["recall"],

        "student_accuracy": metrics["accuracy"],

        "student_log_loss": metrics["log_loss"],

        "route_continuity": continuity,
    }


# --------------------------------------------------
# 5. MAIN FUNCTION
# --------------------------------------------------

def main():

    start = time.time()

    C.RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    C.ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Load current feature dataset
    train, test = load_data()

    print(
        f"Training: {len(train):,} candidate rows, "
        f"{train['trip'].nunique()} trips"
    )

    print(
        f"Student test: {len(test):,} candidate rows, "
        f"{test['trip'].nunique()} trips"
    )

    print(
        f"Using {len(FEATURE_COLS)} features: {FEATURE_COLS}"
    )

    # GroupKFold prevents observations from the same trip
    # appearing in both training and validation folds.
    folds = list(
        GroupKFold(
            n_splits=C.CV_FOLDS
        ).split(
            train[FEATURE_COLS],
            train["label"],
            groups=train["trip"],
        )
    )

    # Create Optuna study
    study = optuna.create_study(
        direction="maximize",

        sampler=optuna.samplers.TPESampler(
            seed=SEED
        ),

        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=2,
        ),
    )

    print(
        f"\nStarting Bayesian optimisation: {N_TRIALS} trials"
    )

    study.optimize(
        objective_factory(train, folds),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    print(
        "\nBest CV macro F1:",
        round(study.best_value, 5),
    )

    print(
        "Best parameters:",
        study.best_params,
    )

    # --------------------------------------------------
    # 6. SAVE OPTIMISATION RESULTS
    # --------------------------------------------------

    trials_path = (
        C.RESULTS_DIR / "bayesian_trials.csv"
    )

    study.trials_dataframe().to_csv(
        trials_path,
        index=False,
    )

    params_path = (
        C.ARTIFACTS_DIR / "lightgbm_best_params.json"
    )

    with open(
        params_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            study.best_params,
            file,
            indent=4,
        )

    # --------------------------------------------------
    # 7. TRAIN FINAL TUNED MODEL
    # --------------------------------------------------

    tuned_model = make_model(
        study.best_params
    )

    tuned_model.fit(
        train[FEATURE_COLS],
        train["label"].astype(int),
    )

    tuned_path = (
        C.ARTIFACTS_DIR / "LightGBM_tuned.joblib"
    )

    joblib.dump(
        tuned_model,
        tuned_path,
    )

    # --------------------------------------------------
    # 8. LOAD ORIGINAL MODEL
    # --------------------------------------------------

    original_path = (
        C.ARTIFACTS_DIR / "LightGBM.joblib"
    )

    if not original_path.exists():

        raise FileNotFoundError(
            f"{original_path} not found. "
            "Run python -m src.models.train_compare first."
        )

    original_model = joblib.load(
        original_path
    )

    # Verify both models use the same feature count.
    for name, model in [
        ("Original LightGBM", original_model),
        ("Tuned LightGBM", tuned_model),
    ]:

        expected = getattr(
            model,
            "n_features_in_",
            None,
        )

        if (
            expected is not None
            and expected != len(FEATURE_COLS)
        ):

            raise ValueError(
                f"{name} expects {expected} features, "
                f"but the current pipeline supplies "
                f"{len(FEATURE_COLS)}. "
                "Check which dataset was used to train "
                "the artifact before comparing."
            )

    # --------------------------------------------------
    # 9. LOAD STUDENT-TEST ROAD NETWORKS
    # --------------------------------------------------

    arcs_by_trip = {

        trip_id: load_trip(
            C.DATASET_DIR,
            "student_test",
            trip_id,
        ).arcs

        for trip_id in test["trip"].unique()
    }

    # --------------------------------------------------
    # 10. COMPARE TUNED VS UNTUNED
    # --------------------------------------------------

    results = [

        evaluate_model(
            "LightGBM_untuned",
            original_model,
            test,
            arcs_by_trip,
        ),

        evaluate_model(
            "LightGBM_tuned",
            tuned_model,
            test,
            arcs_by_trip,
        ),
    ]

    comparison = pd.DataFrame(results)

    comparison_path = (
        C.RESULTS_DIR / "tuned_vs_untuned.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    print("\nTuned vs untuned results:")

    print(
        comparison.round(5).to_string(
            index=False
        )
    )

    # --------------------------------------------------
    # 11. OUTPUT FILES
    # --------------------------------------------------

    print("\nSaved:")

    print(f"- {trials_path}")

    print(f"- {params_path}")

    print(f"- {tuned_path}")

    print(f"- {comparison_path}")

    print(
        f"\nTotal time: "
        f"{(time.time() - start) / 60:.1f} minutes"
    )


if __name__ == "__main__":
    main()
