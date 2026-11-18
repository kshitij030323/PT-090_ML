"""Leakage-safe, trip-grouped CV for candidate arc classifiers.

Run with: python -m src.models.cv
The reported scores measure candidate classification, not final route accuracy.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
try:
    from lightgbm import LGBMClassifier
    _LIGHTGBM_ERROR = None
except Exception as exc:  # Missing native runtime should not block other models.
    LGBMClassifier = None
    _LIGHTGBM_ERROR = str(exc)
try:
    from xgboost import XGBClassifier
    _XGBOOST_ERROR = None
except Exception as exc:
    XGBClassifier = None
    _XGBOOST_ERROR = str(exc)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed/train_features.parquet"
OUT = ROOT / "experiments/results/cv_candidate_models.csv"
TARGET = "is_route"
GROUP = "trip"


def features(frame):
    """Construct numeric, target-independent inputs; remove identifiers."""
    drop = {TARGET, GROUP, "partition", "obs_idx", "arc_id", "true_arc_id", "match_dist"}
    x = frame.drop(columns=[c for c in drop if c in frame], errors="ignore").select_dtypes(include=[np.number]).copy()
    # Bearings are circular: encode angles without a discontinuity at +/- pi.
    for col in ("arc_bearing", "gps_bearing", "hdg_diff"):
        if col in x:
            a = x.pop(col).to_numpy()
            x[f"{col}_sin"] = np.sin(a)
            x[f"{col}_cos"] = np.cos(a)
    return x.replace([np.inf, -np.inf], np.nan)


def models():
    result = {
        "LogisticRegression": make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=300, class_weight="balanced")),
        "RandomForest": RandomForestClassifier(n_estimators=120, min_samples_leaf=2, max_features="sqrt", class_weight="balanced_subsample", n_jobs=-1, random_state=42),
        "MLP": make_pipeline(SimpleImputer(), StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True, max_iter=80, random_state=42)),
    }
    if XGBClassifier is not None:
        result["XGBoost"] = XGBClassifier(n_estimators=180, max_depth=7, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, eval_metric="logloss", tree_method="hist", n_jobs=-1, random_state=42)
    if LGBMClassifier is not None:
        result["LightGBM"] = LGBMClassifier(n_estimators=180, num_leaves=31, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, class_weight="balanced", verbosity=-1, n_jobs=-1, random_state=42)
    return result


def main():
    df = pd.read_parquet(DATA)
    if LGBMClassifier is None:
        print(f"Skipping LightGBM: {_LIGHTGBM_ERROR}")
    if XGBClassifier is None:
        print(f"Skipping XGBoost: {_XGBOOST_ERROR}")
    if TARGET not in df or GROUP not in df:
        raise ValueError(f"Expected {TARGET!r} and {GROUP!r} columns in {DATA}")
    x, y, groups = features(df), df[TARGET].astype(int).to_numpy(), df[GROUP].to_numpy()
    cv = GroupKFold(n_splits=5)
    rows = []
    for name, model in models().items():
        print(f"Evaluating {name}", flush=True)
        for fold, (tr, va) in enumerate(cv.split(x, y, groups), 1):
            model.fit(x.iloc[tr], y[tr])
            pred = model.predict(x.iloc[va])
            score = model.predict_proba(x.iloc[va])[:, 1] if hasattr(model, "predict_proba") else model.decision_function(x.iloc[va])
            row = dict(model=name, fold=fold, accuracy=accuracy_score(y[va], pred), balanced_accuracy=balanced_accuracy_score(y[va], pred), precision=precision_score(y[va], pred, zero_division=0), recall=recall_score(y[va], pred, zero_division=0), f1=f1_score(y[va], pred, zero_division=0), roc_auc=roc_auc_score(y[va], score), n_train=len(tr), n_valid=len(va), valid_trips=len(np.unique(groups[va])))
            rows.append(row)
            print(f"  fold {fold}: acc={row['accuracy']:.4f} balanced={row['balanced_accuracy']:.4f} f1={row['f1']:.4f} auc={row['roc_auc']:.4f}", flush=True)
    result = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT, index=False)
    print("\nMean by model:\n", result.groupby("model")[['accuracy','balanced_accuracy','precision','recall','f1','roc_auc']].mean().sort_values("f1", ascending=False).to_string())
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
