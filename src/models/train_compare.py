"""Train and compare 5 model families with trip-grouped cross-validation.

CV: GroupKFold over the 80 training trips (no trip is split across folds).
Final fit on all 80 training trips; development-stage evaluation on the 10 student-test trips.
The evaluator-test partition is never touched.

Usage: python -m src.models.train_compare
"""
import time, json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import lightgbm as lgb
from src import config as C
from src.features.features import FEATURE_COLS
from src.data.trip_loader import load_trip
from src.models.metrics import candidate_metrics, top1_accuracy, route_continuity

SEED = C.SEED


def make_models():
    return {
        "LogReg": dict(family="Linear / probabilistic", cfg="C=1.0, L2, lbfgs, standardised features",
                       build=lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=SEED))),
        "KNN": dict(family="Instance-based", cfg="k=25, distance-weighted, standardised, <=60k train rows",
                    build=lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=2)),
                    max_train_rows=60000),
        "RandomForest": dict(family="Ensemble (bagging)", cfg="150 trees, max_depth=14, min_samples_leaf=5",
                             build=lambda: RandomForestClassifier(n_estimators=150, max_depth=14, min_samples_leaf=5, n_jobs=2, random_state=SEED)),
        "LightGBM": dict(family="Boosting", cfg="300 trees, lr=0.05, num_leaves=31, subsample=0.8, colsample=0.8",
                         build=lambda: lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8,
                                                          subsample_freq=1, colsample_bytree=0.8, random_state=SEED, n_jobs=2, verbose=-1)),
        "MLP": dict(family="Neural network", cfg="hidden=(64,32), relu, adam, alpha=1e-4, early stopping",
                    build=lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, early_stopping=True,
                                                                                 max_iter=200, n_iter_no_change=10, random_state=SEED))),
    }


def subsample_by_obs(df, max_rows, seed):
    if max_rows is None or len(df) <= max_rows:
        return df
    keys = df[["trip", "obs"]].drop_duplicates()
    frac = max_rows / len(df)
    keys = keys.sample(frac=frac, random_state=seed)
    return df.merge(keys, on=["trip", "obs"])


def predict_proba(model, X):
    return model.predict_proba(X)[:, 1]


def main():
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True); C.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    tab = pd.read_parquet(C.PROCESSED_DIR / "candidates_features.parquet")
    train = tab[tab.partition == "train"].reset_index(drop=True)
    test = tab[tab.partition == "student_test"].reset_index(drop=True)
    print("train rows", len(train), "trips", train.trip.nunique(), "| student rows", len(test), "trips", test.trip.nunique(), flush=True)
    arcs_by_trip = {t: load_trip(C.DATASET_DIR, "student_test", t).arcs for t in test.trip.unique()}
    models = make_models()
    folds = list(GroupKFold(n_splits=C.CV_FOLDS).split(train, groups=train["trip"]))
    fold_rows, summary, cms, histories = [], [], {}, {}

    for name, spec in models.items():
        print(f"=== {name}", flush=True)
        for k, (tr_i, va_i) in enumerate(folds):
            tr, va = train.iloc[tr_i], train.iloc[va_i]
            tr_fit = subsample_by_obs(tr, spec.get("max_train_rows"), SEED)
            m = spec["build"]()
            t0 = time.time(); m.fit(tr_fit[FEATURE_COLS], tr_fit["label"]); fit_s = time.time() - t0
            p = predict_proba(m, va[FEATURE_COLS])
            met = candidate_metrics(va["label"], p)
            c, n = top1_accuracy(va, p)
            met.update(model=name, fold=k, top1_acc=c / n, fit_seconds=fit_s)
            fold_rows.append(met)
            print(f"  fold {k}: macroF1={met['macro_f1']:.4f} top1={met['top1_acc']:.4f} logloss={met['log_loss']:.4f} fit={fit_s:.1f}s", flush=True)

        # final fit on all 80 training trips
        tr_fit = subsample_by_obs(train, spec.get("max_train_rows"), SEED)
        m = spec["build"]()
        t0 = time.time(); m.fit(tr_fit[FEATURE_COLS], tr_fit["label"]); train_s = time.time() - t0
        p_tr = predict_proba(m, train[FEATURE_COLS])
        tr_met = candidate_metrics(train["label"], p_tr)
        t0 = time.time(); p_te = predict_proba(m, test[FEATURE_COLS]); inf_s = time.time() - t0
        te_met = candidate_metrics(test["label"], p_te)
        c, n = top1_accuracy(test, p_te)
        n_obs_total = test[["trip", "obs"]].drop_duplicates().shape[0]
        cont = route_continuity(test, p_te, arcs_by_trip)
        cms[name] = confusion_matrix(test["label"], (p_te >= 0.5).astype(int))
        cvm = pd.DataFrame([r for r in fold_rows if r["model"] == name])
        summary.append({
            "model": name, "family": spec["family"], "key_config": spec["cfg"],
            "cv_macro_f1_mean": cvm.macro_f1.mean(), "cv_macro_f1_std": cvm.macro_f1.std(),
            "cv_top1_mean": cvm.top1_acc.mean(), "cv_logloss_mean": cvm.log_loss.mean(),
            "train_macro_f1": tr_met["macro_f1"], "train_logloss": tr_met["log_loss"],
            "student_top1_acc": c / n_obs_total, "student_macro_f1": te_met["macro_f1"],
            "student_precision": te_met["precision"], "student_recall": te_met["recall"],
            "student_pos_f1": te_met["pos_f1"], "student_accuracy": te_met["accuracy"], "student_log_loss": te_met["log_loss"],
            "route_continuity": cont, "train_seconds": train_s, "inference_seconds_total": inf_s,
            "inference_ms_per_obs": 1000 * inf_s / n_obs_total,
        })
        print("  FINAL", {k: round(v, 4) if isinstance(v, float) else v for k, v in summary[-1].items() if k not in ("family", "key_config")}, flush=True)
        joblib.dump(m, C.ARTIFACTS_DIR / f"{name}.joblib")
        np.save(C.PROCESSED_DIR / f"student_proba_{name}.npy", p_te)
        # epoch histories (only for models that really produce them)
        if name == "MLP":
            net = m.steps[-1][1]
            histories[name] = {"train_loss": list(map(float, net.loss_curve_)), "val_score": list(map(float, net.validation_scores_))}
        if name == "LightGBM":
            tr0, va0 = folds[0]
            mm = lgb.LGBMClassifier(**m.get_params())
            mm.fit(train.iloc[tr0][FEATURE_COLS], train.iloc[tr0]["label"],
                   eval_set=[(train.iloc[tr0][FEATURE_COLS], train.iloc[tr0]["label"]), (train.iloc[va0][FEATURE_COLS], train.iloc[va0]["label"])],
                   eval_names=["train", "valid"], eval_metric="binary_logloss")
            histories[name] = {"train_loss": mm.evals_result_["train"]["binary_logloss"], "val_loss": mm.evals_result_["valid"]["binary_logloss"]}

    pd.DataFrame(fold_rows).to_csv(C.RESULTS_DIR / "cv_fold_results.csv", index=False)
    pd.DataFrame(summary).to_csv(C.RESULTS_DIR / "model_comparison.csv", index=False)
    json.dump(histories, open(C.RESULTS_DIR / "training_histories.json", "w"))
    json.dump({k: v.tolist() for k, v in cms.items()}, open(C.RESULTS_DIR / "confusion_matrices_student.json", "w"))
    print(pd.DataFrame(summary).drop(columns=["family", "key_config"]).round(4).to_string())


if __name__ == "__main__":
    main()
