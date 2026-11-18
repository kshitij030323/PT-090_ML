"""Plots built ONLY from the saved experiment outputs in experiments/results/."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src import config as C


def main():
    OUT = C.ROOT / "visualisations"
    OUT.mkdir(exist_ok=True)
    summ = pd.read_csv(C.RESULTS_DIR / "model_comparison.csv")
    folds = pd.read_csv(C.RESULTS_DIR / "cv_fold_results.csv")
    names = summ["model"].tolist()
    col = plt.cm.tab10(np.arange(len(names)))


    def bar(ax, vals, errs=None, title="", ylabel=""):
        ax.bar(names, vals, yerr=errs, color=col, capsize=4)
        ax.set_title(title); ax.set_ylabel(ylabel); ax.tick_params(axis="x", rotation=20)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    fig, ax = plt.subplots(figsize=(7, 4))
    bar(ax, summ.cv_macro_f1_mean, summ.cv_macro_f1_std, "Cross-validation macro F1 (5-fold, grouped by trip; mean +/- std)", "macro F1")
    ax.set_ylim(min(summ.cv_macro_f1_mean) - 0.05, 1.0); fig.tight_layout(); fig.savefig(OUT / "cv_macro_f1_comparison.png", dpi=150); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    bar(axs[0], summ.student_top1_acc, title="Student-test observation-level Top-1 accuracy", ylabel="Top-1 accuracy")
    bar(axs[1], summ.student_macro_f1, title="Student-test macro F1", ylabel="macro F1")
    for a in axs: a.set_ylim(min(summ.student_top1_acc.min(), summ.student_macro_f1.min()) - 0.08, 1.0)
    fig.tight_layout(); fig.savefig(OUT / "student_test_top1_f1.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    w = 0.38; x = np.arange(len(names))
    ax.bar(x - w / 2, summ.cv_logloss_mean, w, label="CV (mean)"); ax.bar(x + w / 2, summ.student_log_loss, w, label="Student-test")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20); ax.set_ylabel("log loss (lower is better)"); ax.legend(); ax.set_title("Log loss")
    fig.tight_layout(); fig.savefig(OUT / "log_loss_comparison.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    bar(ax, summ.route_continuity, title="Student-test route continuity (classifier only)", ylabel="fraction of connected consecutive predictions")
    ax.set_ylim(min(summ.route_continuity) - 0.05, 1.0); fig.tight_layout(); fig.savefig(OUT / "route_continuity_comparison.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot([folds[folds.model == n].macro_f1 for n in names], tick_labels=names); ax.set_ylabel("macro F1 per fold"); ax.set_title("Fold-to-fold stability (CV macro F1)")
    fig.tight_layout(); fig.savefig(OUT / "cv_fold_stability.png", dpi=150); plt.close(fig)

    cms = json.load(open(C.RESULTS_DIR / "confusion_matrices_student.json"))
    fig, axs = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.2))
    for a, n in zip(np.atleast_1d(axs), names):
        cm = np.array(cms[n]); a.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2): a.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="black", fontsize=8)
        a.set_title(n); a.set_xticks([0, 1]); a.set_yticks([0, 1]); a.set_xlabel("predicted"); a.set_ylabel("true")
    fig.suptitle("Candidate-level confusion matrices on the student-test trips (threshold 0.5)")
    fig.tight_layout(); fig.savefig(OUT / "confusion_matrices_student_test.png", dpi=150); plt.close(fig)

    h = json.load(open(C.RESULTS_DIR / "training_histories.json"))
    fig, axs = plt.subplots(1, len(h), figsize=(5.5 * len(h), 4))
    for a, (n, v) in zip(np.atleast_1d(axs), h.items()):
        a.plot(v["train_loss"], label="train loss")
        if "val_loss" in v: a.plot(v["val_loss"], label="validation loss (fold 0)")
        a.set_title(f"{n}: training history"); a.set_xlabel("iteration / epoch"); a.set_ylabel("log loss"); a.legend()
        if "val_score" in v:
            b = a.twinx(); b.plot(v["val_score"], color="tab:red", label="val. accuracy"); b.set_ylabel("validation accuracy"); b.legend(loc="center right")
    fig.tight_layout(); fig.savefig(OUT / "training_histories.png", dpi=150); plt.close(fig)
    print("plots written to", OUT)


if __name__ == "__main__":
    main()
