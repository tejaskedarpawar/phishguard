"""
src/evaluation.py
-----------------
Evaluation utilities, metric computation, plotting, and artifact persistence.

All metrics are computed strictly on the held-out test set.
Positive class = -1 (Phishing) for precision, recall, and F1-score.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

# Phishing is explicitly the positive class of interest
POS_LABEL = -1
NEG_LABEL = 1

# Color palette for charts
COLORS = {
    "Decision Tree": "#2196F3",
    "K-Nearest Neighbors": "#FF9800",
    "Random Forest": "#4CAF50",
    "Support Vector Machine": "#9C27B0",
}
DEFAULT_COLOR = "#607D8B"


def evaluate_model(
    name: str,
    pipeline: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """Train a pipeline and compute all metrics on the independent test set.

    Parameters
    ----------
    name:
        Human-readable model name.
    pipeline:
        Scikit-learn Pipeline.
    X_train, y_train:
        Training data.
    X_test, y_test:
        Held-out test data.

    Returns
    -------
    dict with metrics, confusion matrix counts, and predictions.
    """
    logger.info("Training pipeline for '%s' ...", name)
    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    training_time = time.perf_counter() - t0
    logger.info("  Trained in %.4f seconds.", training_time)

    y_pred = pipeline.predict(X_test)

    # Probabilities / decision scores for ROC analysis
    y_prob: Optional[np.ndarray] = None
    if hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba(X_test)
            classes = list(pipeline.classes_)
            pos_idx = classes.index(POS_LABEL)
            y_prob = proba[:, pos_idx]
        except Exception as exc:
            logger.debug("predict_proba failed for %s: %s", name, exc)
    elif hasattr(pipeline, "decision_function"):
        try:
            # For SVM decision function: higher score normally favors class with higher index
            # If classes_ is [-1, 1], decision_function > 0 favors 1.
            # Invert so higher score represents higher likelihood of POS_LABEL (-1).
            classes = list(pipeline.classes_)
            raw_scores = pipeline.decision_function(X_test)
            if classes == [-1, 1]:
                y_prob = -raw_scores
            else:
                y_prob = raw_scores
        except Exception as exc:
            logger.debug("decision_function failed for %s: %s", name, exc)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=POS_LABEL, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=POS_LABEL, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=POS_LABEL, zero_division=0)

    roc_auc: Optional[float] = None
    if y_prob is not None:
        try:
            y_test_binary = (y_test == POS_LABEL).astype(int)
            roc_auc = float(roc_auc_score(y_test_binary, y_prob))
        except Exception as exc:
            logger.debug("ROC-AUC computation failed for %s: %s", name, exc)

    # Detailed Confusion Matrix breakdown:
    # labels=[-1, 1] means row/col 0 is -1 (Phishing), row/col 1 is 1 (Legitimate)
    cm = confusion_matrix(y_test, y_pred, labels=[POS_LABEL, NEG_LABEL])
    tp = int(cm[0, 0])  # Actual Phishing, Pred Phishing
    fn = int(cm[0, 1])  # Actual Phishing, Pred Legitimate
    fp = int(cm[1, 0])  # Actual Legitimate, Pred Phishing
    tn = int(cm[1, 1])  # Actual Legitimate, Pred Legitimate

    result: Dict[str, Any] = {
        "name": name,
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4) if roc_auc is not None else None,
        "training_time_s": round(float(training_time), 4),
        "cm_stats": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
        "confusion_matrix": cm.tolist(),
        "y_pred": y_pred,
        "y_prob": y_prob,
    }

    logger.info(
        "  [%s] Acc=%.4f | Prec=%.4f | Rec=%.4f | F1=%.4f | AUC=%s | TP=%d FN=%d FP=%d TN=%d",
        name,
        accuracy,
        precision,
        recall,
        f1,
        f"{roc_auc:.4f}" if roc_auc is not None else "N/A",
        tp, fn, fp, tn,
    )
    return result


def select_best_model(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Select the best performing model.

    Criterion: Highest F1-score on the Phishing class (-1).
    Tiebreaker: Highest ROC-AUC score, followed by training time (faster is better).
    """
    def sort_key(r: Dict[str, Any]) -> Tuple[float, float, float]:
        auc_val = r["roc_auc"] if r["roc_auc"] is not None else 0.0
        time_penalty = -r["training_time_s"]  # faster is better in tie
        return (r["f1"], auc_val, time_penalty)

    best = max(results, key=sort_key)
    logger.info(
        "Selected Best Model: %s (F1=%.4f, ROC-AUC=%s)",
        best["name"],
        best["f1"],
        f"{best['roc_auc']:.4f}" if best["roc_auc"] is not None else "N/A",
    )
    return best


# ---------------------------------------------------------------------------
# Plotting Functions
# ---------------------------------------------------------------------------


def plot_confusion_matrices(
    results: List[Dict[str, Any]],
    y_test: pd.Series,
    results_dir: Path,
) -> None:
    """Save 2x2 grid of confusion matrices formatted with TP, FN, FP, TN."""
    results_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()

    display_labels = ["Phishing (-1)", "Legitimate (1)"]

    for idx, res in enumerate(results):
        ax = axes[idx]
        cm = np.array(res["confusion_matrix"])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format="d")
        color = COLORS.get(res["name"], DEFAULT_COLOR)
        ax.set_title(res["name"], fontsize=12, fontweight="bold", color=color, pad=8)
        ax.tick_params(labelsize=9)

    plt.suptitle("Confusion Matrices — Held-out Test Set (20%)", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    out_path = results_dir / "confusion_matrices.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confusion matrices plot to '%s'", out_path)


def plot_roc_curves(
    results: List[Dict[str, Any]],
    y_test: pd.Series,
    results_dir: Path,
) -> None:
    """Save combined ROC curves plot for models with probability estimates."""
    results_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Chance (AUC = 0.50)")

    y_test_binary = (y_test == POS_LABEL).astype(int)
    any_plotted = False

    for res in results:
        if res.get("y_prob") is None:
            continue
        try:
            fpr, tpr, _ = roc_curve(y_test_binary, res["y_prob"])
            auc_val = res["roc_auc"] if res["roc_auc"] is not None else auc(fpr, tpr)
            color = COLORS.get(res["name"], DEFAULT_COLOR)
            ax.plot(fpr, tpr, color=color, lw=2.2, label=f"{res['name']} (AUC = {auc_val:.4f})")
            any_plotted = True
        except Exception as exc:
            logger.warning("Could not plot ROC for %s: %s", res["name"], exc)

    if not any_plotted:
        ax.text(0.5, 0.5, "ROC unavailable", ha="center", va="center")

    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    ax.set_title("ROC Curves — Phishing Detection (Positive Class: -1)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9.5, frameon=True)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.02])
    ax.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()

    out_path = results_dir / "roc_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved ROC curves plot to '%s'", out_path)


def plot_performance_comparison(
    results: List[Dict[str, Any]],
    results_dir: Path,
) -> None:
    """Save bar charts comparing F1, Recall, Precision, and Accuracy."""
    results_dir.mkdir(parents=True, exist_ok=True)
    names = [r["name"] for r in results]
    metrics = ["F1-Score", "Recall", "Precision", "Accuracy"]

    df_data = {
        "Model": names,
        "F1-Score": [r["f1"] for r in results],
        "Recall": [r["recall"] for r in results],
        "Precision": [r["precision"] for r in results],
        "Accuracy": [r["accuracy"] for r in results],
    }
    df = pd.DataFrame(df_data)

    x = np.arange(len(names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bar_colors = ["#2563eb", "#3b82f6", "#60a5fa", "#93c5fd"]

    for i, metric in enumerate(metrics):
        offset = (i - 1.5) * width
        rects = ax.bar(x + offset, df[metric], width, label=metric, color=bar_colors[i], edgecolor="white")
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=7.5, fontweight="bold",
            )

    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Performance Metrics by Classifier (Phishing Class: -1)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.legend(loc="lower right", fontsize=9.5)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    out_main = results_dir / "performance_comparison.png"
    fig.savefig(out_main, dpi=150, bbox_inches="tight")

    # Also save as f1_comparison.png for backward compatibility
    out_f1 = results_dir / "f1_comparison.png"
    fig.savefig(out_f1, dpi=150, bbox_inches="tight")

    plt.close(fig)
    logger.info("Saved performance comparison plot to '%s' and '%s'", out_main, out_f1)


def plot_feature_importance(
    pipelines: Dict[str, Any],
    feature_names: List[str],
    results_dir: Path,
    top_n: int = 12,
) -> None:
    """Save feature importance charts for tree-based models."""
    results_dir.mkdir(parents=True, exist_ok=True)

    tree_models = {}
    for name, pipe in pipelines.items():
        clf = pipe.named_steps.get("clf", pipe)
        if hasattr(clf, "feature_importances_"):
            tree_models[name] = clf.feature_importances_

    if not tree_models:
        logger.info("No tree models with feature_importances_ found.")
        return

    # Combined figure for tree models
    n_models = len(tree_models)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 6), sharey=False)
    if n_models == 1:
        axes = [axes]

    for idx, (name, importances) in enumerate(tree_models.items()):
        ax = axes[idx]
        sorted_indices = np.argsort(importances)[::-1][:top_n]
        sorted_names = [feature_names[i] for i in sorted_indices]
        sorted_vals = importances[sorted_indices]

        color = COLORS.get(name, "#2563eb")
        ax.barh(sorted_names[::-1], sorted_vals[::-1], color=color, edgecolor="white")
        ax.set_xlabel("Relative Importance", fontsize=10)
        ax.set_title(f"Top {top_n} Features — {name}", fontsize=11, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", linestyle="--", alpha=0.35)

        # Save individual plot as well
        indiv_fig, indiv_ax = plt.subplots(figsize=(8, 5.5))
        indiv_ax.barh(sorted_names[::-1], sorted_vals[::-1], color=color, edgecolor="white")
        indiv_ax.set_xlabel("Relative Importance", fontsize=10)
        indiv_ax.set_title(f"Top {top_n} Features — {name}", fontsize=12, fontweight="bold")
        indiv_ax.spines["top"].set_visible(False)
        indiv_ax.spines["right"].set_visible(False)
        indiv_ax.grid(axis="x", linestyle="--", alpha=0.35)
        plt.tight_layout()
        slug = name.replace(" ", "_").lower()
        indiv_path = results_dir / f"feature_importance_{slug}.png"
        indiv_fig.savefig(indiv_path, dpi=150, bbox_inches="tight")
        plt.close(indiv_fig)

    fig.tight_layout()
    comb_path = results_dir / "feature_importance.png"
    fig.savefig(comb_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved combined feature importance to '%s'", comb_path)


# ---------------------------------------------------------------------------
# Persistence Helpers
# ---------------------------------------------------------------------------


def save_comparison_csv(results: List[Dict[str, Any]], results_dir: Path) -> Path:
    """Save model comparison table to CSV."""
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in results:
        rows.append({
            "Model": r["name"],
            "Accuracy": r["accuracy"],
            "Precision": r["precision"],
            "Recall": r["recall"],
            "F1-Score": r["f1"],
            "ROC-AUC": r["roc_auc"] if r["roc_auc"] is not None else "N/A",
            "Training Time (s)": r["training_time_s"],
        })
    df = pd.DataFrame(rows)
    out_path = results_dir / "model_comparison.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved model comparison CSV to '%s'", out_path)
    return out_path


def save_experiment_summary(
    results: List[Dict[str, Any]],
    best_model: Dict[str, Any],
    dataset_info: Dict[str, Any],
    cv_summary: Optional[Dict[str, Dict[str, float]]],
    results_dir: Path,
) -> Path:
    """Save comprehensive experiment summary JSON as required by Section 4."""
    results_dir.mkdir(parents=True, exist_ok=True)

    model_metrics = {}
    for r in results:
        model_metrics[r["name"]] = {
            "accuracy": r["accuracy"],
            "precision": r["precision"],
            "recall": r["recall"],
            "f1": r["f1"],
            "roc_auc": r["roc_auc"],
            "training_time_s": r["training_time_s"],
            "confusion_matrix": r["confusion_matrix"],
            "cm_stats": r["cm_stats"],
            "cross_validation_train": cv_summary.get(r["name"]) if cv_summary else None,
        }

    summary = {
        "dataset": {
            "name": "UCI Phishing Websites Dataset",
            "source": "https://archive.ics.uci.edu/dataset/327/phishing+websites",
            "total_instances": dataset_info.get("original_shape", [0])[0],
            "cleaned_instances": dataset_info.get("cleaned_shape", [0])[0],
            "duplicates_removed": dataset_info.get("duplicates_removed", 0),
            "feature_count": dataset_info.get("feature_count", 30),
            "target_mapping": {
                "-1": "Phishing (Positive Class)",
                "1": "Legitimate (Negative Class)",
            },
            "class_distribution": dataset_info.get("class_distribution", {}),
            "split_configuration": {
                "train_ratio": 0.80,
                "test_ratio": 0.20,
                "stratified": True,
                "random_state": 42,
                "train_instances": dataset_info.get("train_size", 0),
                "test_instances": dataset_info.get("test_size", 0),
            },
        },
        "selected_best_model": {
            "name": best_model["name"],
            "selection_criterion": "Highest F1-Score on phishing class (-1) with ROC-AUC tiebreaker",
            "test_accuracy": best_model["accuracy"],
            "test_precision": best_model["precision"],
            "test_recall": best_model["recall"],
            "test_f1": best_model["f1"],
            "test_roc_auc": best_model["roc_auc"],
            "training_time_s": best_model["training_time_s"],
        },
        "models": model_metrics,
    }

    out_path = results_dir / "experiment_summary.json"
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Saved experiment summary JSON to '%s'", out_path)

    # Also save best_model.json
    best_info = {
        "best_model": best_model["name"],
        "selection_criterion": "Highest F1-score on phishing class (−1); ROC-AUC as tiebreaker",
        "f1": best_model["f1"],
        "roc_auc": best_model["roc_auc"],
        "accuracy": best_model["accuracy"],
        "precision": best_model["precision"],
        "recall": best_model["recall"],
        "training_time_s": best_model["training_time_s"],
    }
    with open(results_dir / "best_model.json", "w") as fh:
        json.dump(best_info, fh, indent=2)

    return out_path
