"""
src/evaluation.py
-----------------
Functions for evaluating classifiers and generating plots / CSVs.

All metrics are computed on the held-out test set only.
Positive class = -1 (Phishing) for precision, recall, F1.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments
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

# Positive class label = Phishing (-1)
POS_LABEL = -1

# Colour palette for plots
COLORS = {
    "Decision Tree": "#2196F3",
    "K-Nearest Neighbors": "#FF9800",
    "Random Forest": "#4CAF50",
    "Support Vector Machine": "#9C27B0",
}
DEFAULT_COLOR = "#607D8B"


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def evaluate_model(
    name: str,
    pipeline: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """Train a pipeline and compute all metrics on the test set.

    Parameters
    ----------
    name:
        Human-readable model name.
    pipeline:
        Unfitted sklearn Pipeline (or estimator).
    X_train, y_train:
        Training data.
    X_test, y_test:
        Held-out test data.

    Returns
    -------
    dict with keys: name, accuracy, precision, recall, f1, roc_auc,
    training_time_s, y_pred, y_prob (or None).
    """
    logger.info("Training: %s …", name)
    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    training_time = time.perf_counter() - t0
    logger.info("  → trained in %.3f s", training_time)

    y_pred = pipeline.predict(X_test)

    # Probability / decision scores for ROC
    y_prob: Optional[np.ndarray] = None
    if hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba(X_test)
            # Column for POS_LABEL (-1)
            classes = list(pipeline.classes_)
            pos_idx = classes.index(POS_LABEL)
            y_prob = proba[:, pos_idx]
        except Exception:
            pass
    elif hasattr(pipeline, "decision_function"):
        try:
            # For SVM: higher decision score → more likely positive class
            # Negate so higher = more phishing
            y_prob = -pipeline.decision_function(X_test)
        except Exception:
            pass

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=POS_LABEL, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=POS_LABEL, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=POS_LABEL, zero_division=0)

    roc_auc: Optional[float] = None
    if y_prob is not None:
        try:
            # y_prob = P(phishing | x).  roc_auc_score treats higher y_score as
            # more likely to be the positive class.  We tell it pos_label=POS_LABEL
            # so it interprets y correctly as {-1, 1}.
            roc_auc = roc_auc_score(
                (y_test == POS_LABEL).astype(int), y_prob
            )
        except Exception:
            pass

    result = {
        "name": name,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "training_time_s": round(training_time, 3),
        "y_pred": y_pred,
        "y_prob": y_prob,
    }
    logger.info(
        "  Acc=%.4f  P=%.4f  R=%.4f  F1=%.4f  AUC=%s",
        accuracy, precision, recall, f1,
        f"{roc_auc:.4f}" if roc_auc else "N/A",
    )
    return result


def select_best_model(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Choose the best model by F1-score (phishing class), breaking ties with ROC-AUC.

    Rationale: In phishing detection, missing a phishing site (false negative)
    is more dangerous than a false alarm.  F1 balances precision and recall,
    so it is a better criterion than accuracy alone.
    """
    def sort_key(r: dict) -> Tuple:
        auc_val = r["roc_auc"] if r["roc_auc"] is not None else 0.0
        return (r["f1"], auc_val)

    best = max(results, key=sort_key)
    logger.info(
        "Best model: %s  (F1=%.4f, AUC=%s)",
        best["name"], best["f1"],
        f"{best['roc_auc']:.4f}" if best["roc_auc"] else "N/A",
    )
    return best


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def plot_confusion_matrices(
    results: List[Dict[str, Any]],
    y_test: pd.Series,
    results_dir: Path,
) -> None:
    """Save a 2×2 grid of confusion matrices as a PNG."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    labels = ["Phishing (−1)", "Legitimate (1)"]
    display_labels = ["Phishing", "Legitimate"]

    for idx, res in enumerate(results):
        ax = axes[idx]
        cm = confusion_matrix(y_test, res["y_pred"], labels=[-1, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        color = COLORS.get(res["name"], DEFAULT_COLOR)
        ax.set_title(res["name"], fontsize=13, fontweight="bold", color=color)
        ax.tick_params(labelsize=9)

    plt.suptitle("Confusion Matrices — Held-out Test Set", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = results_dir / "confusion_matrices.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)


def plot_roc_curves(
    results: List[Dict[str, Any]],
    y_test: pd.Series,
    results_dir: Path,
) -> None:
    """Save ROC curves for models that provide probability scores."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.50)")

    any_plotted = False
    for res in results:
        if res["y_prob"] is None:
            continue
        fpr, tpr, _ = roc_curve(
            (y_test == POS_LABEL).astype(int), res["y_prob"]
        )
        auc_val = res["roc_auc"] if res["roc_auc"] else auc(fpr, tpr)
        color = COLORS.get(res["name"], DEFAULT_COLOR)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{res['name']} (AUC = {auc_val:.4f})")
        any_plotted = True

    if not any_plotted:
        ax.text(0.5, 0.5, "ROC not available for selected models",
                ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves — Phishing Detection", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    plt.tight_layout()
    out = results_dir / "roc_curves.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)


def plot_f1_comparison(results: List[Dict[str, Any]], results_dir: Path) -> None:
    """Save a bar chart comparing F1-scores across models."""
    names = [r["name"] for r in results]
    f1_scores = [r["f1"] for r in results]
    colors = [COLORS.get(n, DEFAULT_COLOR) for n in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, f1_scores, color=colors, width=0.5, edgecolor="white")

    for bar, val in zip(bars, f1_scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1-Score (Phishing class)", fontsize=11)
    ax.set_title("Model Comparison — F1-Score", fontsize=13, fontweight="bold")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = results_dir / "f1_comparison.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)


def plot_feature_importance(
    pipeline: Any,
    feature_names: List[str],
    model_name: str,
    results_dir: Path,
    top_n: int = 15,
) -> None:
    """Save a feature-importance bar chart (Decision Tree / Random Forest)."""
    estimator = pipeline.named_steps.get("clf", pipeline)
    if not hasattr(estimator, "feature_importances_"):
        logger.info("Model '%s' does not expose feature importances — skipping.", model_name)
        return

    importances = estimator.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in indices]
    top_vals = importances[indices]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top_names[::-1], top_vals[::-1], color="#2196F3", edgecolor="white")
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = results_dir / f"feature_importance_{model_name.replace(' ', '_').lower()}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)


# ---------------------------------------------------------------------------
# CSV / JSON persistence
# ---------------------------------------------------------------------------


def save_comparison_csv(results: List[Dict[str, Any]], results_dir: Path) -> None:
    """Write model comparison table to CSV."""
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
    out = results_dir / "model_comparison.csv"
    df.to_csv(out, index=False)
    logger.info("Saved: %s", out)


def save_best_model_info(best: Dict[str, Any], criterion: str, results_dir: Path) -> None:
    """Persist best-model name and selection criterion as JSON."""
    info = {
        "best_model": best["name"],
        "selection_criterion": criterion,
        "f1": best["f1"],
        "roc_auc": best["roc_auc"],
        "accuracy": best["accuracy"],
        "precision": best["precision"],
        "recall": best["recall"],
        "training_time_s": best["training_time_s"],
    }
    out = results_dir / "best_model.json"
    with open(out, "w") as fh:
        json.dump(info, fh, indent=2)
    logger.info("Saved: %s", out)
