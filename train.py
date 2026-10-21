"""
train.py
--------
Training script for PhishGuard.

Usage:
    python train.py

Pipeline steps:
1. Loads and validates raw URL dataset (data/urls.csv).
2. Extracts 16 lexical and structural features from raw URLs.
3. Performs stratified 80/20 train/test split (random_state=42).
4. Runs 5-fold Stratified Cross-Validation on training data only.
5. Trains four comparative models: Random Forest, Decision Tree, KNN, SVM.
6. Evaluates on independent test set and selects the optimal model.
7. Persists fitted model pipelines to models/ and metrics to results/.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import get_dataset_summary, load_dataset
from src.model_pipeline import (
    build_model_pipelines,
    run_cross_validation,
    train_and_select_best,
)
from src.preprocessing import extract_dataset_features, split_data

DATA_PATH = PROJECT_ROOT / "data" / "urls.csv"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


def train_pipeline() -> dict:
    """Execute the complete training and evaluation pipeline."""
    print("=" * 68)
    print("                PHISHGUARD MODEL TRAINING PIPELINE")
    print("=" * 68)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load dataset
    print(f"\n[1/5] Loading and validating dataset from '{DATA_PATH.relative_to(PROJECT_ROOT)}' ...")
    df = load_dataset(DATA_PATH)
    summary = get_dataset_summary(df)
    print(f"      Total URLs: {summary['total_instances']:,}")
    print(f"      Legitimate: {summary['legitimate_instances']:,} ({summary['legitimate_percentage']}%)")
    print(f"      Phishing  : {summary['phishing_instances']:,} ({summary['phishing_percentage']}%)")

    # 2. Extract features
    print("\n[2/5] Extracting 16 lexical & structural URL features ...")
    X, y = extract_dataset_features(df)
    print(f"      Extracted feature matrix: {X.shape[0]:,} rows x {X.shape[1]} features")

    # 3. Train/test split
    print("\n[3/5] Performing stratified 80/20 train/test split (random_state=42) ...")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.20, random_state=42)
    print(f"      Training set: {len(X_train):,} samples")
    print(f"      Held-out test set: {len(X_test):,} samples (strictly independent)")

    # 4. Cross-validation
    print("\n[4/5] Running 5-fold Stratified Cross-Validation on X_train ...")
    pipelines = build_model_pipelines(random_state=42)
    cv_summary = run_cross_validation(pipelines, X_train, y_train, n_splits=5, random_state=42)

    for m_name, cv_s in cv_summary.items():
        print(f"      {m_name:24s} | CV F1: {cv_s['cv_f1_mean']:.4f} (±{cv_s['cv_f1_std']:.4f}) | CV Acc: {cv_s['cv_accuracy_mean']:.4f}")

    # 5. Fit on X_train, evaluate on held-out X_test, save models
    print("\n[5/5] Training pipelines and evaluating on independent test set ...")
    results, best = train_and_select_best(pipelines, X_train, y_train, X_test, y_test, MODELS_DIR)

    # 6. Save results
    comparison_rows = []
    cm_rows = []
    for r in results:
        comparison_rows.append({
            "Model": r["name"],
            "Accuracy": r["accuracy"],
            "Precision": r["precision"],
            "Recall": r["recall"],
            "F1-Score": r["f1"],
            "ROC-AUC": r["roc_auc"] if r["roc_auc"] is not None else "N/A",
            "FPR": r["fpr"],
            "FNR": r["fnr"],
            "Training Time (s)": r["training_time_s"],
        })
        cm = r["confusion_matrix"]
        cm_rows.append({
            "Model": r["name"],
            "True Negatives (TN)": cm["tn"],
            "False Positives (FP)": cm["fp"],
            "False Negatives (FN)": cm["fn"],
            "True Positives (TP)": cm["tp"],
        })

    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    cm_df = pd.DataFrame(cm_rows)
    cm_df.to_csv(RESULTS_DIR / "confusion_matrix.csv", index=False)

    metrics_payload = {
        "dataset_summary": summary,
        "split_config": {"train_samples": len(X_train), "test_samples": len(X_test), "random_state": 42},
        "cross_validation": cv_summary,
        "test_results": {r["name"]: {k: v for k, v in r.items() if k != "pipeline"} for r in results},
        "selected_model": {
            "name": best["name"],
            "accuracy": best["accuracy"],
            "precision": best["precision"],
            "recall": best["recall"],
            "f1": best["f1"],
            "roc_auc": best["roc_auc"],
            "fpr": best["fpr"],
            "fnr": best["fnr"],
            "artifact": "models/best_model.joblib",
        },
    }
    with open(RESULTS_DIR / "metrics.json", "w") as fh:
        json.dump(metrics_payload, fh, indent=2)

    print("\n" + "=" * 68)
    print("                    INDEPENDENT TEST SET RESULTS")
    print("=" * 68)
    print(comp_df.to_string(index=False))
    print("=" * 68)
    print(f"SELECTED PRODUCTION MODEL: {best['name']}")
    print(f"  Test Accuracy : {best['accuracy']:.4f}")
    print(f"  Test Recall   : {best['recall']:.4f} (Phishing Detection Rate)")
    print(f"  Test Precision: {best['precision']:.4f}")
    print(f"  Test F1-Score : {best['f1']:.4f}")
    print(f"  Test ROC-AUC  : {best['roc_auc']:.4f}")
    print(f"  Saved To      : models/best_model.joblib")
    print("=" * 68 + "\n")

    return metrics_payload


if __name__ == "__main__":
    train_pipeline()
