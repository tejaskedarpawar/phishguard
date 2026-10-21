"""
train.py
--------
End-to-end training and evaluation script for PhishGuard.

Usage:
    python train.py

Pipeline steps:
1. Loads and validates the UCI Phishing Websites dataset (data/phishing.csv).
2. Cleans the data (schema check, duplicate detection and removal, type validation).
3. Splits into training (80%) and independent test (20%) sets using stratified sampling.
4. Executes 5-fold Stratified Cross-Validation on the training data only.
5. Trains all four Scikit-Learn pipelines:
   - Decision Tree (Information Gain / Entropy)
   - K-Nearest Neighbors (with StandardScaler inside Pipeline)
   - Random Forest (150 trees)
   - Support Vector Machine (RBF kernel, StandardScaler, Calibrated probabilities)
6. Evaluates all models on the held-out test set (Accuracy, Precision, Recall, F1, ROC-AUC).
7. Selects the best model based on F1-score on the Phishing class (-1).
8. Persists models to models/*.joblib and metrics/plots to results/.
9. Extracts verified held-out test samples into data/demo_samples.json for faculty demonstration.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "phishing.csv"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
DEMO_SAMPLES_PATH = PROJECT_ROOT / "data" / "demo_samples.json"

sys.path.insert(0, str(PROJECT_ROOT))

from src.data_cleaning import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    clean_dataset,
    load_dataset,
    split_data,
)
from src.evaluation import (
    plot_confusion_matrices,
    plot_feature_importance,
    plot_performance_comparison,
    plot_roc_curves,
    save_comparison_csv,
    save_experiment_summary,
)
from src.train_models import (
    build_pipelines,
    perform_cross_validation,
    train_and_evaluate_all,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")

RANDOM_STATE = 42


def generate_demo_samples(X_test, y_test, pipelines, out_path: Path) -> None:
    """Extract verified legitimate and phishing examples from the held-out test set."""
    # Find indices for legitimate (1) and phishing (-1)
    legit_indices = y_test[y_test == 1].index.tolist()
    phish_indices = y_test[y_test == -1].index.tolist()

    best_pipeline = pipelines.get("Decision Tree") or list(pipelines.values())[0]

    samples = {
        "legitimate_samples": [],
        "phishing_samples": [],
        "tested_urls": [
            {
                "url": "https://www.google.com",
                "expected_type": "Legitimate popular service",
                "note": "Lexical heuristics only. Live page attributes (PageRank, DNS, Traffic) are approximated offline.",
            },
            {
                "url": "http://192.168.1.100/paypal-update/login.php",
                "expected_type": "Phishing pattern",
                "note": "Contains raw IP address, target keyword in path, and suspicious sub-domain structure.",
            },
            {
                "url": "https://secure-login-account-verification-alert.com/signin",
                "expected_type": "Phishing pattern",
                "note": "Domain uses hyphenated deceptive brand keywords with long path.",
            },
            {
                "url": "http://user@evil-phishing-bank.net/auth",
                "expected_type": "Phishing pattern",
                "note": "Uses '@' credential spoofing symbol in authority section.",
            },
        ],
    }

    # Extract 3 verified legitimate test samples
    for i, idx in enumerate(legit_indices[:3], 1):
        features_row = X_test.loc[idx].to_dict()
        pred = int(best_pipeline.predict(X_test.loc[[idx]])[0])
        samples["legitimate_samples"].append({
            "id": f"legit_sample_{i}",
            "description": f"Verified Held-out Test Sample #{i} (Ground Truth: Legitimate)",
            "ground_truth": 1,
            "ground_truth_label": "Likely Legitimate",
            "model_prediction": pred,
            "features": features_row,
        })

    # Extract 3 verified phishing test samples
    for i, idx in enumerate(phish_indices[:3], 1):
        features_row = X_test.loc[idx].to_dict()
        pred = int(best_pipeline.predict(X_test.loc[[idx]])[0])
        samples["phishing_samples"].append({
            "id": f"phish_sample_{i}",
            "description": f"Verified Held-out Test Sample #{i} (Ground Truth: Phishing)",
            "ground_truth": -1,
            "ground_truth_label": "Potentially Phishing",
            "model_prediction": pred,
            "features": features_row,
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(samples, fh, indent=2)
    logger.info("Saved %d demo test fixtures to '%s'", len(samples["legitimate_samples"]) + len(samples["phishing_samples"]), out_path)


def main() -> None:
    logger.info("=" * 65)
    logger.info("PhishGuard — Reproducible Training and Evaluation Pipeline")
    logger.info("=" * 65)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load dataset
    logger.info("Step 1: Loading dataset from '%s' ...", DATA_PATH)
    try:
        df = load_dataset(DATA_PATH)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    # 2. Clean dataset
    logger.info("Step 2: Cleaning dataset & checking schema ...")
    X, y, cleaning_summary = clean_dataset(df, results_dir=RESULTS_DIR)
    logger.info(
        "  Cleaned dataset: %d rows x %d features", len(X), X.shape[1]
    )
    logger.info(
        "  Class balance: Phishing (-1) = %d (%.2f%%) | Legitimate (1) = %d (%.2f%%)",
        cleaning_summary["class_distribution"]["Phishing (−1)"],
        cleaning_summary["class_balance_pct"]["Phishing (−1)"],
        cleaning_summary["class_distribution"]["Legitimate (1)"],
        cleaning_summary["class_balance_pct"]["Legitimate (1)"],
    )

    # 3. Train / test split (80 / 20 stratified)
    logger.info("Step 3: Performing stratified 80/20 train/test split ...")
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    logger.info("  Training instances: %d | Held-out Test instances: %d", len(X_train), len(X_test))

    # Persist dataset info
    dataset_info = {
        **cleaning_summary,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "random_state": RANDOM_STATE,
        "test_split_ratio": 0.20,
    }
    with open(RESULTS_DIR / "dataset_info.json", "w") as fh:
        json.dump(dataset_info, fh, indent=2)

    # 4. Build pipelines
    pipelines = build_pipelines(random_state=RANDOM_STATE)

    # 5. Cross-Validation on X_train only
    logger.info("Step 4: Running 5-fold Stratified Cross-Validation on X_train ...")
    cv_summary = perform_cross_validation(
        pipelines=pipelines,
        X_train=X_train,
        y_train=y_train,
        n_splits=5,
        random_state=RANDOM_STATE,
    )

    # 6. Fit and evaluate on held-out test set
    logger.info("Step 5: Training pipelines and evaluating on independent test set ...")
    results, best_model = train_and_evaluate_all(
        pipelines=pipelines,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        models_dir=MODELS_DIR,
    )

    # 7. Generate plots and persistence artifacts
    logger.info("Step 6: Generating evaluation plots and summaries ...")
    save_comparison_csv(results, RESULTS_DIR)
    plot_confusion_matrices(results, y_test, RESULTS_DIR)
    plot_roc_curves(results, y_test, RESULTS_DIR)
    plot_performance_comparison(results, RESULTS_DIR)
    plot_feature_importance(pipelines, list(FEATURE_COLUMNS), RESULTS_DIR)

    save_experiment_summary(
        results=results,
        best_model=best_model,
        dataset_info=dataset_info,
        cv_summary=cv_summary,
        results_dir=RESULTS_DIR,
    )

    # 8. Generate demonstration fixtures
    logger.info("Step 7: Creating verified demo fixtures from held-out test set ...")
    generate_demo_samples(X_test, y_test, pipelines, DEMO_SAMPLES_PATH)

    logger.info("=" * 65)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("=" * 65)
    logger.info("Selected Best Model: %s", best_model["name"])
    logger.info("  Test Accuracy : %.4f", best_model["accuracy"])
    logger.info("  Test Precision: %.4f", best_model["precision"])
    logger.info("  Test Recall   : %.4f", best_model["recall"])
    logger.info("  Test F1-Score : %.4f", best_model["f1"])
    logger.info("  Test ROC-AUC  : %s", f"{best_model['roc_auc']:.4f}" if best_model["roc_auc"] else "N/A")
    logger.info("  Training Time : %.4f s", best_model["training_time_s"])
    logger.info("Artifacts saved to: %s and %s", MODELS_DIR, RESULTS_DIR)
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
