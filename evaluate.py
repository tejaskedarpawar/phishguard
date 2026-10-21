"""
evaluate.py
-----------
Comprehensive model evaluation script for PhishGuard.

Evaluates the trained model on:
1. Held-out test set metrics from results/model_comparison.csv.
2. Curated benchmark evaluation set from data/test_urls.csv, explicitly
   evaluating github.com variations and known phishing attacks.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.prediction import predict_url

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
BENCHMARK_PATH = PROJECT_ROOT / "data" / "test_urls.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

logging.basicConfig(level=logging.WARNING)


def evaluate_benchmark() -> dict:
    """Evaluate benchmark test URLs and display results."""
    print("=" * 78)
    print("                PHISHGUARD BENCHMARK URL EVALUATION")
    print("=" * 78)

    if not MODEL_PATH.exists():
        print(f"\n[ERROR] Trained model not found at: {MODEL_PATH}")
        print("Please train the model first by running: python main.py --train\n")
        return {}

    # 1. Print Held-out Test Set Metrics if available
    comp_file = RESULTS_DIR / "model_comparison.csv"
    if comp_file.exists():
        print("\n--- Summary of 4 Comparative Classifiers (Held-Out Test Set) ---")
        df_comp = pd.read_csv(comp_file)
        print(df_comp.to_string(index=False))
        print("-" * 78)

    # 2. Evaluate Curated Test URLs
    if not BENCHMARK_PATH.exists():
        print(f"\n[WARNING] Benchmark test set not found at: {BENCHMARK_PATH}")
        return {}

    test_df = pd.read_csv(BENCHMARK_PATH)
    print(f"\nEvaluating {len(test_df)} Diverse Test URLs (Legitimate + Phishing Attacks) ...\n")

    results_table = []
    benchmark_records = []
    correct_count = 0
    fp_count = 0
    fn_count = 0

    for _, row in test_df.iterrows():
        url = str(row["url"]).strip()
        expected_code = int(row["label"])
        expected_label = "Phishing" if expected_code == 1 else "Legitimate"
        category = row.get("category", "Test")

        res = predict_url(url, MODEL_PATH)

        if res["status"] != "success":
            pred_label = "Error"
            confidence = 0.0
            is_correct = False
        else:
            pred_code = res["prediction"]
            pred_label = res["predicted_label"]
            confidence = res.get("confidence") or 0.0
            is_correct = (pred_code == expected_code)

            if not is_correct:
                if expected_code == 0 and pred_code == 1:
                    fp_count += 1
                elif expected_code == 1 and pred_code == 0:
                    fn_count += 1

        if is_correct:
            correct_count += 1

        status_symbol = "PASS" if is_correct else "FAIL"

        results_table.append({
            "URL": url if len(url) <= 45 else url[:42] + "...",
            "Category": category,
            "Expected": expected_label,
            "Predicted": pred_label,
            "Confidence": f"{confidence:.1%}" if confidence else "N/A",
            "Result": status_symbol,
        })

        benchmark_records.append({
            "url": url,
            "category": category,
            "expected_label": expected_label,
            "predicted_label": pred_label,
            "confidence": confidence,
            "correct": is_correct,
            "normalized_url": res.get("normalized_url", ""),
        })

    display_df = pd.DataFrame(results_table)
    print(display_df.to_string(index=False))

    total = len(test_df)
    accuracy = (correct_count / total) * 100

    print("=" * 78)
    print(f"BENCHMARK ACCURACY: {correct_count}/{total} ({accuracy:.1f}%)")
    print(f"False Positives: {fp_count} | False Negatives: {fn_count}")
    print("=" * 78 + "\n")

    # Save benchmark records
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "benchmark_urls_results.json"
    with open(out_path, "w") as fh:
        json.dump({
            "total_urls": total,
            "correct": correct_count,
            "accuracy_pct": round(accuracy, 2),
            "false_positives": fp_count,
            "false_negatives": fn_count,
            "records": benchmark_records,
        }, fh, indent=2)

    return {"accuracy": accuracy, "correct": correct_count, "total": total}


if __name__ == "__main__":
    evaluate_benchmark()
