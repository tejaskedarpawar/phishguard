"""
main.py
-------
PhishGuard — Command-Line Interface (CLI).

Usage:
    python main.py --url https://github.com
    python main.py --train
    python main.py --evaluate
    python main.py            (Interactive menu)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluate import evaluate_benchmark
from src.data_loader import get_dataset_summary, load_dataset
from src.prediction import predict_url
from train import train_pipeline

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "urls.csv"
RESULTS_DIR = PROJECT_ROOT / "results"


def print_banner() -> None:
    print("""
============================================================
                        PHISHGUARD
            ML-Based URL Risk Classifier (CLI)
============================================================""")


def display_url_analysis(url: str) -> None:
    """Analyze a URL and render clean, professional terminal output."""
    print_banner()
    print(f"\nAnalyzing: {url}\n" + "-" * 60)

    res = predict_url(url, MODEL_PATH)

    if res["status"] == "error":
        print(f"[!] Analysis Failed: {res.get('error_message')}")
        print("=" * 60 + "\n")
        return

    confidence_str = f"{res['confidence']:.1%}" if res.get("confidence") is not None else "N/A"
    phish_prob = f"{res['phishing_probability']:.1%}" if res.get("phishing_probability") is not None else "N/A"
    legit_prob = f"{res['legitimate_probability']:.1%}" if res.get("legitimate_probability") is not None else "N/A"

    print(f"Normalized URL : {res['normalized_url']}")
    print(f"Hostname       : {res['hostname']}")
    print(f"Model Result   : {res['predicted_label']}")
    print(f"Risk Status    : {res['risk_status']}")
    print(f"Confidence     : {confidence_str} (Phishing: {phish_prob} | Legit: {legit_prob})")
    print(f"Classifier     : {res['model_name']} (models/best_model.joblib)")
    print("-" * 60)
    print("Lexical Feature Indicators:")
    for k, v in res["feature_highlights"].items():
        print(f"  • {k:22s}: {v}")
    print("-" * 60)
    print(f"{res['disclaimer']}")
    print("=" * 60 + "\n")


def display_dataset_info() -> None:
    """Print dataset statistics and class distribution."""
    print_banner()
    if not DATA_PATH.exists():
        print(f"\n[!] Dataset file not found at: {DATA_PATH}\n")
        return

    df = load_dataset(DATA_PATH)
    summary = get_dataset_summary(df)

    print("\n--- DATASET INFORMATION & ATTRIBUTION ---")
    print(f"Dataset File   : {DATA_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Total Instances: {summary['total_instances']:,} balanced URLs")
    print(f"Legitimate (0) : {summary['legitimate_instances']:,} ({summary['legitimate_percentage']}%)")
    print(f"Phishing (1)   : {summary['phishing_instances']:,} ({summary['phishing_percentage']}%)")
    print("Sources        : OpenDNS Top Domains, UNB ISCX-URL-2016, PhishTank Database")
    print("Exclusions     : Benchmark domains (e.g. github.com) strictly held out")
    print("Feature Count  : 16 pure lexical and structural attributes")
    print("=" * 60 + "\n")


def display_model_evaluation() -> None:
    """Show saved model evaluation comparison table."""
    print_banner()
    comp_file = RESULTS_DIR / "model_comparison.csv"
    metrics_file = RESULTS_DIR / "metrics.json"

    if not comp_file.exists():
        print("\n[!] No evaluation results found. Please train models first using: python main.py --train\n")
        return

    print("\n--- MODEL PERFORMANCE COMPARISON (Held-Out Test Set) ---")
    df = pd.read_csv(comp_file)
    print(df.to_string(index=False))

    if metrics_file.exists():
        with open(metrics_file) as fh:
            data = json.load(fh)
        sel = data.get("selected_model", {})
        print("-" * 60)
        print(f"Selected Production Model: {sel.get('name')} (F1 = {sel.get('f1')})")
        print(f"  Accuracy : {sel.get('accuracy'):.4f} | Recall: {sel.get('recall'):.4f} | Precision: {sel.get('precision'):.4f}")
        print(f"  FPR      : {sel.get('fpr'):.4f} | FNR   : {sel.get('fnr'):.4f}")
    print("=" * 60 + "\n")


def interactive_menu() -> None:
    """Render interactive CLI terminal menu."""
    while True:
        print_banner()
        print("  1. Analyze a URL")
        print("  2. Train the model")
        print("  3. View model evaluation results")
        print("  4. View dataset information")
        print("  5. Exit")
        print("-" * 60)

        choice = input("Enter option [1-5]: ").strip()

        if choice == "1":
            target = input("\nEnter a URL to inspect: ").strip()
            if target:
                display_url_analysis(target)
            else:
                print("[!] URL cannot be empty.\n")
            input("Press Enter to return to menu...")
        elif choice == "2":
            print("\nStarting model training...")
            train_pipeline()
            input("Press Enter to return to menu...")
        elif choice == "3":
            display_model_evaluation()
            evaluate_benchmark()
            input("Press Enter to return to menu...")
        elif choice == "4":
            display_dataset_info()
            input("Press Enter to return to menu...")
        elif choice == "5" or choice.lower() in ("q", "exit", "quit"):
            print("\nExiting PhishGuard. Goodbye!\n")
            break
        else:
            print("[!] Invalid option. Please enter a number between 1 and 5.\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PhishGuard — ML-Based URL Phishing Detection System (CLI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--url", type=str, help="Analyze a specific URL for phishing risk")
    parser.add_argument("--train", action="store_true", help="Train model pipelines and evaluate on test set")
    parser.add_argument("--evaluate", action="store_true", help="Run benchmark evaluation on diverse test URLs")
    parser.add_argument("--info", action="store_true", help="Display dataset statistics and schema details")

    args = parser.parse_args()

    if args.url:
        display_url_analysis(args.url)
    elif args.train:
        train_pipeline()
    elif args.evaluate:
        evaluate_benchmark()
    elif args.info:
        display_dataset_info()
    else:
        interactive_menu()


if __name__ == "__main__":
    main()
