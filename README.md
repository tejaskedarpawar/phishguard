# PhishGuard — ML-Based Phishing Website Detection System

An academically rigorous machine learning project for detecting phishing websites using four comparative classifiers trained on the UCI Phishing Websites dataset. Built with Python, Scikit-Learn, and Streamlit.

---

## 1. Project Overview & Problem Statement

Phishing websites impersonate legitimate organizations (banks, email providers, payment gateways) to harvest credentials, session tokens, and financial data. Static blacklists fail against modern phishing campaigns that rapidly spin up ephemeral domains. 

**PhishGuard** evaluates machine learning classifiers on pre-extracted website and URL characteristics, implementing a complete, leak-free data science pipeline from data hygiene and cross-validation to interactive inference.

---

## 2. Dataset Information & Attribution

- **Dataset**: [Phishing Websites Dataset](https://archive.ics.uci.edu/dataset/327/phishing+websites)
- **Repository**: UCI Machine Learning Repository
- **Authors**: Rami Mohammad, Fadi Thabtah, Lee McCluskey (2015)
- **DOI**: `10.24432/C51W2X`
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Raw Dimensions**: 2,456 instances × 31 columns (30 input features + 1 target column `Result`)
- **Cleaned Dimensions**: 1,716 instances after deduplication (740 duplicate rows removed)
- **Target Variable**: `Result`
  - `Result = -1`: **Phishing** (Positive Class of Interest)
  - `Result = 1`: **Legitimate** (Negative Class)

---

## 3. Data Cleaning & ML Pipeline

```
phishing.csv (2,456 rows)
       │
       ▼
 Schema Check & Typo Handling (src/data_cleaning.py)
       │
       ▼
 Deduplication: 740 exact duplicate rows removed (1,716 unique rows)
       │
       ▼
 Stratified 80/20 Train/Test Split (random_state=42)
       ├── Training Set: 1,372 instances (5-fold Stratified CV)
       └── Test Set: 344 instances (Held-out, strictly unseen during training)
       │
       ▼
 Scikit-Learn Pipelines (src/train_models.py)
       ├── Decision Tree (Entropy / Information Gain)
       ├── K-Nearest Neighbors (StandardScaler fitted inside Pipeline)
       ├── Random Forest (150 trees, max_depth=15)
       └── Support Vector Machine (StandardScaler + Calibrated RBF Kernel)
       │
       ▼
 Evaluation & Artifacts (src/evaluation.py)
       ├── results/model_comparison.csv
       ├── results/confusion_matrices.png
       ├── results/roc_curves.png
       ├── results/performance_comparison.png
       ├── results/feature_importance.png
       └── results/experiment_summary.json
```

---

## 4. Experimental Results

All four classifiers were evaluated on the same independent test set of 344 instances (180 Phishing, 164 Legitimate).

| Model | Test Accuracy | Precision (Phishing) | Recall (Phishing) | F1-Score (Phishing) | ROC-AUC | Training Time (s) |
|---|---|---|---|---|---|---|
| **Random Forest** ⭐ | **0.9448** | **0.9351** | **0.9611** | **0.9479** | **0.9891** | 0.1825 s |
| **Decision Tree** | 0.9419 | 0.9396 | 0.9500 | 0.9448 | 0.9720 | 0.0042 s |
| **Support Vector Machine** | 0.9331 | 0.9337 | 0.9389 | 0.9363 | 0.9868 | 0.0915 s |
| **K-Nearest Neighbors** | 0.9128 | 0.9213 | 0.9111 | 0.9162 | 0.9766 | 0.0048 s |

*Selection Criterion*: In cybersecurity, missing a phishing site (**False Negative**) is far more damaging than flagging a legitimate site (**False Positive**). **Random Forest** achieved the highest F1-Score (**0.9479**), catching 173 of 180 phishing attacks with only 7 false negatives (Recall = **0.9611**), and achieved the highest discriminative ability (ROC-AUC = **0.9891**).

### 5-Fold Stratified Cross-Validation (on Training Data only)
- **Random Forest**: CV F1 = `0.9563 (±0.0105)` | CV Accuracy = `0.9541 (±0.0102)`
- **Support Vector Machine**: CV F1 = `0.9436 (±0.0116)` | CV Accuracy = `0.9410 (±0.0118)`
- **Decision Tree**: CV F1 = `0.9372 (±0.0178)` | CV Accuracy = `0.9351 (±0.0169)`
- **K-Nearest Neighbors**: CV F1 = `0.9216 (±0.0111)` | CV Accuracy = `0.9176 (±0.0116)`

---

## 5. Repository Structure

```
PhishGuard/
├── app.py                      # Presentation-ready Streamlit web application
├── train.py                    # Complete reproducible training and evaluation pipeline
├── requirements.txt            # Python dependencies
├── README.md                   # System documentation and demo checklist
├── .gitignore                  # Git exclusions for environment and temporary files
├── src/
│   ├── __init__.py             # Package exports
│   ├── data_cleaning.py        # Loading, schema validation, deduplication, splitting
│   ├── feature_extraction.py   # URL lexical feature extractor (10 features)
│   ├── train_models.py         # Model pipeline configurations and 5-fold CV
│   ├── evaluation.py           # Metrics calculation, plots, and JSON persistence
│   └── prediction.py           # Inference engine for URLs and full 30-feature records
├── tests/
│   ├── test_data.py            # Unit tests for loading, schema, and splitting
│   ├── test_models.py          # Unit tests for pipeline scaling, CV, and training
│   └── test_prediction.py      # Unit tests for URL heuristics and record prediction
├── data/
│   ├── README.md               # Dataset download and schema details
│   ├── phishing.csv            # Cleaned UCI dataset
│   └── demo_samples.json       # Verified held-out test set fixtures for presentation
├── models/
│   ├── decision_tree.joblib    # Fitted Decision Tree pipeline
│   ├── k-nearest_neighbors.joblib
│   ├── random_forest.joblib    # Best model pipeline
│   └── support_vector_machine.joblib
├── results/
│   ├── model_comparison.csv    # Evaluated metric table
│   ├── confusion_matrices.png  # 2x2 grid of confusion matrices
│   ├── roc_curves.png          # Discriminative ROC plots
│   ├── performance_comparison.png # F1, Precision, Recall, Accuracy bar chart
│   ├── feature_importance.png  # Top feature importances
│   ├── experiment_summary.json # Comprehensive experiment details
│   ├── dataset_info.json       # Preprocessing summary
│   └── best_model.json         # Best model metadata
└── report/
    └── report_outline.md       # 10-page academic lab report outline
```

---

## 6. Faculty Demonstration Checklist (Viva Guide)

Follow this 9-step sequence during project evaluation:

1. **Launch Dashboard**: Open [http://localhost:8501](http://localhost:8501) and explain the 4 model comparison cards and project scope.
2. **Review Dataset Hygiene**: Open **Dataset Insights**; demonstrate that the raw 2,456 instances contained 740 duplicates which were purged to prevent train/test leakage.
3. **Review Cleaning & Splitting**: Point out the stratified 80/20 split and that scaling was fitted strictly inside Pipelines (`X_train` only).
4. **Compare All Four Classifiers**: Open **Model Performance**; present the comparison table and explain why **Random Forest** was selected over KNN, SVM, and Decision Tree.
5. **Analyze the Confusion Matrix**: Explain the True Positives (173 phishing caught) vs. False Negatives (only 7 missed by Random Forest) and the business cost of false negatives.
6. **Benchmark Test Fixture (Ground Truth Legitimate)**: In **URL Analyzer & Inspector** -> **Verified Test Record Benchmarks**, select `Verified Held-out Test Sample #1 (Ground Truth: Legitimate)` and run prediction to show correct classification (`1`).
7. **Benchmark Test Fixture (Ground Truth Phishing)**: Select `Verified Held-out Test Sample #1 (Ground Truth: Phishing)` and run prediction to show correct detection (`-1`).
8. **Demonstrate URL Lexical Analysis**: In **Live URL Heuristic Analysis**, enter an address (e.g. `http://192.168.1.100/paypal-update/login.php` or `https://www.google.com`), show the 10 extracted address-bar features, and explain why full 30-feature assessment requires active web scraping.
9. **Show Code Quality & Automated Tests**: Show the terminal running `pytest -v` (58/58 tests passing).

---

## 7. Quickstart & Local Installation

### Prerequisites
- Python 3.10+ (tested on Python 3.14)
- Git

### Installation
```bash
# Clone the repository
git clone https://github.com/tejaskedarpawar/phishguard.git
cd phishguard

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Reproduce Training & Evaluation
```bash
python train.py
```

### Run Automated Tests
```bash
pytest -v
```

### Launch Web Application
```bash
streamlit run app.py
```

---

## 8. Limitations & Scope

1. **Lexical vs. Full-Page Detection**: Only 10 of the 30 features are derivable from the raw URL string without active crawling. For zero-risk offline demonstration, the live URL analyzer extracts these 10 lexical features and holds page-content features at neutral baseline.
2. **No Live Scraping**: To maintain user privacy, security, and offline reliability, the application does not visit or execute untrusted URLs.
3. **Temporal Drift**: The UCI dataset was published in 2015. Modern attacks utilizing novel homoglyphs or multi-stage redirections require retraining on newer threat intelligence feeds.

---

## 9. License

This project is licensed under the MIT License. The dataset is provided by the UCI Machine Learning Repository under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.
