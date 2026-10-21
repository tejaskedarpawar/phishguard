# PhishGuard — 10-Page Report Outline

## Title Page
- Title: PhishGuard: ML-Based Phishing Website Detection System
- Student name, roll number, course, institution, date

---

## Abstract (≈ 150 words)
Brief summary of problem, approach, dataset, models, key results, and conclusion.

---

## 1. Introduction (≈ 0.5–0.75 page)

### 1.1 Background
- Definition and prevalence of phishing attacks.
- Why blacklist-based approaches are insufficient.
- Role of ML in proactive phishing detection.

### 1.2 Problem Statement
- Given a set of pre-extracted URL and website features, classify whether
  a website is phishing or legitimate.

### 1.3 Objectives
1. Implement a reproducible ML pipeline on the UCI Phishing dataset.
2. Train and compare four classifiers: Decision Tree, KNN, Random Forest, SVM.
3. Evaluate using accuracy, precision, recall, F1, ROC-AUC.
4. Deploy a web-based URL analyzer.

### 1.4 Scope and Limitations
- Binary classification only.
- Trained on pre-extracted features; live tool uses URL-derivable features only.
- Educational prototype; not a production security tool.

---

## 2. Literature Review (≈ 1.5 pages)

### 2.1 Overview
Summarise previous work on phishing detection using ML.

### 2.2 Literature Review Table

| Paper Title | Authors & Year | Problem | Dataset | Models/Methods | Key Findings | Limitations | Relevance to PhishGuard |
|-------------|---------------|---------|---------|---------------|-------------|-------------|------------------------|
| _Search query: "phishing detection machine learning random forest 2020 2021 2022"_ | Pending verification | — | — | — | — | — | — |
| _Search query: "phishing URL detection deep learning 2021 2022 2023"_ | Pending verification | — | — | — | — | — | — |
| _Search query: "UCI phishing dataset classification comparison 2019 2020"_ | Pending verification | — | — | — | — | — | — |
| _Search query: "SVM decision tree phishing website detection IEEE 2020"_ | Pending verification | — | — | — | — | — | — |
| _Search query: "feature importance phishing detection ensemble methods 2021"_ | Pending verification | — | — | — | — | — | — |

> **Note**: The table above contains placeholder rows pending literature search.
> Use Google Scholar, IEEE Xplore, or ACM DL with the provided search queries.
> Do NOT fabricate paper titles, author names, DOIs, or results.

### 2.3 Identified Research Gap
Existing work rarely provides a direct four-model comparison (DT, KNN, RF, SVM)
with full reproducibility on the same UCI dataset split.  PhishGuard addresses
this gap for a focused academic demonstration.

---

## 3. Methodology (≈ 1 page)

### 3.1 System Architecture

```
Raw Dataset (phishing.csv)
         │
         ▼
  Data Loading & Validation
  (src/data_cleaning.py)
         │
         ▼
  Cleaning & EDA
  (duplicate removal, class balance check)
         │
         ▼
  Feature / Target Separation
  X (30 integer features) | y (Result: -1, 1)
         │
         ▼
  Stratified Train/Test Split (80/20)
         │
  ┌──────┴───────────────────────┐
  │                              │
  X_train                     X_test
  y_train                     y_test
  │
  ▼
  Model Pipelines (train.py)
  ┌────────┬─────┬──────────┬─────┐
  DT      KNN   RF         SVM
  │        │     │           │
  (StandardScaler applied to KNN & SVM only)
  │        │     │           │
  └────────┴─────┴───────────┘
         │
         ▼
  Evaluation on X_test / y_test
  (Acc, P, R, F1, AUC, confusion matrix, ROC)
         │
         ▼
  Best Model Selection (F1 criterion)
         │
         ▼
  Save Artifacts (models/*.joblib, results/*.*)
         │
         ▼
  Streamlit App (app.py)
  URL → Feature Extraction → Pipeline → Prediction
```

### 3.2 Experimental Setup
- Python 3.x, scikit-learn, pandas, NumPy, Matplotlib, Streamlit.
- `random_state=42` for all stochastic components.
- Stratified split ensures class proportions preserved in train and test sets.
- StandardScaler fitted on X_train only (prevents data leakage).

---

## 4. Data Cleaning and Preprocessing (≈ 1.5 pages)

### 4.1 Dataset Description
- UCI Phishing Websites Dataset (Mohammad et al., 2015).
- 11,055 instances, 30 features, 1 target column.
- All values are integer codes: −1 (phishing indicator), 0 (suspicious), 1 (legitimate).

### 4.2 Loading and Schema Validation
- CSV loaded with pandas; column names stripped of whitespace.
- Unnamed index column removed if present (common in Kaggle mirrors).
- Case-insensitive column name matching for robustness.

### 4.3 Missing Value Analysis
- Report actual results from `cleaning_summary.json` after running `train.py`.

### 4.4 Duplicate Row Analysis
- Report number of duplicate rows detected and removed.

### 4.5 Class Balance
- Report actual class distribution from `dataset_info.json`.
- Include class distribution bar chart (results/class_distribution.png).

### 4.6 Feature Encoding
- All 30 features are already integer-encoded; no one-hot encoding required.
- StandardScaler applied only for KNN and SVM inside their respective Pipelines.

### 4.7 Train/Test Split
- 80% training, 20% test; stratified by target class; `random_state=42`.
- Report actual split sizes from `dataset_info.json`.

---

## 5. Results and Discussion (≈ 2 pages)

### 5.1 Model Performance Table

> Dataset: UCI Phishing Websites (1716 instances after dedup; 80/20 stratified split; random_state=42)
> Positive class = −1 (Phishing)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Train Time (s) |
|-------|----------|-----------|--------|----------|---------|----------------|
| **Decision Tree** | 0.9477 | 0.9551 | 0.9444 | **0.9497** | 0.9666 | 0.003 |
| K-Nearest Neighbors | 0.9128 | 0.9213 | 0.9111 | 0.9162 | 0.9766 | 0.002 |
| Random Forest | 0.9448 | 0.9351 | 0.9611 | 0.9479 | 0.9891 | 0.173 |
| Support Vector Machine | 0.9331 | 0.9337 | 0.9389 | 0.9363 | 0.9868 | 0.090 |

### 5.2 Confusion Matrices
> Insert `results/confusion_matrices.png`.

### 5.3 ROC Curves
> Insert `results/roc_curves.png`.

### 5.4 F1-Score Comparison Chart
> Insert `results/f1_comparison.png`.

### 5.5 Feature Importance
> Insert `results/feature_importance_decision_tree.png` and
> `results/feature_importance_random_forest.png`.

### 5.6 Best Model Selection
- Selected: [best model from `results/best_model.json`].
- Criterion: F1-score (phishing class, −1); ROC-AUC as tiebreaker.
- Rationale: In phishing detection, recall (catching phishing) is prioritised
  over precision (reducing false alarms).  F1 balances both.

### 5.7 Discussion
- Compare model strengths and weaknesses.
- Discuss recall vs. precision trade-off in security context.
- Note any unexpected results and hypothesise causes.

---

## 6. State-of-the-Art Comparison (≈ 1 page)

### 6.1 Baseline vs. Advanced Methods

| Category | Methods | Typical Performance | Notes |
|----------|---------|--------------------|----|
| Traditional ML (this project) | DT, KNN, RF, SVM | ~95–97% accuracy on UCI dataset | Interpretable, fast to train |
| Ensemble / Boosting | XGBoost, LightGBM, AdaBoost | ~97–99% | Typically outperform single-tree models |
| Deep Learning | CNN, LSTM, BERT on URL text | ~97–99% | Requires more data; less interpretable |
| Graph-based | URL graph + GNN | ~98%+ | High complexity |

> **Clarification**: The four classifiers implemented here are baseline models used
> for comparison and educational demonstration.  They are **not** claimed to be
> state-of-the-art.

### 6.2 Why Our Approach Is Appropriate for This Context
- Directly reproducible on a student machine in minutes.
- All models and their decision boundaries are explainable.
- Dataset is standardised and widely cited.

---

## 7. UI Development and Deployment (≈ 0.75 page)

### 7.1 Application Design
- Five-page Streamlit dashboard.
- Minimalist design: Inter font, restrained blue accent, red/green for risk indicators.
- Responsive layout; works on mobile browsers.

### 7.2 URL Analysis Feature
- Features derived from URL string without visiting the target site.
- Model pipeline loaded from disk; prediction is sub-second.
- Confidence score displayed where supported (predict_proba).

### 7.3 Deployment Readiness
- Local: `pip install -r requirements.txt && python train.py && streamlit run app.py`
- Cloud: Streamlit Community Cloud (GitHub repo + requirements.txt).
- Model artifacts (*.joblib) must be committed or generated prior to deployment.

---

## 8. Conclusion and Limitations (≈ 0.5 page)

### 8.1 Conclusion
- Successfully trained four ML classifiers on the UCI Phishing Websites dataset.
- [Best model] achieved the highest F1-score of [value] on the held-out test set.
- A functional web interface enables URL analysis and result visualisation.

### 8.2 Limitations
1. Pre-extracted features: many features require live website inspection.
2. Static 2015 dataset: may not reflect current phishing tactics.
3. URL-only inference in live tool limits feature completeness.
4. Binary classification; does not distinguish phishing subtypes.
5. Fixed decision threshold at 0.5.

### 8.3 Future Work
- Fine-tune decision threshold for higher recall.
- Add live feature extraction (opt-in, with user consent).
- Retrain on a more recent or continuously updated dataset.
- Experiment with XGBoost / LightGBM / BERT-based URL classifiers.
- Add SHAP explanations for individual predictions.

---

## 9. References

> Add at least 5 verified references after completing the literature review.
> Recommended format: IEEE citation style.

[1] R. M. Mohammad, F. Thabtah, and L. McCluskey, "Phishing Websites Features,"
    UCI Machine Learning Repository, 2015.
    DOI: 10.24432/C51W2X

[2–5] [Pending literature review — see Section 2.2 search queries]

---

*End of Report Outline*
