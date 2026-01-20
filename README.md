# Subscription Churn Reduction (End-to-End Data Science)

An end-to-end, **international** Data Science project that predicts **subscription churn** and turns model outputs into **business actions**.

This repository uses a **realistic synthetic dataset** (no sensitive data) and demonstrates the full workflow:

- Synthetic data generation
- SQL analytics (SQLite)
- Data validation & cleaning
- EDA & KPI analysis
- Feature engineering
- Machine Learning (baseline + boosted model)
- Threshold optimization (cost-based decision)
- Model interpretability (SHAP)
- Final report with business recommendations

## Quickstart

### 1) Create environment

```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\\Scripts\\activate)
pip install -r requirements.txt
```

### 2) Run the full pipeline

```bash
python main.py
```

Artifacts will be saved to:
- `data/synthetic/subscription_churn_synthetic.csv`
- `data/processed/processed_dataset.csv`
- `sql/subscription_churn.db`
- `models/` (trained models)
- `reports/` (figures + report)

## Project structure

```
churn-reduction-ml/
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
├── models/
├── notebooks/
├── reports/
│   └── figures/
├── sql/
├── src/
│   ├── config.py
│   ├── data_generation.py
│   ├── data_validation.py
│   ├── sql_pipeline.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   ├── thresholding.py
│   └── explain.py
├── requirements.txt
├── main.py
└── LICENSE
```

## Notes

- This dataset is synthetic but engineered to mimic real churn drivers (engagement decline, support friction, payment issues, refund behavior, pricing sensitivity).
- The pipeline avoids data leakage and uses proper train/validation/test evaluation.

