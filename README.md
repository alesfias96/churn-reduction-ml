# CHURN REDUCTION ML

End-to-end churn analysis and prediction project.
Goal: identify customers at risk of churn, understand churn drivers, and provide actionable insights
to improve customer retention.

## Goals
- Create a churn dataset from raw customer/activity data
- Perform SQL analytics to build reliable features
- Train a churn prediction model
- Interpret model results (main churn drivers)
- Deliver an actionable output (risk list + key insights)

## Planned pipeline
1. Data model definition (customers, transactions, interactions)
2. SQL-based feature engineering (RFM metrics, usage patterns, tickets, etc.)
3. EDA and validation checks
4. Baseline churn model (LogReg / RandomForest / XGBoost)
5. Evaluation (F1, ROC-AUC, precision/recall tradeoff)
6. Explainability (feature importance / SHAP)
7. Final business-style report

## Tech stack
- SQL (PostgreSQL)
- Python (Pandas, NumPy)
- scikit-learn
- Jupyter Notebooks

## Project structure
```text
churn-reduction-ml/
  sql/                  # schema.sql + queries.sql
  notebooks/
  src/
  data/                 # empty (dataset not included)
  reports/
  README.md
  requirements.txt
```

## Status
Work in progress.

## Output (planned)
- A table/list of customers ranked by churn risk
- Summary of churn drivers with suggested retention actions
