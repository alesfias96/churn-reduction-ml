# Subscription Churn Reduction — Final Report

## 1) SQL KPI Snapshot (SQLite)

### churn_rate_overall

|   churn_rate |
|-------------:|
|      0.71775 |

### churn_by_plan

| plan     |   customers |   churn_rate |   avg_fee |
|:---------|------------:|-------------:|----------:|
| Basic    |        7041 |     0.718364 |   9.17658 |
| Premium  |        3983 |     0.718052 |  16.5634  |
| Standard |        8976 |     0.717135 |  12.8507  |

### churn_by_billing_cycle

| billing_cycle   |   customers |   churn_rate |
|:----------------|------------:|-------------:|
| Monthly         |       15969 |     0.744755 |
| Annual          |        4031 |     0.610767 |

### churn_by_channel

| acquisition_channel   |   customers |   churn_rate |
|:----------------------|------------:|-------------:|
| Organic               |        5889 |     0.72542  |
| Paid Search           |        4416 |     0.724864 |
| Affiliate             |        1991 |     0.723255 |
| Email                 |        1605 |     0.719626 |
| Social                |        3707 |     0.703804 |
| Referral              |        2392 |     0.701505 |

### churn_by_tenure_bucket

| tenure_bucket   |   customers |   churn_rate |
|:----------------|------------:|-------------:|
| 24+             |       12284 |     0.715484 |
| 12-23           |        3997 |     0.715537 |
| 6-11            |        1874 |     0.72572  |
| 3-5             |         975 |     0.717949 |
| <3              |         870 |     0.742529 |

## 2) EDA Highlights

- churn_by_plan: `reports/figures/eda_churn_by_plan.png`
- engagement_vs_churn: `reports/figures/eda_engagement_vs_churn.png`

## 3) Model Performance

### Baseline (Logistic Regression)

- ROC-AUC: 0.6638
- PR-AUC: 0.8244

### XGBoost

- ROC-AUC: 0.6493
- PR-AUC: 0.8140

## 4) Cost-based Thresholding

Using costs: FN=50.0, FP=5.0

{
  "threshold": 0.265,
  "expected_cost": 5640.0,
  "fp": 1128,
  "fn": 0
}

## 5) Interpretability

Method used: **Permutation**

Notes: SHAP import failed in this runtime; using permutation importance instead.

Global importance plot saved to: `reports/figures/global_importance.png`

## 6) Business Recommendations (example)

- Prioritize retention offers for customers with *low engagement*, *high support friction*, and *payment risk*.
- Reduce support resolution time for high-ticket segments (strong churn driver).
- For discounted users, test controlled re-pricing & targeted content recommendations to prevent discount-churn rebound.
- Use the cost-optimized threshold to align the churn model with retention budget constraints.
