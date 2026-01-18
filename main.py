import os
import json
import pandas as pd
import matplotlib.pyplot as plt

from src.config import Config
from src.data_generation import generate_synthetic_subscription_churn, save_dataset
from src.data_validation import run_all_validations
from src.sql_pipeline import create_sqlite_db_from_csv, run_sql_queries
from src.preprocessing import basic_cleaning
from src.feature_engineering import add_features
from src.train import train_models, save_models
from src.evaluate import evaluate_model, save_metrics
from src.thresholding import find_optimal_threshold
from src.explain import compute_global_importance


def ensure_dirs(cfg: Config) -> None:
    os.makedirs(cfg.SYNTHETIC_DIR, exist_ok=True)
    os.makedirs(cfg.PROCESSED_DIR, exist_ok=True)
    os.makedirs(cfg.SQL_DIR, exist_ok=True)
    os.makedirs(cfg.MODELS_DIR, exist_ok=True)
    os.makedirs(cfg.REPORTS_DIR, exist_ok=True)
    os.makedirs(cfg.FIGURES_DIR, exist_ok=True)


def plot_basic_eda(df: pd.DataFrame, cfg: Config) -> dict:
    """Generate a few EDA plots and return their paths."""

    paths = {}

    # Churn rate by plan
    grp = df.groupby("plan")["churned"].mean().sort_values(ascending=False)
    plt.figure()
    grp.plot(kind="bar")
    plt.title("Churn rate by plan")
    plt.ylabel("Churn rate")
    out1 = os.path.join(cfg.FIGURES_DIR, "eda_churn_by_plan.png")
    plt.tight_layout()
    plt.savefig(out1, dpi=160)
    plt.close()
    paths["churn_by_plan"] = out1

    # Churn vs engagement index
    plt.figure()
    df.boxplot(column="engagement_index", by="churned")
    plt.title("Engagement index vs churn")
    plt.suptitle("")
    plt.xlabel("Churned")
    out2 = os.path.join(cfg.FIGURES_DIR, "eda_engagement_vs_churn.png")
    plt.tight_layout()
    plt.savefig(out2, dpi=160)
    plt.close()
    paths["engagement_vs_churn"] = out2

    return paths


def write_report(
    cfg: Config,
    sql_results: dict,
    metrics_baseline: dict,
    metrics_xgb: dict,
    threshold_info: dict,
    explain_path: str,
    explain_method: str,
    explain_notes: str,
    eda_paths: dict,
) -> str:
    report_path = os.path.join(cfg.REPORTS_DIR, "churn_report.md")

    def df_to_md(d: pd.DataFrame, max_rows: int = 10) -> str:
        return d.head(max_rows).to_markdown(index=False)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Subscription Churn Reduction — Final Report\n\n")

        f.write("## 1) SQL KPI Snapshot (SQLite)\n\n")
        for name, df_res in sql_results.items():
            f.write(f"### {name}\n\n")
            f.write(df_to_md(df_res))
            f.write("\n\n")

        f.write("## 2) EDA Highlights\n\n")
        for k, p in eda_paths.items():
            f.write(f"- {k}: `{p}`\n")
        f.write("\n")

        f.write("## 3) Model Performance\n\n")
        f.write("### Baseline (Logistic Regression)\n\n")
        f.write(f"- ROC-AUC: {metrics_baseline['roc_auc']:.4f}\n")
        f.write(f"- PR-AUC: {metrics_baseline['pr_auc']:.4f}\n\n")

        f.write("### XGBoost\n\n")
        f.write(f"- ROC-AUC: {metrics_xgb['roc_auc']:.4f}\n")
        f.write(f"- PR-AUC: {metrics_xgb['pr_auc']:.4f}\n\n")

        f.write("## 4) Cost-based Thresholding\n\n")
        f.write(f"Using costs: FN={cfg.COST_FALSE_NEGATIVE}, FP={cfg.COST_FALSE_POSITIVE}\n\n")
        f.write(json.dumps(threshold_info, indent=2))
        f.write("\n\n")

        f.write("## 5) Interpretability\n\n")
        f.write(f"Method used: **{explain_method}**\n\n")
        if explain_notes:
            f.write(f"Notes: {explain_notes}\n\n")
        f.write(f"Global importance plot saved to: `{explain_path}`\n\n")

        f.write("## 6) Business Recommendations (example)\n\n")
        f.write("- Prioritize retention offers for customers with *low engagement*, *high support friction*, and *payment risk*.\n")
        f.write("- Reduce support resolution time for high-ticket segments (strong churn driver).\n")
        f.write("- For discounted users, test controlled re-pricing & targeted content recommendations to prevent discount-churn rebound.\n")
        f.write("- Use the cost-optimized threshold to align the churn model with retention budget constraints.\n")

    return report_path


def main():
    cfg = Config()
    ensure_dirs(cfg)

    # PHASE 1: data generation
    df_raw = generate_synthetic_subscription_churn(cfg)
    raw_path = save_dataset(df_raw, cfg)

    # PHASE 3: validation
    run_all_validations(df_raw)

    # PHASE 2: SQL pipeline
    db_path = create_sqlite_db_from_csv(raw_path, cfg)
    sql_file = os.path.join(cfg.SQL_DIR, "sql_queries.sql")
    sql_results = run_sql_queries(db_path, sql_file)

    # PHASE 3/5: preprocessing + feature engineering
    df = basic_cleaning(df_raw)
    df = add_features(df)

    # Save processed dataset
    processed_path = os.path.join(cfg.PROCESSED_DIR, cfg.PROCESSED_FILENAME)
    df.to_csv(processed_path, index=False)

    # PHASE 4: EDA plots
    eda_paths = plot_basic_eda(df, cfg)

    # PHASE 6: train models
    artifacts = train_models(df, cfg)
    model_paths = save_models(artifacts, cfg)

    # PHASE 6: evaluate models at default threshold
    metrics_baseline = evaluate_model(artifacts.baseline_pipeline, artifacts.X_test, artifacts.y_test, threshold=0.5)
    metrics_xgb_default = evaluate_model(artifacts.xgb_pipeline, artifacts.X_test, artifacts.y_test, threshold=0.5)

    baseline_metrics_path = save_metrics(metrics_baseline, cfg, "metrics_baseline.json")
    xgb_metrics_path = save_metrics(metrics_xgb_default, cfg, "metrics_xgb_default_threshold.json")

    # PHASE 7: threshold optimization (cost-based)
    y_proba = pd.Series(metrics_xgb_default["proba"])  # stored list
    threshold_info = find_optimal_threshold(
        y_true=artifacts.y_test.values,
        y_proba=y_proba.values,
        cost_fn=cfg.COST_FALSE_NEGATIVE,
        cost_fp=cfg.COST_FALSE_POSITIVE,
        grid_size=201,
    )

    # Evaluate again at optimal threshold
    metrics_xgb_opt = evaluate_model(
        artifacts.xgb_pipeline,
        artifacts.X_test,
        artifacts.y_test,
        threshold=threshold_info["threshold"],
    )
    xgb_opt_metrics_path = save_metrics(metrics_xgb_opt, cfg, "metrics_xgb_optimal_threshold.json")

    # PHASE 8: Interpretability (best-effort SHAP, fallback to permutation)
    explain_plot_path = os.path.join(cfg.FIGURES_DIR, "global_importance.png")
    explain_res = compute_global_importance(
        model=artifacts.xgb_pipeline,
        X=artifacts.X_test,
        y=artifacts.y_test,
        out_path=explain_plot_path,
        max_features=20,
        random_state=cfg.RANDOM_SEED,
    )

    # PHASE 9: report
    report_path = write_report(
        cfg,
        sql_results=sql_results,
        metrics_baseline=metrics_baseline,
        metrics_xgb=metrics_xgb_opt,
        threshold_info=threshold_info,
        explain_path=explain_plot_path,
        explain_method=explain_res.method,
        explain_notes=explain_res.notes,
        eda_paths=eda_paths,
    )

    print("\n=== PIPELINE COMPLETED ===")
    print(f"Synthetic CSV: {raw_path}")
    print(f"SQLite DB:     {db_path}")
    print(f"Processed:     {processed_path}")
    print(f"Models:        {model_paths}")
    print(f"Metrics:       {baseline_metrics_path}, {xgb_metrics_path}, {xgb_opt_metrics_path}")
    print(f"Interpret.:    {explain_plot_path} ({explain_res.method})")
    print(f"Report:        {report_path}")


if __name__ == "__main__":
    main()
