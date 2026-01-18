import os
import json
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
)

from src.config import Config


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.5) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)

    roc_auc = float(roc_auc_score(y_test, proba))
    pr_auc = float(average_precision_score(y_test, proba))

    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()

    report = classification_report(y_test, preds, output_dict=True)

    precision, recall, th = precision_recall_curve(y_test, proba)

    return {
        "threshold": float(threshold),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "classification_report": report,
        "precision_recall_curve": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": th.tolist(),
        },
        "proba": proba.tolist(),
    }


def save_metrics(metrics: dict, cfg: Config, filename: str) -> str:
    os.makedirs(cfg.REPORTS_DIR, exist_ok=True)
    out_path = os.path.join(cfg.REPORTS_DIR, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return out_path
