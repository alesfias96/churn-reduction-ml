# src/explain.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance


@dataclass
class ExplainabilityResult:
    method: str
    global_importance: pd.DataFrame
    notes: str = ""


def _try_import_shap() -> Optional[Any]:
    """Try importing SHAP.

    In some environments, SHAP can fail due to binary/runtime constraints (e.g., numba/coverage
    incompatibilities). We treat SHAP as a best-effort optional enhancement.
    """
    try:
        import shap  # type: ignore
        return shap
    except Exception as e:  # noqa: BLE001
        return None


def compute_global_importance(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    out_path: str,
    max_features: int = 20,
    random_state: int = 42,
) -> ExplainabilityResult:
    """Compute global feature importance.

    Preference order:
      1) SHAP TreeExplainer (if SHAP imports cleanly)
      2) Permutation importance (always works, model-agnostic)

    Saves a bar plot to out_path.
    """

    shap = _try_import_shap()

    if shap is not None:
        try:
            # TreeExplainer works for tree-based models (XGBoost/LightGBM/Sklearn trees)
            explainer = shap.TreeExplainer(model)
            # Sample to keep runtime reasonable
            sample = X.sample(min(len(X), 2000), random_state=random_state)
            shap_values = explainer.shap_values(sample)

            # shap_values can be list for multiclass; for binary it can be array
            if isinstance(shap_values, list):
                shap_vals = np.array(shap_values[1])
            else:
                shap_vals = np.array(shap_values)

            mean_abs = np.abs(shap_vals).mean(axis=0)
            imp = (
                pd.DataFrame({"feature": sample.columns, "importance": mean_abs})
                .sort_values("importance", ascending=False)
                .head(max_features)
                .reset_index(drop=True)
            )

            _plot_importance(imp, out_path, title="Global Feature Importance (SHAP | mean(|value|))")
            return ExplainabilityResult(method="SHAP", global_importance=imp)

        except Exception as e:  # noqa: BLE001
            # fall through to permutation importance
            notes = f"SHAP available but failed at runtime: {type(e).__name__}: {e}. Falling back to permutation importance."

    else:
        notes = "SHAP import failed in this runtime; using permutation importance instead."

    # Permutation importance (model-agnostic)
    r = permutation_importance(model, X, y, n_repeats=8, random_state=random_state, scoring="roc_auc")
    imp = (
        pd.DataFrame({"feature": X.columns, "importance": r.importances_mean})
        .sort_values("importance", ascending=False)
        .head(max_features)
        .reset_index(drop=True)
    )
    _plot_importance(imp, out_path, title="Global Feature Importance (Permutation | ROC-AUC drop)")
    return ExplainabilityResult(method="Permutation", global_importance=imp, notes=notes)


def _plot_importance(imp: pd.DataFrame, out_path: str, title: str) -> None:
    plt.figure(figsize=(9, 6))
    plt.barh(imp["feature"][::-1], imp["importance"][::-1])
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
