import numpy as np


def find_optimal_threshold(y_true, y_proba, cost_fn: float, cost_fp: float, grid_size: int = 101) -> dict:
    """Cost-based thresholding.

    - False Negative: churned customer not flagged -> cost_fn
    - False Positive: non-churn customer flagged -> cost_fp
    """

    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)

    thresholds = np.linspace(0.0, 1.0, grid_size)
    best = {"threshold": 0.5, "expected_cost": float("inf")}

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        cost = fn * cost_fn + fp * cost_fp
        if cost < best["expected_cost"]:
            best = {"threshold": float(t), "expected_cost": float(cost), "fp": fp, "fn": fn}

    return best
