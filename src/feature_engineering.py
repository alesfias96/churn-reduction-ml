import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Tenure buckets
    df["tenure_bucket"] = pd.cut(
        df["tenure_months"],
        bins=[0, 2, 5, 11, 23, 10_000],
        labels=["<3", "3-5", "6-11", "12-23", "24+"],
        include_lowest=True,
    ).astype(str)

    # Engagement proxy
    df["engagement_index"] = (
        (df["usage_days_30"] / 30.0) * 0.50
        + (df["avg_session_minutes"] / 180.0) * 0.30
        + (df["content_diversity"] / 15.0) * 0.20
    ).clip(0, 1)

    # Support friction
    df["support_friction"] = (
        (df["support_tickets_6m"] / 8.0) * 0.6
        + (df["avg_ticket_resolution_hours"] / 72.0) * 0.4
    ).clip(0, 1)

    # Payment risk
    df["payment_risk"] = (df["late_payments_6m"] / 3.0).clip(0, 1)

    # Price sensitivity
    df["discount_intensity"] = df["total_discount_pct"].clip(0, 1)

    # Simple interactions
    df["low_engagement_flag"] = (df["engagement_index"] < 0.35).astype(int)
    df["high_support_flag"] = (df["support_tickets_6m"] >= 2).astype(int)

    # Log transforms (safe)
    df["log_monthly_fee"] = np.log1p(df["monthly_fee"])
    df["log_tenure"] = np.log1p(df["tenure_months"])

    return df
