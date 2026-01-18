import pandas as pd


class DataValidationError(Exception):
    pass


def validate_schema(df: pd.DataFrame) -> None:
    required_cols = {
        "customer_id",
        "signup_date",
        "age",
        "country",
        "device",
        "acquisition_channel",
        "plan",
        "billing_cycle",
        "payment_method",
        "base_price",
        "total_discount_pct",
        "monthly_fee",
        "late_payments_6m",
        "usage_days_30",
        "avg_session_minutes",
        "content_diversity",
        "engagement_trend",
        "support_tickets_6m",
        "avg_ticket_resolution_hours",
        "had_streaming_issues",
        "refund_requests_12m",
        "tenure_months",
        "churned",
    }

    missing = required_cols - set(df.columns)
    if missing:
        raise DataValidationError(f"Missing required columns: {sorted(missing)}")


def validate_ranges(df: pd.DataFrame) -> None:
    if df["age"].min() < 16 or df["age"].max() > 100:
        raise DataValidationError("Age out of expected range")

    if df["usage_days_30"].min() < 0 or df["usage_days_30"].max() > 31:
        raise DataValidationError("usage_days_30 out of expected range")

    if df["monthly_fee"].min() <= 0:
        raise DataValidationError("monthly_fee must be > 0")

    if not set(df["churned"].unique()).issubset({0, 1}):
        raise DataValidationError("churned must be binary 0/1")


def validate_missingness(
    df: pd.DataFrame,
    max_missing_pct: float = 0.02,
    allowed_missing_cols: set[str] | None = None,
) -> None:
    """Check missing values.

    Some columns are *expected* to have missing values (e.g., churn month for non-churners).
    """
    if allowed_missing_cols is None:
        allowed_missing_cols = {"churned_month"}

    missing_pct = df.isna().mean()
    # ignore allowed missing columns
    missing_pct = missing_pct.drop(labels=[c for c in allowed_missing_cols if c in missing_pct.index], errors="ignore")

    too_missing = missing_pct[missing_pct > max_missing_pct]
    if len(too_missing) > 0:
        cols = {k: float(v) for k, v in too_missing.to_dict().items()}
        raise DataValidationError(f"Too many missing values: {cols}")


def run_all_validations(df: pd.DataFrame) -> None:
    validate_schema(df)
    validate_ranges(df)
    validate_missingness(df)
