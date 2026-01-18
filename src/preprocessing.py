import pandas as pd

# Columns that would leak target information (or are too target-adjacent)
LEAKAGE_COLUMNS = [
    "churn_probability_true",  # synthetic generation helper
    "churned_month",  # target-adjacent proxy (only populated for churners)
]


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning, type normalization, and leakage prevention."""
    df = df.copy()

    # Parse dates
    df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")

    # Date features (keep them but drop raw datetime to avoid mixed dtypes in ML)
    df["signup_year"] = df["signup_date"].dt.year
    df["signup_month"] = df["signup_date"].dt.month
    df["signup_dayofweek"] = df["signup_date"].dt.dayofweek
    df = df.drop(columns=["signup_date"])

    # Drop leakage columns if present
    for col in LEAKAGE_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Ensure correct dtypes
    df["churned"] = df["churned"].astype(int)

    return df
