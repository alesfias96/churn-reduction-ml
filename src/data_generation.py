import os
import numpy as np
import pandas as pd

from src.config import Config


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_synthetic_subscription_churn(cfg: Config) -> pd.DataFrame:
    """Generate a realistic synthetic dataset for subscription churn modeling."""

    rng = np.random.default_rng(cfg.RANDOM_SEED)
    n = cfg.N_CUSTOMERS

    customer_id = np.arange(1, n + 1)

    signup_dates = pd.to_datetime(
        rng.choice(
            pd.date_range(cfg.START_DATE, cfg.END_DATE, freq="D"),
            size=n,
            replace=True,
        )
    )

    age = rng.integers(18, 70, size=n)
    country = rng.choice(
        ["IT", "UK", "DE", "FR", "ES", "US", "CA", "NL", "SE"],
        size=n,
        p=[0.14, 0.11, 0.11, 0.10, 0.10, 0.18, 0.08, 0.10, 0.08],
    )

    device = rng.choice(["Mobile", "Desktop", "Tablet", "SmartTV"], size=n, p=[0.45, 0.25, 0.10, 0.20])
    acquisition_channel = rng.choice(
        ["Organic", "Paid Search", "Social", "Referral", "Affiliate", "Email"],
        size=n,
        p=[0.30, 0.22, 0.18, 0.12, 0.10, 0.08],
    )

    # Subscription & billing
    plan = rng.choice(["Basic", "Standard", "Premium"], size=n, p=[0.35, 0.45, 0.20])
    base_price = np.select([plan == "Basic", plan == "Standard", plan == "Premium"], [9.99, 13.99, 17.99])

    billing_cycle = rng.choice(["Monthly", "Annual"], size=n, p=[0.80, 0.20])
    annual_discount = np.where(billing_cycle == "Annual", rng.uniform(0.10, 0.25, size=n), 0.0)

    promo_discount = rng.choice([0.0, 0.05, 0.10, 0.15, 0.20], size=n, p=[0.60, 0.12, 0.12, 0.10, 0.06])
    total_discount = np.clip(promo_discount + annual_discount, 0, 0.40)

    monthly_fee = base_price * (1 - total_discount)

    payment_method = rng.choice(
        ["Credit Card", "Debit Card", "PayPal", "Apple Pay", "Google Pay", "Prepaid"],
        size=n,
        p=[0.36, 0.18, 0.20, 0.10, 0.10, 0.06],
    )

    late_payment_prob = np.select(
        [payment_method == "Prepaid", payment_method == "Debit Card", payment_method == "Credit Card"],
        [0.18, 0.10, 0.06],
        default=0.08,
    )
    late_payments_6m = rng.binomial(n=3, p=late_payment_prob, size=n)

    # Usage behavior
    usage_days_30 = np.clip(rng.normal(loc=15, scale=8, size=n), 0, 30).round().astype(int)
    avg_session_minutes = np.clip(rng.normal(loc=45, scale=20, size=n), 5, 180)
    content_diversity = np.clip(rng.normal(loc=6, scale=3, size=n), 1, 15).round().astype(int)
    engagement_trend = np.clip(rng.normal(loc=0.0, scale=1.0, size=n), -2.5, 2.5)

    # Support / issues
    support_tickets_6m = np.clip(rng.poisson(lam=0.6, size=n), 0, 8)
    avg_ticket_resolution_hours = np.clip(rng.normal(loc=18, scale=10, size=n), 1, 72)
    had_streaming_issues = rng.binomial(n=1, p=0.18, size=n)
    refund_requests_12m = np.clip(rng.poisson(lam=0.25, size=n), 0, 4)

    # Tenure
    today = pd.to_datetime(cfg.END_DATE)
    tenure_days = np.clip(np.asarray((today - signup_dates).days, dtype=float), 1, None)
    tenure_months = np.clip(np.rint(tenure_days / 30.44).astype(int), 1, None)

    # Churn label (probabilistic)
    engagement_score = (
        (usage_days_30 / 30) * 0.45
        + (avg_session_minutes / 180) * 0.25
        + (content_diversity / 15) * 0.15
        + ((engagement_trend + 2.5) / 5.0) * 0.15
    )

    price_sensitivity = total_discount
    ticket_risk = support_tickets_6m / 8
    resolution_risk = avg_ticket_resolution_hours / 72
    late_risk = late_payments_6m / 3
    refund_risk = refund_requests_12m / 4

    is_monthly = (billing_cycle == "Monthly").astype(int)

    churn_logit = (
        -2.2
        + 3.0 * (1 - engagement_score)
        + 1.1 * (ticket_risk + resolution_risk)
        + 1.2 * late_risk
        + 1.6 * refund_risk
        + 0.9 * is_monthly
        + 0.8 * had_streaming_issues
        + 1.0 * price_sensitivity
        + rng.normal(0, 0.35, size=n)
    )

    churn_probability_true = sigmoid(churn_logit)
    churned = rng.binomial(n=1, p=churn_probability_true, size=n)
    churned_month = np.where(churned == 1, rng.integers(1, 13, size=n), np.nan)

    df = pd.DataFrame(
        {
            "customer_id": customer_id,
            "signup_date": signup_dates,
            "age": age,
            "country": country,
            "device": device,
            "acquisition_channel": acquisition_channel,
            "plan": plan,
            "billing_cycle": billing_cycle,
            "payment_method": payment_method,
            "base_price": base_price,
            "total_discount_pct": total_discount,
            "monthly_fee": monthly_fee.round(2),
            "late_payments_6m": late_payments_6m,
            "usage_days_30": usage_days_30,
            "avg_session_minutes": avg_session_minutes.round(1),
            "content_diversity": content_diversity,
            "engagement_trend": np.round(engagement_trend, 2),
            "support_tickets_6m": support_tickets_6m,
            "avg_ticket_resolution_hours": avg_ticket_resolution_hours.round(1),
            "had_streaming_issues": had_streaming_issues,
            "refund_requests_12m": refund_requests_12m,
            "tenure_months": tenure_months,
            # For validation only (must be dropped from modeling to avoid leakage)
            "churn_probability_true": np.round(churn_probability_true, 4),
            "churned": churned,
            "churned_month": churned_month,
        }
    )

    return df


def save_dataset(df: pd.DataFrame, cfg: Config) -> str:
    os.makedirs(cfg.SYNTHETIC_DIR, exist_ok=True)
    out_path = os.path.join(cfg.SYNTHETIC_DIR, cfg.SYNTHETIC_FILENAME)
    df.to_csv(out_path, index=False)
    return out_path
