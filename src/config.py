from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Reproducibility
    RANDOM_SEED: int = 42

    # Data
    N_CUSTOMERS: int = 20000
    START_DATE: str = "2021-01-01"
    END_DATE: str = "2025-12-31"

    # Paths
    SYNTHETIC_DIR: str = "data/synthetic"
    SYNTHETIC_FILENAME: str = "subscription_churn_synthetic.csv"

    PROCESSED_DIR: str = "data/processed"
    PROCESSED_FILENAME: str = "processed_dataset.csv"

    SQL_DIR: str = "sql"
    SQL_DB_FILENAME: str = "subscription_churn.db"

    MODELS_DIR: str = "models"
    REPORTS_DIR: str = "reports"
    FIGURES_DIR: str = "reports/figures"

    # Modeling
    TEST_SIZE: float = 0.20
    VALID_SIZE: float = 0.20

    # Business thresholding (example costs)
    COST_FALSE_NEGATIVE: float = 50.0  # churn missed
    COST_FALSE_POSITIVE: float = 5.0   # retention offer wasted
