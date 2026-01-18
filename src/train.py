import os
import joblib
import numpy as np
import pandas as pd

from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.config import Config


@dataclass
class TrainArtifacts:
    baseline_pipeline: Pipeline
    xgb_pipeline: Pipeline
    X_test: pd.DataFrame
    y_test: pd.Series
    feature_columns: list


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    return preprocessor


def train_models(df: pd.DataFrame, cfg: Config) -> TrainArtifacts:
    df = df.copy()

    y = df["churned"].astype(int)
    X = df.drop(columns=["churned"])

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=cfg.TEST_SIZE,
        random_state=cfg.RANDOM_SEED,
        stratify=y,
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_trainval,
        y_trainval,
        test_size=cfg.VALID_SIZE,
        random_state=cfg.RANDOM_SEED,
        stratify=y_trainval,
    )

    preprocessor = build_preprocessor(X_train)

    baseline = LogisticRegression(max_iter=2000, n_jobs=None)
    baseline_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", baseline)])

    xgb = XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=cfg.RANDOM_SEED,
        n_jobs=4,
        eval_metric="logloss",
    )

    xgb_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", xgb)])

    # Fit baseline
    baseline_pipeline.fit(X_train, y_train)

    # Fit XGB (no early stopping to keep compatibility across XGBoost versions)
    xgb_pipeline.fit(X_train, y_train)

    return TrainArtifacts(
        baseline_pipeline=baseline_pipeline,
        xgb_pipeline=xgb_pipeline,
        X_test=X_test,
        y_test=y_test,
        feature_columns=X.columns.tolist(),
    )


def save_models(artifacts: TrainArtifacts, cfg: Config) -> dict:
    os.makedirs(cfg.MODELS_DIR, exist_ok=True)

    baseline_path = os.path.join(cfg.MODELS_DIR, "baseline_logreg.joblib")
    xgb_path = os.path.join(cfg.MODELS_DIR, "xgb_model.joblib")

    joblib.dump(artifacts.baseline_pipeline, baseline_path)
    joblib.dump(artifacts.xgb_pipeline, xgb_path)

    return {"baseline": baseline_path, "xgb": xgb_path}
