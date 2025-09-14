"""Data loading and preprocessing utilities for emission prediction."""

from __future__ import annotations

import sqlite3
from typing import Dict, Tuple

import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_csv(path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    return pd.read_csv(path)


def load_sqlite(db_path: str, table: str) -> pd.DataFrame:
    """Load data from a SQLite database."""
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql(f"SELECT * FROM {table}", conn)


def quality_report(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Return basic data quality metrics."""
    return {
        "missing_rate": df.isna().mean().to_dict(),
        "n_rows": len(df),
    }


def clean_data(df: pd.DataFrame, z_thresh: float = 3.0) -> pd.DataFrame:
    """Fill missing values and remove outliers based on z-score."""
    df = df.copy()
    df = df.fillna(df.median(numeric_only=True))
    numeric = df.select_dtypes("number")
    z = ((numeric - numeric.mean()) / numeric.std(ddof=0)).abs()
    df = df[(z <= z_thresh).all(axis=1)]
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple polynomial features for numeric columns."""
    df = df.copy()
    numeric_cols = df.select_dtypes("number").columns
    for col in numeric_cols:
        if col != "emission":
            df[f"{col}_sq"] = df[col] ** 2
    return df


def preprocess(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, StandardScaler, Dict[str, Dict[str, float]]]:
    """Clean dataframe, engineer features and return scaled matrix, target and report."""
    report = quality_report(df)
    df = clean_data(df)
    df = engineer_features(df)
    features = df.drop(columns=["emission"])
    target = df["emission"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    return X_scaled, target, scaler, report
