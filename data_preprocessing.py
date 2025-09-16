"""Data loading and preprocessing utilities for emission prediction."""

from __future__ import annotations

import sqlite3
from typing import Dict, Tuple

import pandas as pd
from sklearn.preprocessing import StandardScaler


COLUMN_ALIASES = {
    "全社会用电量(亿千瓦时)": "electricity",
    "常住人口(万人)": "population",
    "GDP总量(亿元)": "gdp_total",
    "城镇化率(%)": "urbanization_rate",
    "粗钢产量(万吨)": "steel_production",
    "煤炭使用量(万吨)": "coal_consumption",
    "碳排放()": "emission",
}


def load_csv(path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    return pd.read_csv(path)


def load_sqlite(db_path: str, table: str) -> pd.DataFrame:
    """Load data from a SQLite database."""
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql(f"SELECT * FROM {table}", conn)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known domain-specific columns to internal names."""

    rename_map = {
        col: alias for col, alias in COLUMN_ALIASES.items() if col in df.columns
    }
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _coerce_numeric_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert object columns that mostly contain numbers into numeric dtype."""

    obj_cols = df.select_dtypes(include="object").columns
    for col in obj_cols:
        cleaned = df[col].astype(str).str.replace(r"[^\d.\-eE+]", "", regex=True)
        numeric = pd.to_numeric(cleaned, errors="coerce")
        if numeric.notna().mean() > 0.5:
            df[col] = numeric
    return df


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
    if len(df) > 1:
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

    df = standardize_columns(df)
    df = _coerce_numeric_like_columns(df)
    report = quality_report(df)
    df = clean_data(df)
    df = engineer_features(df)
    features = df.drop(columns=["emission"])
    target = df["emission"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    return X_scaled, target, scaler, report
