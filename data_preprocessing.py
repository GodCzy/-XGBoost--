"""Data loading and preprocessing utilities for emission prediction."""

from __future__ import annotations

import sqlite3
from typing import Dict, List, Tuple

import numpy as np
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
    if len(df) > 1 and not numeric.empty:
        z = ((numeric - numeric.mean()) / numeric.std(ddof=0)).abs()
        z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
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


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.replace(0, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denom
    return result.replace([np.inf, -np.inf], np.nan)


def domain_specific_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create ratio and log features that capture domain heuristics."""

    df = df.copy()
    if {"electricity", "gdp_total"}.issubset(df.columns):
        df["electricity_intensity"] = _safe_divide(df["electricity"], df["gdp_total"])
    if {"coal_consumption", "gdp_total"}.issubset(df.columns):
        df["coal_intensity"] = _safe_divide(df["coal_consumption"], df["gdp_total"])
    if {"gdp_total", "population"}.issubset(df.columns):
        df["gdp_per_capita"] = _safe_divide(df["gdp_total"], df["population"])
    if {"electricity", "population"}.issubset(df.columns):
        df["electricity_per_capita"] = _safe_divide(df["electricity"], df["population"])
    if "gdp_total" in df.columns:
        df["log_gdp_total"] = np.log1p(df["gdp_total"].clip(lower=0))
    if "electricity" in df.columns:
        df["log_electricity"] = np.log1p(df["electricity"].clip(lower=0))
    return df


def _high_correlation_pairs(
    df: pd.DataFrame, threshold: float = 0.9
) -> List[Dict[str, float]]:
    numeric = df.select_dtypes("number")
    if numeric.empty:
        return []
    corr = numeric.corr().abs()
    pairs: List[Dict[str, float]] = []
    cols = corr.columns
    for i, col_a in enumerate(cols[:-1]):
        for j in range(i + 1, len(cols)):
            value = corr.iloc[i, j]
            if value > threshold:
                pairs.append(
                    {
                        "feature_a": col_a,
                        "feature_b": cols[j],
                        "correlation": float(value),
                    }
                )
    pairs.sort(key=lambda item: item["correlation"], reverse=True)
    return pairs


def reduce_multicollinearity(df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """Drop highly correlated columns to stabilise downstream models."""

    numeric = df.select_dtypes("number")
    if numeric.empty:
        return df
    corr = numeric.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop: List[str] = [
        column for column in upper.columns if any(upper[column] > threshold)
    ]
    return df.drop(columns=to_drop, errors="ignore")


def _feature_summary(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    numeric = df.select_dtypes("number")
    if numeric.empty:
        return summary
    desc = numeric.describe().transpose()
    for feature, stats in desc.iterrows():
        summary[feature] = {
            "mean": float(stats["mean"]),
            "std": float(stats["std"]),
            "min": float(stats["min"]),
            "max": float(stats["max"]),
        }
    return summary


def build_feature_frame(df: pd.DataFrame, drop_target: bool = True) -> pd.DataFrame:
    """Apply feature engineering pipeline without scaling."""

    engineered = domain_specific_features(df)
    engineered = engineer_features(engineered)
    if drop_target:
        return engineered.drop(columns=["emission"], errors="ignore")
    return engineered


def preprocess(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, StandardScaler, Dict[str, Dict[str, float]]]:
    """Clean dataframe, engineer features and return scaled matrix, target and report."""

    df = standardize_columns(df)
    df = _coerce_numeric_like_columns(df)
    report = quality_report(df)
    df = clean_data(df)
    df = domain_specific_features(df)
    df = engineer_features(df)
    features_raw = df.drop(columns=["emission"])
    report["high_correlation_pairs"] = _high_correlation_pairs(features_raw)
    features = reduce_multicollinearity(features_raw)
    target = df["emission"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    report["retained_features"] = list(features.columns)
    report["feature_summary"] = _feature_summary(features)
    return X_scaled, target, scaler, report
