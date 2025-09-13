"""Data loading and preprocessing utilities for emission prediction."""
import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess(df: pd.DataFrame):
    """Clean dataframe and return scaled features and target.

    Parameters
    ----------
    df: pd.DataFrame
        Input data containing an 'emission' column and feature columns.

    Returns
    -------
    X_scaled: ndarray
        Scaled feature matrix.
    y: Series
        Target emission values.
    scaler: StandardScaler
        Fitted scaler for transforming new data.
    """
    df = df.copy()
    df = df.fillna(df.median(numeric_only=True))
    features = df.drop(columns=["emission"])
    target = df["emission"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    return X_scaled, target, scaler
