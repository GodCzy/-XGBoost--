"""Demonstration of the minimal emission prediction and optimization framework."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data_preprocessing import preprocess
from emission_predictor import EmissionPredictor
from monitoring import ProcessMonitor
from optimization import pso


def generate_synthetic_data(n: int = 200):
    rng = np.random.default_rng(42)
    data = pd.DataFrame(
        {
            "electricity": rng.normal(100, 10, n),
            "gdp": rng.normal(50, 5, n),
            "coal": rng.normal(30, 3, n),
        }
    )
    data["emission"] = (
        0.3 * data["electricity"]
        + 0.5 * data["gdp"]
        + 0.2 * data["coal"]
        + rng.normal(0, 2, n)
    )
    return data


def emission_objective(params):
    # Placeholder objective: linear combination of parameters
    temp, pressure = params
    return 0.5 * temp + 0.3 * pressure


def main():
    data = generate_synthetic_data()
    X, y, _, report = preprocess(data)
    print("Data quality report:", report)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    predictor = EmissionPredictor()
    predictor.train(X_train, y_train)
    metrics = predictor.evaluate(X_test, y_test)
    print("Evaluation metrics:", metrics)

    try:
        shap_vals = predictor.shap_values(X_train[:10])
        print("Computed SHAP values with shape", shap_vals.shape)
    except Exception as exc:  # pragma: no cover - shap optional
        print("SHAP computation failed:", exc)

    monitor = ProcessMonitor(
        threshold=50,
        optimizer=pso,
        objective=emission_objective,
        bounds=[(300, 1000), (1, 10)],
    )
    params, val = monitor.adjust(current_emission=60)
    print("Optimization result:", params, val)


if __name__ == "__main__":
    main()
