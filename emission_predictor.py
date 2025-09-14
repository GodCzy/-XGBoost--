"""Unified model manager for emission prediction and analysis."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

try:  # pragma: no cover - optional dependency
    import shap
except ImportError:  # pragma: no cover - optional dependency
    shap = None


@dataclass
class ModelManager:
    """Manage multiple models and provide common utilities."""

    seed: int = 42
    models: dict[str, object] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        np.random.seed(self.seed)
        self.models = {
            "rf": RandomForestRegressor(n_estimators=100, random_state=self.seed),
            "xgb": XGBRegressor(objective="reg:squarederror", random_state=self.seed),
            "mlp": MLPRegressor(
                hidden_layer_sizes=(64, 32), max_iter=500, random_state=self.seed
            ),
        }

    # ------------------------------------------------------------------
    # Training & prediction
    # ------------------------------------------------------------------
    def train(self, X, y) -> None:
        for model in self.models.values():
            model.fit(X, y)

    def predict(self, X):
        preds = np.column_stack([m.predict(X) for m in self.models.values()])
        return preds.mean(axis=1)

    # ------------------------------------------------------------------
    # Evaluation & interpretability
    # ------------------------------------------------------------------
    def evaluate(self, X, y) -> dict:
        ensemble_pred = self.predict(X)
        metrics = {
            "ensemble": {
                "r2": r2_score(y, ensemble_pred),
                "mape": mean_absolute_percentage_error(y, ensemble_pred),
            }
        }
        for name, model in self.models.items():
            preds = model.predict(X)
            metrics[name] = {
                "r2": r2_score(y, preds),
                "mape": mean_absolute_percentage_error(y, preds),
            }
        return metrics

    def shap_values(self, X, model_name: str = "xgb"):
        if shap is None:
            raise ImportError("shap is required for computing SHAP values")
        explainer = shap.TreeExplainer(self.models[model_name])
        return explainer.shap_values(X)

    def permutation_importance(self, X, y, model_name: str = "rf", n_repeats: int = 5):
        result = permutation_importance(
            self.models[model_name], X, y, n_repeats=n_repeats, random_state=self.seed
        )
        return result.importances_mean

    # ------------------------------------------------------------------
    # Versioning utilities
    # ------------------------------------------------------------------
    def save(self, version: str, path: str = "models") -> None:
        os.makedirs(path, exist_ok=True)
        for name, model in self.models.items():
            joblib.dump(model, os.path.join(path, f"{name}_{version}.joblib"))

    def load(self, version: str, path: str = "models") -> None:
        for name in self.models:
            self.models[name] = joblib.load(
                os.path.join(path, f"{name}_{version}.joblib")
            )


# Backwards compatibility -------------------------------------------------
EmissionPredictor = ModelManager
