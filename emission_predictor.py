"""Model wrapper combining RandomForest and XGBoost for emission prediction."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from xgboost import XGBRegressor

try:
    import shap
except ImportError:  # pragma: no cover - optional dependency
    shap = None


class EmissionPredictor:
    """Train and evaluate ensemble models for emission prediction."""

    def __init__(self):
        self.rf = RandomForestRegressor(n_estimators=100, random_state=42)
        self.xgb = XGBRegressor(objective="reg:squarederror", random_state=42)

    def train(self, X, y):
        self.rf.fit(X, y)
        self.xgb.fit(X, y)

    def predict(self, X):
        rf_pred = self.rf.predict(X)
        xgb_pred = self.xgb.predict(X)
        return np.mean([rf_pred, xgb_pred], axis=0)

    def evaluate(self, X, y):
        preds = self.predict(X)
        return {
            "r2": r2_score(y, preds),
            "mape": mean_absolute_percentage_error(y, preds),
        }

    def shap_values(self, X):
        """Return SHAP values for XGBoost model if shap is available."""
        if shap is None:
            raise ImportError("shap is required for computing SHAP values")
        explainer = shap.TreeExplainer(self.xgb)
        return explainer.shap_values(X)
