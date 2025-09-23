"""Unified model manager for emission prediction and analysis."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict

import joblib
import numpy as np
from scipy.stats import t
from sklearn.base import clone
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import RidgeCV
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
    ensemble_weights_: Dict[str, np.ndarray] = field(default_factory=dict, init=False)
    meta_model: StackingRegressor | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        np.random.seed(self.seed)
        self.models = {
            "rf": RandomForestRegressor(n_estimators=100, random_state=self.seed),
            "xgb": XGBRegressor(objective="reg:squarederror", random_state=self.seed),
            "mlp": MLPRegressor(
                hidden_layer_sizes=(64, 32), max_iter=500, random_state=self.seed
            ),
            "gbr": GradientBoostingRegressor(random_state=self.seed),
            "hgb": HistGradientBoostingRegressor(random_state=self.seed),
        }
        self.ensemble_weights_ = {}
        self.meta_model = None

    # ------------------------------------------------------------------
    # Training & prediction
    # ------------------------------------------------------------------
    def train(self, X, y, validation_fraction: float = 0.2) -> None:
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)

        if 0 < validation_fraction < 1 and len(X_arr) > 1:
            X_train, X_val, y_train, y_val = train_test_split(
                X_arr, y_arr, test_size=validation_fraction, random_state=self.seed
            )
        else:
            X_train, y_train = X_arr, y_arr
            X_val = y_val = None

        for model in self.models.values():
            model.fit(X_train, y_train)

        self._compute_self_adaptive_weights(X_val, y_val)

        for model in self.models.values():
            model.fit(X_arr, y_arr)

        self._compute_residual_weights(X_arr, y_arr)
        self._train_stacking_model(X_arr, y_arr)

    def _base_predictions(self, X) -> Dict[str, np.ndarray]:
        return {name: model.predict(X) for name, model in self.models.items()}

    def predict(self, X, strategy: str = "mean"):
        preds = self._base_predictions(X)

        if strategy == "mean":
            return np.column_stack(list(preds.values())).mean(axis=1)

        if strategy in {"equal", "residual", "self_adaption"}:
            weights = self.ensemble_weights_.get(strategy)
            if weights is None:
                weights = np.array([0.5, 0.5])
            base = np.column_stack([preds["xgb"], preds["rf"]])
            return base @ weights

        if strategy == "stacking":
            if self.meta_model is None:
                raise ValueError("Stacking model is not available before training")
            return self.meta_model.predict(X)

        if strategy in preds:
            return preds[strategy]

        raise ValueError(f"Unknown strategy '{strategy}'")

    # ------------------------------------------------------------------
    # Evaluation & interpretability
    # ------------------------------------------------------------------
    def evaluate(self, X, y) -> dict:
        results = {
            "base_models": {},
            "ensembles": {},
        }
        base_preds = self._base_predictions(X)
        for name, preds in base_preds.items():
            results["base_models"][name] = self._metric_summary(y, preds)

        results["ensembles"]["mean"] = self._metric_summary(
            y, self.predict(X, strategy="mean")
        )
        for strategy in ("equal", "residual", "self_adaption", "stacking"):
            results["ensembles"][strategy] = self._metric_summary(
                y, self.predict(X, strategy=strategy)
            )
        return results

    def shap_values(self, X, model_name: str = "xgb", strategy: str | None = None):
        if shap is None:
            raise ImportError("shap is required for computing SHAP values")
        if strategy is not None:
            weights = self.ensemble_weights_.get(strategy)
            if weights is None:
                raise ValueError(f"Unknown strategy '{strategy}'")
            shap_xgb = shap.TreeExplainer(self.models["xgb"]).shap_values(X)
            shap_rf = shap.TreeExplainer(self.models["rf"]).shap_values(X)
            return weights[0] * shap_xgb + weights[1] * shap_rf
        if model_name not in self.models:
            raise ValueError(f"Unknown model '{model_name}'")
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
        if self.meta_model is not None:
            joblib.dump(self.meta_model, os.path.join(path, f"stack_{version}.joblib"))

    def load(self, version: str, path: str = "models") -> None:
        for name in self.models:
            self.models[name] = joblib.load(
                os.path.join(path, f"{name}_{version}.joblib")
            )
        stack_path = os.path.join(path, f"stack_{version}.joblib")
        if os.path.exists(stack_path):
            self.meta_model = joblib.load(stack_path)
        else:
            self.meta_model = None

    # ------------------------------------------------------------------
    def available_strategies(self) -> list[str]:
        return [
            "mean",
            "equal",
            "residual",
            "self_adaption",
            "stacking",
            *self.models.keys(),
        ]

    def ensemble_weights(self) -> Dict[str, Dict[str, float]]:
        weights = {}
        for key, value in self.ensemble_weights_.items():
            weights[key] = {
                "xgb": float(value[0]),
                "rf": float(value[1]),
            }
        return weights

    def stacking_weights(self) -> Dict[str, float]:
        if self.meta_model is None or not hasattr(self.meta_model, "final_estimator_"):
            return {}
        estimator = self.meta_model.final_estimator_
        if not hasattr(estimator, "coef_"):
            return {}
        coefs = np.atleast_1d(estimator.coef_).ravel()
        base_names = [name for name, _ in self.meta_model.estimators]
        if coefs.size != len(base_names):
            return {}
        norm = np.sum(np.abs(coefs)) or 1.0
        return {name: float(coef / norm) for name, coef in zip(base_names, coefs)}

    def predict_with_uncertainty(
        self, X, strategy: str = "self_adaption", confidence: float = 0.9
    ) -> Dict[str, np.ndarray]:
        """Return prediction along with simple t-interval uncertainty bounds."""

        central = np.asarray(self.predict(X, strategy=strategy))
        base_preds = self._base_predictions(X)
        matrix = np.column_stack(list(base_preds.values()))
        if matrix.ndim != 2 or matrix.shape[1] < 2:
            return {
                "prediction": central,
                "lower": central,
                "upper": central,
            }
        std = matrix.std(axis=1, ddof=1)
        se = np.nan_to_num(std / np.sqrt(matrix.shape[1]), nan=0.0)
        df = max(matrix.shape[1] - 1, 1)
        critical = t.ppf((1 + confidence) / 2, df)
        lower = central - critical * se
        upper = central + critical * se
        return {
            "prediction": central,
            "lower": lower,
            "upper": upper,
        }

    # ------------------------------------------------------------------
    def _compute_self_adaptive_weights(self, X_val, y_val) -> None:
        default = np.array([0.5, 0.5])
        if X_val is None or y_val is None or len(y_val) == 0:
            self.ensemble_weights_["equal"] = default
            self.ensemble_weights_["self_adaption"] = default
            return

        xgb_pred = self.models["xgb"].predict(X_val)
        rf_pred = self.models["rf"].predict(X_val)
        best_score = float("inf")
        best_weight = 0.5
        for weight in np.linspace(0.0, 1.0, 21):
            combined = weight * xgb_pred + (1 - weight) * rf_pred
            score = mean_squared_error(y_val, combined)
            if score < best_score:
                best_score = score
                best_weight = weight
        weights = np.array([best_weight, 1 - best_weight])
        self.ensemble_weights_["equal"] = default
        self.ensemble_weights_["self_adaption"] = weights

    def _compute_residual_weights(self, X, y) -> None:
        xgb_pred = self.models["xgb"].predict(X)
        rf_pred = self.models["rf"].predict(X)
        mse_xgb = mean_squared_error(y, xgb_pred)
        mse_rf = mean_squared_error(y, rf_pred)
        total = mse_xgb + mse_rf
        if total == 0:
            weights = np.array([0.5, 0.5])
        else:
            weights = np.array([mse_rf / total, mse_xgb / total])
        self.ensemble_weights_["residual"] = weights

    def _train_stacking_model(self, X, y) -> None:
        if len(X) < 5:
            self.meta_model = None
            return
        estimators = [(name, clone(model)) for name, model in self.models.items()]
        final_estimator = RidgeCV(alphas=np.logspace(-3, 3, 7))
        stacking = StackingRegressor(
            estimators=estimators,
            final_estimator=final_estimator,
            passthrough=False,
            cv=5,
        )
        stacking.fit(X, y)
        self.meta_model = stacking

    @staticmethod
    def _metric_summary(y_true, y_pred) -> Dict[str, float]:
        mse = mean_squared_error(y_true, y_pred)
        return {
            "r2": float(r2_score(y_true, y_pred)),
            "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mse)),
        }


# Backwards compatibility -------------------------------------------------
EmissionPredictor = ModelManager
