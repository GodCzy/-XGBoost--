from __future__ import annotations

import logging
from threading import RLock
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.model_selection import train_test_split

from config import Config
from data_preprocessing import (
    build_feature_frame,
    clean_data,
    frame_from_records,
    load_dataset,
    preprocess,
)
from emission_predictor import ModelManager
from experiment_manager import ExperimentManager
from monitoring import ProcessMonitor
from optimization import bayesian_optimization, genetic_algorithm, pso
from main import emission_objective, generate_synthetic_data

config = Config()
logging.basicConfig(
    level=config.logging_level(),
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Data preparation helpers
# ---------------------------------------------------------------------------


def _load_training_frame(cfg: Config) -> pd.DataFrame:
    if cfg.dataset_path:
        try:
            logger.info("Loading dataset from %s", cfg.dataset_path)
            return load_dataset(cfg.dataset_path, table=cfg.dataset_table)
        except FileNotFoundError:
            logger.warning(
                "Dataset path %s not found. Falling back to synthetic data.",
                cfg.dataset_path,
            )
        except ValueError as exc:
            logger.warning(
                "Failed to load dataset %s: %s. Falling back to synthetic data.",
                cfg.dataset_path,
                exc,
            )
    logger.info("Generating synthetic training data")
    return generate_synthetic_data()


def _base_feature_names(feature_names: List[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for name in feature_names:
        base = name[:-3] if name.endswith("_sq") else name
        if base not in seen:
            seen.add(base)
            ordered.append(base)
    return ordered


def _aggregate_feature_values(
    values: Any, feature_names: List[str]
) -> List[Dict[str, float]]:
    totals: Dict[str, float] = {}
    for name, value in zip(feature_names, values):
        base = name[:-3] if name.endswith("_sq") else name
        totals[base] = totals.get(base, 0.0) + float(value)
    return [
        {"feature": feature, "importance": totals[feature]}
        for feature in sorted(totals, key=lambda key: abs(totals[key]), reverse=True)
    ]


def _run_experiment(
    manager: ExperimentManager,
    name: str,
    optimizer,
    **kwargs: Any,
) -> Dict[str, Any]:
    params, value = manager.run(
        name,
        optimizer,
        emission_objective,
        [(300, 1000), (1, 10)],
        **kwargs,
    )
    params_array = np.atleast_1d(params)
    return {
        "algorithm": name,
        "best_params": [float(v) for v in params_array],
        "best_val": float(value),
    }


def _prepare_training_context(frame: pd.DataFrame) -> Dict[str, Any]:
    logger.info("Training model for API service")
    frame = frame.copy()
    X_scaled, y, scaler, report = preprocess(frame)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = ModelManager(seed=42)
    model.train(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)

    feature_names = list(scaler.feature_names_in_)
    base_features = _base_feature_names(feature_names)

    permutation_raw = model.permutation_importance(
        X_test, y_test, model_name="rf", n_repeats=8
    )
    permutation_importance = _aggregate_feature_values(permutation_raw, feature_names)

    shap_summary: Dict[str, List[Dict[str, float]]] | None = None
    try:
        sample_size = min(100, len(X_train))
        if sample_size > 0:
            shap_sample = X_train[:sample_size]
            shap_summary = {
                "xgb": _aggregate_feature_values(
                    np.abs(model.shap_values(shap_sample, model_name="xgb")).mean(
                        axis=0
                    ),
                    feature_names,
                ),
                "rf": _aggregate_feature_values(
                    np.abs(model.shap_values(shap_sample, model_name="rf")).mean(
                        axis=0
                    ),
                    feature_names,
                ),
                "self_adaption": _aggregate_feature_values(
                    np.abs(
                        model.shap_values(shap_sample, strategy="self_adaption")
                    ).mean(axis=0),
                    feature_names,
                ),
            }
    except Exception:  # pragma: no cover - shap may fail at runtime
        logger.exception("Failed to compute SHAP summary for service")
        shap_summary = None

    monitor = ProcessMonitor(
        threshold=config.threshold,
        optimizer=pso,
        objective=emission_objective,
        bounds=[(300, 1000), (1, 10)],
        window=5,
    )
    monitor_log: List[Dict[str, Any]] = []
    preds = model.predict(X_test[: min(5, len(X_test))], strategy="self_adaption")
    for idx, (actual, predicted) in enumerate(zip(y_test[: len(preds)], preds)):
        params, value = monitor.step(float(actual), float(predicted))
        monitor_log.append(
            {
                "step": idx,
                "actual": float(actual),
                "predicted": float(predicted),
                "action": (
                    None
                    if params is None
                    else [float(v) for v in np.atleast_1d(params)]
                ),
                "value": float(value),
            }
        )

    params, value = monitor.adjust(current_emission=float(config.threshold + 5))
    monitor_log.append(
        {
            "step": len(monitor_log),
            "actual": None,
            "predicted": None,
            "action": (
                None if params is None else [float(v) for v in np.atleast_1d(params)]
            ),
            "value": float(value),
            "forced": True,
        }
    )

    experiments: List[Dict[str, Any]] = []
    experiment_summary: List[Dict[str, Any]] = []
    experiment_manager = ExperimentManager(log_file="experiments_service.csv")
    try:
        experiments.append(
            _run_experiment(experiment_manager, "pso", pso, iterations=5)
        )
        experiments.append(
            _run_experiment(
                experiment_manager, "bayesian", bayesian_optimization, iterations=5
            )
        )
        experiments.append(
            _run_experiment(experiment_manager, "ga", genetic_algorithm, generations=5)
        )
        experiment_summary = experiment_manager.compare().to_dict(orient="records")
    except Exception:  # pragma: no cover - optimizer may fail
        logger.exception("Experiment execution failed")

    return {
        "frame": frame.reset_index(drop=True),
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "model": model,
        "metrics": metrics,
        "permutation": permutation_importance,
        "shap": shap_summary,
        "report": report,
        "base_features": base_features,
        "feature_names": feature_names,
        "ensemble_weights": model.ensemble_weights(),
        "stacking_weights": model.stacking_weights(),
        "strategies": model.available_strategies(),
        "monitor_log": monitor_log,
        "experiments": experiments,
        "experiment_summary": experiment_summary,
    }


def build_training_context(
    cfg: Config,
    frame: pd.DataFrame | None = None,
) -> Dict[str, Any]:
    source_frame = frame if frame is not None else _load_training_frame(cfg)
    return _prepare_training_context(source_frame)


class TrainingState:
    """Thread-safe container for the service training context."""

    def __init__(self, cfg: Config):
        self._cfg = cfg
        self._lock = RLock()
        self._context = build_training_context(cfg)

    def context(self) -> Dict[str, Any]:
        with self._lock:
            return self._context

    def update(self, frame: pd.DataFrame | None = None) -> Dict[str, Any]:
        context = build_training_context(self._cfg, frame)
        with self._lock:
            self._context = context
        return context


TRAINING_STATE = TrainingState(config)


def _training_context() -> Dict[str, Any]:
    return TRAINING_STATE.context()


def _resolve_training_payload(payload: Any) -> pd.DataFrame:
    if isinstance(payload, dict):
        if "dataset_path" in payload:
            table = payload.get("dataset_table") or payload.get("table")
            return load_dataset(
                payload["dataset_path"],
                table=table or config.dataset_table,
            )
        records = (
            payload.get("records")
            or payload.get("rows")
            or payload.get("data")
        )
        if records is None:
            raise ValueError("Payload must include records or a dataset path")
    else:
        records = payload
    return frame_from_records(records)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index() -> str:
    """Serve the interactive dashboard."""
    return render_template("index.html")


@app.route("/metadata")
def metadata():
    context = _training_context()
    return jsonify(
        {
            "features": context["base_features"],
            "report": context["report"],
            "strategies": context["strategies"],
            "ensemble_weights": context["ensemble_weights"],
            "stacking_weights": context.get("stacking_weights", {}),
        }
    )


@app.route("/metrics")
def metrics():
    return jsonify(_training_context()["metrics"])


@app.route("/feature-insights")
def feature_insights():
    context = _training_context()
    return jsonify(
        {
            "permutation_importance": context["permutation"],
            "shap_summary": context["shap"],
        }
    )


@app.route("/optimization")
def optimization():
    context = _training_context()
    return jsonify(
        {
            "experiments": context["experiments"],
            "summary": context["experiment_summary"],
        }
    )


@app.route("/monitor-sample")
def monitor_sample():
    return jsonify(
        {
            "log": _training_context()["monitor_log"],
            "threshold": float(config.threshold),
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    """Return emission predictions for provided features."""
    try:
        payload = request.get_json(force=True)
        strategy = request.args.get("strategy", "mean")
        all_strategies = request.args.get("all_strategies", "false").lower() == "true"
        include_uncertainty = request.args.get("uncertainty", "false").lower() == "true"
        if isinstance(payload, dict) and not all(
            isinstance(v, (int, float)) for v in payload.values()
        ):
            strategy = payload.pop("strategy", strategy)
        records = payload if isinstance(payload, list) else [payload]
        df = pd.DataFrame(records)
        df = clean_data(df)
        if df.empty:
            raise ValueError("No valid input data")
        df = build_feature_frame(df, drop_target=False)
        features = df.drop(columns=["emission"], errors="ignore")
        context = _training_context()
        features = features.reindex(
            columns=context["scaler"].feature_names_in_, fill_value=0.0
        )
        X_in = context["scaler"].transform(features)
        model: ModelManager = context["model"]
        if all_strategies:
            preds = {
                name: model.predict(X_in, strategy=name).tolist()
                for name in model.available_strategies()
            }
            response_payload: Dict[str, Any] = {"predictions": preds}
            if include_uncertainty:
                interval = model.predict_with_uncertainty(X_in, strategy=strategy)
                response_payload["uncertainty"] = {
                    "strategy": strategy,
                    **{key: value.tolist() for key, value in interval.items()},
                }
            return jsonify(response_payload)
        preds = model.predict(X_in, strategy=strategy)
        if include_uncertainty:
            interval = model.predict_with_uncertainty(X_in, strategy=strategy)
            return jsonify(
                {
                    "prediction": preds.tolist(),
                    "strategy": strategy,
                    "lower": interval["lower"].tolist(),
                    "upper": interval["upper"].tolist(),
                }
            )
        return jsonify({"prediction": preds.tolist(), "strategy": strategy})
    except Exception as exc:  # pragma: no cover - runtime safety
        logger.exception("Prediction failed")
        return jsonify({"error": str(exc)}), 400


@app.route("/train", methods=["POST"])
def train_endpoint():
    """Retrain the service models with user-provided data."""

    try:
        payload = request.get_json(force=True, silent=True)
        if payload is None:
            raise ValueError("A JSON payload is required for training")
        frame = _resolve_training_payload(payload)
        context = TRAINING_STATE.update(frame)
        return jsonify(
            {
                "status": "trained",
                "rows": int(len(context["frame"])),
                "features": context["base_features"],
                "strategies": context["strategies"],
                "metrics": context["metrics"],
            }
        )
    except Exception as exc:  # pragma: no cover - runtime safety
        logger.exception("Training update failed")
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
