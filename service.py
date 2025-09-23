from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.model_selection import train_test_split

from config import Config
from data_preprocessing import build_feature_frame, clean_data, preprocess
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
        dataset_path = Path(cfg.dataset_path)
        if dataset_path.exists():
            logger.info("Loading dataset from %s", dataset_path)
            return pd.read_csv(dataset_path)
        logger.warning(
            "Dataset path %s not found. Falling back to synthetic data.", dataset_path
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


def _prepare_training_context() -> Dict[str, Any]:
    logger.info("Training model for API service")
    frame = _load_training_frame(config)
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
        "frame": frame,
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


TRAINING_CONTEXT = _prepare_training_context()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index() -> str:
    """Serve the interactive dashboard."""
    return render_template("index.html")


@app.route("/metadata")
def metadata():
    return jsonify(
        {
            "features": TRAINING_CONTEXT["base_features"],
            "report": TRAINING_CONTEXT["report"],
            "strategies": TRAINING_CONTEXT["strategies"],
            "ensemble_weights": TRAINING_CONTEXT["ensemble_weights"],
            "stacking_weights": TRAINING_CONTEXT.get("stacking_weights", {}),
        }
    )


@app.route("/metrics")
def metrics():
    return jsonify(TRAINING_CONTEXT["metrics"])


@app.route("/feature-insights")
def feature_insights():
    return jsonify(
        {
            "permutation_importance": TRAINING_CONTEXT["permutation"],
            "shap_summary": TRAINING_CONTEXT["shap"],
        }
    )


@app.route("/optimization")
def optimization():
    return jsonify(
        {
            "experiments": TRAINING_CONTEXT["experiments"],
            "summary": TRAINING_CONTEXT["experiment_summary"],
        }
    )


@app.route("/monitor-sample")
def monitor_sample():
    return jsonify(
        {
            "log": TRAINING_CONTEXT["monitor_log"],
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
        features = features.reindex(
            columns=TRAINING_CONTEXT["scaler"].feature_names_in_, fill_value=0.0
        )
        X_in = TRAINING_CONTEXT["scaler"].transform(features)
        model: ModelManager = TRAINING_CONTEXT["model"]
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
