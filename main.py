"""Demonstration of the minimal emission prediction and optimization framework."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import Config
from data_preprocessing import load_dataset, preprocess
from emission_predictor import ModelManager
from experiment_manager import ExperimentManager
from monitoring import ProcessMonitor
from optimization import (
    bayesian_optimization,
    genetic_algorithm,
    pso,
)


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


def load_training_frame(config: Config) -> pd.DataFrame:
    """Return dataframe for training based on configuration."""

    logger = logging.getLogger(__name__)
    if config.dataset_path:
        try:
            return load_dataset(
                config.dataset_path,
                table=config.dataset_table,
            )
        except FileNotFoundError:
            logger.warning(
                "Dataset path %s not found. Falling back to synthetic data.",
                config.dataset_path,
            )
        except ValueError as exc:
            logger.warning(
                "Failed to load dataset %s: %s. Falling back to synthetic data.",
                config.dataset_path,
                exc,
            )
    return generate_synthetic_data()


def main():
    config = Config()
    logging.basicConfig(
        level=config.logging_level(),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    data = load_training_frame(config)
    X, y, _, report = preprocess(data)
    logger.info("Data quality report: %s", report)
    if report.get("feature_summary"):
        top_features = list(report["feature_summary"].items())[:3]
        logger.info("Top feature summary: %s", top_features)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    predictor = ModelManager(seed=42)
    predictor.train(X_train, y_train)
    metrics = predictor.evaluate(X_test, y_test)
    logger.info("Evaluation metrics: %s", metrics)
    logger.info("Ensemble weights: %s", predictor.ensemble_weights())
    logger.info("Stacking weights: %s", predictor.stacking_weights())

    try:
        shap_vals = predictor.shap_values(X_train[:10], strategy="self_adaption")
        logger.info("Computed ensemble SHAP values with shape %s", shap_vals.shape)
    except Exception:  # pragma: no cover - shap optional
        logger.exception("SHAP computation failed")

    perm_imp = predictor.permutation_importance(
        X_test, y_test, model_name="rf", n_repeats=5
    )
    logger.info("Permutation importance: %s", perm_imp)

    interval = predictor.predict_with_uncertainty(
        X_test[:5], strategy="stacking", confidence=0.9
    )
    logger.info(
        "Stacking predictions with uncertainty: central=%s lower=%s upper=%s",
        interval["prediction"],
        interval["lower"],
        interval["upper"],
    )

    predictor.save(version="v1", path=config.models_dir)
    predictor.load(version="v1", path=config.models_dir)

    monitor = ProcessMonitor(
        threshold=config.threshold,
        optimizer=pso,
        objective=emission_objective,
        bounds=[(300, 1000), (1, 10)],
        window=5,
    )

    # 使用预测结果进行逐步监控
    preds = predictor.predict(X_test[:5], strategy="self_adaption")
    for actual, pred in zip(y_test[:5], preds):
        params, val = monitor.step(actual_emission=actual, predicted_emission=pred)
        print("Monitor step:", params, val)

    # 触发异常优化示例
    params, val = monitor.step(actual_emission=150.0, predicted_emission=60.0)
    print("Anomaly triggered optimization:", params, val)

    # 直接调用 adjust 进行额外调优并记录日志
    params, val = monitor.adjust(current_emission=config.threshold + 10)
    logger.info("Optimization result: %s %s", params, val)

    # 运行多种实验以比较不同优化策略
    manager = ExperimentManager()
    bounds = [(300, 1000), (1, 10)]
    manager.run("pso", pso, emission_objective, bounds, iterations=10)
    manager.run(
        "bayes", bayesian_optimization, emission_objective, bounds, iterations=10
    )
    manager.run("ga", genetic_algorithm, emission_objective, bounds, generations=10)
    print("Experiment summary:\n", manager.compare())


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.getLogger(__name__).exception("Application failed")
