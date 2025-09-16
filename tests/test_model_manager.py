import numpy as np
from sklearn.model_selection import train_test_split

from data_preprocessing import preprocess
from emission_predictor import ModelManager
from main import generate_synthetic_data


def test_model_manager_train_evaluate_predict():
    data = generate_synthetic_data(80)
    X, y, _, _ = preprocess(data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    manager = ModelManager(seed=0)
    manager.train(X_train, y_train)

    preds = manager.predict(X_test)
    assert preds.shape[0] == y_test.shape[0]

    metrics = manager.evaluate(X_test, y_test)
    assert "base_models" in metrics and "ensembles" in metrics
    assert "mean" in metrics["ensembles"]
    assert isinstance(metrics["ensembles"]["mean"]["r2"], float)

    weights = manager.ensemble_weights()
    assert "self_adaption" in weights and "residual" in weights

    strategies = manager.available_strategies()
    for strategy in ("equal", "self_adaption", "rf"):
        assert strategy in strategies
        preds_strategy = manager.predict(X_test, strategy=strategy)
        assert preds_strategy.shape[0] == y_test.shape[0]
