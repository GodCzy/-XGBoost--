import numpy as np
from sklearn.model_selection import train_test_split

from data_preprocessing import preprocess
from emission_predictor import ModelManager
from main import generate_synthetic_data


def test_model_manager_train_evaluate_predict():
    data = generate_synthetic_data(50)
    X, y, _, _ = preprocess(data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    manager = ModelManager(seed=0)
    manager.train(X_train, y_train)
    preds = manager.predict(X_test)
    assert preds.shape[0] == y_test.shape[0]
    metrics = manager.evaluate(X_test, y_test)
    assert "ensemble" in metrics and "rf" in metrics
    assert isinstance(metrics["ensemble"]["r2"], float)
