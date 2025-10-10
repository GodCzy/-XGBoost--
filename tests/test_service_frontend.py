from service import app


def test_index_route():
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert "智能排放管控驾驶舱" in response.get_data(as_text=True)


def test_predict_route_returns_value():
    client = app.test_client()
    payload = {"electricity": 100, "gdp": 50, "coal": 30}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.get_json() or "predictions" in response.get_json()


def test_predict_route_returns_uncertainty():
    client = app.test_client()
    payload = {"electricity": 100, "gdp": 50, "coal": 30}
    response = client.post("/predict?strategy=stacking&uncertainty=true", json=payload)
    data = response.get_json()
    assert response.status_code == 200
    assert "lower" in data and "upper" in data


def test_metadata_and_metrics_routes():
    client = app.test_client()
    metadata = client.get("/metadata").get_json()
    assert metadata["features"]
    assert metadata["strategies"]
    assert "ensemble_weights" in metadata
    assert "stacking_weights" in metadata
    assert "feature_summary" in metadata["report"]

    metrics = client.get("/metrics").get_json()
    assert "base_models" in metrics and "ensembles" in metrics


def test_feature_insights_and_monitor_routes():
    client = app.test_client()
    insights = client.get("/feature-insights").get_json()
    assert "permutation_importance" in insights
    monitor = client.get("/monitor-sample").get_json()
    assert "log" in monitor and isinstance(monitor["log"], list)


def test_optimization_route():
    client = app.test_client()
    optimization = client.get("/optimization").get_json()
    assert "experiments" in optimization


def test_train_route_retrains_model():
    client = app.test_client()
    payload = [
        {
            "electricity": 100 + idx,
            "gdp": 50 + idx,
            "coal": 30 + idx,
            "emission": 80 + idx,
        }
        for idx in range(6)
    ]
    response = client.post("/train", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "trained"
    assert data["rows"] >= 3
    predict_resp = client.post("/predict", json=payload[0])
    assert predict_resp.status_code == 200
    prediction_payload = predict_resp.get_json()
    assert "prediction" in prediction_payload or "predictions" in prediction_payload
