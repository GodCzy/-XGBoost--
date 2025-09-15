from service import app


def test_index_route():
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert "排放预测面板" in response.get_data(as_text=True)


def test_predict_route_returns_value():
    client = app.test_client()
    payload = {"electricity": 100, "gdp": 50, "coal": 30}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.get_json()
