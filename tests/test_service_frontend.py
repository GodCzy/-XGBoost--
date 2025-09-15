from service import app


def test_index_route():
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert "排放预测面板" in response.get_data(as_text=True)
