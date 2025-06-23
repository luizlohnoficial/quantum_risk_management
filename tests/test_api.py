from fastapi.testclient import TestClient
from risk_credit_quantum.src.api.main import app

client = TestClient(app)

def test_predict_pd():
    response = client.post("/predict_pd", json={"features": [0.1, 0.2, 0.3]})
    assert response.status_code == 200
    assert "pd" in response.json()

def test_optimize_portfolio():
    response = client.post(
        "/optimize_portfolio",
        json={"returns": [0.1, 0.2], "risks": [0.05, 0.03], "budget": 1},
    )
    assert response.status_code == 200
    assert "selection" in response.json()

def test_simulate_default():
    response = client.post(
        "/simulate_default",
        json={"probabilities": [0.1, 0.2], "trials": 10},
    )
    assert response.status_code == 200
    assert "default_rate" in response.json()
