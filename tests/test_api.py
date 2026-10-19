import pytest
from fastapi.testclient import TestClient

from heartml import config
from heartml.api import app

SAMPLE = {"age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1,
          "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 0,
          "ca": 0, "thal": 1}

pytestmark = pytest.mark.skipif(not config.MODEL_PATH.exists(), reason="train the model first")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_prediction_and_confidence(client):
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1.0
    assert 0 <= body["probability_disease"] <= 1


def test_predict_rejects_missing_field(client):
    bad = {k: v for k, v in SAMPLE.items() if k != "age"}
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_rejects_out_of_range(client):
    assert client.post("/predict", json={**SAMPLE, "sex": 5}).status_code == 422


def test_metrics_endpoint_exposes_counters(client):
    client.post("/predict", json=SAMPLE)
    text = client.get("/metrics").text
    assert "api_requests_total" in text
    assert "model_predictions_total" in text
