
"""Pruebas automatizadas de la API REST MiBici P1."""

import pytest
from fastapi.testclient import TestClient

from mibici.api.main import MODEL_NAME, create_app


class FakeModelService:
    """Simula el servicio de inferencia de Oscar."""

    def __init__(self):
        self.loaded = False
        self.load_count = 0
        self.records = None
        self.fail = False
        self.not_ready = False
        self.predictions = [12.5]

    def load_model(self):
        self.loaded = True
        self.load_count += 1

    def is_ready(self):
        return self.loaded and not self.not_ready

    def predict(self, records):
        self.records = records

        if self.fail:
            raise ValueError("Error simulado de inferencia")

        return self.predictions


@pytest.fixture
def fake_service():
    """Servicio de inferencia simulado."""
    return FakeModelService()


@pytest.fixture
def client(fake_service):
    """Cliente HTTP con el ciclo de vida de FastAPI activo."""
    app = create_app(model_service=fake_service)

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_payload():
    """Entrada provisional de ocho características numéricas."""
    return {
        "features": {
            f"feature_{i}": float(i)
            for i in range(1, 9)
        }
    }


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready(client):
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"ready": True}


def test_prediction(client, valid_payload, fake_service):
    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] == 12.5
    assert data["model"] == MODEL_NAME

    assert fake_service.records == [
        valid_payload["features"]
    ]


def test_missing_feature(client, valid_payload):
    del valid_payload["features"]["feature_1"]

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 422


def test_invalid_data_type(client, valid_payload):
    valid_payload["features"]["feature_1"] = "invalid"

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 422


def test_extra_feature(client, valid_payload):
    valid_payload["features"]["feature_9"] = 9.0

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 422


def test_empty_body(client):
    response = client.post("/predict", json={})

    assert response.status_code == 422


def test_malformed_json(client):
    response = client.post(
        "/predict",
        content='{"features":}',
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422


def test_unknown_endpoint(client):
    response = client.get("/unknown")

    assert response.status_code == 404


def test_prediction_error(client, valid_payload, fake_service):
    fake_service.fail = True

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 500
    assert response.json() == {
        "detail": "No fue posible realizar la predicción."
    }


def test_openapi(client):
    response = client.get("/openapi.json")

    assert response.status_code == 200

    paths = response.json()["paths"]

    assert "/predict" in paths
    assert "/health" in paths
    assert "/ready" in paths


def test_startup_loads_model_once(fake_service):
    app = create_app(model_service=fake_service)

    with TestClient(app) as test_client:
        assert fake_service.load_count == 1
        assert test_client.get("/ready").status_code == 200
        assert test_client.get("/ready").status_code == 200

    assert fake_service.load_count == 1


def test_model_not_ready(client, valid_payload, fake_service):
    fake_service.not_ready = True

    response = client.get("/ready")

    assert response.status_code == 503

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 503


def test_invalid_prediction_count(
    client,
    valid_payload,
    fake_service,
):
    fake_service.predictions = [12.5, 14.0]

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 500


@pytest.mark.parametrize(
    "invalid_value",
    ["invalid", None, True],
)
def test_invalid_feature_values(
    client,
    valid_payload,
    invalid_value,
):
    valid_payload["features"]["feature_1"] = invalid_value

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 422
