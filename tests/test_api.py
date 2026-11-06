
"""Pruebas automatizadas de la API REST MiBici P1.

Semana 8 - Integración de FastAPI, Pydantic y MLflow.

Las pruebas utilizan FakeModelService para validar los endpoints
sin depender de un servidor MLflow externo.

Contrato de entrada basado en:
docs/week8/model_registry_contract.md
"""

import copy
import math

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from mibici.api.main import MODEL_NAME, create_app
from mibici.api.schemas import P1_FEATURES, PredictRequest

from mibici.api.model_service import (
    ModelInputError,
    ModelNotReadyError,
    ModelPredictionError,
)


# ============================================================
# 1. Servicio simulado de inferencia
# ============================================================

class FakeModelService:
    """Simula el ModelService de Oscar para las pruebas de FastAPI."""

    def __init__(self):
        self.loaded = False
        self.load_count = 0
        self.records = None
        self.fail = False
        self.not_ready = False
        self.predictions = [12.5]

    def load_model(self):
        """Simula la carga del modelo desde MLflow Registry."""
        self.loaded = True
        self.load_count += 1

    def is_ready(self):
        """Indica si el modelo simulado está disponible."""
        return self.loaded and not self.not_ready

    def predict(self, records):
        """Devuelve predicciones simuladas."""
        self.records = records

        if self.fail:
            raise ValueError("Error simulado de inferencia")

        return self.predictions


# ============================================================
# 2. Fixtures de PyTest
# ============================================================

@pytest.fixture
def fake_service():
    """Crea un servicio de inferencia independiente por prueba."""
    return FakeModelService()


@pytest.fixture
def client(fake_service):
    """Inicia FastAPI con el ciclo de vida activo."""
    app = create_app(model_service=fake_service)

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_payload():
    """Entrada oficial P1, según la firma del Registry."""
    return {
        "features": {
            "hour_of_day": 8,
            "weekday": 2,
            "fin_semana": 0,
            "salidas_lag_1h": 3,
            "salidas_lag_24h": 4,
            "salidas_lag_168h": 5,
            "media_estacion": 2.1,
            "media_estacion_dia_hora": 3.4,
        }
    }


# ============================================================
# 3. Pruebas existentes: salud y disponibilidad
# ============================================================

def test_health(client):
    """Verifica que /health responde correctamente."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready(client):
    """Verifica que /ready confirma la disponibilidad."""
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"ready": True}


# ============================================================
# 4. Pruebas existentes: predicciones
# ============================================================

def test_prediction(client, valid_payload, fake_service):
    """Verifica la predicción y el registro enviado al servicio."""
    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] == 12.5
    assert data["model"] == MODEL_NAME

    expected_features = {
        name: float(value)
        for name, value in valid_payload["features"].items()
    }

    assert fake_service.records == [expected_features]

    assert list(fake_service.records[0]) == list(P1_FEATURES)

    assert all(
        isinstance(value, float)
        for value in fake_service.records[0].values()
    )


def test_missing_feature(client, valid_payload):
    """Rechaza una característica obligatoria faltante."""
    del valid_payload["features"]["salidas_lag_1h"]

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 422


def test_invalid_data_type(client, valid_payload):
    """Rechaza una característica con un valor no numérico."""
    valid_payload["features"]["hour_of_day"] = "invalid"

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 422


def test_extra_feature(client, valid_payload):
    """Rechaza características adicionales."""
    valid_payload["features"]["estacion_id"] = 10

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 422


def test_empty_body(client):
    """Rechaza solicitudes sin el campo features."""
    response = client.post("/predict", json={})

    assert response.status_code == 422


def test_malformed_json(client):
    """Rechaza JSON mal formado."""
    response = client.post(
        "/predict",
        content='{"features":}',
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422


def test_unknown_endpoint(client):
    """Devuelve 404 para una ruta inexistente."""
    response = client.get("/unknown")

    assert response.status_code == 404


def test_prediction_error(client, valid_payload, fake_service):
    """Controla un error simulado durante la inferencia."""
    fake_service.fail = True

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 500
    assert response.json() == {
        "detail": "No fue posible realizar la predicción."
    }


def test_openapi(client):
    """Comprueba la generación de OpenAPI."""
    response = client.get("/openapi.json")

    assert response.status_code == 200

    paths = response.json()["paths"]

    assert "/predict" in paths
    assert "/health" in paths
    assert "/ready" in paths


def test_startup_loads_model_once(fake_service):
    """Comprueba que FastAPI carga el modelo una sola vez."""
    app = create_app(model_service=fake_service)

    with TestClient(app) as test_client:
        assert fake_service.load_count == 1

        assert test_client.get("/ready").status_code == 200
        assert test_client.get("/ready").status_code == 200

    assert fake_service.load_count == 1


def test_model_not_ready(client, valid_payload, fake_service):
    """Devuelve HTTP 503 cuando el modelo no está listo."""
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
    """Rechaza una salida con más predicciones de las esperadas."""
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
    """Rechaza cadenas, nulos y booleanos."""
    valid_payload["features"]["hour_of_day"] = invalid_value

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 422


# ============================================================
# 5. Nuevas pruebas: contrato oficial P1
# ============================================================

def test_p1_feature_names(valid_payload):
    """Comprueba que existen exactamente las ocho columnas oficiales."""
    assert tuple(valid_payload["features"]) == P1_FEATURES
    assert len(valid_payload["features"]) == 8


def test_unknown_feature_same_count(client, valid_payload):
    """Rechaza un nombre desconocido aunque existan ocho campos."""
    features = valid_payload["features"]

    del features["media_estacion"]
    features["variable_desconocida"] = 2.1

    assert len(features) == 8

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 422


def test_missing_features_object(client):
    """Rechaza una solicitud sin objeto features."""
    response = client.post(
        "/predict",
        json={"otro_campo": {}},
    )

    assert response.status_code == 422


def test_empty_features(client):
    """Rechaza un diccionario sin características."""
    response = client.post(
        "/predict",
        json={"features": {}},
    )

    assert response.status_code == 422


def test_extra_request_field(client, valid_payload):
    """Rechaza propiedades adicionales fuera de features."""
    valid_payload["campo_adicional"] = "no permitido"

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 422


# ============================================================
# 6. Nuevas pruebas: rangos de tiempo
# ============================================================

@pytest.mark.parametrize(
    "hour",
    [-1, -3, 24, 30],
)
def test_invalid_hour_of_day(client, valid_payload, hour):
    """Rechaza horas fuera del intervalo 0-23."""
    valid_payload["features"]["hour_of_day"] = hour

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 422


@pytest.mark.parametrize(
    "hour",
    [0, 8, 23],
)
def test_valid_hour_of_day(client, valid_payload, hour):
    """Acepta horas dentro del intervalo permitido."""
    valid_payload["features"]["hour_of_day"] = hour

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 200


@pytest.mark.parametrize(
    "weekday",
    [-1, 7, 10],
)
def test_invalid_weekday(client, valid_payload, weekday):
    """Rechaza días fuera del intervalo 0-6."""
    valid_payload["features"]["weekday"] = weekday

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 422


@pytest.mark.parametrize(
    "weekday, weekend",
    [
        (0, 0),
        (4, 0),
        (5, 1),
        (6, 1),
    ],
)
def test_valid_weekday_and_weekend(
    client,
    valid_payload,
    weekday,
    weekend,
):
    """Acepta combinaciones coherentes de día y fin de semana."""
    valid_payload["features"]["weekday"] = weekday
    valid_payload["features"]["fin_semana"] = weekend

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 200


@pytest.mark.parametrize(
    "weekend",
    [-1, 2, 0.5],
)
def test_invalid_fin_semana(client, valid_payload, weekend):
    """Rechaza valores de fin_semana distintos de 0 o 1."""
    valid_payload["features"]["fin_semana"] = weekend

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 422


@pytest.mark.parametrize(
    "weekday, weekend",
    [
        (0, 1),
        (2, 1),
        (5, 0),
        (6, 0),
    ],
)
def test_inconsistent_weekend(
    client,
    valid_payload,
    weekday,
    weekend,
):
    """Rechaza fin_semana cuando no coincide con weekday."""
    valid_payload["features"]["weekday"] = weekday
    valid_payload["features"]["fin_semana"] = weekend

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 422


# ============================================================
# 7. Nuevas pruebas: rezagos y medias históricas
# ============================================================

@pytest.mark.parametrize(
    "feature",
    [
        "salidas_lag_1h",
        "salidas_lag_24h",
        "salidas_lag_168h",
        "media_estacion",
        "media_estacion_dia_hora",
    ],
)
def test_negative_features(client, valid_payload, feature):
    """Rechaza rezagos y medias con valores negativos."""
    valid_payload["features"][feature] = -1.0

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 422


@pytest.mark.parametrize(
    "feature",
    [
        "salidas_lag_1h",
        "salidas_lag_24h",
        "salidas_lag_168h",
        "media_estacion",
        "media_estacion_dia_hora",
    ],
)
def test_zero_non_negative_features(
    client,
    valid_payload,
    feature,
):
    """Acepta cero en rezagos y medias históricas."""
    valid_payload["features"][feature] = 0

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 200


# ============================================================
# 8. Nuevas pruebas: tipos y conversión de datos
# ============================================================

def test_integer_values_are_accepted(
    client,
    valid_payload,
    fake_service,
):
    """Comprueba que los enteros se convierten a float."""
    valid_payload["features"]["salidas_lag_1h"] = 3
    valid_payload["features"]["hour_of_day"] = 8

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 200
    assert fake_service.records is not None

    for value in fake_service.records[0].values():
        assert isinstance(value, float)


def test_numeric_string_rejected(client, valid_payload):
    """Rechaza cadenas numéricas, incluso si parecen válidas."""
    valid_payload["features"]["hour_of_day"] = "8"

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 422


@pytest.mark.parametrize(
    "invalid_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_non_finite_values(valid_payload, invalid_value):
    """Rechaza NaN e infinitos directamente en Pydantic."""
    features = copy.deepcopy(valid_payload["features"])
    features["media_estacion"] = invalid_value

    with pytest.raises(ValidationError):
        PredictRequest(features=features)


def test_invalid_features_not_sent_to_model(
    client,
    valid_payload,
    fake_service,
):
    """Las entradas inválidas nunca deben alcanzar ModelService."""
    valid_payload["features"]["hour_of_day"] = -1

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 422
    assert fake_service.records is None


# ============================================================
# 9. Nuevas pruebas: respuesta del modelo
# ============================================================

def test_non_negative_prediction(client, valid_payload):
    """Verifica que una predicción positiva es aceptada."""
    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 200
    assert response.json()["prediction"] >= 0


def test_prediction_is_numeric(client, valid_payload):
    """Verifica el tipo numérico de la respuesta."""
    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 200
    assert isinstance(response.json()["prediction"], float)


def test_prediction_response_contains_model(client, valid_payload):
    """Verifica que la respuesta identifica el modelo P1."""
    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 200
    assert response.json()["model"] == MODEL_NAME


# ============================================================
# 10. Nuevas pruebas: validación directa del esquema
# ============================================================

def test_pydantic_preserves_feature_order(valid_payload):
    """Comprueba que Pydantic ordena según la firma MLflow."""
    reversed_features = dict(
        reversed(list(valid_payload["features"].items()))
    )

    payload = PredictRequest(features=reversed_features)

    assert list(payload.features) == list(P1_FEATURES)


def test_pydantic_converts_all_values_to_float(valid_payload):
    """Comprueba que la representación final usa float."""
    payload = PredictRequest.model_validate(valid_payload)

    assert len(payload.features) == 8
    assert all(
        type(value) is float
        for value in payload.features.values()
    )


def test_pydantic_rejects_boolean_values(valid_payload):
    """Los booleanos no deben interpretarse como números."""
    features = copy.deepcopy(valid_payload["features"])
    features["fin_semana"] = True

    with pytest.raises(ValidationError):
        PredictRequest(features=features)


def test_pydantic_rejects_infinite_prediction():
    """Rechaza valores infinitos en el esquema de respuesta."""
    from mibici.api.schemas import PredictResponse

    with pytest.raises(ValidationError):
        PredictResponse(
            prediction=math.inf,
            model=MODEL_NAME,
        )


def test_pydantic_rejects_negative_prediction():
    """Rechaza predicciones negativas en el esquema de respuesta."""
    from mibici.api.schemas import PredictResponse

    with pytest.raises(ValidationError):
        PredictResponse(
            prediction=-1.0,
            model=MODEL_NAME,
        )


# ============================================================
# 11. Pruebas de integración: manejo de errores ModelService
# ============================================================

def test_model_input_error_returns_422(
    valid_payload,
    fake_service,
):
    """ModelInputError debe producir HTTP 422."""

    def predict_invalid_input(records):
        raise ModelInputError(
            "Las columnas no cumplen la firma P1."
        )

    fake_service.predict = predict_invalid_input

    app = create_app(model_service=fake_service)

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json=valid_payload,
        )

    assert response.status_code == 422
    assert response.json()["detail"] == (
        "Los datos de entrada no son válidos."
    )


def test_model_not_ready_error_returns_503(
    valid_payload,
    fake_service,
):
    """ModelNotReadyError durante predict debe producir 503."""

    def predict_unavailable(records):
        raise ModelNotReadyError(
            "El modelo dejó de estar disponible."
        )

    fake_service.predict = predict_unavailable

    app = create_app(model_service=fake_service)

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json=valid_payload,
        )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "El modelo no está disponible."
    )


def test_model_prediction_error_returns_500(
    valid_payload,
    fake_service,
):
    """ModelPredictionError debe producir HTTP 500."""

    def predict_failed(records):
        raise ModelPredictionError(
            "Error interno durante la inferencia."
        )

    fake_service.predict = predict_failed

    app = create_app(model_service=fake_service)

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json=valid_payload,
        )

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "No fue posible realizar la predicción."
    )


def test_health_when_model_loading_fails(fake_service):
    """La API debe responder /health aunque falle MLflow."""

    def failed_load():
        raise ModelNotReadyError(
            "No fue posible conectar con MLflow."
        )

    fake_service.load_model = failed_load

    app = create_app(model_service=fake_service)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok"
    }


def test_ready_when_model_loading_fails(fake_service):
    """Si la carga falla, /ready debe devolver HTTP 503."""

    def failed_load():
        raise ModelNotReadyError(
            "No fue posible recuperar el modelo champion."
        )

    fake_service.load_model = failed_load

    app = create_app(model_service=fake_service)

    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "El modelo no está disponible."
    )
