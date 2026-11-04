"""Pruebas de recuperación del modelo P1 para Semana 8."""

from unittest.mock import MagicMock

import pytest

from mibici.config import MODEL_PRODUCTION_ALIAS, REGISTERED_MODELS
from mibici.modeling import validate_registry


def simular_registry(monkeypatch):
    # Reemplazo el cliente y las URIs de MLflow para no depender del servidor.
    client = MagicMock()
    client.get_model_version_by_alias.return_value.version = "1"
    client.get_model_version_by_alias.return_value.run_id = "run-test"

    monkeypatch.setattr(validate_registry, "MlflowClient", lambda: client)
    monkeypatch.setattr(validate_registry.mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(validate_registry.mlflow, "set_registry_uri", lambda uri: None)
    return client


def test_week8_model_name():
    assert REGISTERED_MODELS["p1"] == "mibici-equipo15-p1-demanda"


def test_week8_model_alias():
    assert MODEL_PRODUCTION_ALIAS == "champion"


def test_week8_load_champion(monkeypatch):
    client = simular_registry(monkeypatch)
    model = MagicMock()
    loader = MagicMock(return_value=model)

    result = validate_registry.validate_registered_model("p1", model_loader=loader)

    assert result is model
    client.get_model_version_by_alias.assert_called_once_with(
        name="mibici-equipo15-p1-demanda",
        alias="champion",
    )
    loader.assert_called_once_with("models:/mibici-equipo15-p1-demanda@champion")


def test_week8_invalid_question():
    with pytest.raises(ValueError, match="Pregunta inválida"):
        validate_registry.validate_registered_model("p4")


def test_week8_model_load_failure(monkeypatch):
    simular_registry(monkeypatch)
    loader = MagicMock(side_effect=RuntimeError("Model unavailable"))

    with pytest.raises(RuntimeError, match="Model unavailable"):
        validate_registry.validate_registered_model("p1", model_loader=loader)
