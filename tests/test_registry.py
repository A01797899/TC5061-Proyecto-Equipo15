"""Pruebas automatizadas del Model Registry de MiBici."""

from unittest.mock import MagicMock, patch

import pytest

from mibici.config import (
    REGISTERED_MODELS,
    MODEL_PRODUCTION_ALIAS,
)
from mibici.modeling import registry


# ============================================================
# 1. Validación de nombres registrados
# ============================================================

def test_registered_model_names():
    """Verifica los nombres oficiales de P1, P2 y P3."""

    assert REGISTERED_MODELS == {
        "p1": "mibici-equipo15-p1-demanda",
        "p2": "mibici-equipo15-p2-riesgo",
        "p3": "mibici-equipo15-p3-grupos",
    }


def test_champion_alias():
    """Verifica el alias de producción."""

    assert MODEL_PRODUCTION_ALIAS == "champion"


# ============================================================
# 2. Validación de parámetros
# ============================================================

def test_register_model_invalid_question():
    """Rechaza preguntas no válidas."""

    with pytest.raises(ValueError, match="Pregunta no válida"):
        registry.register_model(
            question="p4",
            model_uri="models:/m-test",
            description="Modelo de prueba",
            data_version="test-version",
        )


def test_register_model_empty_uri():
    """Rechaza una URI vacía."""

    with pytest.raises(ValueError, match="Se requiere la URI"):
        registry.register_model(
            question="p1",
            model_uri="",
            description="Modelo de prueba",
            data_version="test-version",
        )


# ============================================================
# 3. Registro de modelos simulados
# ============================================================

@patch.object(registry, "get_commit", return_value="abc123")
@patch.object(registry, "configure_mlflow")
@patch.object(registry.mlflow, "register_model")
def test_register_model_p1(
    mock_register,
    mock_configure,
    mock_commit,
):
    """Comprueba el registro y metadatos de P1."""

    client = MagicMock()
    mock_configure.return_value = client

    mock_register.return_value.version = "1"
    mock_register.return_value.run_id = "run-test-001"

    result = registry.register_model(
        question="p1",
        model_uri="models:/m-test-p1",
        description="Modelo P1 de prueba",
        data_version="7e1470ac0632",
    )

    mock_register.assert_called_once_with(
        model_uri="models:/m-test-p1",
        name="mibici-equipo15-p1-demanda",
    )

    client.update_registered_model.assert_called_once()
    client.update_model_version.assert_called_once_with(
        name="mibici-equipo15-p1-demanda",
        version="1",
        description="Modelo P1 de prueba",
    )

    expected_tags = {
        "team": "15",
        "question": "P1",
        "experiment": "mibici-equipo15",
        "data_version": "7e1470ac0632",
        "registration_commit": "abc123",
        "stage_of_work": "candidate",
    }

    for key, value in expected_tags.items():
        client.set_model_version_tag.assert_any_call(
            name="mibici-equipo15-p1-demanda",
            version="1",
            key=key,
            value=value,
        )

    assert result.version == "1"
    assert result.run_id == "run-test-001"


# ============================================================
# 4. Pruebas de promoción a champion
# ============================================================

@patch.object(registry, "configure_mlflow")
def test_promote_p1_to_champion(mock_configure):
    """Verifica la asignación del alias champion."""

    client = MagicMock()
    mock_configure.return_value = client

    registry.promote_to_champion(
        question="p1",
        version="2",
    )

    client.set_registered_model_alias.assert_called_once_with(
        name="mibici-equipo15-p1-demanda",
        alias="champion",
        version="2",
    )

    client.set_model_version_tag.assert_called_once_with(
        name="mibici-equipo15-p1-demanda",
        version="2",
        key="stage_of_work",
        value="production_selected",
    )


@pytest.mark.parametrize("question", ["p1", "p2", "p3"])
@patch.object(registry, "configure_mlflow")
def test_promote_all_questions(mock_configure, question):
    """Verifica que P1, P2 y P3 admiten champion."""

    client = MagicMock()
    mock_configure.return_value = client

    registry.promote_to_champion(
        question=question,
        version="1",
    )

    client.set_registered_model_alias.assert_called_once_with(
        name=REGISTERED_MODELS[question],
        alias="champion",
        version="1",
    )


def test_promote_invalid_question():
    """Rechaza promociones de preguntas inexistentes."""

    with pytest.raises(ValueError, match="Pregunta no válida"):
        registry.promote_to_champion(
            question="p4",
            version="1",
        )


# ============================================================
# 5. Verificar que registrar no asigna champion
# ============================================================

@patch.object(registry, "get_commit", return_value="abc123")
@patch.object(registry, "configure_mlflow")
@patch.object(registry.mlflow, "register_model")
def test_register_does_not_assign_champion(
    mock_register,
    mock_configure,
    mock_commit,
):
    """Registrar un candidato no debe promoverlo automáticamente."""

    client = MagicMock()
    mock_configure.return_value = client

    mock_register.return_value.version = "1"
    mock_register.return_value.run_id = "run-test"

    registry.register_model(
        question="p3",
        model_uri="models:/m-test-p3",
        description="Candidato P3",
        data_version="7e1470ac0632",
    )

    client.set_registered_model_alias.assert_not_called()


# ============================================================
# 6. Verificación del URI por alias
# ============================================================

@patch(
    "mibici.modeling.validate_registry.mlflow.pyfunc.load_model"
)
@patch(
    "mibici.modeling.validate_registry.MlflowClient"
)
@patch(
    "mibici.modeling.validate_registry.mlflow.set_registry_uri"
)
@patch(
    "mibici.modeling.validate_registry.mlflow.set_tracking_uri"
)
def test_load_champion_p1(
    mock_tracking,
    mock_registry_uri,
    mock_client_class,
    mock_load,
):
    """Verifica la carga de un champion mediante su URI."""

    from mibici.modeling.validate_registry import (
        validate_registered_model,
    )

    client = mock_client_class.return_value
    client.get_model_version_by_alias.return_value.version = "1"
    client.get_model_version_by_alias.return_value.run_id = "run-001"

    expected_model = MagicMock()
    mock_load.return_value = expected_model

    result = validate_registered_model("p1")

    client.get_model_version_by_alias.assert_called_once_with(
        name="mibici-equipo15-p1-demanda",
        alias="champion",
    )

    mock_load.assert_called_once_with(
        "models:/mibici-equipo15-p1-demanda@champion"
    )

    assert result is expected_model
