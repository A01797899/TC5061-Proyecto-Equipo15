"""Pruebas del servicio de inferencia P1."""

from unittest.mock import MagicMock

import numpy as np
import pytest
from mlflow.types import ColSpec, Schema

from mibici.api import model_service
from mibici.api.model_service import (
    ModelInputError,
    ModelNotReadyError,
    ModelPredictionError,
    ModelService,
)

COLUMNS = [
    "hour_of_day",
    "weekday",
    "fin_semana",
    "salidas_lag_1h",
    "salidas_lag_24h",
    "salidas_lag_168h",
    "media_estacion",
    "media_estacion_dia_hora",
]

RECORD = {
    "hour_of_day": 8,
    "weekday": 2,
    "fin_semana": 0,
    "salidas_lag_1h": 3,
    "salidas_lag_24h": 4,
    "salidas_lag_168h": 5,
    "media_estacion": 2.1,
    "media_estacion_dia_hora": 3.4,
}


@pytest.fixture
def fake_model():
    model = MagicMock()
    model.metadata.get_input_schema.return_value = Schema(
        [ColSpec("double", name) for name in COLUMNS]
    )
    model.predict.side_effect = lambda frame: np.full(len(frame), 3.5)
    return model


@pytest.fixture
def registry(monkeypatch, fake_model):
    client = MagicMock()
    client.get_model_version_by_alias.return_value.version = "1"
    loader = MagicMock(return_value=fake_model)

    monkeypatch.setattr(model_service, "MlflowClient", lambda: client)
    monkeypatch.setattr("mlflow.pyfunc.load_model", loader)
    monkeypatch.setattr(model_service.mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(model_service.mlflow, "set_registry_uri", lambda uri: None)
    return client, loader


@pytest.fixture
def service(registry):
    instance = ModelService()
    instance.load_model()
    return instance


def test_instancia_sin_argumentos_no_carga():
    instance = ModelService()
    assert not instance.is_ready()
    assert instance.model_name == "mibici-equipo15-p1-demanda"
    assert instance.model_alias == "champion"


def test_carga_resuelve_alias_y_version(service, registry):
    client, loader = registry
    client.get_model_version_by_alias.assert_called_once_with(
        "mibici-equipo15-p1-demanda", "champion"
    )
    loader.assert_called_once_with("models:/mibici-equipo15-p1-demanda/1")
    assert service.is_ready()
    assert service.model_version == "1"
    assert service.input_columns == COLUMNS


def test_registry_no_disponible(monkeypatch, registry):
    client, _ = registry
    client.get_model_version_by_alias.side_effect = Exception("connection refused")
    instance = ModelService()
    with pytest.raises(ModelNotReadyError, match="No se pudo cargar") as info:
        instance.load_model()
    assert info.value.__cause__ is not None
    assert not instance.is_ready()


def test_modelo_sin_firma(registry, fake_model):
    fake_model.metadata.get_input_schema.return_value = None
    with pytest.raises(ModelNotReadyError, match="firma"):
        ModelService().load_model()


def test_prediccion_valida(service):
    assert service.predict([RECORD]) == [3.5]


def test_dataframe_en_orden_y_float64(service, fake_model):
    reordered = dict(reversed(list(RECORD.items())))
    service.predict([reordered])
    frame = fake_model.predict.call_args.args[0]
    assert list(frame.columns) == COLUMNS
    assert (frame.dtypes == "float64").all()


def test_varios_registros(service):
    assert service.predict([RECORD, RECORD, RECORD]) == [3.5, 3.5, 3.5]


def test_recorta_negativos_a_cero(service, fake_model):
    fake_model.predict.side_effect = lambda frame: np.array([-0.4])
    assert service.predict([RECORD]) == [0.0]


def test_salida_tipos_python(service):
    result = service.predict([RECORD])
    assert type(result) is list
    assert type(result[0]) is float


def test_sin_modelo_cargado():
    with pytest.raises(ModelNotReadyError):
        ModelService().predict([RECORD])


def test_lista_vacia(service):
    with pytest.raises(ModelInputError, match="al menos un registro"):
        service.predict([])


def test_registro_no_es_objeto(service):
    with pytest.raises(ModelInputError, match="no es un objeto"):
        service.predict([[1, 2, 3]])


def test_columna_faltante(service):
    record = {k: v for k, v in RECORD.items() if k != "weekday"}
    with pytest.raises(ModelInputError, match="Faltan columnas"):
        service.predict([record])


def test_columna_extra(service):
    with pytest.raises(ModelInputError, match="no esperadas"):
        service.predict([{**RECORD, "estacion_id": 10}])


@pytest.mark.parametrize("value", ["8", None, True, [1]])
def test_valor_no_numerico(service, value):
    with pytest.raises(ModelInputError, match="no numérico"):
        service.predict([{**RECORD, "hour_of_day": value}])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_valor_no_finito(service, value):
    with pytest.raises(ModelInputError, match="NaN o infinito"):
        service.predict([{**RECORD, "media_estacion": value}])


def test_falla_del_modelo_conserva_causa(service, fake_model):
    fake_model.predict.side_effect = RuntimeError("boom")
    with pytest.raises(ModelPredictionError, match="inferencia") as info:
        service.predict([RECORD])
    assert isinstance(info.value.__cause__, RuntimeError)


def test_cantidad_de_predicciones(service, fake_model):
    fake_model.predict.side_effect = lambda frame: np.array([1.0, 2.0])
    with pytest.raises(ModelPredictionError, match="Se esperaban 1"):
        service.predict([RECORD])


def test_salida_no_finita(service, fake_model):
    fake_model.predict.side_effect = lambda frame: np.array([np.nan])
    with pytest.raises(ModelPredictionError, match="NaN o infinitos"):
        service.predict([RECORD])


def test_excepciones_compatibles_con_main():
    # main.py de la API captura ValueError como error de entrada.
    assert issubclass(ModelInputError, ValueError)
    assert issubclass(ModelNotReadyError, RuntimeError)
    assert issubclass(ModelPredictionError, RuntimeError)


def test_registry_local_con_modelo_real(tmp_path, monkeypatch):
    """Registra un modelo sklearn pequeño en un Registry temporal y lo sirve."""

    import mlflow
    import pandas as pd
    from mlflow import MlflowClient
    from mlflow.models import infer_signature
    from sklearn.linear_model import LinearRegression

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    experiment_id = mlflow.create_experiment(
        "prueba-servicio", artifact_location=(tmp_path / "artifacts").as_uri()
    )

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((50, len(COLUMNS))), columns=COLUMNS)
    y = X.sum(axis=1)
    estimator = LinearRegression().fit(X, y)

    with mlflow.start_run(experiment_id=experiment_id):
        info = mlflow.sklearn.log_model(
            estimator,
            name="model",
            signature=infer_signature(X, estimator.predict(X)),
            registered_model_name="mibici-equipo15-p1-demanda",
        )

    MlflowClient().set_registered_model_alias(
        "mibici-equipo15-p1-demanda", "champion", info.registered_model_version
    )

    instance = ModelService(tracking_uri=uri)
    instance.load_model()

    # Enteros como los que envía un cliente JSON; la firma exige double.
    result = instance.predict([RECORD])
    expected = float(estimator.predict(pd.DataFrame([RECORD], columns=COLUMNS, dtype=float))[0])

    assert instance.model_version == "1"
    assert result == pytest.approx([max(expected, 0.0)])
