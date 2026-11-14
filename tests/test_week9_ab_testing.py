
"""Pruebas de Semana 9 para evaluación A/B P1."""

import numpy as np
import pandas as pd
import pytest
from scripts.week9.compare_models import (
    calculate_metrics,
    compare_predictions,
    improvement_percentage,
    load_evaluation,
    predict_in_batches,
    save_champion_predictions,
)


def test_mae_rmse_perfectos():
    y = np.array([1.0, 2.0, 3.0])

    result = calculate_metrics(y, y)

    assert result["mae"] == 0.0
    assert result["rmse"] == 0.0


def test_mae_rmse_con_errores():
    actual = np.array([1.0, 2.0, 3.0])
    predicted = np.array([2.0, 2.0, 1.0])

    result = calculate_metrics(
        actual, predicted
    )

    assert result["mae"] == pytest.approx(1.0)
    assert result["rmse"] == pytest.approx(
        np.sqrt(5.0 / 3.0)
    )


def test_mejora_porcentual():
    result = improvement_percentage(
        1.0, 0.9
    )

    assert result == pytest.approx(10.0)


def test_degradacion_porcentual():
    result = improvement_percentage(
        1.0, 1.2
    )

    assert result == pytest.approx(-20.0)


def test_error_champion_cero():
    assert improvement_percentage(0.0, 0.0) is None


def test_comparacion_challenger_mejor():
    actual = np.array([2.0, 4.0, 6.0])
    champion = np.array([1.0, 3.0, 5.0])
    challenger = np.array([2.0, 4.0, 6.0])

    result = compare_predictions(
        actual,
        champion,
        challenger,
    )

    assert result["champion"]["mae"] == 1.0
    assert result["challenger"]["mae"] == 0.0
    assert result["improvement_pct"]["mae"] == 100.0


def test_rechazo_dimensiones_incorrectas():
    with pytest.raises(ValueError):
        calculate_metrics(
            [1.0, 2.0],
            [1.0],
        )


def test_rechazo_arreglo_vacio():
    with pytest.raises(ValueError):
        calculate_metrics([], [])


def test_rechazo_valores_no_finitos():
    with pytest.raises(ValueError):
        calculate_metrics(
            [1.0, np.nan],
            [1.0, 2.0],
        )


class FakeModel:
    def predict(self, X):
        return np.full(
            len(X),
            -1.0,
            dtype="float64",
        )


def test_recorte_predicciones_negativas():
    X = pd.DataFrame({
        "feature": [1.0, 2.0, 3.0]
    })

    result = predict_in_batches(
        FakeModel(),
        X,
        batch_size=2,
    )

    assert len(result) == 3
    assert np.all(result == 0.0)


def test_batch_size_invalido():
    X = pd.DataFrame({"feature": [1.0]})

    with pytest.raises(ValueError):
        predict_in_batches(
            FakeModel(),
            X,
            batch_size=0,
        )


def test_dataset_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_evaluation(
            tmp_path / "no_existe.parquet"
        )


def test_dataset_fuera_de_marzo(tmp_path):
    from mibici.api.schemas import P1_FEATURES

    data = {
        feature: [1.0]
        for feature in P1_FEATURES
    }

    data.update({
        "station": [1],
        "hour": [pd.Timestamp("2025-04-01")],
        "y": [2.0],
    })

    path = tmp_path / "evaluation.parquet"

    pd.DataFrame(data).to_parquet(
        path,
        index=False,
    )

    with pytest.raises(ValueError):
        load_evaluation(path)


# ============================================================
# Pruebas de integración MLflow con mocks
# ============================================================

from types import SimpleNamespace

from scripts.week9 import compare_models

from mibici.api.schemas import P1_FEATURES


class FakeSchema:
    """Simula la firma de entrada de MLflow."""

    def __init__(self, features):
        self.features = list(features)

    def input_names(self):
        return self.features


class FakeMLflowModel:
    """Simula un modelo pyfunc registrado."""

    def __init__(self, features=None, signature=True):
        if features is None:
            features = P1_FEATURES

        schema = (
            FakeSchema(features)
            if signature
            else None
        )

        self.metadata = SimpleNamespace(
            get_input_schema=lambda: schema
        )

    def predict(self, X):
        return np.full(
            len(X),
            2.0,
            dtype="float64",
        )


class FakeRegistryClient:
    """Simula la consulta de versiones del Registry."""

    def __init__(self, version="1", error=None):
        self.version = version
        self.error = error

    def get_model_version_by_alias(
        self,
        name,
        alias,
    ):
        if self.error is not None:
            raise self.error

        return SimpleNamespace(
            version=self.version,
            run_id="fake-run-id",
        )



def test_registry_resuelve_champion(monkeypatch):
    """Comprueba alias, versión y URI sin acceder a MLflow real."""

    loaded_uris = []

    def fake_load_model(uri):
        loaded_uris.append(uri)
        return FakeMLflowModel()

    # Sustituimos la referencia mlflow del módulo comparador.
    # De esta forma no se descargan artefactos reales.
    fake_mlflow = SimpleNamespace(
        pyfunc=SimpleNamespace(
            load_model=fake_load_model
        )
    )

    monkeypatch.setattr(
        compare_models,
        "mlflow",
        fake_mlflow,
    )

    client = FakeRegistryClient(version="1")

    model, info = compare_models.resolve_model(
        client,
        "mibici-equipo15-p1-demanda",
        "champion",
    )

    assert model is not None
    assert info["alias"] == "champion"
    assert info["version"] == "1"
    assert info["run_id"] == "fake-run-id"

    expected_uri = (
        "models:/mibici-equipo15-p1-demanda/1"
    )

    assert info["model_uri"] == expected_uri

    assert loaded_uris == [expected_uri]




def test_registry_rechaza_firma_incompatible(monkeypatch):
    """Rechaza un Challenger con características incompatibles."""

    invalid_features = list(P1_FEATURES)[:-1]

    def fake_load_model(uri):
        return FakeMLflowModel(
            features=invalid_features
        )

    # Sustituimos MLflow exclusivamente dentro del comparador.
    fake_mlflow = SimpleNamespace(
        pyfunc=SimpleNamespace(
            load_model=fake_load_model
        )
    )

    monkeypatch.setattr(
        compare_models,
        "mlflow",
        fake_mlflow,
    )

    client = FakeRegistryClient(version="2")

    with pytest.raises(
        ValueError,
        match="no coincide con P1",
    ):
        compare_models.resolve_model(
            client,
            "mibici-equipo15-p1-demanda",
            "challenger",
        )



def test_registry_rechaza_modelo_sin_firma(monkeypatch):
    """Rechaza modelos sin una firma MLflow válida."""

    def fake_load_model(uri):
        return FakeMLflowModel(
            signature=False
        )

    fake_mlflow = SimpleNamespace(
        pyfunc=SimpleNamespace(
            load_model=fake_load_model
        )
    )

    monkeypatch.setattr(
        compare_models,
        "mlflow",
        fake_mlflow,
    )

    client = FakeRegistryClient(version="2")

    with pytest.raises(
        ValueError,
        match="no tiene firma",
    ):
        compare_models.resolve_model(
            client,
            "mibici-equipo15-p1-demanda",
            "challenger",
        )


def test_registry_maneja_alias_inexistente(
    monkeypatch,
):
    """Comprueba que un error del Registry no se oculta."""

    client = FakeRegistryClient(
        error=RuntimeError(
            "Alias challenger no encontrado"
        )
    )

    with pytest.raises(
        RuntimeError,
        match="Alias challenger no encontrado",
    ):
        compare_models.resolve_model(
            client,
            "mibici-equipo15-p1-demanda",
            "challenger",
        )


def test_registry_rechaza_misma_version(
    monkeypatch,
    tmp_path,
):
    """Evita una comparación A/B del modelo consigo mismo."""

    features = list(P1_FEATURES)

    dataset = pd.DataFrame({
        feature: [1.0]
        for feature in features
    })

    dataset["station"] = [1]
    dataset["hour"] = [
        pd.Timestamp("2025-03-01 08:00:00")
    ]
    dataset["y"] = [2.0]

    monkeypatch.setattr(
        compare_models,
        "load_evaluation",
        lambda path: dataset,
    )

    monkeypatch.setattr(
        compare_models,
        "file_sha256",
        lambda path: "fake-sha256",
    )

    monkeypatch.setattr(
        compare_models.mlflow,
        "set_tracking_uri",
        lambda uri: None,
    )

    monkeypatch.setattr(
        compare_models.mlflow,
        "set_registry_uri",
        lambda uri: None,
    )

    monkeypatch.setattr(
        compare_models,
        "MlflowClient",
        lambda: FakeRegistryClient(version="1"),
    )

    monkeypatch.setattr(
        compare_models.mlflow.pyfunc,
        "load_model",
        lambda uri: FakeMLflowModel(),
    )

    with pytest.raises(
        ValueError,
        match="son la misma versión",
    ):
        compare_models.run_evaluation(
            dataset_path=tmp_path / "fake.parquet",
            output_dir=tmp_path / "results",
            tracking_uri="http://127.0.0.1:5000",
            mode="both",
        )

def test_exportaciones_champion(tmp_path):
    """Comprueba los CSV y Parquet generados."""

    hours = pd.date_range(
        "2025-03-01",
        periods=4,
        freq="h",
    )

    df = pd.DataFrame({
        "station": [1, 1, 2, 2],
        "hour": hours,
        "y": [1.0, 3.0, 2.0, 4.0],
    })

    predictions = np.array([
        1.0, 2.0, 2.0, 5.0
    ])

    metadata = save_champion_predictions(
        df,
        predictions,
        tmp_path,
    )

    assert metadata["prediction_rows"] == 4
    assert metadata["days"] == 1
    assert metadata["stations"] == 2

    assert (
        tmp_path / "champion_predictions.parquet"
    ).exists()

    assert (
        tmp_path / "champion_daily_metrics.csv"
    ).exists()

    assert (
        tmp_path / "champion_station_metrics.csv"
    ).exists()


def test_mae_diario_exportado(tmp_path):
    """Comprueba el MAE y RMSE de los agregados diarios."""

    df = pd.DataFrame({
        "station": [1, 1, 2],
        "hour": pd.to_datetime([
            "2025-03-01 08:00",
            "2025-03-01 09:00",
            "2025-03-01 10:00",
        ]),
        "y": [1.0, 3.0, 5.0],
    })

    predictions = np.array([
        1.0, 2.0, 3.0
    ])

    save_champion_predictions(
        df,
        predictions,
        tmp_path,
    )

    daily = pd.read_csv(
        tmp_path / "champion_daily_metrics.csv"
    )

    # Errores absolutos: 0, 1 y 2 -> MAE = 1.0
    assert daily.loc[0, "champion_mae"] == pytest.approx(
        1.0
    )

    # Errores cuadrados: 0, 1 y 4 -> promedio = 5/3
    assert daily.loc[0, "champion_rmse"] == pytest.approx(
        np.sqrt(5.0 / 3.0)
    )


def test_exportacion_rechaza_predicciones_incompletas(tmp_path):
    """Rechaza predicciones con longitud incorrecta."""

    df = pd.DataFrame({
        "station": [1, 2],
        "hour": pd.to_datetime([
            "2025-03-01 08:00",
            "2025-03-01 09:00",
        ]),
        "y": [1.0, 2.0],
    })

    with pytest.raises(
        ValueError,
        match="no corresponden al dataset",
    ):
        save_champion_predictions(
            df,
            np.array([1.0]),
            tmp_path,
        )