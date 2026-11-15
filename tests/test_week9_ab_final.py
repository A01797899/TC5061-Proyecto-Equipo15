"""Semana 9: pruebas del reporte A/B completo por estación.

Pruebas aisladas: no necesitan servidor MLflow ni datos reales de MiBici.
Verifican la exportación añadida a run_evaluation(mode='both').
"""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mibici.api.schemas import P1_FEATURES
from scripts.week9 import compare_models


class ModeloFalso:
    """Devuelve predicciones deterministas para el lote recibido."""

    def __init__(self, predicciones):
        self.predicciones = np.asarray(predicciones, dtype="float64")
        self.posicion = 0

    def predict(self, X):
        inicio = self.posicion
        fin = inicio + len(X)
        resultado = self.predicciones[inicio:fin]
        assert len(resultado) == len(X)
        self.posicion = fin
        return resultado


def _datos(stations, targets):
    n = len(stations)
    inicio = pd.Timestamp("2025-03-01 00:00")
    # Fechas sin duplicados por estación y suficientes para el conjunto de prueba.
    horas = [inicio + pd.Timedelta(hours=i) for i in range(n)]
    data = {feature: np.ones(n, dtype="float64") for feature in P1_FEATURES}
    data.update({"station": stations, "hour": horas, "y": targets})
    return pd.DataFrame(data)


def _ejecutar_ab(monkeypatch, tmp_path, df, champion, challenger):
    """Ejecuta el comparador real con modelos y Registry simulados."""
    assert len(df) == len(champion) == len(challenger)

    modelos = {
        "champion": (ModeloFalso(champion), "1"),
        "challenger": (ModeloFalso(challenger), "2"),
    }

    def fake_resolve_model(client, model_name, alias):
        model, version = modelos[alias]
        info = {
            "alias": alias,
            "version": version,
            "run_id": f"run-falso-{alias}",
            "model_uri": f"models:/{model_name}/{version}",
        }
        return model, info

    monkeypatch.setattr(compare_models, "load_evaluation", lambda _: df)
    monkeypatch.setattr(compare_models, "file_sha256", lambda _: "hash-de-prueba")
    monkeypatch.setattr(compare_models, "resolve_model", fake_resolve_model)
    monkeypatch.setattr(compare_models, "MlflowClient", lambda: object())
    monkeypatch.setattr(compare_models.mlflow, "set_tracking_uri", lambda _: None)
    monkeypatch.setattr(compare_models.mlflow, "set_registry_uri", lambda _: None)

    salida = tmp_path / "ab-final"
    result = compare_models.run_evaluation(
        dataset_path=tmp_path / "evaluacion-simulada.parquet",
        output_dir=salida,
        tracking_uri="http://servidor-falso",
        mode="both",
        batch_size=200,
    )
    return salida, result


def test_ab_final_exporta_metricas_correctas_por_estacion(monkeypatch, tmp_path):
    """Comprueba MAE, RMSE, observaciones y porcentajes por estación."""
    df = _datos([1, 1, 2, 2], [2.0, 4.0, 0.0, 2.0])
    salida, _ = _ejecutar_ab(
        monkeypatch, tmp_path, df,
        champion=[1.0, 2.0, 0.0, 1.0],
        challenger=[2.0, 3.0, 1.0, 2.0],
    )

    archivo = salida / "station_comparison_metrics.csv"
    assert archivo.is_file(), "Falta exportar station_comparison_metrics.csv"
    estaciones = pd.read_csv(archivo).set_index("station")
    assert set(estaciones.index) == {1, 2}
    assert estaciones.index.is_unique
    assert estaciones["observations"].to_dict() == {1: 2, 2: 2}

    uno = estaciones.loc[1]
    assert uno["champion_mae"] == pytest.approx(1.5)
    assert uno["challenger_mae"] == pytest.approx(0.5)
    assert uno["champion_rmse"] == pytest.approx(np.sqrt(2.5))
    assert uno["challenger_rmse"] == pytest.approx(np.sqrt(0.5))
    assert uno["mae_improvement_pct"] == pytest.approx(100 * (1.5 - 0.5) / 1.5)
    assert uno["rmse_improvement_pct"] == pytest.approx(
        100 * (np.sqrt(2.5) - np.sqrt(0.5)) / np.sqrt(2.5)
    )

    dos = estaciones.loc[2]
    assert dos["champion_mae"] == pytest.approx(0.5)
    assert dos["challenger_mae"] == pytest.approx(0.5)
    assert dos["mae_improvement_pct"] == pytest.approx(0.0)


def test_ab_final_porcentaje_indefinido_si_champion_sin_error(monkeypatch, tmp_path):
    """Si Champion tiene error cero, su mejora porcentual es indefinida."""
    df = _datos([10, 10, 11, 11], [1.0, 3.0, 1.0, 3.0])
    salida, _ = _ejecutar_ab(
        monkeypatch, tmp_path, df,
        champion=[1.0, 3.0, 0.0, 2.0],
        challenger=[0.0, 3.0, 1.0, 3.0],
    )
    metrics = pd.read_csv(salida / "station_comparison_metrics.csv").set_index("station")
    assert metrics.loc[10, "champion_mae"] == 0.0
    assert pd.isna(metrics.loc[10, "mae_improvement_pct"])
    assert pd.isna(metrics.loc[10, "rmse_improvement_pct"])
    assert metrics.loc[11, "mae_improvement_pct"] == pytest.approx(100.0)
    assert metrics.loc[11, "rmse_improvement_pct"] == pytest.approx(100.0)


def test_ab_final_usa_todo_dataset_no_muestra_1000(monkeypatch, tmp_path):
    """El CSV por estación debe usar todas las filas, no predictions_sample.csv."""
    n = 1005
    # 1000 primeras filas: error cero; las cinco restantes: error cuatro.
    df = _datos([10] * n, [4.0] * n)
    champion = [4.0] * 1000 + [0.0] * 5
    challenger = [4.0] * n
    salida, resultados = _ejecutar_ab(
        monkeypatch, tmp_path, df, champion, challenger,
    )
    estaciones = pd.read_csv(salida / "station_comparison_metrics.csv")
    muestra = pd.read_csv(salida / "predictions_sample.csv")

    assert len(muestra) == 1000
    assert len(estaciones) == 1
    assert int(estaciones.loc[0, "observations"]) == n
    assert estaciones.loc[0, "champion_mae"] == pytest.approx(20 / n)
    assert estaciones.loc[0, "champion_rmse"] == pytest.approx(np.sqrt(80 / n))
    assert estaciones.loc[0, "challenger_mae"] == pytest.approx(0.0)
    assert resultados["champion"]["mae"] == pytest.approx(20 / n)


def test_ab_final_conserva_exportaciones_y_metricas_globales(monkeypatch, tmp_path):
    """Añadir el CSV de estaciones no debe alterar las salidas anteriores."""
    df = _datos([1, 1, 2, 2], [2.0, 4.0, 0.0, 2.0])
    salida, resultado = _ejecutar_ab(
        monkeypatch, tmp_path, df,
        champion=[1.0, 2.0, 0.0, 1.0],
        challenger=[2.0, 3.0, 1.0, 2.0],
    )
    for nombre in [
        "comparison_metrics.json",
        "daily_metrics.csv",
        "predictions_sample.csv",
        "station_comparison_metrics.csv",
    ]:
        assert (salida / nombre).is_file(), nombre

    with (salida / "comparison_metrics.json").open(encoding="utf-8") as f:
        guardado = json.load(f)
    estaciones = pd.read_csv(salida / "station_comparison_metrics.csv")
    diario = pd.read_csv(salida / "daily_metrics.csv")

    assert guardado["evaluation_rows"] == len(df)
    assert guardado["champion"]["mae"] == pytest.approx(resultado["champion"]["mae"])
    assert guardado["challenger"]["rmse"] == pytest.approx(resultado["challenger"]["rmse"])
    assert estaciones["observations"].sum() == len(df)
    assert diario["observations"].sum() == len(df)
