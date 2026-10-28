
"""Pruebas unitarias de entrenamiento de MiBici - Semana 7."""

import sys
from unittest.mock import Mock

import numpy as np
import pytest

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from mibici.config import RANDOM_SEED
from mibici.modeling import p1_p2_candidates as candidates
from mibici.modeling import train


# 1. Pruebas de hiperparametros

@pytest.mark.parametrize(
    "modelo, clave",
    [
        ("rf", "n_estimators"),
        ("hgb", "max_iter"),
        ("logreg", "C"),
    ],
)
def test_hiperparametros_predeterminados(modelo, clave):
    hp = candidates.resolver_hiperparametros(modelo)
    assert isinstance(hp, dict)
    assert clave in hp
    assert hp[clave] > 0


def test_hiperparametros_personalizados():
    hp = candidates.resolver_hiperparametros(
        "rf", n_estimators=25, max_depth=5
    )
    assert hp["n_estimators"] == 25
    assert hp["max_depth"] == 5


def test_hiperparametro_cero_conserva_default():
    original = candidates.resolver_hiperparametros("rf")
    resultado = candidates.resolver_hiperparametros(
        "rf", n_estimators=0
    )
    assert resultado["n_estimators"] == original["n_estimators"]


def test_modelo_sin_configuracion_falla():
    with pytest.raises(KeyError):
        candidates.resolver_hiperparametros("inexistente")


# 2. Construccion de modelos

@pytest.mark.parametrize(
    "pregunta,modelo,parametros,tipo",
    [
        ("p1", "rf", {"n_estimators": 10}, RandomForestRegressor),
        ("p2", "rf", {"n_estimators": 10}, RandomForestClassifier),
        ("p1", "hgb", {"max_iter": 10}, HistGradientBoostingRegressor),
        ("p2", "hgb", {"max_iter": 10}, HistGradientBoostingClassifier),
    ],
)
def test_crear_modelo_tipo_correcto(
    pregunta, modelo, parametros, tipo
):
    estimador = candidates.crear_modelo(pregunta, modelo, parametros)
    assert isinstance(estimador, tipo)
    assert estimador.random_state == RANDOM_SEED


def test_crear_regresion_logistica_p2():
    modelo = candidates.crear_modelo(
        "p2", "logreg", {"C": 1.0, "max_iter": 200}
    )
    assert isinstance(modelo, Pipeline)
    estimador = modelo.steps[-1][1]
    assert isinstance(estimador, LogisticRegression)
    assert estimador.class_weight == "balanced"
    assert estimador.random_state == RANDOM_SEED


@pytest.mark.parametrize(
    "pregunta,modelo",
    [
        ("p1", "logreg"),
        ("p1", "modelo_desconocido"),
        ("p2", "modelo_desconocido"),
    ],
)
def test_modelos_no_soportados(pregunta, modelo):
    with pytest.raises(ValueError, match="no soportado"):
        candidates.crear_modelo(pregunta, modelo, {})


# 3. Entrenamiento y predicciones

@pytest.mark.parametrize("pregunta", ["p1", "p2"])
def test_random_forest_entrena_y_predice(pregunta):
    rng = np.random.default_rng(RANDOM_SEED)
    X = rng.normal(size=(60, 4))

    if pregunta == "p1":
        y = 3 * X[:, 0] - X[:, 1] + 2
    else:
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

    modelo = candidates.crear_modelo(
        pregunta, "rf", {"n_estimators": 12}
    )
    modelo.fit(X, y)
    predicciones = modelo.predict(X)

    assert len(predicciones) == len(y)
    assert np.isfinite(predicciones).all()

    if pregunta == "p2":
        assert set(np.unique(predicciones)).issubset({0, 1})


# 4. Metricas de evaluacion

def test_mejor_umbral_maximiza_f1():
    y = np.array([0, 0, 1, 1])
    prob = np.array([0.1, 0.3, 0.7, 0.9])
    umbrales = np.array([0.2, 0.5, 0.8])

    resultado = candidates.mejor_umbral(y, prob, umbrales)
    assert resultado == pytest.approx(0.5)


@pytest.mark.parametrize(
    "baseline,candidato,menor_es_mejor,esperado",
    [
        (10.0, 8.0, True, 20.0),
        (0.5, 0.6, False, 20.0),
        (10.0, 12.0, True, -20.0),
        (0.0, 5.0, True, 0.0),
    ],
)
def test_mejora_porcentual(
    baseline, candidato, menor_es_mejor, esperado
):
    resultado = candidates.mejora_pct(
        baseline, candidato, menor_es_mejor
    )
    assert resultado == pytest.approx(esperado)


# 5. Orquestacion de modelos

@pytest.mark.parametrize(
    "pregunta,modulo",
    [
        ("p1", "mibici.modeling.p1_baseline"),
        ("p2", "mibici.modeling.p2_baseline"),
        ("p3", "mibici.modeling.p3_baseline"),
    ],
)
def test_run_baseline_selecciona_modulo(
    monkeypatch, pregunta, modulo
):
    ejecutar = Mock()
    monkeypatch.setattr(train.subprocess, "run", ejecutar)

    train.run_baseline(pregunta, "http://localhost:5000")

    ejecutar.assert_called_once()
    comando = ejecutar.call_args.args[0]
    opciones = ejecutar.call_args.kwargs

    assert comando[:3] == [sys.executable, "-m", modulo]
    assert opciones["check"] is True
    assert opciones["env"]["MLFLOW_TRACKING_URI"] == (
        "http://localhost:5000"
    )
    assert "MLFLOW_RUN_ID" not in opciones["env"]


@pytest.mark.parametrize(
    "pregunta,modulo",
    [
        ("p1", "mibici.modeling.p1_p2_search"),
        ("p2", "mibici.modeling.p1_p2_search"),
        ("p3", "mibici.modeling.p3_search"),
    ],
)
def test_run_search_selecciona_modulo(
    monkeypatch, pregunta, modulo
):
    ejecutar = Mock()
    monkeypatch.setattr(train.subprocess, "run", ejecutar)

    train.run_search(pregunta, "http://localhost:5000")

    ejecutar.assert_called_once()
    comando = ejecutar.call_args.args[0]

    assert comando[:3] == [sys.executable, "-m", modulo]


def test_run_search_rechaza_pregunta_invalida(monkeypatch):
    ejecutar = Mock()
    monkeypatch.setattr(train.subprocess, "run", ejecutar)

    with pytest.raises(NotImplementedError):
        train.run_search("p4", "http://localhost:5000")

    ejecutar.assert_not_called()
