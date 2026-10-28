
"""Pruebas de reproducibilidad de modelos MiBici - Semana 7."""

import numpy as np
import pytest

from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    r2_score,
)

from mibici.config import RANDOM_SEED
from mibici.modeling.p1_p2_candidates import crear_modelo


@pytest.fixture
def datos_regresion():
    """Genera datos deterministas para P1."""
    rng = np.random.default_rng(RANDOM_SEED)
    X = rng.normal(size=(120, 5))
    ruido = rng.normal(scale=0.1, size=120)

    y = (
        2.5 * X[:, 0]
        - 1.2 * X[:, 1]
        + 0.8 * X[:, 2]
        + ruido
    )

    return X[:90], X[90:], y[:90], y[90:]


@pytest.fixture
def datos_clasificacion():
    """Genera datos deterministas para P2."""
    rng = np.random.default_rng(RANDOM_SEED)
    X = rng.normal(size=(140, 5))
    y = (
        1.5 * X[:, 0]
        - X[:, 1]
        + 0.5 * X[:, 2]
        > 0
    ).astype(int)

    return X[:100], X[100:], y[:100], y[100:]


@pytest.mark.parametrize(
    "modelo, parametros",
    [
        ("rf", {"n_estimators": 15, "max_depth": 5}),
        ("hgb", {"max_iter": 20, "max_depth": 4}),
    ],
)
def test_reproducibilidad_p1(
    datos_regresion, modelo, parametros
):
    """Compara dos entrenamientos de regresión P1."""
    X_train, X_test, y_train, y_test = datos_regresion

    modelo_1 = crear_modelo("p1", modelo, parametros)
    modelo_2 = crear_modelo("p1", modelo, parametros)

    modelo_1.fit(X_train, y_train)
    modelo_2.fit(X_train, y_train)

    pred_1 = modelo_1.predict(X_test)
    pred_2 = modelo_2.predict(X_test)

    np.testing.assert_allclose(
        pred_1, pred_2, rtol=1e-10, atol=1e-10
    )

    mae_1 = mean_absolute_error(y_test, pred_1)
    mae_2 = mean_absolute_error(y_test, pred_2)

    r2_1 = r2_score(y_test, pred_1)
    r2_2 = r2_score(y_test, pred_2)

    assert mae_1 == pytest.approx(mae_2)
    assert r2_1 == pytest.approx(r2_2)


@pytest.mark.parametrize(
    "modelo, parametros",
    [
        ("rf", {"n_estimators": 15, "max_depth": 5}),
        ("hgb", {"max_iter": 20, "max_depth": 4}),
        ("logreg", {"C": 1.0, "max_iter": 200}),
    ],
)
def test_reproducibilidad_p2(
    datos_clasificacion, modelo, parametros
):
    """Compara dos entrenamientos de clasificación P2."""
    X_train, X_test, y_train, y_test = datos_clasificacion

    modelo_1 = crear_modelo("p2", modelo, parametros)
    modelo_2 = crear_modelo("p2", modelo, parametros)

    modelo_1.fit(X_train, y_train)
    modelo_2.fit(X_train, y_train)

    pred_1 = modelo_1.predict(X_test)
    pred_2 = modelo_2.predict(X_test)

    np.testing.assert_array_equal(pred_1, pred_2)

    accuracy_1 = accuracy_score(y_test, pred_1)
    accuracy_2 = accuracy_score(y_test, pred_2)

    assert accuracy_1 == pytest.approx(accuracy_2)

    prob_1 = modelo_1.predict_proba(X_test)[:, 1]
    prob_2 = modelo_2.predict_proba(X_test)[:, 1]

    np.testing.assert_allclose(
        prob_1, prob_2, rtol=1e-10, atol=1e-10
    )


def test_generacion_datos_misma_semilla():
    """La misma semilla debe producir idénticos datos."""
    rng_1 = np.random.default_rng(RANDOM_SEED)
    rng_2 = np.random.default_rng(RANDOM_SEED)

    X_1 = rng_1.normal(size=(30, 4))
    X_2 = rng_2.normal(size=(30, 4))

    np.testing.assert_array_equal(X_1, X_2)


def test_modelos_utilizan_semilla_configurada():
    """Los modelos respetan RANDOM_SEED de config.py."""
    estimador_p1 = crear_modelo(
        "p1", "rf", {"n_estimators": 10}
    )
    estimador_p2 = crear_modelo(
        "p2", "hgb", {"max_iter": 10}
    )

    assert estimador_p1.random_state == RANDOM_SEED
    assert estimador_p2.random_state == RANDOM_SEED
