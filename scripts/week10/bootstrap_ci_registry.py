#!/usr/bin/env python3
"""Inicializa un Registry temporal de MLflow para CI (nunca el de producción).

Ejecutar desde la imagen API, conectada a MLflow mediante Compose:
    docker compose run --rm --no-deps api python scripts/week10/bootstrap_ci_registry.py

El modelo es sintético, determinista y respeta el contrato de 8 variables P1.
No se entrena ni pretende reemplazar el Champion real del proyecto.
"""

from __future__ import annotations

import os

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature

MODEL_NAME = "mibici-equipo15-p1-demanda"
ALIAS = "champion"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

FEATURES = (
    "hour_of_day",
    "weekday",
    "fin_semana",
    "salidas_lag_1h",
    "salidas_lag_24h",
    "salidas_lag_168h",
    "media_estacion",
    "media_estacion_dia_hora",
)

EXAMPLE = {
    "hour_of_day": 8.0,
    "weekday": 2.0,
    "fin_semana": 0.0,
    "salidas_lag_1h": 3.0,
    "salidas_lag_24h": 4.0,
    "salidas_lag_168h": 5.0,
    "media_estacion": 2.1,
    "media_estacion_dia_hora": 3.4,
}


class DeterministicP1(mlflow.pyfunc.PythonModel):
    """Modelo de prueba que devuelve una demanda no negativa reproducible."""

    def predict(self, context, model_input: pd.DataFrame, params=None):
        # Usa columnas del contrato P1; devuelve un vector de float64.
        prediction = (
            0.4 * model_input["salidas_lag_1h"]
            + 0.2 * model_input["media_estacion_dia_hora"]
            + 0.1 * model_input["hour_of_day"]
            + 0.05 * model_input["media_estacion"]
        )
        return np.maximum(0.0, prediction.to_numpy(dtype=np.float64))


def main() -> None:
    if os.getenv("MIBICI_CI_BOOTSTRAP") != "1":
        raise SystemExit(
            "Protección: solo se permite registrar el modelo de prueba con "
            "MIBICI_CI_BOOTSTRAP=1. No ejecutar contra el Registry real."
        )
    if TRACKING_URI.rstrip("/") != "http://mlflow:5000":
        raise SystemExit(
            "Protección: el bootstrap de CI requiere MLFLOW_TRACKING_URI=http://mlflow:5000."
        )

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(TRACKING_URI)
    client = MlflowClient()

    # En CI usamos un Registry efímero. Evitamos sobrescribir un modelo existente.
    try:
        client.get_registered_model(MODEL_NAME)
    except mlflow.exceptions.RestException as exc:
        if getattr(exc, "error_code", None) != "RESOURCE_DOES_NOT_EXIST":
            raise
    else:
        raise SystemExit(
            f"Protección: ya existe {MODEL_NAME}; no se modifica un Registry existente."
        )

    sample = pd.DataFrame([EXAMPLE], columns=FEATURES, dtype="float64")
    model = DeterministicP1()
    expected = float(model.predict(None, sample)[0])
    signature = infer_signature(sample, model.predict(None, sample))

    mlflow.set_experiment("mibici-equipo15-ci-week10")
    with mlflow.start_run(run_name="bootstrap_ci_p1_deterministic"):
        info = mlflow.pyfunc.log_model(
            name="p1_ci_deterministic",
            python_model=model,
            signature=signature,
            input_example=sample,
        )
        version = mlflow.register_model(info.model_uri, MODEL_NAME)

    if version.status != "READY":
        version = client.get_model_version(MODEL_NAME, version.version)
    if version.status != "READY":
        raise RuntimeError(f"Modelo de CI no READY: {version.status}")

    client.set_registered_model_alias(MODEL_NAME, ALIAS, version.version)

    # Prueba de lectura del Registry y carga a través de MLflow (no solo del objeto en memoria).
    stored = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{ALIAS}")
    actual = float(stored.predict(sample)[0])
    if not np.isclose(actual, expected, rtol=0.0, atol=1e-10):
        raise RuntimeError(f"Predicción CI inesperada: {actual} != {expected}")

    print(f"Registry temporal inicializado: {MODEL_NAME}@{ALIAS}")
    print(f"Version: {version.version}; status: {version.status}")
    print(f"Columnas P1: {', '.join(FEATURES)}")
    print(f"Predicción esperada CI: {expected:.15f}")
    print("BOOTSTRAP CI COMPLETADO - OK")


if __name__ == "__main__":
    main()
