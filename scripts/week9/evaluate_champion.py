
"""Semana 9: evaluación del Champion P1 sobre marzo de 2025."""

import json
import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import mean_absolute_error, mean_squared_error

from mibici.api.schemas import P1_FEATURES
from mibici.config import REGISTERED_MODELS


TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)

DATASET = Path(
    "data/processed/week9/evaluation.parquet"
)

OUTPUT_DIR = Path(
    "reports/week9/ab_testing"
)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(TRACKING_URI)

    print("MLflow:", TRACKING_URI)

    client = MlflowClient()
    model_name = REGISTERED_MODELS["p1"]

    version = client.get_model_version_by_alias(
        model_name,
        "champion",
    )

    model_uri = (
        f"models:/{model_name}/{version.version}"
    )

    print("Modelo:", model_name)
    print("Versión:", version.version)
    print("Run ID:", version.run_id)
    print("Model URI:", model_uri)

    model = mlflow.pyfunc.load_model(model_uri)

    df = pd.read_parquet(DATASET)

    features = list(P1_FEATURES)
    X = df[features].astype("float64")
    y = df["y"].to_numpy(dtype="float64")

    if df.empty:
        raise ValueError("Dataset de evaluación vacío")

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Características no finitas")

    if not np.isfinite(y).all():
        raise ValueError("Etiquetas no finitas")

    print("Observaciones:", len(df))
    print("Estaciones:", df["station"].nunique())
    print("Inicio:", df["hour"].min())
    print("Fin:", df["hour"].max())

    predictions = []

    for start in range(0, len(X), 10000):
        batch = X.iloc[start:start + 10000]

        pred = np.asarray(
            model.predict(batch),
            dtype="float64",
        )

        if pred.shape != (len(batch),):
            raise ValueError("Forma de predicción incorrecta")

        if not np.isfinite(pred).all():
            raise ValueError("Predicciones no finitas")

        predictions.append(
            np.clip(pred, 0.0, None)
        )

    y_pred = np.concatenate(predictions)

    metrics = {
        "model": model_name,
        "model_version": str(version.version),
        "run_id": version.run_id,
        "model_uri": model_uri,
        "evaluation_period": "2025-03-01/2025-03-31",
        "observations": int(len(df)),
        "stations": int(df["station"].nunique()),
        "mae": float(mean_absolute_error(y, y_pred)),
        "rmse": float(
            np.sqrt(mean_squared_error(y, y_pred))
        ),
    }

    output_file = OUTPUT_DIR / "champion_march_2025.json"

    output_file.write_text(
        json.dumps(
            metrics,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nRESULTADOS DEL CHAMPION")
    print(json.dumps(metrics, indent=2))
    print("\nArchivo generado:", output_file)


if __name__ == "__main__":
    main()
