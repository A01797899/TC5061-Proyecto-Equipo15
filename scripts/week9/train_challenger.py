"""Entrena el Challenger P1 con enero-febrero de 2025 y lo deja en MLflow.

No abre el conjunto de evaluación de marzo. La firma y los hiperparámetros
siguen al Champion registrado en la versión 1.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import skops.io as sio

from mibici.api.schemas import P1_FEATURES
from mibici.config import MLFLOW_EXPERIMENT, RANDOM_SEED

FEATURES = list(P1_FEATURES)
TRAIN_PERIOD = "2025-01-01/2025-02-28"
EVALUATION_START = pd.Timestamp("2025-03-01")
MAX_ITER = 300
LEARNING_RATE = 0.1
MAX_DEPTH = None
EARLY_STOPPING = False


def rechazar_conjunto_reservado(ruta: Path) -> None:
    """Impide ajustar el modelo con el Parquet de marzo."""
    if "evaluation" in ruta.as_posix():
        raise ValueError(
            "El Challenger no se entrena con el conjunto reservado de marzo."
        )


def cargar_entrenamiento(ruta: Path) -> pd.DataFrame:
    """Lee el Parquet de enero-febrero y comprueba el contrato de P1."""
    rechazar_conjunto_reservado(ruta)
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe el dataset de entrenamiento: {ruta}")

    datos = pd.read_parquet(ruta)
    requeridas = set(FEATURES) | {"y", "hour"}
    faltantes = requeridas - set(datos.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas: {sorted(faltantes)}")
    if datos.empty:
        raise ValueError("El dataset de entrenamiento está vacío.")

    datos = datos.copy()
    datos["hour"] = pd.to_datetime(datos["hour"], errors="raise")
    hora_maxima = datos["hour"].max()
    if hora_maxima >= EVALUATION_START:
        raise ValueError(
            f"La hora máxima {hora_maxima} entra en marzo. El fit se detiene."
        )

    valores = datos[FEATURES + ["y"]].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(valores.to_numpy(dtype="float64")).all():
        raise ValueError("El entrenamiento contiene NaN o infinitos.")
    datos[FEATURES + ["y"]] = valores
    return datos


def crear_challenger() -> HistGradientBoostingRegressor:
    """Instancia el mismo HistGradientBoosting del Champion."""
    return HistGradientBoostingRegressor(
        max_iter=MAX_ITER,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        early_stopping=EARLY_STOPPING,
        random_state=RANDOM_SEED,
    )


def entrenar(ruta: Path, tracking_uri: str) -> str:
    """Ajusta el Challenger, registra la corrida y devuelve el run_id."""
    datos = cargar_entrenamiento(ruta)
    caracteristicas = datos[FEATURES].astype("float64")
    objetivo = datos["y"].to_numpy(dtype="float64")
    hora_maxima = datos["hour"].max()

    modelo = crear_challenger()
    modelo.fit(caracteristicas, objetivo)
    prediccion = modelo.predict(caracteristicas)
    mae = float(mean_absolute_error(objetivo, prediccion))
    rmse = float(root_mean_squared_error(objetivo, prediccion))

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name="P1_challenger_2025_01_02") as run:
        mlflow.log_params(
            {
                "max_iter": MAX_ITER,
                "learning_rate": LEARNING_RATE,
                "max_depth": str(MAX_DEPTH),
                "early_stopping": EARLY_STOPPING,
                "random_state": RANDOM_SEED,
                "train_rows": len(datos),
                "train_period": TRAIN_PERIOD,
                "features": ",".join(FEATURES),
            }
        )
        mlflow.log_metrics({"train_mae": mae, "train_rmse": rmse})
        mlflow.set_tags(
            {
                "team": "15",
                "question": "P1",
                "stage_of_work": "candidate",
                "train_period": TRAIN_PERIOD,
            }
        )
        ejemplo = caracteristicas.head(5)
        firma = infer_signature(ejemplo, modelo.predict(ejemplo))
        confiables = sio.get_untrusted_types(data=sio.dumps(modelo))
        mlflow.sklearn.log_model(
            modelo,
            name="model",
            signature=firma,
            input_example=ejemplo,
            skops_trusted_types=confiables,
        )
        run_id = run.info.run_id

    print(f"filas {len(datos)}")
    print(f"hora_maxima {hora_maxima}")
    print(f"max_iter {MAX_ITER}")
    print(f"learning_rate {LEARNING_RATE}")
    print(f"max_depth {MAX_DEPTH}")
    print(f"early_stopping {EARLY_STOPPING}")
    print(f"random_state {RANDOM_SEED}")
    print(f"train_mae {mae}")
    print(f"train_rmse {rmse}")
    print(f"run_id {run_id}")
    return run_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena el Challenger P1 de enero-febrero 2025.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--tracking-uri", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entrenar(args.data, args.tracking_uri)


if __name__ == "__main__":
    main()
