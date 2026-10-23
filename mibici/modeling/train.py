
"""Punto de entrada de entrenamiento del MLflow Project MiBici."""

import argparse
import os
import subprocess
import sys

from mibici.config import (
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    RANDOM_SEED,
    UMBRAL_RIESGO_CAPACIDAD,
    DEFAULT_N_ESTIMATORS,
    P3_MIN_DEPARTURES,
    P3_K_MIN,
    P3_K_MAX,
)


BASELINE_MODULES = {
    "p1": "mibici.modeling.p1_baseline",
    "p2": "mibici.modeling.p2_baseline",
    "p3": "mibici.modeling.p3_baseline",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="MiBici MLflow Project - Entrenamiento"
    )

    parser.add_argument(
        "--question",
        choices=["p1", "p2", "p3"],
        required=True,
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Modelo solicitado: baseline, rf, etc.",
    )

    parser.add_argument(
        "--n-estimators",
        type=int,
        default=0,
        help="0 utiliza el valor predeterminado de config.py",
    )

    return parser.parse_args()


def run_baseline(question, tracking_uri):
    module = BASELINE_MODULES[question]

    command = [
        sys.executable,
        "-m",
        module,
        "--tracking-uri",
        tracking_uri,
    ]

    if question == "p2":
        command.extend([
            "--threshold",
            str(UMBRAL_RIESGO_CAPACIDAD),
        ])

    if question == "p3":
        command.extend([
            "--experiment",
            MLFLOW_EXPERIMENT,
            "--seed",
            str(RANDOM_SEED),
            "--min-departures",
            str(P3_MIN_DEPARTURES),
            "--k-min",
            str(P3_K_MIN),
            "--k-max",
            str(P3_K_MAX),
        ])

    print(f"Executing baseline: {question}", flush=True)

    # Copiar las variables de entorno del proceso actual.
    env = os.environ.copy()

    # Evitar reutilizar el Run ID creado por MLflow Project.
    env.pop("MLFLOW_RUN_ID", None)

    # Mantener la conexión al servidor central de MLflow.
    env["MLFLOW_TRACKING_URI"] = tracking_uri

    # Ejecutar el baseline como un proceso independiente.
    subprocess.run(
        command,
        check=True,
        env=env,
    )



def main():
    args = parse_args()

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        MLFLOW_TRACKING_URI,
    )

    n_estimators = (
        args.n_estimators
        if args.n_estimators > 0
        else DEFAULT_N_ESTIMATORS
    )

    print(f"Question: {args.question}", flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Tracking URI: {tracking_uri}", flush=True)
    print(f"n_estimators: {n_estimators}", flush=True)

    if args.model == "baseline":
        run_baseline(args.question, tracking_uri)
        return

    raise NotImplementedError(
        f"Model '{args.model}' for {args.question} "
        "has not yet been integrated. "
        "Coordinate integration with the model developer."
    )


if __name__ == "__main__":
    main()
