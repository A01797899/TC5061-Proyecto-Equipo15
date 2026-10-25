"""Gestión de modelos registrados para MiBici Equipo 15."""

from __future__ import annotations

import argparse
import subprocess

import mlflow
from mlflow import MlflowClient

from mibici.config import (
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    REGISTERED_MODELS,
    MODEL_PRODUCTION_ALIAS,
)


def get_commit() -> str:
    """Obtiene el commit actual de Git."""

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode == 0:
        return result.stdout.strip()

    return "unknown"


def configure_mlflow() -> MlflowClient:
    """Configura Tracking y Model Registry."""

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_registry_uri(MLFLOW_TRACKING_URI)

    return MlflowClient()


def register_model(
    question: str,
    model_uri: str,
    description: str,
    data_version: str,
):
    """Registra un modelo previamente almacenado en MLflow."""

    if question not in REGISTERED_MODELS:
        raise ValueError(f"Pregunta no válida: {question}")

    if not model_uri:
        raise ValueError("Se requiere la URI del modelo.")

    client = configure_mlflow()

    model_name = REGISTERED_MODELS[question]

    print(f"Registering model: {model_name}")
    print(f"Source URI: {model_uri}")

    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
    )

    version = model_version.version

    client.update_registered_model(
        name=model_name,
        description=(
            f"Modelo MiBici Equipo 15 para {question.upper()}. "
            f"Experimento: {MLFLOW_EXPERIMENT}."
        ),
    )

    client.update_model_version(
        name=model_name,
        version=version,
        description=description,
    )

    tags = {
        "team": "15",
        "question": question.upper(),
        "experiment": MLFLOW_EXPERIMENT,
        "data_version": data_version,
        "registration_commit": get_commit(),
        "stage_of_work": "candidate",
    }

    for key, value in tags.items():
        client.set_model_version_tag(
            name=model_name,
            version=version,
            key=key,
            value=value,
        )

    print(f"Registered model: {model_name}")
    print(f"Version: {version}")
    print(f"Source run: {model_version.run_id}")

    return model_version


def promote_to_champion(
    question: str,
    version: str,
):
    """Asigna champion a una versión ya evaluada."""

    if question not in REGISTERED_MODELS:
        raise ValueError(f"Pregunta no válida: {question}")

    client = configure_mlflow()

    model_name = REGISTERED_MODELS[question]

    client.set_registered_model_alias(
        name=model_name,
        alias=MODEL_PRODUCTION_ALIAS,
        version=version,
    )

    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="stage_of_work",
        value="production_selected",
    )

    print(
        f"Champion assigned: {model_name}, "
        f"version {version}"
    )


def main():

    parser = argparse.ArgumentParser(
        description="MiBici MLflow Model Registry"
    )

    parser.add_argument(
        "--question",
        choices=["p1", "p2", "p3"],
        required=True,
    )

    parser.add_argument(
        "--model-uri",
        required=True,
    )

    parser.add_argument(
        "--description",
        required=True,
    )

    parser.add_argument(
        "--data-version",
        required=True,
    )

    args = parser.parse_args()

    register_model(
        question=args.question,
        model_uri=args.model_uri,
        description=args.description,
        data_version=args.data_version,
    )


if __name__ == "__main__":
    main()
