"""Validación de modelos registrados en MLflow."""

import argparse

import mlflow
from mlflow import MlflowClient

from mibici.config import (
    MLFLOW_TRACKING_URI,
    REGISTERED_MODELS,
    MODEL_PRODUCTION_ALIAS,
)


def validate_registered_model(question: str):

    if question not in REGISTERED_MODELS:
        raise ValueError(f"Pregunta inválida: {question}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_registry_uri(MLFLOW_TRACKING_URI)

    client = MlflowClient()

    model_name = REGISTERED_MODELS[question]

    model_version = client.get_model_version_by_alias(
        name=model_name,
        alias=MODEL_PRODUCTION_ALIAS,
    )

    model_uri = (
        f"models:/{model_name}@{MODEL_PRODUCTION_ALIAS}"
    )

    print("Question:", question)
    print("Registered model:", model_name)
    print("Version:", model_version.version)
    print("Source run:", model_version.run_id)
    print("Model URI:", model_uri)

    model = mlflow.pyfunc.load_model(model_uri)

    print("Model loaded successfully")

    return model


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--question",
        choices=["p1", "p2", "p3"],
        required=True,
    )

    args = parser.parse_args()

    validate_registered_model(args.question)


if __name__ == "__main__":
    main()