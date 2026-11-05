"""Servicio de inferencia del modelo P1 de MiBici con MLflow pyfunc."""

from __future__ import annotations

import math
from numbers import Real

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from mibici.config import (
    MLFLOW_TRACKING_URI,
    MODEL_PRODUCTION_ALIAS,
    REGISTERED_MODELS,
)


class ModelNotReadyError(RuntimeError):
    """El modelo no está cargado o no se pudo recuperar del Registry."""


class ModelInputError(ValueError):
    """Los registros no cumplen la firma del modelo."""


class ModelPredictionError(RuntimeError):
    """La ejecución del modelo falló o devolvió una salida inválida."""


class ModelService:
    """Carga el modelo P1 champion una vez y ejecuta predicciones."""

    def __init__(
        self,
        question: str = "p1",
        alias: str = MODEL_PRODUCTION_ALIAS,
        tracking_uri: str = MLFLOW_TRACKING_URI,
    ):
        self.model_name = REGISTERED_MODELS[question]
        self.model_alias = alias
        self.tracking_uri = tracking_uri
        self.model_version: str | None = None
        self.model_uri: str | None = None
        self.input_columns: list[str] = []
        self.model = None

    def load_model(self) -> None:
        """Resuelve el alias a una versión y carga esa versión del Registry."""

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_registry_uri(self.tracking_uri)

        try:
            version = MlflowClient().get_model_version_by_alias(
                self.model_name, self.model_alias
            )
            # Se carga la versión resuelta para que la versión reportada sea la usada.
            model_uri = f"models:/{self.model_name}/{version.version}"
            model = mlflow.pyfunc.load_model(model_uri)
        except Exception as exc:
            raise ModelNotReadyError(
                f"No se pudo cargar {self.model_name}@{self.model_alias} "
                f"desde {self.tracking_uri}."
            ) from exc

        schema = model.metadata.get_input_schema()
        if schema is None or not schema.has_input_names():
            raise ModelNotReadyError("El modelo no tiene una firma con nombres de columnas.")

        self.input_columns = list(schema.input_names())
        self.model_version = str(version.version)
        self.model_uri = model_uri
        self.model = model

    def is_ready(self) -> bool:
        """Indica si hay un modelo cargado."""

        return self.model is not None

    def prepare_input(self, records: list[dict]) -> pd.DataFrame:
        """Valida los registros y arma el DataFrame en el orden de la firma."""

        if not self.input_columns:
            raise ModelNotReadyError("No se conoce la firma del modelo.")

        if not isinstance(records, list) or not records:
            raise ModelInputError("Se requiere al menos un registro.")

        expected = set(self.input_columns)
        rows = []

        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ModelInputError(f"El registro {index} no es un objeto.")

            missing = [c for c in self.input_columns if c not in record]
            if missing:
                raise ModelInputError(f"Faltan columnas en el registro {index}: {missing}")

            extra = sorted(set(record) - expected)
            if extra:
                raise ModelInputError(f"Columnas no esperadas en el registro {index}: {extra}")

            row = []
            for column in self.input_columns:
                value = record[column]
                # bool es subclase de int y se rechaza aparte.
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise ModelInputError(f"Valor no numérico en {column} (registro {index}).")
                value = float(value)
                if not math.isfinite(value):
                    raise ModelInputError(f"Valor NaN o infinito en {column} (registro {index}).")
                row.append(value)

            rows.append(row)

        # La firma exige double; int64 no se convierte automáticamente.
        return pd.DataFrame(rows, columns=self.input_columns, dtype="float64")

    def predict(self, records: list[dict]) -> list[float]:
        """Devuelve una predicción de viajes por registro, recortada a cero."""

        if not self.is_ready():
            raise ModelNotReadyError("El modelo todavía no está cargado.")

        frame = self.prepare_input(records)

        try:
            raw = self.model.predict(frame)
        except Exception as exc:
            raise ModelPredictionError("Error durante la inferencia.") from exc

        try:
            values = np.asarray(raw, dtype="float64")
        except (TypeError, ValueError) as exc:
            raise ModelPredictionError("La salida del modelo no es numérica.") from exc

        if values.ndim != 1 or len(values) != len(frame):
            raise ModelPredictionError(
                f"Se esperaban {len(frame)} predicciones y se obtuvo la forma {values.shape}."
            )

        if not np.isfinite(values).all():
            raise ModelPredictionError("La salida contiene valores NaN o infinitos.")

        # Igual que en la evaluación del entrenamiento: la demanda no es negativa.
        return [float(v) for v in np.clip(values, 0.0, None)]
