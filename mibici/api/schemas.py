
"""Esquemas de entrada y salida para la API MiBici P1.

Contrato provisional hasta confirmar la firma MLflow del modelo.
"""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


NumericFeature = Annotated[
    float,
    Field(strict=True, allow_inf_nan=False),
]


class PredictRequest(BaseModel):
    """Solicitud de predicción para P1."""

    model_config = ConfigDict(extra="forbid")

    features: dict[str, NumericFeature] = Field(
        min_length=8,
        max_length=8,
        description="Ocho características numéricas del modelo P1.",
    )


class PredictResponse(BaseModel):
    """Respuesta de inferencia del modelo P1."""

    prediction: float = Field(
        allow_inf_nan=False,
        description="Demanda estimada por el modelo P1.",
    )

    model: str = Field(
        description="Nombre del modelo registrado en MLflow.",
    )


class HealthResponse(BaseModel):
    """Estado básico del servicio FastAPI."""

    status: str


class ReadyResponse(BaseModel):
    """Estado de disponibilidad del modelo."""

    ready: bool
