

"""Esquemas de entrada y salida para la API MiBici P1.

Contrato basado en la firma registrada en MLflow Model Registry
para el modelo mibici-equipo15-p1-demanda, alias champion.
"""

from typing import Annotated

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
)


# Columnas oficiales de entrada, en el orden de la firma MLflow.
P1_FEATURES = (
    "hour_of_day",
    "weekday",
    "fin_semana",
    "salidas_lag_1h",
    "salidas_lag_24h",
    "salidas_lag_168h",
    "media_estacion",
    "media_estacion_dia_hora",
)


def validar_numero(value):
    """Acepta enteros y decimales, pero rechaza booleanos y cadenas."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            "La característica debe ser un número entero o decimal."
        )

    return float(value)


NumericFeature = Annotated[
    float,
    BeforeValidator(validar_numero),
    Field(allow_inf_nan=False),
]


class PredictRequest(BaseModel):
    """Solicitud de predicción con las ocho variables oficiales de P1."""

    model_config = ConfigDict(extra="forbid")

    features: dict[str, NumericFeature] = Field(
        min_length=8,
        max_length=8,
        description=(
            "Ocho características obligatorias del modelo P1, "
            "según la firma registrada en MLflow."
        ),
    )

    @field_validator("features")
    @classmethod
    def validar_features(cls, features):
        """Verifica nombres, rangos y consistencia del calendario."""

        expected = set(P1_FEATURES)
        received = set(features)

        missing = sorted(expected - received)
        extra = sorted(received - expected)

        if missing:
            raise ValueError(
                f"Faltan características obligatorias: {missing}"
            )

        if extra:
            raise ValueError(
                f"Características no reconocidas: {extra}"
            )

        hour = features["hour_of_day"]
        weekday = features["weekday"]
        weekend = features["fin_semana"]

        if not 0 <= hour <= 23:
            raise ValueError(
                "hour_of_day debe estar entre 0 y 23."
            )

        if not 0 <= weekday <= 6:
            raise ValueError(
                "weekday debe estar entre 0 y 6."
            )

        if weekend not in (0.0, 1.0):
            raise ValueError(
                "fin_semana debe ser 0 o 1."
            )

        expected_weekend = 1.0 if weekday >= 5 else 0.0

        if weekend != expected_weekend:
            raise ValueError(
                "fin_semana no coincide con weekday."
            )

        non_negative_features = (
            "salidas_lag_1h",
            "salidas_lag_24h",
            "salidas_lag_168h",
            "media_estacion",
            "media_estacion_dia_hora",
        )

        for name in non_negative_features:
            if features[name] < 0:
                raise ValueError(
                    f"{name} debe ser mayor o igual a 0."
                )

        return {
            name: features[name]
            for name in P1_FEATURES
        }


class PredictResponse(BaseModel):
    """Respuesta de inferencia del modelo P1."""

    prediction: float = Field(
        allow_inf_nan=False,
        ge=0,
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
