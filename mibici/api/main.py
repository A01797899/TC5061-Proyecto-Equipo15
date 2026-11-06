
"""API REST de inferencia para el modelo P1 de MiBici.

Semana 8 - Equipo 15.

Integra:
- FastAPI: endpoints de salud, disponibilidad y predicción.
- Pydantic: validación del contrato P1.
- MLflow: recuperación del modelo champion mediante ModelService.
- Manejo de errores HTTP de inferencia y disponibilidad.
"""

import logging

from contextlib import asynccontextmanager
from typing import Protocol

from fastapi import FastAPI, HTTPException, Request
from pydantic import ValidationError

from mibici.api.model_service import (
    ModelInputError,
    ModelNotReadyError,
    ModelPredictionError,
)

from mibici.api.schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    ReadyResponse,
)

from mibici.config import (
    MODEL_PRODUCTION_ALIAS,
    REGISTERED_MODELS,
)


# ============================================================
# 1. Configuración y constantes
# ============================================================

logger = logging.getLogger(__name__)

MODEL_NAME = REGISTERED_MODELS["p1"]
MODEL_ALIAS = MODEL_PRODUCTION_ALIAS

MODEL_UNAVAILABLE_MESSAGE = "El modelo no está disponible."
PREDICTION_ERROR_MESSAGE = "No fue posible realizar la predicción."


# ============================================================
# 2. Contrato del servicio de inferencia
# ============================================================

class ModelServiceProtocol(Protocol):
    """Interfaz compatible con ModelService de Oscar."""

    def load_model(self) -> None:
        """Carga el modelo desde MLflow."""
        ...

    def is_ready(self) -> bool:
        """Indica si el modelo está disponible."""
        ...

    def predict(self, records: list[dict]) -> list[float]:
        """Ejecuta las predicciones solicitadas."""
        ...


# ============================================================
# 3. Construcción de la aplicación FastAPI
# ============================================================

def create_app(
    model_service: ModelServiceProtocol | None = None,
) -> FastAPI:
    """Construye la API permitiendo inyectar un servicio de pruebas."""

    # ========================================================
    # 3.1 Ciclo de vida de FastAPI
    # ========================================================

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Inicializa ModelService durante el arranque."""

        service = model_service

        if service is None:
            from mibici.api.model_service import ModelService

            service = ModelService()

        app.state.model_service = service
        app.state.model_load_error = None

        try:
            logger.info(
                "Inicializando modelo MLflow: %s@%s",
                MODEL_NAME,
                MODEL_ALIAS,
            )

            service.load_model()

            logger.info(
                "Inicialización del modelo completada."
            )

        except ModelNotReadyError as exc:
            # La API sigue disponible para responder /health.
            # /ready y /predict indicarán que el modelo no está listo.
            app.state.model_load_error = str(exc)

            logger.exception(
                "No fue posible cargar el modelo P1 desde MLflow."
            )

        yield

    # ========================================================
    # 3.2 Configuración de FastAPI
    # ========================================================

    app = FastAPI(
        title="MiBici Equipo 15 - Prediction API",
        description=(
            "API REST para estimar la demanda de viajes por "
            "estación y hora utilizando el modelo P1 champion "
            "registrado en MLflow."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # ========================================================
    # 4. Endpoint /health
    # ========================================================

    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["Health"],
    )
    def health():
        """Verifica que la aplicación HTTP está funcionando."""

        return HealthResponse(status="ok")

    # ========================================================
    # 5. Endpoint /ready
    # ========================================================

    @app.get(
        "/ready",
        response_model=ReadyResponse,
        tags=["Health"],
        responses={
            503: {
                "description": "Modelo no disponible",
            },
        },
    )
    def ready(request: Request):
        """Comprueba si el modelo MLflow está cargado."""

        service = request.app.state.model_service

        if not service.is_ready():
            raise HTTPException(
                status_code=503,
                detail=MODEL_UNAVAILABLE_MESSAGE,
            )

        return ReadyResponse(ready=True)

    # ========================================================
    # 6. Endpoint /predict
    # ========================================================

    @app.post(
        "/predict",
        response_model=PredictResponse,
        tags=["Predictions"],
        responses={
            422: {
                "description": "Solicitud de predicción inválida",
            },
            503: {
                "description": "Modelo no disponible",
            },
            500: {
                "description": "Error interno de inferencia",
            },
        },
    )
    def predict(
        payload: PredictRequest,
        request: Request,
    ):
        """Valida una entrada P1 y realiza una predicción."""

        service = request.app.state.model_service

        # ----------------------------------------------------
        # 6.1 Disponibilidad
        # ----------------------------------------------------

        if not service.is_ready():
            raise HTTPException(
                status_code=503,
                detail=MODEL_UNAVAILABLE_MESSAGE,
            )

        # ----------------------------------------------------
        # 6.2 Preparación del registro
        # ----------------------------------------------------

        records = [payload.features]

        # ----------------------------------------------------
        # 6.3 Inferencia y manejo de errores
        # ----------------------------------------------------

        try:
            predictions = service.predict(records)

            if len(predictions) != 1:
                raise ModelPredictionError(
                    "Se esperaba exactamente una predicción."
                )

            response = PredictResponse(
                prediction=float(predictions[0]),
                model=MODEL_NAME,
            )

            return response

        except ModelInputError as exc:
            # Error de datos detectado por ModelService.
            logger.warning(
                "Entrada rechazada por ModelService: %s",
                exc,
            )

            raise HTTPException(
                status_code=422,
                detail="Los datos de entrada no son válidos.",
            ) from exc

        except ModelNotReadyError as exc:
            # El modelo dejó de estar disponible.
            logger.warning(
                "Modelo MLflow no disponible: %s",
                exc,
            )

            raise HTTPException(
                status_code=503,
                detail=MODEL_UNAVAILABLE_MESSAGE,
            ) from exc

        except ModelPredictionError as exc:
            # El modelo recibió la solicitud, pero falló.
            logger.exception(
                "Error de inferencia del modelo P1."
            )

            raise HTTPException(
                status_code=500,
                detail=PREDICTION_ERROR_MESSAGE,
            ) from exc

        except (
            ValueError,
            TypeError,
            OverflowError,
            ValidationError,
        ) as exc:
            # Conserva compatibilidad con los tests anteriores
            # y captura errores de conversión o respuesta.
            logger.exception(
                "Error al procesar la respuesta del modelo."
            )

            raise HTTPException(
                status_code=500,
                detail=PREDICTION_ERROR_MESSAGE,
            ) from exc

        except Exception as exc:
            # Última protección para errores inesperados.
            # No se exponen trazas internas al cliente HTTP.
            logger.exception(
                "Error inesperado durante la predicción P1."
            )

            raise HTTPException(
                status_code=500,
                detail=PREDICTION_ERROR_MESSAGE,
            ) from exc

    return app


# ============================================================
# 7. Aplicación utilizada por Uvicorn
# ============================================================

app = create_app()
