
"""API REST de inferencia para el modelo P1 de MiBici."""

from contextlib import asynccontextmanager
from typing import Protocol

from fastapi import FastAPI, HTTPException, Request

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


MODEL_NAME = REGISTERED_MODELS["p1"]
MODEL_ALIAS = MODEL_PRODUCTION_ALIAS


class ModelServiceProtocol(Protocol):
    """Interfaz esperada del servicio desarrollado por Oscar."""

    def load_model(self) -> None:
        ...

    def is_ready(self) -> bool:
        ...

    def predict(self, records: list[dict]) -> list:
        ...


def create_app(
    model_service: ModelServiceProtocol | None = None,
) -> FastAPI:
    """Construye FastAPI permitiendo inyectar un servicio de pruebas."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Inicializa el servicio de inferencia al arrancar."""

        service = model_service

        if service is None:
            # Importación diferida para no depender de Oscar
            # durante las pruebas con mocks.
            from mibici.api.model_service import ModelService

            service = ModelService()

        app.state.model_service = service
        service.load_model()

        yield

    app = FastAPI(
        title="MiBici Equipo 15 - Prediction API",
        description="API REST de inferencia del modelo P1.",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["Health"],
    )
    def health():
        """Comprueba el funcionamiento del servicio HTTP."""

        return HealthResponse(status="ok")

    @app.get(
        "/ready",
        response_model=ReadyResponse,
        tags=["Health"],
    )
    def ready(request: Request):
        """Comprueba si el modelo está disponible."""

        service = request.app.state.model_service

        if not service.is_ready():
            raise HTTPException(
                status_code=503,
                detail="El modelo no está disponible.",
            )

        return ReadyResponse(ready=True)

    @app.post(
        "/predict",
        response_model=PredictResponse,
        tags=["Predictions"],
    )
    def predict(
        payload: PredictRequest,
        request: Request,
    ):
        """Valida los datos y ejecuta una predicción P1."""

        service = request.app.state.model_service

        if not service.is_ready():
            raise HTTPException(
                status_code=503,
                detail="El modelo no está disponible.",
            )

        try:
            predictions = service.predict(
                [payload.features]
            )

            if len(predictions) != 1:
                raise ValueError(
                    "Se esperaba exactamente una predicción."
                )

            return PredictResponse(
                prediction=float(predictions[0]),
                model=MODEL_NAME,
            )

        except (ValueError, TypeError, OverflowError) as exc:
            raise HTTPException(
                status_code=500,
                detail="No fue posible realizar la predicción.",
            ) from exc

    return app


app = create_app()
