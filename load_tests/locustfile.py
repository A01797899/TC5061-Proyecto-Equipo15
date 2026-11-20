"""Prueba de carga del endpoint POST /predict del Champion P1.

Uso:
    locust -f load_tests/locustfile.py --host http://127.0.0.1:8000
"""

import json
import math
import random
from pathlib import Path

from locust import HttpUser, between, task

PAYLOAD_FILE = Path(__file__).resolve().parent / "payloads_p1.json"
MODEL_NAME = "mibici-equipo15-p1-demanda"

with PAYLOAD_FILE.open(encoding="utf-8") as file:
    PAYLOADS = json.load(file)["payloads"]


class MiBiciUser(HttpUser):
    """Usuario virtual que solicita predicciones de demanda P1."""

    wait_time = between(1, 2)

    @task
    def predict(self):
        payload = random.choice(PAYLOADS)

        with self.client.post(
            "/predict",
            json=payload,
            name="POST /predict",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
                return

            try:
                body = response.json()
            except ValueError:
                response.failure("Respuesta no es JSON")
                return

            prediction = body.get("prediction")

            if body.get("model") != MODEL_NAME:
                response.failure(f"Modelo inesperado: {body.get('model')}")
            elif not isinstance(prediction, (int, float)):
                response.failure("Predicción no numérica")
            elif not math.isfinite(prediction) or prediction < 0:
                response.failure(f"Predicción inválida: {prediction}")
            else:
                response.success()
