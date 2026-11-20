#!/usr/bin/env python3
"""Smoke tests end-to-end de MiBici - Semana 10, Equipo 15.

Valida MLflow y la API FastAPI levantados con Podman Compose.
No modifica el Model Registry ni requiere bibliotecas externas.

Ejemplo:
    python scripts/week10/smoke_integration.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from copy import deepcopy
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAYLOAD = PROJECT_ROOT / "postman" / "examples" / "p1-valid.json"
MODEL_NAME = "mibici-equipo15-p1-demanda"
EXPECTED_PREDICTION = 3.694163560787005
P1_FEATURES = (
    "hour_of_day", "weekday", "fin_semana", "salidas_lag_1h",
    "salidas_lag_24h", "salidas_lag_168h", "media_estacion",
    "media_estacion_dia_hora",
)


def request(method: str, url: str, *, payload: dict | None = None, timeout: float = 15.0) -> tuple[int, object]:
    """Devuelve (HTTP status, cuerpo JSON o texto), incluyendo errores 4xx."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(req, timeout=timeout) as response:
            status = response.status
            body = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        status = exc.code
        body = exc.read().decode("utf-8", errors="replace")
    except (URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"No se pudo conectar con {url}: {exc}") from exc
    try:
        return status, json.loads(body)
    except ValueError:
        return status, body


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def prediction_of(response: object) -> float:
    require(isinstance(response, dict), "Respuesta /predict no es un objeto JSON")
    require(response.get("model") == MODEL_NAME, f"Modelo inesperado: {response.get('model')!r}")
    value = response.get("prediction")
    require(isinstance(value, (int, float)) and not isinstance(value, bool), "La prediccion no es numerica")
    number = float(value)
    require(math.isfinite(number) and number >= 0, "La prediccion no es finita o es negativa")
    return number


def main() -> int:
    parser = argparse.ArgumentParser(description="Pruebas smoke de la integración MiBici FastAPI + MLflow")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--mlflow-url", default="http://127.0.0.1:5000")
    parser.add_argument("--payload", type=Path, default=DEFAULT_PAYLOAD)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--expected", type=float, default=EXPECTED_PREDICTION,
                        help="Prediccion esperada del ejemplo P1 Champion v1")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    try:
        require(args.timeout > 0, "--timeout debe ser positivo")
        require(args.tolerance >= 0, "--tolerance no puede ser negativa")
        payload = json.loads(args.payload.read_text(encoding="utf-8"))
        require(isinstance(payload, dict) and isinstance(payload.get("features"), dict),
                "El archivo JSON debe contener el objeto features")
        require(set(payload["features"]) == set(P1_FEATURES),
                "El JSON no tiene exactamente las 8 caracteristicas P1")
    except (OSError, ValueError, AssertionError) as exc:
        print(f"[ERROR] No se pudo preparar el JSON: {exc}")
        return 1

    api = args.api_url.rstrip("/")
    mlflow = args.mlflow_url.rstrip("/")
    results: list[tuple[str, bool, str]] = []

    def check(name: str, func) -> None:
        try:
            detail = func()
            results.append((name, True, str(detail or "OK")))
            print(f"[PASS] {name}: {detail or 'OK'}")
        except (AssertionError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            results.append((name, False, str(exc)))
            print(f"[FAIL] {name}: {exc}")

    def expect_http(method: str, url: str, code: int, body: dict | None = None, data: dict | None = None) -> str:
        status, result = request(method, url, payload=data, timeout=args.timeout)
        require(status == code, f"HTTP {status}, se esperaba {code}; respuesta={result}")
        if body is not None:
            require(result == body, f"Respuesta inesperada: {result!r}")
        return f"HTTP {code}"

    print("=== MiBici | Smoke Integration | Semana 10 ===")
    print(f"API: {api} | MLflow: {mlflow}")
    print(f"Payload: {args.payload}")

    check("01 MLflow health", lambda: expect_http("GET", f"{mlflow}/health", 200))
    check("02 FastAPI health", lambda: expect_http("GET", f"{api}/health", 200, {"status": "ok"}))
    check("03 FastAPI ready", lambda: expect_http("GET", f"{api}/ready", 200, {"ready": True}))

    predictions: list[float] = []

    def predict_once() -> str:
        status, body = request("POST", f"{api}/predict", payload=payload, timeout=args.timeout)
        require(status == 200, f"HTTP {status}; respuesta={body}")
        number = prediction_of(body)
        predictions.append(number)
        return f"HTTP 200; model={MODEL_NAME}; prediction={number:.15f}"

    check("04 Inferencia P1 Champion", predict_once)

    def prediction_reference() -> str:
        require(predictions, "No hay una prediccion valida para comparar")
        require(math.isclose(predictions[0], args.expected, rel_tol=0, abs_tol=args.tolerance),
                f"Prediccion {predictions[0]} != {args.expected} (tolerancia={args.tolerance})")
        return f"valor de referencia {args.expected:.15f}"

    check("05 Prediccion de referencia", prediction_reference)
    check("06 Segunda inferencia", predict_once)

    def repeatability() -> str:
        require(len(predictions) == 2, "Falta una de las dos predicciones")
        require(math.isclose(predictions[0], predictions[1], rel_tol=0, abs_tol=1e-10),
                f"Predicciones diferentes: {predictions}")
        return "mismo resultado en dos ejecuciones"

    check("07 Reproducibilidad", repeatability)
    check("08 Ruta inexistente", lambda: expect_http("GET", f"{api}/endpoint-inexistente", 404))
    check("09 JSON vacio", lambda: expect_http("POST", f"{api}/predict", 422, data={}))

    invalid_hour = deepcopy(payload)
    invalid_hour["features"]["hour_of_day"] = 30
    check("10 Hora fuera de rango", lambda: expect_http("POST", f"{api}/predict", 422, data=invalid_hour))

    missing = deepcopy(payload)
    del missing["features"]["media_estacion_dia_hora"]
    check("11 Caracteristica faltante", lambda: expect_http("POST", f"{api}/predict", 422, data=missing))

    passed = sum(ok for _, ok, _ in results)
    print("=" * 52)
    print(f"RESULTADO: {passed}/{len(results)} pruebas aprobadas")
    if passed == len(results):
        print("SMOKE INTEGRATION COMPLETADA - OK")
        return 0
    print("SMOKE INTEGRATION FALLIDA")
    return 1


if __name__ == "__main__":
    sys.exit(main())
