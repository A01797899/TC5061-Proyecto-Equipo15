"""Prueba de inferencia con el modelo P1 real del Model Registry."""

import math

from mibici.api.model_service import ModelInputError, ModelService

# Entrada documentada en docs/week8/model_registry_contract.md.
RECORD = {
    "hour_of_day": 8,
    "weekday": 2,
    "fin_semana": 0,
    "salidas_lag_1h": 3,
    "salidas_lag_24h": 4,
    "salidas_lag_168h": 5,
    "media_estacion": 2.1,
    "media_estacion_dia_hora": 3.4,
}
EXPECTED_CONTRACT = 3.69416356


def main():
    service = ModelService()

    print("Tracking URI:", service.tracking_uri)
    print(f"Cargando {service.model_name}@{service.model_alias}...")
    service.load_model()

    print("Modelo cargado:", service.is_ready())
    print("Versión:", service.model_version)
    print("Model URI:", service.model_uri)
    print("Run ID:", service.model.metadata.run_id)
    print("Columnas de la firma:", service.input_columns)

    if len(service.input_columns) != 8:
        raise RuntimeError("La firma P1 no tiene las 8 columnas esperadas.")

    print("\nEntrada (enteros, como llegarían en JSON):", RECORD)
    first = service.predict([RECORD])
    print("Predicción:", first)

    second = service.predict([RECORD])
    print("Segunda ejecución:", second)

    if first != second:
        raise RuntimeError("Las predicciones no son reproducibles.")
    if not math.isclose(first[0], EXPECTED_CONTRACT, abs_tol=1e-6):
        raise RuntimeError(f"La predicción no coincide con el contrato ({EXPECTED_CONTRACT}).")
    print("Coincide con el contrato del Registry:", EXPECTED_CONTRACT)

    batch = service.predict([RECORD, {**RECORD, "hour_of_day": 18}, {**RECORD, "hour_of_day": 3}])
    print("\nLote de 3 registros (8 h, 18 h, 3 h):", batch)

    try:
        service.predict([{k: v for k, v in RECORD.items() if k != "weekday"}])
    except ModelInputError as exc:
        print("Registro incompleto rechazado:", exc)

    print("\nINFERENCIA PYFUNC COMPLETADA")


if __name__ == "__main__":
    main()
