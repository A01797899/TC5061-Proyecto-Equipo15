## Semana 8 - Recuperación del modelo desde MLflow Registry

**Objetivo.** Verificar que el modelo P1 registrado en la Semana 6 se puede recuperar desde el Model Registry por nombre y alias, y documentar su contrato de entradas y salidas para el servicio de inferencia y la API FastAPI. No se reentrenó, registró ni promovió ningún modelo.

**Modelo seleccionado.** `mibici-equipo15-p1-demanda` (P1, demanda por estación y hora, regresión con `HistGradientBoostingRegressor`).

| Dato | Valor |
|---|---|
| Versión | `1` (`READY`) |
| Alias | `champion` |
| Run ID | `6e3429f222a0407aa159dd9f46699365` |
| URI de MLflow | `models:/mibici-equipo15-p1-demanda@champion` |
| Tracking URI local | `http://127.0.0.1:5000` |
| Tracking URI desde Podman | `http://host.containers.internal:5000` |

**Variables de entrada.** Ocho columnas `double`, todas obligatorias y en este orden: `hour_of_day`, `weekday`, `fin_semana`, `salidas_lag_1h`, `salidas_lag_24h`, `salidas_lag_168h`, `media_estacion` y `media_estacion_dia_hora`. Los rangos y el significado de cada una están en [`docs/week8/model_registry_contract.md`](model_registry_contract.md).

**Firma de salida.** `Tensor('float64', (-1,))`: una predicción de viajes por fila.

**Servidor.** El Registry no está en Git. Se usa la base `mlflow.db` y la carpeta `mlartifacts/` del entorno donde se registraron los modelos, fuera del repositorio:

```bash
.venv/bin/mlflow server \
  --backend-store-uri sqlite:///../mlflow-jm/mlflow.db \
  --serve-artifacts \
  --artifacts-destination ../mlflow-jm/mlartifacts \
  --host 127.0.0.1 --port 5000
```

Para que la API en Podman llegue al Registry, el servidor se levanta con `--host 0.0.0.0` y `--allowed-hosts "host.containers.internal:5000,localhost:5000,127.0.0.1:5000"`; si no, MLflow responde `403 Invalid Host header` (detalle en el contrato).

En otra terminal:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

**Comando de inspección.** Lista las versiones, valida el alias `champion` y su estado `READY`, y muestra la firma:

```bash
python scripts/week8/inspect_registry.py
```

**Comando de validación de carga.** Reutiliza `validate_registered_model()` y carga el modelo con `mlflow.pyfunc.load_model()`:

```bash
python -m mibici.modeling.validate_registry --question p1
```

**Pruebas automatizadas.** `tests/test_week8_registry.py` valida el nombre, el alias, la llamada al cargador con la URI correcta, el rechazo de preguntas inválidas y la propagación de un fallo de carga. Usa mocks y no necesita el servidor:

```bash
python -m pytest tests/test_week8_registry.py -v
python -m pytest tests/test_registry.py -v
python -m pytest tests/ -v
```

**Evidencias.** En `reports/week8/evidence/registry/`:

| Archivo | Evidencia |
|---|---|
| `01_registry_model.png` | Modelo P1 en la interfaz de MLflow |
| `02_champion_alias.png` | Versión 1 y alias `champion` |
| `03_model_signature.png` | Firma de entrada y salida |
| `04_registry_inspection.png` | Ejecución de `inspect_registry.py` |
| `05_model_loaded.png` | Carga correcta con `validate_registry` |
| `06_registry_tests.png` | Pruebas automatizadas |

**Relación con el servicio pyfunc.** `mibici/api/model_service.py` carga `models:/mibici-equipo15-p1-demanda@champion` con `mlflow.pyfunc.load_model()` y la API FastAPI valida las 8 variables con Pydantic antes de llamar a `predict()`. Al cargar el modelo se comprobó que:

- la firma rechaza valores enteros (`int64`), así que la API debe convertir las entradas a `float`;
- las columnas faltantes producen error y las extra se ignoran;
- los nulos y los valores fuera de rango no producen error, así que esa validación le corresponde a Pydantic.
