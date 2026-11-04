# MiBici P1 - Contrato del modelo registrado

Contrato técnico del modelo P1 (demanda) para el servicio de inferencia (`mibici/api/model_service.py`) y los esquemas Pydantic de la API FastAPI. Todos los datos se obtuvieron del Registry real con `scripts/week8/inspect_registry.py` y `mibici.modeling.validate_registry`.

## Modelo

`mibici-equipo15-p1-demanda`

Estima el número de viajes iniciados en una estación durante una hora (regresión).

## Alias

`champion`

## Versión

`1`, estado `READY`, registrada el 25 de octubre de 2026.

## Run ID

`6e3429f222a0407aa159dd9f46699365`

## Tracking URI

Servidor MLflow local con backend SQLite y artefactos servidos por HTTP:

```bash
mlflow server \
  --backend-store-uri sqlite:///<carpeta>/mlflow.db \
  --serve-artifacts \
  --artifacts-destination <carpeta>/mlartifacts \
  --host 127.0.0.1 --port 5000
```

| Desde | `MLFLOW_TRACKING_URI` |
|---|---|
| La máquina local | `http://127.0.0.1:5000` |
| Un contenedor Podman | `http://host.containers.internal:5000` |
| Un contenedor Docker | `http://host.docker.internal:5000` |

Con `--host 127.0.0.1`, MLflow 3.16 solo acepta peticiones cuyo encabezado `Host` sea `localhost` o `127.0.0.1`. Desde un contenedor responde `403 Invalid Host header`, así que para la API en Podman el servidor se levanta así:

```bash
mlflow server \
  --backend-store-uri sqlite:///<carpeta>/mlflow.db \
  --serve-artifacts \
  --artifacts-destination <carpeta>/mlartifacts \
  --host 0.0.0.0 --port 5000 \
  --allowed-hosts "host.containers.internal:5000,localhost:5000,127.0.0.1:5000"
```

Los artefactos se registraron con el esquema `mlflow-artifacts:/`, por lo que el servidor debe arrancar con `--serve-artifacts`. La carpeta de artefactos del registro original se llama `mlartifacts/`, no `mlflow-artifacts/`. Ni `mlflow.db` ni `mlartifacts/` están en Git.

## Modelo URI

`models:/mibici-equipo15-p1-demanda@champion`

```python
import mlflow

model = mlflow.pyfunc.load_model("models:/mibici-equipo15-p1-demanda@champion")
```

## Variables de entrada

La firma exige un DataFrame con estas 8 columnas, en este orden:

| # | Variable | Tipo MLflow | Obligatoria | Significado | Rango esperado |
|---|---|---|---|---|---|
| 1 | `hour_of_day` | `double` | Sí | Hora del día | 0 a 23 |
| 2 | `weekday` | `double` | Sí | Día de la semana (0 = lunes) | 0 a 6 |
| 3 | `fin_semana` | `double` | Sí | 1 si `weekday` ≥ 5, 0 si no | 0 o 1 |
| 4 | `salidas_lag_1h` | `double` | Sí | Salidas de la estación 1 hora antes | ≥ 0 |
| 5 | `salidas_lag_24h` | `double` | Sí | Salidas de la estación 24 horas antes | ≥ 0 |
| 6 | `salidas_lag_168h` | `double` | Sí | Salidas de la estación 168 horas (7 días) antes | ≥ 0 |
| 7 | `media_estacion` | `double` | Sí | Media histórica de salidas por hora de la estación (train) | ≥ 0 |
| 8 | `media_estacion_dia_hora` | `double` | Sí | Media histórica de la estación para ese día y hora (train) | ≥ 0 |

Ejemplo registrado con el modelo (`serving_input_example.json`):

```json
{
  "dataframe_split": {
    "columns": ["hour_of_day", "weekday", "fin_semana", "salidas_lag_1h",
                "salidas_lag_24h", "salidas_lag_168h", "media_estacion",
                "media_estacion_dia_hora"],
    "data": [[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.8466529846191406, 0.11428571492433548]]
  }
}
```

## Firma de salida

`Tensor('float64', (-1,))`: un arreglo de NumPy con una predicción por fila de entrada (viajes esperados en la hora).

| Entrada | Salida de `model.predict()` |
|---|---|
| `hour_of_day=8, weekday=2, fin_semana=0, salidas_lag_1h=3, salidas_lag_24h=4, salidas_lag_168h=5, media_estacion=2.1, media_estacion_dia_hora=3.4` | `[3.69416356]` |

## Comportamiento comprobado al cargar el modelo

| Caso | Resultado | Qué debe hacer la API |
|---|---|---|
| Valores enteros (`int64`) | `MlflowException`: no convierte `int64` a `float64` | Convertir las 8 variables a `float` antes de `predict()` |
| Falta una columna | `MlflowException`: `Model is missing inputs` | Declarar los 8 campos como obligatorios en Pydantic |
| Columnas extra | Se ignoran | Rechazarlas en Pydantic (`extra="forbid"`) |
| Valores nulos (`NaN`) | Predice sin error | Rechazarlos: el entrenamiento descartó las filas con nulos |
| Valores fuera de rango (por ejemplo, `hour_of_day=-3`) | Predice sin error | Validar los rangos de la tabla de entradas |

## Origen del modelo

| Dato | Valor |
|---|---|
| Experimento | `mibici-equipo15` |
| Corrida padre | `P1_search_candidates` (`84b3d18fea4a49dbb1c212e9311da9c7`) |
| Corrida | `P1_candidate_hgb_max_iter=300_learning_rate=0.1_max_depth=None` |
| Algoritmo | `HistGradientBoostingRegressor` (`max_iter=300`, `learning_rate=0.1`, `max_depth=None`) |
| Logged Model | `m-04774b8f25ff468b933a18756c24b068` |
| Serialización | `skops` (`model.skops`), flavor `mlflow.sklearn` |
| Entorno | Python 3.12.15, mlflow 3.16.1, scikit-learn 1.9.1, numpy 2.5.3, pandas 3.0.6, skops 0.16.0 |
| Periodos | train 2024-01-01 a 2024-09-01, val 2024-09-01 a 2024-11-01, test 2024-11-01 a 2025-01-01 |
| Filas de train | 2,059,056 |
| Commit | `ad81f296c6aea3c97935399dd389c6a02c25d713` |

Métricas de la corrida:

| Conjunto | MAE | RMSE |
|---|---|---|
| Train | 0.781 | 1.322 |
| Validación | 0.840 | 1.428 |
| Test | 0.828 | 1.393 |
| Baseline (test) | 0.856 | 1.464 |

En test mejora al baseline 3.2% en MAE y 4.8% en RMSE.

Artefactos de la corrida: `metrics/metrics.json`, `config/` (`config.py`, `MLproject`, `python_env.yaml`, `run_params.json`), `data/` (`manifiesto.json`, `csv_manifest.json`) y `plots/` (`p1_test_real_vs_predicho.png`, `p1_test_residuales.png`).

Artefactos del modelo: `MLmodel`, `model.skops`, `conda.yaml`, `python_env.yaml`, `requirements.txt`, `input_example.json` y `serving_input_example.json`.

## Validación

| Comprobación | Resultado |
|---|---|
| Versiones registradas | 1 versión, `READY` |
| Alias `champion` | Apunta a la versión 1 |
| Firma | 8 entradas `double` obligatorias y salida `float64` |
| `python -m mibici.modeling.validate_registry --question p1` | `Model loaded successfully` |
| `pytest tests/test_week8_registry.py` | 5 passed |

No fue necesario reentrenar, registrar ni promover el modelo.
