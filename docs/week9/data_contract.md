# Semana 9 - Contrato de datos del Champion P1

Contrato de entrada del modelo P1 que usan el monitoreo de drift (Evidently), el reentrenamiento del Challenger y el A/B testing. Todo se tomó del Registry, de `mibici/config.py`, de `mibici/modeling/p1_p2_candidates.py` y de los artefactos de la corrida del Champion.

## Modelo Champion

| Dato                    | Valor                                                                                   |
| ----------------------- | --------------------------------------------------------------------------------------- |
| Modelo registrado       | `mibici-equipo15-p1-demanda`                                                            |
| Versión y alias         | `1`, `champion`, estado `READY`                                                         |
| URI de MLflow           | `models:/mibici-equipo15-p1-demanda@champion`                                           |
| Run ID                  | `6e3429f222a0407aa159dd9f46699365`                                                      |
| Algoritmo               | `HistGradientBoostingRegressor` (`max_iter=300`, `learning_rate=0.1`, `max_depth=None`) |
| Versión de datos        | `7e1470ac0632`                                                                          |
| Commit de entrenamiento | `ad81f296c6aea3c97935399dd389c6a02c25d713`                                              |
| Contrato de la semana 8 | [`docs/week8/model_registry_contract.md`](../week8/model_registry_contract.md)          |

## Variable objetivo

`y`: número de viajes que inician (salidas) en una estación durante una hora. Es un conteo entero ≥ 0. Las horas sin viajes valen 0 porque el panel estación-hora es completo.

## Variables de entrada

Las 8 son obligatorias, de tipo `double` en la firma y van en este orden (`P1_FEATURES` en `mibici/api/schemas.py`):

| #   | Variable                  | Rango             | Origen                                                                          |
| --- | ------------------------- | ----------------- | ------------------------------------------------------------------------------- |
| 1   | `hour_of_day`             | 0 a 23            | Hora de `hour`                                                                  |
| 2   | `weekday`                 | 0 a 6 (0 = lunes) | `dayofweek` de `hour`                                                           |
| 3   | `fin_semana`              | 0 o 1             | `weekday >= 5` (`asignar_split()`)                                              |
| 4   | `salidas_lag_1h`          | ≥ 0               | `y` de la misma estación 1 hora antes (`agregar_rezagos()` → `lag_features()`)  |
| 5   | `salidas_lag_24h`         | ≥ 0               | `y` de la misma estación 24 horas antes                                         |
| 6   | `salidas_lag_168h`        | ≥ 0               | `y` de la misma estación 168 horas (7 días) antes                               |
| 7   | `media_estacion`          | ≥ 0               | Media de `y` de la estación, solo con el entrenamiento (`agregar_historicos()`) |
| 8   | `media_estacion_dia_hora` | ≥ 0               | Media de `y` por estación, `weekday` y `hour_of_day`, solo con el entrenamiento |

Reglas que se deben conservar:

- Los rezagos se buscan por fecha, no por posición: un hueco deja `NaN` y nunca mezcla estaciones ni usa horas futuras.
- Las medias se calculan una sola vez con las filas de entrenamiento (enero a agosto de 2024, antes de quitar nulos) y quedan congeladas. Una estación sin historial toma la media global de `y`; una combinación estación-día-hora sin historial toma `media_estacion`.
- El modelo rechaza enteros (`int64`): las 8 columnas se entregan como `float64`.

## Periodo original de entrenamiento

Los límites de `mibici/config.py` son exclusivos:

| Conjunto      | Periodo             | Límite en `config.py`                                     | Filas estación-hora                     |
| ------------- | ------------------- | --------------------------------------------------------- | --------------------------------------- |
| Entrenamiento | 1 ene - 31 ago 2024 | `FECHA_INICIO = "2024-01-01"`, `TRAIN_END = "2024-09-01"` | 2,119,872 (2,059,056 tras quitar nulos) |
| Validación    | 1 sep - 31 oct 2024 | `VAL_END = "2024-11-01"`                                  | 529,968                                 |
| Prueba        | 1 nov - 31 dic 2024 | `TEST_END = "2025-01-01"`                                 | 529,968                                 |

## Construcción original del panel

1. `p1_baseline.aggregate_monthly()` lee los 12 CSV de `data/raw/datos_abiertos_2024_MM.csv` con su encoding y formato de fecha por mes (julio en `utf-8`, septiembre y octubre con `%d/%m/%Y %H:%M`, el resto con `%Y-%m-%d %H:%M:%S`). Conserva solo los viajes con fecha y origen válidos dentro de su mes y cuenta salidas por estación y hora.
2. Las estaciones del modelo son las 362 con al menos un viaje en el entrenamiento.
3. `p1_baseline.panel()` completa el panel estación-hora del 1 de enero de 2024 al 1 de enero de 2025 y rellena con 0 las horas sin viajes.
4. `asignar_split()` marca train, val y test, y crea `fin_semana`.
5. `agregar_rezagos()` crea los 3 rezagos y `agregar_historicos()` las 2 medias con train.
6. `run_candidate()` quita del entrenamiento las filas con alguna variable nula: las primeras 168 horas de cada estación, 362 × 168 = 60,816 filas. Luego convierte las 8 variables a `float64`.

## Ingesta histórica (`data/manifiesto.json`)

| Concepto                           | Valor          |
| ---------------------------------- | -------------- |
| Año                                | 2024           |
| Archivos mensuales                 | 12             |
| Registros procesados               | 4,645,921      |
| Fechas inválidas y orígenes vacíos | 0              |
| Versión de datos                   | `7e1470ac0632` |

El manifiesto histórico no se modifica en la semana 9.

## Datasets de la semana 9

| Archivo                                          | Periodo             | Uso                                 |
| ------------------------------------------------ | ------------------- | ----------------------------------- |
| `data/processed/week9/reference_p1.parquet`      | 1 ene - 31 ago 2024 | Referencia de drift                 |
| `data/processed/week9/current_p1.parquet`        | 1 ene - 31 mar 2025 | Datos actuales para drift           |
| `data/processed/week9/train_alternative.parquet` | 1 ene - 28 feb 2025 | Entrenamiento del Challenger        |
| `data/processed/week9/evaluation.parquet`        | 1 - 31 mar 2025     | Evaluación de Champion y Challenger |

Los cuatro tienen las columnas `station`, `hour`, `y` y las 8 variables de P1 en `float64`, con una fila por estación y hora y sin nulos en las variables. Los parquets no se suben a Git; se generan con `mibici/monitoring/prepare_data.py`, y sus conteos y hashes quedan en `reports/week9/data_manifest.json`.
