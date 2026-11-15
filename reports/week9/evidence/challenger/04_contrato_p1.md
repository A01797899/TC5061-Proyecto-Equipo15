# Actividad 3.2. Parámetros, características y métricas de P1

Fecha: 2026-11-15. Rama: `feature/week9-challenger-JLMM`. No se entrenó ningún modelo en este paso.

Fuentes: `docs/week8/model_registry_contract.md`, `mibici/api/schemas.py` (`P1_FEATURES`), `docs/week9/data_preparation.md` sección 9 y `docs/week9/ab_testing_analysis.md` sección 4. La actividad 3.1 confirmó el mismo run en el Registry vivo.

## Modelo vigente

| Dato | Valor |
|---|---|
| Nombre | `mibici-equipo15-p1-demanda` |
| Alias | `champion` |
| Versión | 1 |
| Estado | `READY` |
| Run | `6e3429f222a0407aa159dd9f46699365` |
| Experimento | `mibici-equipo15` |
| Algoritmo | `HistGradientBoostingRegressor` |
| `max_iter` | 300 |
| `learning_rate` | 0.1 |
| `max_depth` | None |
| Serialización | `skops`, flavor `mlflow.sklearn` |
| Filas de train | 2,059,056 |
| Periodo de train | 2024-01-01 a 2024-09-01, exclusivo |

## Ocho características, en el orden de la firma

1. `hour_of_day`
2. `weekday`
3. `fin_semana`
4. `salidas_lag_1h`
5. `salidas_lag_24h`
6. `salidas_lag_168h`
7. `media_estacion`
8. `media_estacion_dia_hora`

La etiqueta es `y`, las salidas reales de la hora. `media_estacion` y `media_estacion_dia_hora` se calcularon solo con el entrenamiento de 2024 y quedan congeladas para 2025.

## Métricas de la corrida del Champion

| Conjunto | MAE | RMSE |
|---|---:|---:|
| Train | 0.7812010963 | 1.3224130282 |
| Validación | 0.8395558696 | 1.4280638316 |
| Prueba, nov–dic 2024 | 0.8279171026 | 1.3930097154 |
| Baseline de prueba | 0.856 | 1.464 |

## Línea base ya publicada en marzo 2025

Sobre 269,328 observaciones, el Champion versión 1 obtuvo MAE `0.8406793191283037` y RMSE `1.4369529768005689`.
