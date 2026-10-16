# P1 — Predicción de demanda horaria por estación (Equipo 15)

## 1. Pregunta y justificación

**Pregunta de negocio:** ¿Cuántos viajes se iniciarán en una estación o zona en un periodo futuro?

**Definición operativa para este baseline:** pronosticar el número de viajes que se iniciarán en cada estación de MiBici durante **la siguiente hora**. La unidad de análisis es **estación × hora**; la variable objetivo es el conteo de inicios de viaje. El resultado permitirá al operador anticipar demanda y orientar la planeación de flota y el rebalanceo.

Se eligieron promedios históricos como baselines porque son interpretables, económicos de ejecutar y constituyen un punto de comparación para futuros modelos (por ejemplo, Random Forest). **No se ha entrenado Random Forest en esta etapa.**

## 2. Datos utilizados y preparación

- **Fuente:** [MiBici, datos abiertos](https://mibici.net/es/datos-abiertos/).
- **Periodo:** enero–diciembre de 2024; 12 CSV mensuales `datos_abiertos_2024_01.csv` a `datos_abiertos_2024_12.csv`.
- **Volumen de origen comprobado en el EDA:** 4,645,921 viajes; 370 identificadores de estación observados como origen o destino.
- **Insumos de P1:** estación de origen y fecha/hora de inicio de viaje. El CSV de nomenclatura no se requiere para este baseline.
- **Normalización:** se manejan las codificaciones y los formatos de fecha diferentes encontrados en el EDA; las salidas se agregan por hora.
- **Horas sin salidas:** se completan con cero dentro del panel estación-hora evaluado. Esto **no confirma** que la estación estuviera operativa todo ese tiempo.
- **Estaciones:** 362 estaciones observadas en entrenamiento. Se reportan 8 estaciones ausentes del periodo de entrenamiento, que quedan fuera de esta evaluación; el baseline aún no predice para estaciones nuevas.
- Los registros con duración igual a cero o superior a 24 horas **no se descartan exclusivamente por su duración**, porque el objetivo de P1 es contar inicios de viaje, no predecir duración.

Los CSV crudos permanecen exclusivamente en `data/raw/` local y se excluyen del historial de Git.

## 3. División temporal y prevención de fuga de información

| Conjunto | Periodo | Filas estación-hora |
|---|---|---:|
| Entrenamiento | Enero–agosto de 2024 | 2,119,872 |
| Validación | Septiembre–octubre de 2024 | 529,968 |
| Prueba | Noviembre–diciembre de 2024 | 529,968 |

Los promedios utilizados para las predicciones se calculan **únicamente** con los datos de entrenamiento. No se utilizan viajes de validación ni de prueba para ajustar las medias. La partición es cronológica y las características del baseline son conocidas al momento de pronosticar (estación, día de semana y hora objetivo).

Este experimento es un baseline histórico fijo, no un sistema de actualización en línea. Las ocho estaciones no observadas durante entrenamiento requieren una estrategia posterior de *cold start*.

## 4. Baselines y métricas

### Modelo A — `station_mean`

Predice en cada estación su **número promedio de salidas por hora** observado en el entrenamiento, independientemente de la hora o del día.

### Modelo B — `station_weekday_hour_mean`

Predice mediante el promedio del entrenamiento para cada combinación **estación + día de la semana + hora**. Capta la periodicidad semanal; sigue siendo un promedio histórico, no un modelo complejo.

**Métricas:** error absoluto medio (MAE) y raíz del error cuadrático medio (RMSE), calculados de manera independiente para entrenamiento, validación y prueba. Ambas se expresan en viajes por estación-hora; un error más pequeño es mejor. RMSE penaliza especialmente las desviaciones grandes.

### Resultados verificados en MLflow (prueba: noviembre–diciembre de 2024)

| Baseline | `test_mae` | `test_rmse` |
|---|---:|---:|
| `P1_baseline_station_mean` | 1.3621 | 2.1185 |
| `P1_baseline_station_weekday_hour_mean` | **0.8556** | **1.4635** |

El modelo B redujo el MAE en aproximadamente **37.2 %** y el RMSE en **30.9 %** con respecto al A. Estos resultados corresponden a los runs ejecutados y compartidos en MLflow; no son estimaciones.

**Interpretación:** el promedio general por estación produce una predicción diaria casi constante y no sigue el ciclo semanal real. El promedio por estación/día/hora sigue mejor dicho ciclo. En la gráfica agregada de prueba persisten discrepancias pronunciadas hacia finales de diciembre; sin información externa adicional **no se les atribuye una causa confirmada**. La gráfica diaria es una visualización auxiliar: las métricas anteriores se calculan sobre observaciones **estación-hora**.

## 5. Evidencia y trazabilidad en MLflow

- **Experimento:** `mibici-equipo15`.
- **Runs:** `P1_baseline_station_mean` y `P1_baseline_station_weekday_hour_mean`, ambos terminados satisfactoriamente.
- **Métricas por run:** `train_mae`, `train_rmse`, `val_mae`, `val_rmse`, `test_mae`, `test_rmse`.
- **Parámetros:** `model`, `train_period`, `val_period`, `test_period`, `chunk_size` (100000), `train_stations` (362), `seed` (42), `missing_hour_policy` (`zero`) y `unseen_station_policy` (`excluded_and_reported`).
- **Etiquetas:** `team=15`, `question=P1`, `dataset=MiBici 2024`, `data_version=sha256_manifest`, `target=trips_started_per_station_hour`, `prediction_horizon=next_hour`, `git_commit`.
- **Artefactos de cada run:** `p1/data_manifest.json` (manifiesto SHA-256 de archivos de origen), `p1/metrics.json` y `p1/p1_test_real_vs_predicho.png`.

## 6. Reproducción desde la raíz del repositorio

Estructura local necesaria:

```text
TC5061-Proyecto-Equipo15/
├── data/raw/                 # 12 CSV de 2024 (no versionados)
├── mibici/modeling/p1_baseline.py
├── references/P1_INSTRUCCIONES.md
└── ...
```

Usar un entorno Python con las dependencias del proyecto instaladas. El entorno de las ejecuciones verificadas fue `mlops` (Python 3.13.15), aunque la reproducibilidad desde un clon limpio requiere fijar y probar las dependencias de `requirements.txt`.

```bash
# Desde la raíz del repositorio, con el entorno Python activado
python -m mibici.modeling.p1_baseline --tracking-uri sqlite:///mlflow.db
```

Para consultar los runs (en otra terminal):

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Abrir `http://127.0.0.1:5000` y seleccionar `mibici-equipo15`. Con este MLflow se utiliza **SQLite** porque el backend de tracking exclusivamente basado en carpetas de archivos está en modo de mantenimiento; no usar el antiguo comando `mlflow ui --backend-store-uri ./mlruns`.

### Archivos que deben permanecer fuera de Git

Comprobar que `.gitignore` excluya los datos, bases de datos, artefactos y archivos temporales. Como mínimo:

```gitignore
/data/**/*.csv
/data/**/*.parquet
/data/**/*.sqlite
/mlflow.db
/mlflow.db-*
/mlruns/
/mlartifacts/
```

Antes de confirmar cambios:

```bash
git branch --show-current
git status --short
git check-ignore -v mlflow.db
git check-ignore -v data/raw/datos_abiertos_2024_01.csv
```

El trabajo se realiza en `feature/A01797899-josemanuel-martinez`, **no** directamente en `main`. El repositorio GitHub permanece privado y la integración se hará por Pull Request revisado.

## 7. Limitaciones y siguientes pasos

1. La evaluación solo comprende estaciones observadas en entrenamiento; quedan ocho estaciones posteriores fuera de la comparación.
2. Completar con cero las horas sin viajes no equivale a demostrar actividad real de cada estación.
3. Los patrones de fin de año todavía no están modelados explícitamente.
4. No se incorporan clima, inventario real, días festivos u otras fuentes externas.
5. Validar un modelo candidato (por ejemplo, `RandomForestRegressor`) usando exclusivamente variables disponibles al pronosticar y compararlo con estos baselines.
6. Finalizar la preparación reproducible desde clon limpio y el vínculo exacto entre cada run y el commit que contiene el código.

**Estado:** baselines P1 ejecutados y resultados registrados en MLflow. P2 y P3 se implementarán y documentarán por separado.
