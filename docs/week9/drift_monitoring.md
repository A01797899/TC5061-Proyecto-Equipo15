# Semana 9 - Monitoreo de drift del Champion P1 con Evidently

Pasos 2 y 3 de la actividad: generar el reporte de drift con Evidently e interpretar los resultados. Usa los datos que prepara `mibici/monitoring/prepare_data.py` (ver [`data_preparation.md`](data_preparation.md) y [`data_contract.md`](data_contract.md)).

## 1. Objetivo

Detectar si los datos que recibe el Champion P1 (`models:/mibici-equipo15-p1-demanda@champion`) en un periodo posterior se alejan de los datos con los que se entrenó, medir qué variables cambian y cuánto, y dar evidencia para el A/B testing y la recomendación de Continuous Training.

## 2. Conjuntos comparados

| Escenario | Referencia | Actual | Para qué sirve |
|---|---|---|---|
| `principal` | `reference_p1.parquet`: entrenamiento del Champion, 8 ene - 31 ago 2024 (2,059,056 filas) | `current_p1.parquet`: 1 ene - 31 mar 2025 (781,920 filas) | Comparación que pide la actividad: datos de entrenamiento contra un periodo posterior. |
| `estacional` | 8 ene - 31 mar 2024, tomado de la misma referencia | 1 ene - 31 mar 2025 | Control de estacionalidad. Compara los mismos meses con un año de diferencia. |
| `mes_2025-01/02/03` | Entrenamiento completo | Cada mes de 2025 por separado | Tendencia: ¿el drift crece con el tiempo? |

**Por qué el control estacional.** La referencia (enero-agosto) incluye vacaciones, la temporada de lluvias y meses con más uso, y el periodo actual es solo invierno. Comparar enero-marzo contra enero-agosto mezcla el cambio real de comportamiento con la estacionalidad normal (limitación señalada en `data_preparation.md`, sección 12). Si una variable deriva en `principal` pero no en `estacional`, el cambio se explica sobre todo por la temporada; si deriva en ambos, hay un cambio año contra año que el modelo no conoce.

Los tres escenarios usan las mismas 362 estaciones del Champion y las mismas transformaciones, así que no hay diferencias de construcción entre conjuntos.

## 3. Configuración de Evidently

- **Versión:** `evidently==0.7.23` (API nueva: `Report`, `Dataset`, `DataDefinition`, `evidently.presets`). Se fijó en `requirements.txt` y `python_env.yaml`.
- **Presets:** `DataDriftPreset` (drift por variable y proporción de columnas con drift) y `DataSummaryPreset` (calidad de datos: filas, duplicados, faltantes, columnas constantes y estadísticas por columna).
- **Columnas monitoreadas:** las 8 variables de entrada de P1 y el objetivo `y`.

| Tipo en Evidently | Columnas | Método | Umbral |
|---|---|---|---|
| Categórica | `hour_of_day`, `weekday`, `fin_semana` | Distancia de Jensen-Shannon | 0.1 |
| Numérica | `salidas_lag_1h`, `salidas_lag_24h`, `salidas_lag_168h`, `media_estacion`, `media_estacion_dia_hora`, `y` | Distancia de Wasserstein normalizada (entre la desviación estándar de la referencia) | 0.1 |
| Identificador / tiempo | `station`, `hour` | No se evalúan para drift | — |

**Decisiones:**

1. **Distancias en lugar de pruebas de hipótesis.** Con millones de filas, pruebas como KS o chi cuadrada dan p-values casi nulos ante diferencias irrelevantes y marcarían drift en todo. Una distancia con umbral mide la magnitud del cambio. Son los métodos que Evidently usa por defecto con más de 1,000 filas; se fijan explícitamente para que el resultado no dependa del tamaño de muestra (por ejemplo, con `--muestra` o en las pruebas).
2. **Hora, día y fin de semana como categóricas.** Son discretas; tratarlas como numéricas mediría una distancia entre 23 h y 0 h que no tiene sentido. Evidently no acepta etiquetas `float` en categóricas, así que solo la copia para el monitoreo se convierte a enteros; los parquets y el contrato del modelo siguen en `float64`.
3. **Umbral 0.1** (default de Evidently). Una Wasserstein normalizada de 0.1 equivale a mover la distribución un 10% de su desviación estándar.
4. **`station` y `hour` como id y timestamp**, para que la prueba de filas duplicadas revise la llave estación-hora real y no filas que solo coinciden en los valores.
5. **Objetivo `y` incluido.** No hay predicciones en producción con etiqueta, pero los datos históricos sí traen `y`: su drift indica *label/prior drift* (la demanda cambió) y anticipa el cambio en el error del modelo.

**Pruebas automáticas de Evidently.** `DataSummaryPreset` genera pruebas que comparan cada estadística con la referencia ±10% (por ejemplo, número de filas o máximo de una columna). Algunas fallan por diseño: el periodo actual es más corto que la referencia, así que la prueba de número de filas siempre falla. Por eso `drift_resumen.json` lista las pruebas que no pasaron y se interpretan, no se toman como errores de datos.

## 4. Revisiones complementarias (pandas)

- **Calidad sobre el contrato de P1:** nulos, duplicados estación-hora, rangos de `hour_of_day` (0-23), `weekday` (0-6) y `fin_semana` (0-1), negativos en rezagos y medias, tipos `float64` y estaciones sin salidas en el periodo.
- **Estadísticas por variable:** media, desviación, mediana, p90 y porcentaje de ceros en ambos conjuntos, con el cambio relativo de la media.
- **Demanda por estación:** `media_estacion` está congelada con el entrenamiento de 2024, así que su distribución casi no puede derivar aunque la demanda real cambie. `demanda_por_estacion.csv` contrasta, por estación, la media congelada con la demanda observada en 2025 (`brecha_congelada`). Esa brecha es el riesgo real de estas dos variables: dejan de representar la estación.

## 5. Cómo ejecutar

Requisitos: `.venv` con `requirements.txt` (incluye Evidently) y los parquets de `data/processed/week9/` (`python -m mibici.monitoring.prepare_data`).

```bash
python -m mibici.monitoring.drift_report               # corrida completa (~2-3 min)
python -m mibici.monitoring.drift_report --muestra 200000 --sin-html   # iteración rápida
python -m pytest tests/test_week9_drift.py -v
ruff check mibici/monitoring/ tests/test_week9_drift.py
```

Opciones: `--datos` (carpeta de parquets), `--salida` (carpeta de resultados), `--muestra N` (N filas al azar por conjunto, semilla 42) y `--sin-html`.

## 6. Artefactos (`reports/week9/drift/`)

| Archivo | Contenido |
|---|---|
| `evidently_drift_principal.html` | Reporte interactivo de Evidently: drift por variable con gráficas y resumen de calidad. Se adjunta al PDF. |
| `evidently_drift_estacional.html` | Mismo reporte para el control estacional. |
| `evidently_drift_*.json` | Snapshot completo de Evidently (métricas y pruebas). |
| `drift_por_variable.csv` | Escenario principal: método, umbral, puntaje, drift y estadísticas descriptivas por variable. |
| `drift_escenarios.csv` | Puntaje y drift por variable en `principal` y `estacional`. |
| `drift_por_mes.csv` | Puntaje y drift por variable para cada mes de 2025. |
| `demanda_por_estacion.csv` | Demanda media por estación en ambos periodos y brecha contra la media congelada. |
| `drift_resumen.json` | Configuración, variables con drift por escenario, pruebas de Evidently, calidad de ambos conjuntos y resumen por estación. |
| `figures/01_drift_por_variable.png` | Puntaje de drift por variable en ambos escenarios contra el umbral. |
| `figures/02_perfil_horario.png` | Demanda media por hora, entre semana y fin de semana, 2024 contra 2025. |
| `figures/03_distribuciones.png` | Histogramas de las variables numéricas y de `y`. |
| `figures/04_drift_por_mes.png` | Mapa de calor del drift por variable y mes. |
| `figures/05_estaciones_media_congelada.png` | Media congelada del Champion contra demanda observada en 2025 por estación. |

## 7. Resultados

Capturas del reporte de Evidently para el PDF en `reports/week9/evidence/drift/`: `01_evidently_drift_principal.png` (tabla de drift por variable con las distribuciones de referencia y actual), `02_evidently_resumen_datos_principal.png` (resumen de calidad del conjunto) y `03_evidently_drift_estacional.png` (control estacional).

Corrida completa con `evidently==0.7.23` sobre los parquets de `prepare_data` (misma versión de datos `7e1470ac0632` y mismos conteos que `reports/week9/data_manifest.json`).

### 7.1 Drift por variable

| Variable | Tipo | Principal | Estacional | Ene 2025 | Feb 2025 | Mar 2025 | Media ref → act |
|---|---|---|---|---|---|---|---|
| `hour_of_day` | cat. | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 11.50 → 11.50 |
| `weekday` | cat. | 0.011 | 0.010 | 0.038 | 0.004 | 0.041 | 2.99 → 3.02 |
| `fin_semana` | cat. | 0.005 | 0.002 | 0.020 | 0.002 | 0.031 | 0.283 → 0.289 |
| `salidas_lag_1h` | num. | 0.006 | 0.015 | 0.030 | 0.015 | 0.007 | 1.476 → 1.465 |
| `salidas_lag_24h` | num. | 0.009 | 0.018 | 0.039 | 0.015 | 0.006 | 1.475 → 1.455 |
| `salidas_lag_168h` | num. | 0.020 | 0.034 | **0.084** | 0.020 | 0.010 | 1.463 → 1.412 |
| `media_estacion` | num. | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.465 → 1.465 |
| `media_estacion_dia_hora` | num. | 0.002 | 0.001 | 0.007 | 0.001 | 0.012 | 1.465 → 1.461 |
| `y` (objetivo) | num. | 0.006 | 0.015 | 0.030 | 0.015 | 0.007 | 1.476 → 1.465 |

Umbral 0.1 en todos los casos. **Ninguna variable presenta drift en ningún escenario** (0 de 9; el conjunto tampoco, con `drift_share` 0.5). El puntaje más alto es 0.084, un 16% por debajo del umbral.

### 7.2 Calidad de datos

- Ambos conjuntos cumplen el contrato: 0 nulos, 0 duplicados estación-hora, 0 valores fuera de rango, 0 negativos y las 8 variables en `float64`.
- Pruebas de Evidently que no pasan:
  - **Número de filas (solo en `principal`):** esperado, porque la referencia cubre casi 8 meses y el periodo actual 3.
  - **Máximo de `y` y de los rezagos:** 86 en la referencia contra 71 en 2025. Son horas pico aisladas; el p99 es 12 en ambos periodos, así que la cola no cambió.
- Las estaciones 104 y 310 no registran salidas en todo el periodo actual.

## 8. Interpretación

**1. No hay drift global en las entradas ni en el objetivo.**
- La demanda media por estación-hora pasa de 1.476 a 1.465 (−0.7%) y la proporción de horas sin viajes de 52.1% a 52.4%.
- El perfil horario entre semana y en fin de semana es prácticamente el mismo (`02_perfil_horario.png`), con los picos de 8 h y 18 h en el mismo lugar y con la misma altura.
- El Champion recibe en 2025 datos dentro de la distribución con la que se entrenó. No hay *covariate shift* ni *label shift* que, por sí solos, anticipen una caída del desempeño agregado.

**2. La estacionalidad pesa más que el paso del tiempo, y aun así es pequeña.**
- Los rezagos y `y` derivan algo más en el control estacional (0.015 contra 0.006) porque enero-marzo de 2024 tuvo más demanda que el promedio de enero-agosto. Por ejemplo, enero de 2024 tuvo 1.557 salidas por estación-hora y enero de 2025 tuvo 1.398, un 10% menos.
- Promediar ocho meses diluye esa diferencia; compararlos mes con mes la deja ver, pero ni así se acerca al umbral.

**3. El pico de enero es un efecto de calendario, no una tendencia.**
- El único puntaje relevante es `salidas_lag_168h` en enero (0.084).
- En la primera semana de enero ese rezago toma la semana de Navidad: 0.84 salidas en promedio contra 1.04 observadas. En febrero y marzo cae a 0.020 y 0.010.
- El drift no se acumula mes a mes, pero el modelo no conoce los días festivos y en esas semanas recibe rezagos atípicos.

**4. El riesgo está por estación, oculto por el agregado.**
- `media_estacion` y `media_estacion_dia_hora` tienen drift casi nulo por construcción: están congeladas con el entrenamiento de 2024. Lo relevante es si siguen representando la demanda de cada estación (`05_estaciones_media_congelada.png`, `demanda_por_estacion.csv`).
- La brecha media absoluta entre la media congelada y la demanda observada es de 0.12 salidas por hora, cerca del 8% de la media.
- 44 de 362 estaciones cambian más de 20%, y 11 más de 50% (3 bajan y 8 suben). Juntas pesan solo el 6% de la demanda, por eso no mueven la distribución global.
- Casos concretos:
  - **Estación 180:** baja de 2.05 a 0.55 salidas por hora (−73%).
  - **Estación 333:** sube de 1.19 a 2.39 (+99%).
  - **Estaciones 389 y 390:** suben más de 260% sobre una base baja.
  - **Estaciones 104 y 310:** sin salidas en 2025, con medias congeladas de 0.09 y 0.17. El modelo seguirá prediciéndoles demanda positiva.
- Como predictor ingenuo, `media_estacion_dia_hora` empeora: su MAE contra `y` sube de 0.794 a 0.866 (+9%). El rezago de 168 h se mantiene estable (1.018 → 0.998). Las medias históricas envejecen aunque las distribuciones no cambien.

**5. Implicaciones para P1.**
- Sin drift global, el desempeño agregado del Champion en 2025 debería parecerse al de su prueba de 2024 (MAE 0.828). Esto lo confirma o lo descarta el A/B testing sobre marzo de 2025.
- La degradación esperable se concentra en las estaciones que cambiaron de nivel. Los rezagos se adaptan solos a la demanda reciente, pero las medias congeladas arrastran el nivel de 2024, de modo que el error crece justo donde la operación cambió.
- Un Challenger que recalcule las medias con enero-febrero de 2025 ataca directamente esa fuente de error. Debe hacerlo sin usar marzo, como indica `data_preparation.md`.

**6. Insumos para la recomendación de Continuous Training.**
- El drift de datos no justifica, por sí solo, reentrenar de urgencia: ninguna variable cruza el umbral y el drift no crece mes a mes.
- Sí justifica:
  1. Recalcular periódicamente las medias históricas, que son la parte del modelo que envejece.
  2. Agregar al monitoreo una vista por estación con alertas, por ejemplo cambio de demanda mayor a 50% o estación sin salidas durante 7 días.
  3. Disparar el reentrenamiento por evidencia: drift ≥ 0.1 en alguna variable o degradación del MAE en el A/B, en lugar de un calendario fijo.
- La decisión final se toma con el resultado del A/B testing.
