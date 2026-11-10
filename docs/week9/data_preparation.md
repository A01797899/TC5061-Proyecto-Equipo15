# Semana 9 - Preparación de datos de referencia y actuales

## 1. Objetivo

Preparar datos confiables y reproducibles de MiBici para tres usos de la semana 9: el monitoreo de drift con Evidently (Óscar), el entrenamiento del Challenger (José Luis) y el A/B testing entre Champion y Challenger (José Manuel). Los datos actuales deben tener exactamente las mismas 8 variables que espera el Champion P1, sin información futura.

## 2. Champion P1 y sus variables

El modelo es `mibici-equipo15-p1-demanda`, versión `1`, alias `champion`, run `6e3429f222a0407aa159dd9f46699365` (URI `models:/mibici-equipo15-p1-demanda@champion`). Predice `y`, el número de salidas de una estación en una hora, con 8 variables `float64`: `hour_of_day`, `weekday`, `fin_semana`, `salidas_lag_1h`, `salidas_lag_24h`, `salidas_lag_168h`, `media_estacion` y `media_estacion_dia_hora`. El detalle de cada una está en [`data_contract.md`](data_contract.md).

## 3. Periodo histórico original

| Conjunto | Periodo | Uso en la semana 9 |
|---|---|---|
| Entrenamiento | 1 ene - 31 ago 2024 | Referencia de drift y origen de las medias congeladas |
| Validación | 1 sep - 31 oct 2024 | No se usa |
| Prueba | 1 nov - 31 dic 2024 | Solo como historia para los rezagos de enero de 2025 |

## 4. Archivos de enero a marzo de 2025

| Archivo | Registros | Encoding | Formato de fecha |
|---|---|---|---|
| `datos_abiertos_2025_01.csv` | 380,184 | `latin1` | `%Y-%m-%d %H:%M:%S` |
| `datos_abiertos_2025_02.csv` | 372,529 | `latin1` | `%Y-%m-%d %H:%M:%S` |
| `datos_abiertos_2025_03.csv` | 405,110 | `utf-8` | `%Y-%m-%d %H:%M:%S` |

## 5. Fuentes y hashes

| Archivo | URL | SHA-256 |
|---|---|---|
| `datos_abiertos_2025_01.csv` | https://mibici.net/site/assets/files/97653/datos_abiertos_2025_01.csv | `90ee75701a69374926af0400d204f3280baed0e3c1a1f97d9e6c0b283757125a` |
| `datos_abiertos_2025_02.csv` | https://mibici.net/site/assets/files/99879/datos_abiertos_2025_02.csv | `f52f13f59eb2f1517e05c72f0a3d3d70c5b0fa6b86626c7325e8ca3b15ff74f8` |
| `datos_abiertos_2025_03.csv` | https://mibici.net/site/assets/files/101986/datos_abiertos_2025_03.csv | `e5aa1b5d197f294022d3f0ff1b55bf553fcbdb697b50d43deaddf8dca36fed64` |

Los datos de 2024 son los de la ingesta histórica (`data/manifiesto.json`, versión `7e1470ac0632`), que no se modifica.

## 6. Diferencias entre los esquemas de 2024 y 2025

Los tres archivos tienen las mismas 8 columnas, en el mismo orden y con el mismo formato de fecha que 2024. La única diferencia es el encoding: marzo viene en `utf-8`, igual que julio de 2024, y el resto en `latin1`. Por eso `prepare_data.py` detecta el encoding y el formato de fecha de cada archivo en lugar de usar las reglas fijas por mes de 2024. La información demográfica sí viene en 2025: género y año de nacimiento vacíos son menos del 1%, igual que en 2024.

## 7. Reglas de limpieza y validación

| Tipo | Regla | Referencia 2024 | Actual 2025 |
|---|---|---|---|
| Descarte | Sin inicio u origen | 0 | 0 |
| Descarte | Inicio fuera del periodo o del mes de su archivo | 0 | 0 |
| Validación | `viaje_id` o filas duplicadas | 0 | 0 |
| Anomalía conservada | Género vacío | 17,664 | 8,897 |
| Anomalía conservada | Año de nacimiento vacío | 2,046 | 456 |
| Anomalía conservada | Duración fuera de [1, 180] min | 110,101 | 28,790 |
| Anomalía conservada | Edad fuera de [10, 90] | 742 | 173 |
| Anomalía conservada | Viajes de estaciones nuevas | 0 | 11,957 |

Las anomalías se conservan porque P1 cuenta inicios de viaje, igual que en el entrenamiento original. El detalle está en `reports/week9/data_quality.json`.

## 8. Panel estación-hora

Se cuentan las salidas por estación y hora, y se completa el panel de las 362 estaciones del Champion (las que tuvieron viajes en el entrenamiento) con `p1_baseline.panel()`, que rellena con 0 las horas sin viajes. Es un solo panel continuo del 1 de enero de 2024 al 31 de marzo de 2025.

- Las 8 estaciones nuevas de 2025 (391 a 398) no entran al panel, porque el Champion nunca las vio; se reportan aparte.
- Las estaciones 104 y 310 no tienen viajes en 2025 y quedan con 0 salidas todo el periodo. Un 0 no prueba que la estación estuviera operando, así que conviene revisarlas antes de interpretar cambios de demanda.

## 9. Transformaciones y medias históricas

Se reutilizan las funciones del entrenamiento del Champion: `asignar_split()`, `agregar_rezagos()` (con `lag_features()`) y `agregar_historicos()`.

- Los rezagos de 1, 24 y 168 horas se buscan por fecha en el panel continuo, así que el 1 de enero de 2025 usa diciembre de 2024 y no se pierde ninguna fila.
- `media_estacion` y `media_estacion_dia_hora` se calculan solo con el entrenamiento de 2024 y quedan congeladas para 2025.
- La referencia descarta las primeras 168 horas sin rezagos completos, igual que el Champion: 2,059,056 filas.

**Comprobación contra el Champion.** Con el modelo registrado y las variables reconstruidas, el MAE y el RMSE coinciden hasta el décimo decimal con los que registró la corrida original (train 0.7812010963 y 1.3224130282, validación 0.8395558696 y 1.4280638316, prueba 0.8279171026 y 1.3930097154). Las variables son idénticas a las que vio el modelo.

## 10. Particiones

| Archivo | Periodo | Filas | Uso |
|---|---|---|---|
| `reference_p1.parquet` | 8 ene - 31 ago 2024 | 2,059,056 | Referencia de drift |
| `current_p1.parquet` | 1 ene - 31 mar 2025 | 781,920 | Datos actuales de drift |
| `train_alternative.parquet` | 1 ene - 28 feb 2025 | 512,592 | Entrenar el Challenger |
| `evaluation.parquet` | 1 - 31 mar 2025 | 269,328 | Comparar Champion y Challenger |

La separación es por fecha, no aleatoria, porque los datos dependen del tiempo. La última hora de entrenamiento es anterior a la primera de evaluación y no se repite ninguna llave estación-hora. Marzo no debe usarse para ajustar el Challenger. Si el Challenger recalcula las medias históricas, debe hacerlo solo con enero y febrero de 2025.

Todos los archivos tienen `station`, `hour`, `y` y las 8 variables `float64` sin nulos, así que sirven tanto para monitoreo como para entrenamiento supervisado.

## 11. Cómo ejecutar el proceso

Requisitos: `.venv` con `requirements.txt`, la ingesta de 2024 en `data/processed/viajes_2024.parquet` (`python -m mibici.dataset`) y los 3 CSV de 2025 en `data/external/week9_2025/`.

```bash
mkdir -p data/external/week9_2025
curl -L --fail --retry 5 --retry-all-errors -o data/external/week9_2025/datos_abiertos_2025_01.csv "https://mibici.net/site/assets/files/97653/datos_abiertos_2025_01.csv"
curl -L --fail --retry 5 --retry-all-errors -o data/external/week9_2025/datos_abiertos_2025_02.csv "https://mibici.net/site/assets/files/99879/datos_abiertos_2025_02.csv"
curl -L --fail --retry 5 --retry-all-errors -o data/external/week9_2025/datos_abiertos_2025_03.csv "https://mibici.net/site/assets/files/101986/datos_abiertos_2025_03.csv"
shasum -a 256 data/external/week9_2025/*.csv

python -m mibici.monitoring.prepare_data
python -m mibici.monitoring.data_quality
python -m pytest tests/test_week9_data.py -v
```

Los hashes deben coincidir con la sección 5. `prepare_data` también acepta `--etapa viajes`, `--etapa p1` o `--etapa particiones` para correr una sola parte.

| Salida | Contenido | En Git |
|---|---|---|
| `data/processed/week9/*.parquet` | Viajes de 2025 y los 4 datasets de P1 | No |
| `reports/week9/data_quality.json` | Calidad de ambos periodos | Sí |
| `reports/week9/data_manifest.json` | Fuentes, hashes, periodos, filas, estaciones, Champion, commit y reglas | Sí |

## 12. Resultados, limitaciones y evidencias

- Los CSV de 2025 son compatibles con 2024, sin descartes ni duplicados.
- Las variables de referencia reproducen exactamente las del Champion.
- Limitación: comparar enero-marzo con enero-agosto mezcla cambios reales con estacionalidad normal, así que conviene segmentar el drift por hora y día de la semana.
- Limitación: las estaciones nuevas no se evalúan con el Champion, y un 0 en el panel no distingue una estación sin demanda de una estación cerrada.

Evidencias en `reports/week9/evidence/data/`:

| Archivo | Evidencia |
|---|---|
| `01_entorno_pruebas_iniciales.png` | Rama, entorno y pruebas iniciales |
| `02_contrato_p1.png` | Las 8 variables y las funciones del entrenamiento |
| `03_periodo_referencia.png` | Periodos de `config.py`, `dropna` y manifiesto de 2024 |
| `04_descarga_compatibilidad_2025.png` | Descarga, hashes y columnas de 2025 contra 2024 |
| `05_carga_periodos.png` | Carga de ambos periodos |
| `06_calidad_datos.png` | Calidad de datos |
| `07_variables_p1.png` | Datasets de P1 y validación del contrato |
| `08_particiones.png` | Particiones sin superposición |
| `09_documentacion_manifiesto.png` | Manifiesto de datos y documentación |
| `10_pruebas_week9.png` | Pruebas de la semana 9, suite completa y ruff |
