# Semana 11 - Consolidación del repositorio y experimentación

**Proyecto:** MiBici - TC5061 · **Equipo:** 15 · **Actividad:** Proyecto integrador final · **Criterio:** Estructura, datos y experimentación (20 puntos)

## 1. Objetivo

Consolidamos y verificamos la estructura, la preparación de datos, la ingeniería de características y el seguimiento de experimentos del proyecto MiBici, aprovechando lo construido en las semanas anteriores. No modificamos código, datos versionados, corridas históricas ni alias del Model Registry: todo se validó con ejecuciones nuevas en un servidor aislado y con consultas de solo lectura al histórico.

## 2. Entorno de validación

| Componente | Resultado |
|---|---|
| Sistema operativo | macOS (Darwin) |
| Versión de Python | 3.12.15 (`.venv` y entornos `virtualenv` de MLflow) |
| Versión de MLflow | 3.16.1 |
| Rama de trabajo | `feature/week11-repository-experimentation-A01252855` |
| Commit de partida | `f5d230e` (`main` actualizado con `git pull origin main`) |
| Validación de dependencias | `pip check`: `No broken requirements found.` |

Trabajamos sobre el repositorio del equipo actualizado con `main`, reutilizando los CSV de `data/raw/` que ya estaban descargados. Las dependencias se reinstalaron desde `requirements.txt`, `requirements-dev.txt` y `requirements-api.txt`, y los dos entry points se ejecutaron con `--env-manager virtualenv`, de modo que MLflow creó desde cero el entorno declarado en `python_env.yaml`.

| Servidor | Puerto | Uso |
|---|---|---|
| Validación (`validation.db`, fuera del repositorio) | 5051 | Ejecuciones nuevas de `ingest` y `train`, y pruebas |
| Histórico (copia de `runtime/mlflow`, fuera del repositorio) | 5000 | Consulta de las corridas y nested runs del equipo, solo lectura |

## 3. Estructura del repositorio

Verificamos la organización basada en Cookiecutter Data Science: `data/` (`raw`, `interim`, `processed`, `external` y `manifiesto.json`), `mibici/` como paquete instalable (`dataset.py`, `features.py`, `config.py`, `data_validation.py`, `modeling/`, `monitoring/`, `api/`), `notebooks/`, `models/`, `tests/`, `docs/`, `reports/` y `references/`, además de `MLproject`, `python_env.yaml`, `pyproject.toml` y los archivos de contenedores y CI/CD.

La separación de responsabilidades permite que la ingesta, las características, el entrenamiento, el monitoreo y la API evolucionen por separado y se prueben de forma independiente. `.gitignore` excluye los datos crudos y procesados, `.venv`, `mlflow.db` y `runtime/`.

## 4. Instalación y reproducibilidad

| Paquete | Versión |
|---|---|
| mlflow | 3.16.1 |
| numpy | 2.5.3 |
| pandas | 3.0.6 |
| scikit-learn | 1.9.1 |
| pyarrow | 25.0.1 |
| pytest | 9.1.1 |
| evidently | 0.7.23 |
| deepchecks | 0.19.1 |
| fastapi | 0.162.0 |

Resultado de la validación: `pip check` sin conflictos e importación correcta de `mibici`, `mlflow`, `sklearn`, `pandas` y `pyarrow`. DeepChecks se importa mediante `mibici.data_validation.cargar_deepchecks()`, que aplica las correcciones de compatibilidad documentadas en el README.

## 5. Ingesta y versionamiento de datos

La ingesta procesa los 12 archivos mensuales de viajes de 2024 y la nomenclatura de estaciones mediante el entry point `ingest`.

| Indicador | Referencia histórica | Validación final |
|---|---|---|
| Archivos mensuales | 12 | 12 + nomenclatura (13 fuentes) |
| Viajes limpios | 4,645,921 | 4,645,921 (8 columnas); 484 estaciones |
| Registros descartados | 0 | 0 sin inicio u origen, 0 duplicados |
| Versión de datos | `7e1470ac0632` | `7e1470ac0632` |
| Validación SHA-256 | - | 13/13 hashes coinciden |
| Ejecución MLflow Project | FINISHED | `aed142aa…` FINISHED (`virtualenv`, 2.6 min) |

Antes de ejecutar respaldamos el manifiesto fuera del repositorio. Al terminar recalculamos el SHA-256 de cada fuente y lo comparamos contra el manifiesto generado y contra el respaldo: coinciden los 13. `git diff -- data/manifiesto.json` no muestra cambios, así que la ingesta es determinista y reproducible.

## 6. Feature Engineering

El proyecto integra tres preguntas de negocio:

- **P1:** predicción de demanda de bicicletas por estación y hora.
- **P2:** clasificación del riesgo operativo mediante una variable proxy.
- **P3:** agrupamiento de estaciones según sus patrones de utilización.

`mibici/features.py` expone 9 funciones: `hourly_departures`, `hourly_arrivals`, `calendar_features`, `hourly_flow_balance`, `proxy_risk_labels`, `hourly_usage_profiles`, `complete_hourly_panel`, `lag_features` y `trip_quality_flags`. Con una muestra de 100,000 viajes comprobamos las salidas por estación y hora, las llegadas, las variables de calendario, el balance de flujos y los perfiles horarios de P3.

| Partición (P1 y P2) | Periodo |
|---|---|
| Entrenamiento | enero a agosto de 2024 (`< 2024-09-01`) |
| Validación | septiembre y octubre (`< 2024-11-01`) |
| Prueba | noviembre y diciembre (`< 2025-01-01`) |

El modelo P1 utiliza 8 variables: `hour_of_day`, `weekday`, `fin_semana`, `salidas_lag_1h`, `salidas_lag_24h`, `salidas_lag_168h`, `media_estacion` y `media_estacion_dia_hora`; su contrato se conserva en la documentación de la semana 8. P3 trabaja con perfiles anuales por estación, tipo de día y hora. `tests/test_features.py`: 32 pruebas aprobadas.

## 7. MLflow Project

Verificamos los dos puntos de entrada declarados en `MLproject` (`name: mibici-equipo15`, `python_env: python_env.yaml`):

- `ingest`: preparación de datos (`python -m mibici.dataset`).
- `train`: ejecución parametrizada de los modelos (`question`, `model`, `n_estimators`, `max_depth`, `learning_rate`, `max_iter`).

Ejecuciones nuevas realizadas en el servidor aislado:

| Ejecución | Run ID | Resultado |
|---|---|---|
| `mlflow run . -e ingest` | `aed142aa…` | FINISHED |
| `mlflow run . -e train -P question=p1 -P model=baseline` | `2c5a7d59…` | FINISHED |
| `P1_baseline_station_mean` | `dba86425…` | test MAE 1.3621 · RMSE 2.1185 |
| `P1_baseline_station_weekday_hour_mean` | `ebcace5d…` | test MAE **0.8556** · RMSE 1.4635 |

El baseline de P1 reproduce exactamente la métrica histórica (MAE test 0.8556), con 362 estaciones en entrenamiento, 8 ausentes y 2,119,872 / 529,968 / 529,968 filas estación-hora en train, validación y prueba. P2 y P3 se validaron con sus corridas históricas y con las pruebas automatizadas, como recomienda la guía.

## 8. MLflow Tracking

Consultamos el experimento `mibici-equipo15` y verificamos la existencia de parámetros, métricas, etiquetas, identificadores de corridas y artefactos.

- Fuente del servidor consultado: copia del registro del equipo (`runtime/mlflow`) servida en `http://127.0.0.1:5000`, sin modificar el original.
- Total de corridas consultadas: 88 en `mibici-equipo15`, todas FINISHED (P1: 9, P2: 9, P3: 70).

Distinguimos las corridas históricas, que consultamos sin modificar, de las 4 corridas nuevas registradas en el servidor aislado (5051).

## 9. Nested Runs y selección de candidatos

| Búsqueda | Run ID padre | Corridas hijas esperadas | Corridas verificadas |
|---|---|---|---|
| P1 | `84b3d18f…` | 3 | 3 |
| P2 | `ccb22777…` | 3 | 3 |
| P3 - K-Means + PCA | `b9cfd2f3…` | 21 | 21 |
| P3 - Jerárquico | `8a5ccaf1…` | 21 | 21 |
| P3 - GMM + PCA | `23fd4529…` | 21 | 21 |
| **Total** | **5** | **69** | **69, 0 errores** |

| Búsqueda | Criterio | Candidato seleccionado |
|---|---|---|
| P1 | Minimizar MAE de validación | HGB `max_iter=300`, `learning_rate=0.1` · val MAE 0.8396 (run `6e3429f2…`, champion) |
| P2 | Maximizar Average Precision de validación | Random Forest `n_estimators=300`, `max_depth=12` · val AP 0.3765 |
| P3 - K-Means + PCA | Silhouette + admisibilidad | `k=2`, 5 componentes · silhouette 0.1276 |
| P3 - Jerárquico | Silhouette + admisibilidad | `k=2`, `ward` · silhouette 0.1063 |
| P3 - GMM + PCA | Silhouette + admisibilidad | `k=2`, 10 componentes, covarianza `full` · silhouette 0.1325 |

En todos los casos el `best_child_run_id` del padre apunta a una hija suya. En P3 no se elige solo por el mayor silhouette: en el jerárquico, `k=2` con `complete` tiene silhouette 0.526, pero un grupo de una sola estación (0.27 %), por debajo del mínimo de 5 %, y quedó como no admisible.

Las consultas de validación no modificaron los modelos registrados ni sus alias de producción.

## 10. Pruebas automatizadas

Ejecutamos las pruebas de ingesta, feature engineering, calidad de datos, entrenamiento, reproducibilidad y búsqueda de candidatos, con el Tracking URI apuntando al servidor aislado.

| Archivo | Aprobadas |
|---|---|
| `test_dataset.py` | 33 |
| `test_features.py` | 32 |
| `test_data_validation.py` (DeepChecks) | 25 |
| `test_model_training.py` | 28 |
| `test_model_reproducibility.py` | 7 |
| `test_p1_p2_search.py` | 7 |

| Indicador | Pruebas de Jorge | Suite completa (`tests/`) |
|---|---|---|
| Pruebas aprobadas | 132 | 315 |
| Pruebas fallidas | 0 | 0 |
| Pruebas omitidas | 0 | 0 |
| Advertencias | 0 | 24 (deprecación de librerías externas) |
| Código de salida | 0 | 0 |

El reporte completo queda en `08_pytest_experimentation.txt` y `pytest_experimentation.xml`.

## 11. Hallazgos y limitaciones

- **Dependencias:** `requirements.txt` declara `psutil` y `skops`, que no están en `python_env.yaml`. No afectan `ingest` ni `train`, pero conviene alinear ambos archivos.
- **DeepChecks:** la versión 0.19.1 no puede importarse directamente con scikit-learn 1.9 porque busca el scorer `max_error`. El proyecto lo resuelve con `cargar_deepchecks()`, y las 25 pruebas de calidad corren sin omitirse.
- **Advertencias:** las 24 de la suite completa vienen de litestar, de SQLAlchemy dentro de MLflow y de evidently con pandas 4; ninguna proviene del código del proyecto.
- **Etiquetas:** las corridas de P3 usan la etiqueta `question` en minúsculas (`p3`), salvo una (`P3`); P1 y P2 usan mayúsculas.
- **Fuente histórica:** la consulta se hizo sobre la copia local del registro del equipo; el `mlflow.db` de la raíz contiene corridas locales repetidas y no se usó como referencia.
- **Alcance:** no reejecutamos las búsquedas completas de candidatos (opcionales en la guía); se verificaron con sus corridas históricas y con `test_p1_p2_search.py`.
- **Datos:** la ingesta reutilizó los CSV ya descargados; los hashes confirman que son idénticos a los versionados.

## 12. Conclusiones

La consolidación nos permitió verificar la estructura del proyecto, la preparación de datos, la ingeniería de características y el seguimiento de experimentos. Reprodujimos la ingesta y el baseline de P1 desde MLflow Projects con entornos `virtualenv` nuevos, con la misma versión de datos (`7e1470ac0632`) y la misma métrica histórica (MAE 0.8556). Las búsquedas de candidatos, los nested runs (5 padres y 69 hijas) y la selección de modelos se verificaron sobre las corridas históricas del equipo, y las 132 pruebas relacionadas, así como la suite completa de 315, se aprobaron sin fallas ni omisiones.

## 13. Documentación relacionada

- [README general del proyecto](../../README.md)
- [Contrato del modelo P1](../week8/model_registry_contract.md)
- [Contrato de datos](../week9/data_contract.md)
- [Preparación de datos](../week9/data_preparation.md)
- [MLflow Project](../../MLproject)
- [Pruebas automatizadas](../../tests/)

## 14. Evidencias

Carpeta: `reports/week11/evidence/experimentation/`

| Archivo | Contenido |
|---|---|
| `00_rama_trabajo.png` | Rama de trabajo y commit de partida |
| `01_estructura_repositorio.png` | Estructura Cookiecutter y módulos de `mibici` |
| `02_entorno_dependencias.png` | Python, versiones, `pip check` y comparación con `python_env.yaml` |
| `03_ingesta_manifiesto.png` | Run de `ingest`, parquets y manifiesto |
| `03b_validacion_hashes.png` | Comparación de hashes SHA-256 |
| `04_feature_engineering.png` | Funciones, features de P1, particiones y salidas |
| `04b_feature_engineering_tests.png` | Pruebas de features |
| `05_mlflow_project.png` | `MLproject`, entry points y `--help` |
| `05b_mlflow_project_execution.png` | Ejecución del baseline de P1 en el servidor aislado |
| `06_mlflow_experiments.png` | Experimento `mibici-equipo15` en MLflow Tracking |
| `07_nested_runs.png` | Corridas padre e hijas |
| `07a_nested_runs_p1_p2.png` | Nested runs de P1 y P2 |
| `07b_nested_runs_p3.png` | Las tres familias de P3 con sus 63 hijas |
| `07c_nested_runs_validation.png` | Validación automatizada: 5 padres y 69 hijas |
| `07d_candidate_selection.png` | Selección de candidatos |
| `08_pruebas_datos_modelos.png` | Pruebas de datos y modelos (132) |
| `08a_pytest_data_validation.png` | Calidad de datos con DeepChecks |
| `08b_pytest_reproducibility.png` | Reproducibilidad del entrenamiento |
| `08c_pytest_full_suite.png` | Suite completa (315) |
| `01_repository_structure.txt`, `02_environment_packages.txt`, `02_pip_check.txt`, `04_feature_engineering_tests.txt`, `08_pytest_experimentation.txt`, `pytest_experimentation.xml` | Salidas en texto de los comandos |
