# Semana 11 - Validación del Model Registry y gobernanza

**Proyecto:** MiBici - TC5061 · **Equipo:** 15 · **Actividad:** Proyecto integrador final · **Criterios:** Modelo y Registry; monitoreo y gobernanza · **Responsable:** José Luis Mendoza Mora (A01732182)

## 1. Objetivo

Consolidamos y verificamos los modelos registrados durante las semanas anteriores, comprobando sus versiones, alias, linaje, firmas, hiperparámetros, métricas y disponibilidad para inferencia, sobre una copia del Registry histórico que no se modifica. Complementamos la documentación del proyecto con el [Model Card de P1](model_card.md), que reúne gobernanza, desempeño, monitoreo, riesgos y uso responsable.

Todas las consultas fueron de solo lectura. No registramos versiones, no movimos los alias `champion` ni `challenger`, no promovimos el Challenger y no editamos el README.

## 2. Ambiente de validación

| Componente | Resultado |
|---|---|
| Rama de trabajo | `feature/week11-registry-model-card-A01732182` |
| Commit de partida | `1230cff4f74d32f3d7ee80f597a758feeb225888` (`main`, PR #44). Antes del commit de esta actividad la rama se adelantó con fast-forward a `9252498` (`main`, PR #45 de Oscar), que solo añade `docs/week11/monitoring_performance_finops.md` |
| Sistema operativo | Ubuntu 24.04.4 LTS, Linux 6.17.0-1008-nvidia, `aarch64` |
| Python | 3.12.13 (`.venv` con `requirements.txt`, `requirements-dev.txt` y `requirements-api.txt`) |
| MLflow | 3.16.1 (scikit-learn 1.9.1, pandas 3.0.6, numpy 2.5.3, skops 0.16.0) |
| `pip check` | Sin conflictos |
| Tracking URI | `http://127.0.0.1:5052` (`mlflow server --serve-artifacts`, backend SQLite) |
| Origen del Registry | Copia del respaldo `mlflow-runtime-week10.tar.gz` de José Manuel, extraída fuera del repositorio en `../mibici-week11-joseluis-registry/` (`mlflow.db` + `mlartifacts/`, 162 archivos de artefactos) |
| Integridad de SQLite | `PRAGMA integrity_check` = `ok` |
| Fecha de la ejecución | 26 de noviembre de 2026 |

Los puertos quedaron separados: `5000` para el Registry original o Compose, `5051` para la validación de Jorge y `5052` para esta validación. Las pruebas automatizadas apuntaron a un directorio temporal (sección 9). Evidencias: `00_git_environment.txt`, `00_environment_packages.txt`, `00_pip_check.txt`, `00_registry_connection.txt`.

## 3. Modelos registrados

Los nombres oficiales de `mibici/config.py` (`REGISTERED_MODELS`) coinciden con los tres modelos del Registry. La selección histórica fue: P1 por MAE de validación (`HistGradientBoostingRegressor`), P2 por Average Precision de validación (`RandomForestClassifier`) y P3 por estabilidad, admisibilidad e interpretación (K-Means + PCA). P1 es el único que sirve FastAPI.

| Modelo | Versiones | Alias | Champion esperado | Champion verificado | Estado | Run ID del Champion |
|---|---:|---|---:|---:|---|---|
| `mibici-equipo15-p1-demanda` | 2 | `champion` → 1, `challenger` → 2 | 1 | 1 | `READY` | `6e3429f222a0407aa159dd9f46699365` |
| `mibici-equipo15-p2-riesgo` | 1 | `champion` → 1 | 1 | 1 | `READY` | `d794555acb0c498987d17fd61c0ae531` |
| `mibici-equipo15-p3-grupos` | 1 | `champion` → 1 | 1 | 1 | `READY` | `27f5684907f64610a4d3718d1178421d` |

La validación automatizada de alias terminó con `Errores encontrados: 0`. El Registry contiene en total cuatro versiones y cuatro alias. Evidencia: `01_registered_models.txt`.

## 4. Versiones Champion y Challenger de P1

| Propiedad | Champion | Challenger |
|---|---|---|
| Versión | 1 | 2 |
| Alias | `champion` | `challenger` |
| Estado | `READY` | `READY` |
| Run ID | `6e3429f222a0407aa159dd9f46699365` | `7074129578d045418cbed0691330d1a7` |
| Source | `models:/m-04774b8f25ff468b933a18756c24b068` | `models:/m-ea1a3789403948d08f509f0a562419e5` |
| Tags de la versión | Ninguno | `team=15`, `question=P1`, `experiment=mibici-equipo15`, `data_version=7e1470ac0632`, `registration_commit=ddedf4beb63e99bdd96deeb6886d33348afa2041`, `stage_of_work=candidate` |
| Validación final | Coincide con la tabla de la sección 3 de la guía | Coincide con la tabla de la sección 3 de la guía |

Las aserciones del script del PDF (versiones 1 y 2, ambas `READY`, Run IDs distintos e iguales a los históricos) se cumplieron: `ALIASES Y VERSIONES VALIDADOS`. `READY` indica disponibilidad en el Registry; la versión de producción es la que señala el alias `champion`. El snapshot inicial quedó en `02_registry_snapshot_before.json` con las cuatro versiones; la salida del script en `02b_alias_validation.txt`.

## 5. Linaje de los modelos

| Propiedad | Champion v1 | Challenger v2 |
|---|---|---|
| Experimento | `mibici-equipo15` (id 1) | `mibici-equipo15` (id 1) |
| Nombre de la corrida | `P1_candidate_hgb_max_iter=300_learning_rate=0.1_max_depth=None` | `P1_challenger_2025_01_02` |
| Estado de la corrida | `FINISHED` | `FINISHED` |
| Corrida padre | `P1_search_candidates` (`84b3d18fea4a49dbb1c212e9311da9c7`) | Ninguna |
| Logged Model | `m-04774b8f25ff468b933a18756c24b068` (`READY`), único con `source_run_id` igual al run | `m-ea1a3789403948d08f509f0a562419e5` (`READY`), único con `source_run_id` igual al run |
| Commit de entrenamiento (`mlflow.source.git.commit`) | `ad81f296c6aea3c97935399dd389c6a02c25d713` (PR #18, 25 de octubre de 2026; existe en el clon) | `ddedf4beb63e99bdd96deeb6886d33348afa2041` (merge del PR #33, 15 de noviembre de 2026; existe en el clon) |
| Rama al entrenar | `main` | `feature/week9-ab-final-A01797899` |
| Script de origen | `mibici/modeling/p1_p2_search.py` | `scripts/week9/train_challenger.py` |
| Versión de datos | Tag de la corrida `data_version=7e1470ac0632`, `dataset=MiBici 2024` | Tag de la versión `data_version=7e1470ac0632`; los CSV de 2025 están en `reports/week9/data_manifest.json` |
| Periodo de entrenamiento | 2024-01-01 a 2024-09-01, 2,059,056 filas | 2025-01-01 a 2025-02-28, 512,592 filas |
| Artefactos de la corrida | `config/` (`MLproject`, `config.py`, `python_env.yaml`, `run_params.json`), `data/` (`manifiesto.json`, `csv_manifest.json`), `metrics/metrics.json`, `plots/` (2 PNG) | Ninguno |
| Artefactos del modelo | `MLmodel`, `model.skops`, `conda.yaml`, `python_env.yaml`, `requirements.txt`, `input_example.json`, `serving_input_example.json` | Los mismos siete archivos |
| Model URI | `models:/mibici-equipo15-p1-demanda/1` | `models:/mibici-equipo15-p1-demanda/2` |

El commit de entrenamiento del Challenger, que el PDF dejaba pendiente, es `ddedf4beb63e99bdd96deeb6886d33348afa2041`; coincide con el tag `registration_commit` de la versión 2. El commit del Champion coincide con el publicado en el contrato de la semana 8. La identidad de los datos de 2025 del Challenger no se demuestra con el tag `7e1470ac0632`, que corresponde a la ingesta de 2024; se apoya en los SHA-256 de los tres CSV de 2025 que registra `reports/week9/data_manifest.json` (`90ee7570…`, `f52f13f5…`, `e5aa1b5d…`). Evidencias: `03_model_lineage.json`, `03_model_lineage.txt`.

## 6. Firmas del modelo

`scripts/week8/inspect_registry.py` terminó con `VALIDACIÓN DEL REGISTRY COMPLETADA` (`04_registry_inspection.txt`): dos versiones `READY`, alias `champion` en la versión 1 y firma con ocho entradas `double` obligatorias y salida `Tensor('float64', (-1,))`.

| Modelo | Entradas esperadas | Resultado en la copia |
|---|---:|---|
| P1 Champion v1 | 8 | Confirmadas: `hour_of_day`, `weekday`, `fin_semana`, `salidas_lag_1h`, `salidas_lag_24h`, `salidas_lag_168h`, `media_estacion`, `media_estacion_dia_hora`; flavors `python_function` y `sklearn` |
| P1 Challenger v2 | 8 | Confirmadas, mismas columnas, mismos tipos, misma salida |
| P2 Champion v1 | 15 | No confirmada: el logged model `m-165c943b3eec4915aabc8e801feb9979` no tiene `MLmodel` en `mlartifacts/1/models/` de la copia |
| P3 Champion v1 | 49 | No confirmada: el logged model `m-041658517cae46178fa30b582bd44b22` no tiene `MLmodel` en `mlartifacts/1/models/` de la copia |

Champion y Challenger de P1 comparten el contrato completo: `signature.to_dict()` es idéntico en ambas versiones, las ocho columnas coinciden en nombre y orden con `P1_FEATURES` de `mibici/api/schemas.py`, todas son `double` y obligatorias: `CONTRATOS CHAMPION Y CHALLENGER COMPATIBLES`. Las cifras de 15 y 49 entradas de P2 y P3 provienen de la descripción de cada modelo registrado y del README (semana 6), no de una lectura de la firma en esta copia.

La firma no sustituye la validación semántica: enteros requieren conversión a `float64`, columnas faltantes se rechazan, columnas extra se ignoran, y valores nulos o fuera de rango producen una predicción. Por eso la API valida con Pydantic y `ModelService` (tipos, nombres, rangos, consistencia de calendario, resultados finitos y recorte de negativos a cero). Evidencia: `04c_signature_comparison.txt`.

## 7. Parámetros y métricas

Hiperparámetros leídos en la corrida y confirmados en el estimador cargado con `mlflow.sklearn.load_model` (`HistGradientBoostingRegressor`, `n_iter_=300`, `n_features_in_=8` en ambas versiones):

| Parámetro | Champion v1 (corrida) | Challenger v2 (corrida) | Estimador cargado, ambas versiones |
|---|---|---|---|
| `max_iter` | `hp_max_iter=300` | `max_iter=300` | 300 |
| `learning_rate` | `hp_learning_rate=0.1` | `learning_rate=0.1` | 0.1 |
| `max_depth` | `hp_max_depth=None` | `max_depth=None` | `None` |
| `early_stopping` | No registrado como parámetro | `early_stopping=False` | `False` |
| `random_state` | `seed=42` | `random_state=42` | 42 |

Métricas de 2024, registradas en la corrida `6e3429f2…` y en su artefacto `metrics/metrics.json`:

| Conjunto | MAE | RMSE |
|---|---:|---:|
| Train, ene-ago 2024, 2,059,056 filas | 0.7812010963 | 1.3224130282 |
| Validación, sep-oct 2024 | 0.8395558696 | 1.4280638316 |
| Prueba, nov-dic 2024 | 0.8279171026 | 1.3930097154 |
| Baseline de prueba | 0.8555637279 | 1.4635037391 |

La mejora de prueba frente al baseline, en el artefacto, es 3.2314 % de MAE y 4.8168 % de RMSE. El MAE de validación sin redondear es 0.8395558695664934; Jorge publica 0.8396 y el contrato de la semana 8 redondea a 0.840.

Métricas de entrenamiento del Challenger (ene-feb 2025, 512,592 filas): `train_mae=0.7943356785`, `train_rmse=1.3130923736`. No se comparan con las métricas de prueba del Champion porque son conjuntos distintos.

Métricas de marzo de 2025, A/B offline de la semana 9 (`reports/week9/ab_testing/comparison_metrics.json`, 269,328 observaciones, 362 estaciones, 31 días):

| Métrica | Champion v1 | Challenger v2 | Mejora del Challenger |
|---|---:|---:|---:|
| MAE | 0.8406793191 | 0.8243329534 | 1.9444 % |
| RMSE | 1.4369529768 | 1.4046436576 | 2.2485 % |

Evidencia: `05_model_parameters_metrics.txt`.

## 8. Validación funcional

| Comprobación | Resultado |
|---|---|
| Carga del Champion P1 (`validate_registry --question p1`) | Versión 1, run `6e3429f2…`, `Model loaded successfully`, exit 0 |
| Carga del Champion P2 (`--question p2`) | Falla, exit 1: `Could not find an "MLmodel" configuration file`; la copia no tiene los artefactos de `m-165c943b…` |
| Carga del Champion P3 (`--question p3`) | Falla, exit 1, misma causa para `m-041658517…` |
| Inferencia Champion real (`scripts.week8.test_pyfunc_inference`) | `3.694163560787005`; coincide con la referencia `3.69416356` (tolerancia 1e-6), exit 0 |
| Inferencia Challenger real (mismo registro, `models:/…/2`) | `3.962376668317585`; carga e infiere, diferencia de +0.2682 frente al Champion |
| Predicción reproducible | Segunda ejecución idéntica; lote de 3 registros (8 h, 18 h, 3 h): `3.694163560787005`, `3.5332597213752526`, `3.78948414909114` |
| Rechazo de entradas incorrectas | `ModelInputError`: `Faltan columnas en el registro 0: ['weekday']` |
| Compatibilidad de firmas | Champion y Challenger con el mismo contrato (sección 6) |
| Champion sintético | No se usó; la predicción no es `2.785` |

Registro de prueba: `hour_of_day=8, weekday=2, fin_semana=0, salidas_lag_1h=3, salidas_lag_24h=4, salidas_lag_168h=5, media_estacion=2.1, media_estacion_dia_hora=3.4`. La inferencia de ambos alias es funcional; no elige un ganador.

El Compose de José Manuel no estaba levantado en `http://127.0.0.1:8000` durante la ejecución (`curl: (7) Failed to connect`), así que `06d_api_champion_prediction.png` no se produjo. Como dato complementario, el contenedor local de la semana 8 `mibici-week8-api` (puerto `8001`, apuntando a `host.containers.internal:5000`) respondió `/health` `{"status":"ok"}`, `/ready` `{"ready":true}` y `/predict` `{"prediction":3.694163560787005,"model":"mibici-equipo15-p1-demanda"}`. Esa coincidencia no demuestra por sí sola la identidad del modelo cargado por el Compose de la semana 11. Evidencias: `06a_models_loaded.txt`, `06_pyfunc_inference.txt`, `06c_champion_challenger_inference.txt`, `06d_api_check.txt`.

## 9. Pruebas automatizadas

Ejecución conjunta con `MLFLOW_TRACKING_URI` y `MLFLOW_REGISTRY_URI` apuntando a `file://$(mktemp -d)`:

| Indicador | Resultado |
|---|---|
| `tests/test_registry.py` | 12 aprobadas |
| `tests/test_week8_registry.py` | 5 aprobadas (igual que el antecedente de la semana 8) |
| `tests/test_model_service.py` | 26 aprobadas |
| Pruebas aprobadas | 43 |
| Pruebas fallidas | 0 |
| Pruebas omitidas | 0 |
| Advertencias | 1 (`SADeprecationWarning` de SQLAlchemy dentro de MLflow, en `test_registry_local_con_modelo_real`) |
| Código de salida | 0 |
| Duración | 5.83 s |

Las pruebas usan mocks; `test_registry_local_con_modelo_real` crea su propio SQLite en `tmp_path` de pytest. El directorio temporal quedó vacío al terminar y el `mlflow.db` histórico no cambió (SHA-256 `ecd6ee3c84e6869455d5af2077c1e26d815d9d9afd3fc5ee205c42ec824cb553` antes y después). Evidencias: `06_registry_tests.txt`, `pytest_registry.xml`.

## 10. Decisión de producción

El Challenger v2 redujo el MAE en 1.9444 % y el RMSE en 2.2485 % sobre las 269,328 observaciones de marzo de 2025, con MAE menor en 28 de 31 días y en 269 de 362 estaciones (recuento verificado en `daily_metrics.csv` y `station_comparison_metrics.csv`). En 93 estaciones el MAE subió. Evidently, con umbral 0.1, no marcó drift en ninguna de las 9 variables; 44 estaciones cambiaron su demanda más de 20 % y 11 más de 50 %.

Mantenemos la decisión documentada en la semana 9: el alias `champion` permanece en la versión 1 y la versión 2 sigue como candidato preferente. La mejora agregada es moderada, la evaluación cubre un solo mes, es offline y sin inferencia estadística que considere dependencia temporal y entre estaciones, y el Champion v1 es la versión con validaciones funcionales, de carga y de FinOps. Esa frase no afirma que la versión 1 prediga mejor en marzo de 2025; afirma que es la versión aprobada para el servicio y que la versión 2 no ha cumplido las condiciones de aceptación. Los diez criterios propuestos para una promoción futura están en la sección 18 del Model Card. Evidencia: `07_ab_drift_finops_checks.txt`.

## 11. Model Card

El [Model Card de P1](model_card.md) consolida identificación, linaje, datos, parámetros, contrato, desempeño de 2024, comparación de marzo de 2025, monitoreo, decisión, riesgos, uso responsable, despliegue, reproducibilidad, FinOps, política de promoción y rollback, seguridad y mantenimiento, con los valores leídos en esta validación.

## 12. Verificación de conservación del Registry

| Elemento | Estado inicial (`02_registry_snapshot_before.json`) | Estado final (`08_registry_snapshot_after.json`) |
|---|---|---|
| Champion P1 | v1 `READY`, run `6e3429f222a0407aa159dd9f46699365` | v1 `READY`, run `6e3429f222a0407aa159dd9f46699365` |
| Challenger P1 | v2 `READY`, run `7074129578d045418cbed0691330d1a7` | v2 `READY`, run `7074129578d045418cbed0691330d1a7` |
| Champion P2 | v1 `READY`, run `d794555acb0c498987d17fd61c0ae531` | v1 `READY`, run `d794555acb0c498987d17fd61c0ae531` |
| Champion P3 | v1 `READY`, run `27f5684907f64610a4d3718d1178421d` | v1 `READY`, run `27f5684907f64610a4d3718d1178421d` |
| Versiones registradas | 4 (2 + 1 + 1) | 4 (2 + 1 + 1) |
| Alias | 4 | 4 |

El script de la actividad 16 comparó ambos snapshots: `REGISTRY CONSERVADO CORRECTAMENTE`.

## 13. Hallazgos y limitaciones

- **Copia local que no sirvió.** `httpclient/mlflow-data/mlflow_equipo15/` tiene la versión 2 de P1 sin alias y con otra corrida (`987a98c61f484b66ab101a308bf3757c`, logged model `m-39a601c2…`); el run `7074129578…` no existe ahí. Se descartó. La fuente de este documento es la copia del respaldo `mlflow-runtime-week10.tar.gz`.
- **P2 y P3 sin artefactos de modelo.** La copia válida conserva los metadatos, los alias y los artefactos de corrida de P2 (`config/`, `data/`, `metrics/`, `plots/`) y P3 (`config/`, `data/`, `plots/`), pero `mlartifacts/1/models/` solo contiene los dos logged models de P1. Ni el respaldo original intacto ni la copia `httpclient/` tienen `MLmodel` para `m-165c943b…` ni `m-041658517…`. Por eso la firma (15 y 49 entradas) y la carga de P2 y P3 no se confirman aquí. José Manuel ya había anotado que no verificó esos artefactos. Antes de dar P2 y P3 por validados, el equipo debe aportar los artefactos de la semana 6 de esos dos logged models.
- **Challenger sin artefactos de corrida.** La corrida `7074129578…` no tiene `metrics/`, `config/` ni `data/`; solo el logged model. Sus métricas de entrenamiento están en la corrida y su manifiesto de datos de 2025 en `reports/week9/data_manifest.json`. El tag `data_version=7e1470ac0632` de la versión 2 hereda la ingesta de 2024 y no identifica los CSV de 2025.
- **Tags asimétricos.** La versión 1 de P1 no tiene tags de Model Registry (`data_version` y `git_commit` están en la corrida); la versión 2 sí los tiene. La corrida del Champion no registra `early_stopping` ni `random_state` como parámetros (registra `seed=42`); ambos se confirmaron en el estimador cargado.
- **`search_model_versions` sin alias.** En MLflow 3.16.1 esa consulta devuelve la lista `aliases` vacía por versión; el snapshot deriva los alias por versión del mapa `aliases` del modelo registrado.
- **API en vivo.** El Compose de José Manuel no estaba arriba en `:8000`; la evidencia `06d` queda pendiente de coordinarla. La consulta al contenedor local de la semana 8 en `:8001` es complementaria.
- **Monitoreo y FinOps.** El resumen de la semana 11 de Oscar ([`monitoring_performance_finops.md`](monitoring_performance_finops.md), PR #45) no aporta mediciones nuevas: repite las cifras de las semanas 9 y 10 (sin drift global, 44 y 11 estaciones, MAE 0.8407 contra 0.8243, 96.9 req/s, P95 76 ms, 0.4157 USD por millón). El Model Card usa esas mismas cifras como histórico. No hay medición nueva de la semana 11 que incorporar.
- **Advertencia de pruebas.** La única advertencia proviene de SQLAlchemy dentro de MLflow, no del código del proyecto.
- **Alcance.** No se reentrenó, no se registraron versiones, no se promovió, no se repitió Locust y no se editó el README. `mlflow.db` y `mlartifacts/` quedan fuera de Git.

## 14. Conclusiones

Verificamos que los tres modelos del equipo están registrados con los nombres oficiales y con el alias `champion` en la versión 1, y que P1 tiene además el alias `challenger` en la versión 2, ambos `READY` y con los Run IDs históricos. Documentamos el linaje completo de las dos versiones de P1 desde la copia válida: experimento, corrida padre del Champion, logged models, commits de entrenamiento (incluido el del Challenger, `ddedf4be…`), versión de datos y artefactos. Confirmamos que Champion y Challenger comparten el mismo contrato de ocho entradas `double` y salida `float64`, que los hiperparámetros coinciden en MLflow y en los estimadores serializados, y que el Champion real reproduce `3.694163560787005` para el registro del contrato, mientras el Challenger carga e infiere `3.962376668317585`. Las 43 pruebas del Registry y del servicio se aprobaron sobre un backend temporal, y los snapshots inicial y final del Registry son idénticos.

La limitación principal es que la copia disponible no contiene los artefactos de modelo de P2 y P3, de modo que sus firmas y su carga quedan sin confirmar hasta que el equipo complete `mlartifacts/`. La decisión de producción se conserva: Champion v1 en el servicio, Challenger v2 como candidato preferente sujeto a los criterios de promoción propuestos en el Model Card.

## 15. Referencias

- [Model Card de P1](model_card.md)
- [Contrato del modelo P1, semana 8](../week8/model_registry_contract.md)
- [Registry de la semana 8](../week8/registry_readme.md)
- [Contrato de datos P1](../week9/data_contract.md)
- [Preparación de datos de 2025](../week9/data_preparation.md)
- [Estrategia del Challenger](../week9/challenger_strategy.md)
- [Análisis A/B](../week9/ab_testing_analysis.md)
- [Monitoreo de drift](../week9/drift_monitoring.md)
- [Pruebas de carga](../week10/load_testing_analysis.md)
- [FinOps](../week10/finops.md)
- [Integración y despliegue, semana 11](deployment_integration.md)
- [Repositorio y experimentación, semana 11](repository_experimentation.md)
- [Monitoreo, A/B testing, rendimiento y FinOps, semana 11](monitoring_performance_finops.md)

## 16. Evidencias

Carpeta: `reports/week11/evidence/registry/`

| Archivo | Contenido |
|---|---|
| `00_git_environment.txt`, `00_environment_packages.txt`, `00_pip_check.txt`, `00_registry_connection.txt` | Rama, commit, paquetes, `pip check` y conexión al servidor `5052` |
| `01_registered_models.png`, `01_registered_models.txt` | Modelos P1, P2 y P3 con sus alias |
| `02_champion_challenger.png`, `02_registry_snapshot_before.json`, `02b_alias_validation.txt` | Versiones 1 y 2 de P1, estado `READY`, alias y snapshot inicial |
| `03_model_lineage.png`, `03_model_lineage.json`, `03_model_lineage.txt` | Corridas, padre, logged models, commits, tags y artefactos |
| `04_model_signature.png`, `04_registry_inspection.txt`, `04c_signature_comparison.txt` | Firma del Champion, firmas de los tres modelos y comparación Champion/Challenger |
| `05_model_parameters_metrics.png`, `05_model_parameters_metrics.txt` | Hiperparámetros y métricas de MLflow y del estimador |
| `06_registry_validation.png`, `06a_models_loaded.txt`, `06_pyfunc_inference.txt`, `06c_champion_challenger_inference.txt`, `06d_api_check.txt` | Carga de los tres modelos, inferencia oficial, inferencia de ambos alias y consulta HTTP |
| `06e_registry_tests.png`, `06_registry_tests.txt`, `pytest_registry.xml` | Pruebas automatizadas |
| `07_ab_drift_finops_checks.txt` | Verificación de los CSV del A/B, resumen de drift y cociente FinOps |
| `08_registry_unchanged.png`, `08_registry_snapshot_after.json`, `08_registry_unchanged.txt` | Snapshot final y comparación |
| `../model_card/07_model_card.png` | Model Card renderizado en GitHub |
