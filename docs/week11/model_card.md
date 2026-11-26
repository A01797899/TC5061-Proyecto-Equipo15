# Model Card - MiBici P1: predicción de demanda por estación y hora

| Campo | Valor |
|---|---|
| Proyecto | TC5061 - Operaciones de Aprendizaje Automático |
| Equipo | 15 |
| Modelo | `mibici-equipo15-p1-demanda` |
| Versión seleccionada para producción | Champion v1 (alias `champion`) |
| Versión alternativa evaluada | Challenger v2 (alias `challenger`) |
| Versión del Model Card | 1.0 |
| Responsable del documento | José Luis Mendoza Mora (A01732182) |
| Fecha de última validación | 26 de noviembre de 2026, sobre la copia del Registry descrita en [Validación del Model Registry](model_registry_validation.md) |

Integrantes: Jorge David Alcázar Rochín (A01252855), José Luis Mendoza Mora (A01732182), José Manuel Martínez Romero (A01797899), Oscar Alejandro Gutiérrez Araiza (A00832992).

## 1. Descripción general

El modelo P1 estima la demanda de bicicletas públicas de MiBici (Guadalajara) como el número de viajes iniciados en una estación durante una hora. Su propósito es apoyar el análisis de patrones de uso y la planeación operativa del sistema.

Forma parte de una arquitectura MLOps que integra ingesta y versionamiento de datos, MLflow Projects, seguimiento de experimentos, Model Registry, inferencia con FastAPI, contenedores, CI/CD, monitoreo de drift, A/B offline, pruebas de carga y análisis de costos.

El alcance de este Model Card es P1. Los modelos P2 (`mibici-equipo15-p2-riesgo`, riesgo operativo proxy) y P3 (`mibici-equipo15-p3-grupos`, agrupamiento de estaciones) son componentes del proyecto integrador y no entran al servicio REST de predicción.

## 2. Identificación y estado del modelo

| Propiedad | Champion | Challenger |
|---|---|---|
| Nombre registrado | `mibici-equipo15-p1-demanda` | `mibici-equipo15-p1-demanda` |
| Versión | 1 | 2 |
| Alias | `champion` | `challenger` |
| Estado en el Registry (verificado el 26 de noviembre de 2026) | `READY` | `READY` |
| Algoritmo | `HistGradientBoostingRegressor` | `HistGradientBoostingRegressor` |
| Tipo de problema | Regresión | Regresión |
| Estado operativo | Seleccionado para el servicio de inferencia | Candidato preferente, en evaluación |

`READY` indica que la versión está disponible en el Registry. La selección de producción la determina el alias `champion`, no el estado. La comprobación de versiones, alias y Run IDs está en la sección 4 de la [Validación del Model Registry](model_registry_validation.md).

## 3. Uso previsto

- Estimar la demanda de viajes iniciados por estación y hora.
- Apoyar el análisis de utilización de estaciones y de patrones temporales.
- Proporcionar información para la planeación operativa.
- Servir como componente predictivo dentro de la arquitectura MLOps del equipo.

La salida es una estimación estadística. No mide la disponibilidad actual de bicicletas en la estación.

## 4. Usos no previstos

- Garantizar la disponibilidad de bicicletas o estimar inventario real sin información adicional de llegadas, salidas, capacidad y disponibilidad.
- Identificar usuarios individuales o inferir características personales.
- Tomar decisiones operativas críticas sin supervisión humana y validación adicional.
- Generalizar directamente a otras ciudades o sistemas de bicicletas públicas.
- Predecir estaciones nuevas sin historia suficiente. Las ocho estaciones de 2025 (391 a 398) no están en el panel del Champion.

## 5. Arquitectura del modelo

Ambas versiones usan `HistGradientBoostingRegressor` de scikit-learn 1.9.1, serializado con `skops` (`model.skops`) y registrado con los flavors `sklearn` y `python_function` de MLflow 3.16.1. El Challenger conserva la familia y los hiperparámetros del Champion; la diferencia es la ventana de entrenamiento.

| Hiperparámetro | Valor | Confirmación de la semana 11 |
|---|---|---|
| `max_iter` | 300 | Parámetros de ambas corridas y `get_params()` de ambos estimadores; `n_iter_=300` |
| `learning_rate` | 0.1 | Ambas corridas y ambos estimadores |
| `max_depth` | `None` | Ambas corridas y ambos estimadores |
| `early_stopping` | `False` | Corrida del Challenger y ambos estimadores (la corrida del Champion no lo registra como parámetro) |
| `random_state` | 42 | Corrida del Champion como `seed=42`, corrida del Challenger como `random_state=42`, y ambos estimadores |

El servicio recupera el modelo desde el Registry con `ModelService` (`mibici/api/model_service.py`), que resuelve el alias `champion` al arrancar, carga la versión con `mlflow.pyfunc.load_model` y la mantiene en memoria.

## 6. Linaje y versionamiento

### 6.1. Champion v1

| Propiedad | Valor |
|---|---|
| Experimento | `mibici-equipo15` |
| Corrida | `P1_candidate_hgb_max_iter=300_learning_rate=0.1_max_depth=None` |
| Run ID | `6e3429f222a0407aa159dd9f46699365` |
| Corrida padre | `P1_search_candidates`, `84b3d18fea4a49dbb1c212e9311da9c7` |
| Logged Model ID | `m-04774b8f25ff468b933a18756c24b068` |
| Commit de entrenamiento | `ad81f296c6aea3c97935399dd389c6a02c25d713` (PR #18) |
| Versión de datos | `7e1470ac0632` (`data/manifiesto.json`, ingesta de 2024) |
| Serialización | `skops` |
| Model URI | `models:/mibici-equipo15-p1-demanda/1` |
| Artefactos de la corrida | `metrics/metrics.json`, `config/`, `data/`, `plots/` |

### 6.2. Challenger v2

| Propiedad | Valor |
|---|---|
| Experimento | `mibici-equipo15` |
| Corrida | `P1_challenger_2025_01_02` |
| Run ID | `7074129578d045418cbed0691330d1a7` |
| Logged Model ID | `m-ea1a3789403948d08f509f0a562419e5` |
| Commit de entrenamiento | `ddedf4beb63e99bdd96deeb6886d33348afa2041` (merge del PR #33; coincide con el tag `registration_commit`) |
| Script | `scripts/week9/train_challenger.py` |
| Versión registrada | 2 |
| Model URI | `models:/mibici-equipo15-p1-demanda/2` |
| Artefactos de la corrida | Ninguno; los siete artefactos del modelo están en el logged model |

Los identificadores se leyeron en la copia válida del Registry el 26 de noviembre de 2026. El tag `data_version=7e1470ac0632` de la versión 2 hereda la ingesta de 2024; el linaje de los datos de 2025 del Challenger está en `reports/week9/data_manifest.json`, que registra fuentes, SHA-256 y periodos de los tres CSV de enero a marzo de 2025 ([Preparación de datos](../week9/data_preparation.md)).

## 7. Datos de entrenamiento y evaluación

### 7.1. Datos originales de MiBici

La ingesta histórica de 2024 procesó doce archivos mensuales y obtuvo 4,645,921 viajes limpios, con versión de datos `7e1470ac0632`. A partir de esos viajes se construyó un panel estación-hora de 362 estaciones, con rezagos de 1, 24 y 168 horas y medias históricas por estación.

### 7.2. Champion

| Conjunto | Periodo | Filas |
|---|---|---:|
| Entrenamiento | Enero a agosto de 2024 (`< 2024-09-01`) | 2,059,056 (después de descartar las primeras 168 horas sin rezagos completos) |
| Validación | Septiembre y octubre de 2024 | - |
| Prueba | Noviembre y diciembre de 2024 | - |

### 7.3. Challenger

| Conjunto | Periodo | Filas |
|---|---|---:|
| Entrenamiento alternativo | 1 de enero a 28 de febrero de 2025 | 512,592 |
| Evaluación reservada | 1 a 31 de marzo de 2025 | 269,328 |

Marzo no se usó para ajustar el Challenger. Las medias históricas (`media_estacion`, `media_estacion_dia_hora`) se mantuvieron congeladas con el entrenamiento de 2024 para conservar la firma del Champion.

### 7.4. Evaluación comparativa

La comparación A/B offline usa las mismas 269,328 observaciones de marzo de 2025, 362 estaciones y 31 días, con predicciones negativas recortadas a cero en ambos modelos.

## 8. Características de entrada y salida

Firma MLflow verificada en ambas versiones: ocho entradas `double` obligatorias, en este orden, y salida `Tensor('float64', (-1,))`.

| # | Variable | Tipo | Significado | Rango esperado |
|---|---|---|---|---|
| 1 | `hour_of_day` | `double` | Hora del día | 0 a 23 |
| 2 | `weekday` | `double` | Día de la semana (0 = lunes) | 0 a 6 |
| 3 | `fin_semana` | `double` | 1 si `weekday` ≥ 5, 0 si no | 0 o 1 |
| 4 | `salidas_lag_1h` | `double` | Salidas de la estación una hora antes | ≥ 0 |
| 5 | `salidas_lag_24h` | `double` | Salidas de la estación 24 horas antes | ≥ 0 |
| 6 | `salidas_lag_168h` | `double` | Salidas de la estación una semana antes | ≥ 0 |
| 7 | `media_estacion` | `double` | Media histórica de salidas por hora de la estación, congelada con el entrenamiento de 2024 | ≥ 0 |
| 8 | `media_estacion_dia_hora` | `double` | Media histórica de la estación para ese día y hora, congelada con el entrenamiento de 2024 | ≥ 0 |

La salida es una predicción por fila: viajes esperados en esa estación y hora. El servicio comprueba que sea finita y recorta los valores negativos a cero.

La firma no valida semántica: enteros requieren conversión a `float64`, columnas extra se ignoran y valores nulos o fuera de rango producen una predicción. FastAPI (Pydantic, `extra="forbid"`, ocho campos numéricos obligatorios) y `ModelService` validan tipos, nombres, rangos, consistencia entre `weekday` y `fin_semana` y resultados finitos.

## 9. Evaluación histórica del Champion (2024)

Valores leídos en la corrida `6e3429f2…` y en su artefacto `metrics/metrics.json` de la copia válida:

| Conjunto | MAE | RMSE |
|---|---:|---:|
| Entrenamiento, ene-ago 2024 | 0.7812010963 | 1.3224130282 |
| Validación, sep-oct 2024 | 0.8395558696 | 1.4280638316 |
| Prueba, nov-dic 2024 | 0.8279171026 | 1.3930097154 |
| Baseline de prueba (media por estación, día y hora) | 0.8555637279 | 1.4635037391 |

Frente al baseline de prueba, el Champion mejora 3.2314 % en MAE y 4.8168 % en RMSE. El MAE de validación sin redondear es 0.8395558695664934; Jorge lo publica como 0.8396 y el contrato de la semana 8 como 0.840. Estas cifras corresponden a 2024 y no a la comparación A/B de marzo de 2025.

## 10. Evaluación comparativa Champion/Challenger (marzo de 2025)

| Métrica | Champion v1 | Challenger v2 | Mejora del Challenger |
|---|---:|---:|---:|
| MAE | 0.8406793191 | 0.8243329534 | 1.9444 % |
| RMSE | 1.4369529768 | 1.4046436576 | 2.2485 % |

El Challenger redujo el MAE en 28 de 31 días y en 269 de 362 estaciones; en 93 estaciones el MAE subió. Es una mejora agregada favorable que no garantiza un desempeño superior en todas las estaciones ni en otros periodos. La evaluación es offline, sin asignación de tráfico entre versiones y sin prueba inferencial que considere la dependencia temporal y entre estaciones; no demuestra impacto causal sobre indicadores de negocio. Fuente: [Análisis A/B](../week9/ab_testing_analysis.md).

## 11. Monitoreo y drift

El monitoreo de la semana 9 comparó con Evidently 0.7.23 la distribución del entrenamiento del Champion (8 de enero a 31 de agosto de 2024, 2,059,056 filas) contra enero a marzo de 2025 (781,920 filas), con un control estacional de los mismos meses de 2024. Umbral 0.1; distancia de Jensen-Shannon para `hour_of_day`, `weekday` y `fin_semana`, y Wasserstein normalizada para los rezagos, las medias y el objetivo `y`.

- Ninguna de las nueve variables superó el umbral en ningún escenario; el puntaje más alto fue 0.084 (`salidas_lag_168h` en enero, efecto de la semana de Navidad).
- 44 estaciones cambiaron su demanda más de 20 % y 11 más de 50 % respecto a la media congelada de 2024; las estaciones 104 y 310 no registran salidas en 2025.
- La ausencia de drift agregado no excluye cambios locales: las medias congeladas envejecen aunque las distribuciones no cambien.

Recomendamos mantener el monitoreo de distribuciones y complementarlo con MAE, RMSE y volumen de demanda por estación. El reporte de la semana 9 es evidencia histórica offline; no es monitoreo continuo de tráfico de producción. Fuentes: [Monitoreo de drift](../week9/drift_monitoring.md) y [resumen de la semana 11](monitoring_performance_finops.md).

## 12. Decisión de producción

Aunque el Challenger presenta menores errores agregados en marzo de 2025, se mantiene el alias `champion` en la versión 1 y la versión 2 continúa como candidato preferente para una siguiente etapa de validación.

| Razón | Justificación |
|---|---|
| Mejora moderada | 1.9444 % de MAE y 2.2485 % de RMSE |
| Deterioro localizado | 93 de 362 estaciones con mayor MAE |
| Alcance temporal | Un solo mes de evaluación |
| Evaluación offline | Sin tráfico real distribuido entre versiones |
| Incertidumbre estadística | Sin pruebas inferenciales con dependencia temporal y entre estaciones |
| Operación validada | El Champion v1 tiene validaciones funcionales, de carga, de integración y de FinOps |
| Gobernanza | La promoción requiere criterios de aceptación y revisión humana |

Esta decisión no afirma que la versión 1 prediga mejor en marzo de 2025. Afirma que la versión 1 es la aprobada para el servicio y que la versión 2 todavía no cumple las condiciones de aceptación de la sección 18. Durante la semana 11 no se promovió ninguna versión ni se movió ningún alias; los snapshots inicial y final del Registry son idénticos.

## 13. Riesgos y limitaciones

| Riesgo o limitación | Impacto potencial | Mitigación propuesta |
|---|---|---|
| Cambios de demanda por estación | Deterioro localizado de predicciones | Monitorear métricas segmentadas; vigilar las 93 estaciones con deterioro y las 44 con cambio mayor a 20 % |
| Estacionalidad | Variación de patrones temporales no vista en la ventana de evaluación | Evaluar periodos equivalentes y ventanas adicionales |
| Estaciones nuevas | Sin historia suficiente (391 a 398 en 2025) | Procedimiento específico de incorporación |
| Rezagos desactualizados o incorrectos | Predicciones con información equivocada | Validar frescura y origen de los datos |
| Medias históricas congeladas | Menor adaptación a cambios estructurales | Revisar periódicamente su vigencia; recalcularlas sin usar el periodo de evaluación |
| Eventos extraordinarios y festivos | Cambios abruptos de demanda (semana de Navidad, 17 de marzo de 2025) | Monitorear anomalías y revisión humana |
| Ausencia de inventario real | Confusión entre demanda y disponibilidad | Limitar el uso a demanda esperada |
| Evaluación temporal limitada | Incertidumbre sobre el desempeño futuro | Ampliar la evaluación a otros periodos |
| Dependencia de bibliotecas | Incompatibilidades o vulnerabilidades (avisos de MLflow 3.16.1 y setuptools en la semana 7) | Controlar versiones y ejecutar pruebas de regresión |
| Artefactos del Registry | La copia revisada no contiene los artefactos de modelo de P2 y P3 | Completar `mlartifacts/` antes de validar esos modelos |

El modelo no incorpora variables meteorológicas, eventos especiales ni disponibilidad de bicicletas en tiempo real.

## 14. Consideraciones éticas y uso responsable

El contrato de inferencia usa características agregadas de estación y tiempo. No requiere identificadores personales, género ni edad. Los datos originales de MiBici sí incluyen `usuario_id`, género y año de nacimiento, por lo que su tratamiento debe respetar las condiciones de uso de la fuente y las prácticas de privacidad del equipo.

La ausencia de variables demográficas en la firma no demuestra ausencia de sesgo. Las diferencias de cobertura, demanda y error entre estaciones pueden producir recomendaciones operativas desiguales; conviene revisar el desempeño por estación y contexto geográfico antes de usar las predicciones para redistribución. Las predicciones son apoyo para decisiones operativas, no sustituto de la supervisión humana.

No se ha realizado una auditoría formal de equidad demográfica, impacto social o privacidad del sistema completo.

## 15. Despliegue y operación

El Champion se recupera desde MLflow Model Registry y se sirve con FastAPI en un contenedor; Compose integra la API y MLflow, con los puertos publicados en loopback (`127.0.0.1:8000` y `127.0.0.1:5000`).

| Endpoint | Función |
|---|---|
| `GET /health` | Funcionamiento del proceso HTTP |
| `GET /ready` | Disponibilidad del modelo cargado |
| `POST /predict` | Predicción con el contrato P1; responde `prediction` y `model` |

El modelo se carga en el arranque y permanece en memoria. Un cambio de alias en el Registry no sustituye el modelo que el proceso ya cargó: una promoción requiere un procedimiento controlado de recarga o reinicio y validación. La respuesta de `/predict` identifica el nombre del modelo, no la versión ni el Run ID; la identidad del modelo cargado se confirma con la configuración de Compose, el Registry usado y los logs de carga. Fuente: [Integración y despliegue](deployment_integration.md).

## 16. Validación funcional y reproducibilidad

Registro de prueba del contrato: `hour_of_day=8, weekday=2, fin_semana=0, salidas_lag_1h=3, salidas_lag_24h=4, salidas_lag_168h=5, media_estacion=2.1, media_estacion_dia_hora=3.4`.

| Comprobación | Resultado de la semana 11 |
|---|---|
| Carga del Champion P1 (`validate_registry --question p1`) | Versión 1, run `6e3429f2…`, `Model loaded successfully` |
| Predicción de referencia del Champion real | `3.694163560787005`, igual a la referencia `3.69416356` con tolerancia 1e-6 |
| Reproducibilidad | Dos ejecuciones idénticas; lote de tres registros `3.694163560787005`, `3.5332597213752526`, `3.78948414909114` |
| Rechazo de entrada incompleta | `Faltan columnas en el registro 0: ['weekday']` |
| Challenger v2 con el mismo registro | Carga e infiere `3.962376668317585` (prueba funcional, no comparativa) |
| Contrato Champion/Challenger | Firmas idénticas |
| Pruebas `test_registry`, `test_week8_registry`, `test_model_service` | 43 aprobadas, 0 fallidas, 0 omitidas, exit 0, sobre un backend temporal |
| Smoke de integración de José Manuel (semana 11) | 11/11 con `3.694163560787005` |

El pipeline de CI usa un Champion sintético con predicción de referencia `2.785`, destinado solo a validar la integración técnica. Ese modelo no representa el desempeño, los resultados A/B ni el costo del Champion real, y no se usó en ninguna evidencia de este documento.

## 17. Rendimiento y FinOps

Punto de operación recomendado para el Champion v1, medido por Oscar en la semana 10 con Locust sobre Compose (4 CPU, 3.8 GiB) y costeado en [FinOps](../week10/finops.md):

| Indicador | S5-150 |
|---|---:|
| Usuarios concurrentes de Locust | 150 |
| Throughput | 96.9 solicitudes/s |
| Latencia P95 | 76.0 ms |
| Errores observados | 0 |
| Inferencias exitosas por hora | 348,840 |
| CPU y memoria de la API | 173.78 %, 308.75 MiB |
| CPU y memoria de MLflow | 4.51 %, 2176.0 MiB |
| Tarifa de referencia | 0.145 USD/h (`c7g.xlarge` Linux, US East N. Virginia, consulta del 22 de noviembre de 2026) |
| Costo por inferencia | 0.145 / (96.9 × 3600) = 0.0000004156633413599357871803692237 USD |
| Costo por millón de inferencias | 0.415663 USD |

El cociente se reimprimió con `Decimal` en la semana 11 y coincide con `reports/week10/finops/cost_by_scenario.csv`. El escenario de 300 usuarios tiene el menor costo unitario (147.1 req/s, 0.273812 USD por millón) pero su P95 sube a 740 ms; a partir de 200 usuarios el P95 pasa a 170 ms. El equipo privilegia el equilibrio entre costo, capacidad y latencia, y por eso el punto de operación es 150 usuarios.

El costo es una estimación con mediciones locales y una tarifa pública; no es una factura ni un benchmark en esa instancia, y no incluye almacenamiento, red, IPv4, soporte ni impuestos. La carga corresponde al Champion v1 y no debe atribuirse al Challenger sin pruebas adicionales. MLflow concentra 87.6 % de la memoria de los dos contenedores y es un costo fijo. Las cifras son las mediciones de la semana 10; el [resumen de la semana 11 de Oscar](monitoring_performance_finops.md) confirma los mismos valores y no añade mediciones nuevas.

## 18. Política propuesta de promoción y rollback

Criterios propuestos para promover un candidato al alias `champion`:

| Criterio | Condición |
|---|---|
| Desempeño global | Mantener o mejorar MAE y RMSE sobre conjuntos reservados |
| Estabilidad temporal | Evaluar periodos adicionales a marzo de 2025 |
| Desempeño localizado | Revisar estaciones críticas y deterioros relevantes |
| Calidad de datos | Comprobar que la evaluación no esté comprometida |
| Compatibilidad | Conservar el contrato de entrada y salida |
| Regresión | Aprobar las pruebas funcionales |
| Rendimiento | Validar latencia y throughput del candidato |
| FinOps | Comprobar el costo operativo del candidato |
| Aprobación | Revisión y aceptación explícita del equipo |
| Rollback | Conservar una versión previa recuperable, con restauración controlada del alias, recarga del servicio y pruebas funcionales |

Esta política es una propuesta de gobernanza. No es una automatización de Continuous Training ni una promoción realizada en esta actividad.

## 19. Seguridad y restricciones operativas

La revisión de la semana 7 ([`reports/week7/security/hallazgos.md`](../../reports/week7/security/hallazgos.md)) ejecutó Bandit sobre `mibici/` (19 hallazgos iniciales anotados y justificados, 0 después) y pip-audit sobre los requisitos. Quedaron avisos sin corregir por decisión documentada: tres de MLflow 3.16.1 (corrección en 3.17.x) y dos de setuptools 80.10.2 (retenido por DeepChecks). Cero hallazgos de Bandit no significan ausencia de vulnerabilidades. No hay secretos en el código; la única variable de entorno es `MLFLOW_TRACKING_URI`.

La arquitectura local es una demostración académica funcional. No incluye autenticación ni autorización de clientes de la API, alta disponibilidad ni monitoreo continuo de tráfico real. Esos controles deben implementarse antes de exponer el servicio fuera del entorno local.

Estado de las funcionalidades, para no presentar el sistema como más avanzado de lo implementado:

| Funcionalidad | Estado |
|---|---|
| Modelos registrados, alias Champion y Challenger, API contenerizada, validación de entradas, drift offline, A/B offline, pruebas de carga, FinOps | Implementado |
| Model Card | En desarrollo hasta el PR |
| Promoción automática | No implementada |
| Continuous Training automatizado | No requerido en esta actividad |
| Monitoreo continuo de tráfico real | No demostrado |
| Auditoría formal de equidad | No realizada en la documentación revisada |
| Alta disponibilidad y autenticación de la API | No demostradas en la implementación revisada |

## 20. Mantenimiento del Model Card

Este documento se vuelve a emitir cuando: se registre una nueva versión del modelo; cambie la versión seleccionada para producción; se modifique la firma; se actualicen los datos, las medias históricas o los hiperparámetros; se identifiquen nuevos riesgos o limitaciones; cambien las condiciones de monitoreo, rendimiento o costo; o se realice una nueva evaluación Champion/Challenger. También debe actualizarse cuando el equipo complete los artefactos de P2 y P3 en el Registry.

## 21. Referencias y documentación técnica

- [README general](../../README.md)
- [Contrato del modelo registrado, semana 8](../week8/model_registry_contract.md)
- [Validación del Model Registry, semana 11](model_registry_validation.md)
- [Instrucciones de P1](../../references/P1_INSTRUCCIONES.md)
- [Contrato de datos P1](../week9/data_contract.md)
- [Preparación de datos de 2025](../week9/data_preparation.md)
- [Estrategia del Challenger](../week9/challenger_strategy.md)
- [Análisis A/B](../week9/ab_testing_analysis.md)
- [Monitoreo de drift](../week9/drift_monitoring.md)
- [Integración FastAPI y MLflow, semana 10](../week10/integration.md)
- [Pipeline CI/CD](../week10/cicd.md)
- [Pruebas de carga](../week10/load_testing_analysis.md)
- [FinOps](../week10/finops.md)
- [Integración y despliegue, semana 11](deployment_integration.md)
- [Repositorio y experimentación, semana 11](repository_experimentation.md)
- [Monitoreo, A/B testing, rendimiento y FinOps, semana 11](monitoring_performance_finops.md)
- Evidencias: `reports/week11/evidence/registry/` y `reports/week11/evidence/model_card/`

Estado del documento: en desarrollo hasta el pull request; sujeto a revisión del Equipo 15. Las cifras de monitoreo, rendimiento y costo coinciden con el resumen de la semana 11 de Oscar.
