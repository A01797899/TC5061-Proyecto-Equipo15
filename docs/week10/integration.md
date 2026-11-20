# Semana 10 — Integración real de FastAPI y MLflow

## 1. Objetivo

Validar de manera reproducible la integración entre la API REST de inferencia FastAPI, el servidor MLflow Tracking/Model Registry y el modelo P1 registrado con el alias `champion`. Confirmar tanto respuestas exitosas como el rechazo de solicitudes incorrectas, ejecutar la suite de regresión y verificar la persistencia al recrear los contenedores.

## 2. Arquitectura y entorno

La orquestación existente utiliza `docker-compose.yml`, con los servicios `api` y `mlflow` en la red `mibici-net`.

| Servicio | Acceso local | Comprobación |
|---|---|---|
| MLflow | `http://127.0.0.1:5000` | `GET /health` |
| FastAPI | `http://127.0.0.1:8000` | `GET /health`, `GET /ready`, `POST /predict` |

FastAPI utiliza `MLFLOW_TRACKING_URI=http://mlflow:5000` dentro de la red Compose. Se verificó la resolución DNS de `mlflow` (IP observada `10.89.0.2`) y una consulta HTTP interna a `/version` que devolvió `3.16.1`. Los archivos de MLflow se montan desde `./runtime/mlflow` hacia `/mlflow` en el contenedor.

Entorno local observado: macOS, Podman 5.8.9, Python 3.12.15, Pytest 9.1.1. `podman-compose` se instaló en un entorno Python independiente.

## 3. Preparación de los artefactos

Se reutilizaron copias de `mlflow.db` y `mlartifacts/` provenientes de la ejecución de la semana 8. Los originales no se modificaron. Se copiaron a:

```text
runtime/mlflow/
├── mlflow.db
└── mlartifacts/
```

El contenido de `runtime/` está excluido del control de versiones mediante `.gitignore`. Se verificó la presencia del directorio de artefactos del modelo registrado y de archivos como `MLmodel`, `requirements.txt` y los archivos del modelo.

## 4. Validación del Registry y del Champion P1

La inspección `scripts/week8/inspect_registry.py`, ejecutada dentro del servicio `api`, confirmó:

| Propiedad | Resultado |
|---|---|
| Nombre registrado | `mibici-equipo15-p1-demanda` |
| Alias | `champion` |
| Versión activa del alias | `1` |
| Estado | `READY` |
| Versiones encontradas | `1` y `2`, ambas `READY` |
| Descarga de artefactos | `100%` |
| Número de características | `8` |

Características exigidas por la firma: `hour_of_day`, `weekday`, `fin_semana`, `salidas_lag_1h`, `salidas_lag_24h`, `salidas_lag_168h`, `media_estacion`, `media_estacion_dia_hora`. La salida está declarada como tensor de tipo `float64`.

## 5. Validaciones funcionales y negativas

Se comprobaron respuestas **HTTP 200** de MLflow `/health`, FastAPI `/health` y `/ready`; este último devolvió `{"ready":true}`. El endpoint `/predict` devolvió una respuesta correcta para la entrada de referencia `postman/examples/p1-valid.json`.

**Predicción de referencia del Champion P1:** `3.694163560787005`.

La prueba directa con MLflow PyFunc cargó la versión 1, repitió la inferencia con el mismo resultado, procesó un lote de tres registros y rechazó un registro incompleto.

También se verificaron las respuestas negativas esperadas:

| Caso | Código observado |
|---|---|
| Ruta inexistente | `404` |
| Cuerpo JSON vacío | `422` |
| `hour_of_day = 30` | `422` |
| Característica obligatoria faltante | `422` |

## 6. Smoke test automatizado

Script creado en `scripts/week10/smoke_integration.py` y ejecutado desde la raíz del repositorio:

```bash
python3 scripts/week10/smoke_integration.py
echo $?
```

**Resultado observado:** `11/11 pruebas aprobadas`, mensaje `SMOKE INTEGRATION COMPLETADA - OK` y **código de salida `0`**. Verifica la disponibilidad de ambos servicios, inferencia, nombre del modelo, valor de referencia, reproducibilidad y manejo de errores HTTP. Esta prueba requiere servicios reales accesibles en los puertos 5000 y 8000 y los artefactos de MLflow preparados.

## 7. Pruebas de regresión

Se ejecutaron:

```bash
python3 -m pytest tests/test_api.py -v
python3 -m pytest tests/ -q
```

| Prueba | Resultado observado |
|---|---|
| Suite FastAPI | `72 passed, 1 warning in 6.48s` |
| Suite completa | `311 passed, 24 warnings in 62.27s` |

No hubo pruebas fallidas. Las advertencias fueron principalmente de deprecación en dependencias y no bloquearon la ejecución. Estas suites validan los casos implementados en el repositorio; no reemplazan el smoke test contra los servicios reales.

## 8. Persistencia y recuperación

Se detuvieron y eliminaron los contenedores con:

```bash
"$COMPOSE" -f docker-compose.yml down
```

Luego se comprobó que seguían presentes `runtime/mlflow/mlflow.db`, `runtime/mlflow/mlartifacts/` y el directorio de artefactos del Champion. Se recrearon los contenedores mediante:

```bash
"$COMPOSE" -f docker-compose.yml up -d
"$COMPOSE" -f docker-compose.yml ps
```

Ambos alcanzaron estado `healthy`. MLflow `/health` y FastAPI `/ready` respondieron HTTP 200. La inspección del Registry volvió a identificar el Champion P1 versión 1 (`READY`) y la ejecución posterior del smoke test volvió a concluir **11/11 aprobadas**, código de salida **0** y predicción de referencia sin cambios.

## 9. Inventario de evidencias

Las capturas originales se guardaron en `reports/week10/evidence/integration/`:

| Evidencia | Archivo |
|---|---|
| Compose activo | `01_compose_services_running.png` |
| Servidor MLflow | `02_mlflow_server.png` |
| Tracking URI | `03_tracking_uri.png` |
| Red interna | `04_internal_network.png` |
| Registry Champion | `05_registry_champion.png` |
| Interfaz MLflow | `06_mlflow_champion_ui.png` |
| FastAPI health | `07_api_health.png` |
| FastAPI ready | `08_api_ready.png` |
| Entrada válida | `09_p1_valid_payload.png` |
| Predicción HTTP | `10_api_predict.png` |
| Inferencia PyFunc | `11_pyfunc_inference.png` |
| Pruebas negativas | `12_api_negative_tests.png` |
| Smoke test | `13_smoke_integration.png` |
| Pytest FastAPI | `14_pytest_fastapi.png` |
| Pytest completo | `15_pytest_full_suite.png` |
| Persistencia con servicios detenidos | `16_mlflow_persistence_down.png` |
| Reinicio y salud | `17a_mlflow_persistence_restart.png` |
| Registry tras reinicio | `17b_mlflow_persistence_restart.png` |
| Smoke test tras reinicio | `17c_mlflow_persistence_restart.png` |

## 10. Consideraciones para CI/CD

El smoke test depende de un servidor MLflow accesible y de los datos del Model Registry. **GitHub Actions no dispone automáticamente de `runtime/mlflow/mlflow.db` ni de `runtime/mlflow/mlartifacts/`**, porque están ignorados por Git. Antes de ejecutar esta prueba en CI debe definirse una estrategia explícita para preparar un Registry de prueba y artefactos adecuados o separar las pruebas unitarias de la validación end-to-end contra servicios aprovisionados.

## 11. Conclusión

Validamos la integración del servicio FastAPI con MLflow, la recuperación del Champion P1 y su contrato de entrada, la predicción reproducible y los códigos de error esperados. Automatizamos once comprobaciones funcionales y verificamos que la suite de regresión existente aprueba 311 pruebas. También comprobamos que, después de recrear los contenedores, se conservan los artefactos y el servicio vuelve a producir la predicción de referencia. Con estos resultados dejamos preparada la base técnica para desarrollar la siguiente actividad de CI/CD.
