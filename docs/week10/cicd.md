# Semana 10 — Pipeline CI/CD de MiBici (Equipo 15)

## 1. Objetivo

En esta actividad implementamos y preparamos **MiBici-pipeline**, un workflow de GitHub Actions para automatizar pruebas de software, construcción de imágenes y validaciones de integración entre FastAPI y MLflow. Diseñamos una ejecución manual para que los integrantes del equipo puedan iniciar pruebas sin generar un nuevo *push* ni abrir un *pull request*.

**Alcance de la evidencia documentada:** la integración se validó localmente en un sandbox independiente con Podman. La ejecución del workflow en un runner de GitHub Actions continúa pendiente y debe verificarse antes de declararlo completamente operativo.

## 2. Archivos y componentes

| Elemento | Función |
|---|---|
| `.github/workflows/MiBici-pipeline.yml` | Define disparadores y jobs del pipeline. |
| `scripts/week10/bootstrap_ci_registry.py` | Registra un modelo P1 sintético, reproducible y exclusivo para CI en un Registry temporal. |
| `scripts/week10/smoke_integration.py` | Ejecuta 11 verificaciones HTTP y de inferencia contra los servicios desplegados. |
| `docker-compose.yml` | Orquestación del proyecto, incorporada previamente por Jorge; no se modificó para esta prueba. |
| `Containerfile` y `Containerfile.mlflow` | Construcción de FastAPI y MLflow. |
| `reports/week10/evidence/cicd/` | Evidencias de las verificaciones locales. |

## 3. Disparadores y modalidades de ejecución

El workflow contempla eventos automáticos `push` y `pull_request`, así como `workflow_dispatch`, que permite iniciarlo manualmente desde GitHub Actions. La ejecución manual ofrece cuatro modalidades:

- **`tests`:** ejecuta pruebas de Python y Pytest.
- **`build`:** construye las imágenes de los servicios.
- **`integration`:** levanta un entorno temporal, inicializa el Registry de CI y ejecuta pruebas end-to-end.
- **`full`:** selecciona las etapas de pruebas, construcción e integración.

La disponibilidad real de las ejecuciones manuales, sus condiciones por rama y el resultado de cada job deberán comprobarse cuando el workflow esté publicado en la rama predeterminada.

## 4. Arquitectura de la validación de integración

Se utilizó un sandbox Git independiente del directorio principal de desarrollo, bajo `week10-ci-sandbox`, para evitar modificar el Registry real del Champion P1.

El flujo de integración es:

1. Iniciar MLflow con base SQLite y directorio de artefactos exclusivos del sandbox.
2. Verificar la disponibilidad HTTP del servidor MLflow.
3. Registrar un modelo determinista con las ocho características del contrato P1 y asignarle el alias `champion`.
4. Validar que el modelo registrado esté en estado `READY` y tenga la firma esperada.
5. Iniciar FastAPI y comprobar su conexión con el Registry.
6. Ejecutar las verificaciones de salud, inferencia, reproducibilidad y manejo de errores.

### 4.1. Puertos utilizados en el sandbox

| Servicio | Puerto en macOS | Puerto interno del contenedor |
|---|---|---|
| MLflow | `127.0.0.1:15000` | `5000` |
| FastAPI | `127.0.0.1:18000` | `8000` |

FastAPI consulta a MLflow mediante `MLFLOW_TRACKING_URI=http://mlflow:5000` dentro de la red interna del sandbox. Los puertos `15000` y `18000` corresponden únicamente a la publicación en el equipo anfitrión.

## 5. Champion de prueba exclusivo para CI

En un Registry inicialmente vacío registramos:

| Propiedad | Valor |
|---|---|
| Modelo | `mibici-equipo15-p1-demanda` |
| Alias | `champion` |
| Versión | `1` |
| Estado | `READY` |
| Variables de entrada | 8, de tipo `double` |
| Predicción de referencia de CI | `2.785` |

Las características son `hour_of_day`, `weekday`, `fin_semana`, `salidas_lag_1h`, `salidas_lag_24h`, `salidas_lag_168h`, `media_estacion` y `media_estacion_dia_hora`.

**Aclaración:** este Champion es un modelo sintético utilizado para comprobar el contrato y las conexiones del sistema. No sustituye al modelo entrenado en el proyecto MiBici ni demuestra su rendimiento predictivo. El Champion real, validado en la actividad de integración anterior, produce una predicción de referencia distinta (`3.694163560787005`).

## 6. Validaciones ejecutadas localmente

Se confirmó que MLflow y FastAPI permanecieron en estado `healthy`. Los endpoints de ambos servicios respondieron HTTP 200; la API produjo una predicción de aproximadamente `2.785` con el modelo sintético. Ejecutamos:

```bash
python3 scripts/week10/smoke_integration.py \
  --api-url http://127.0.0.1:18000 \
  --mlflow-url http://127.0.0.1:15000 \
  --expected 2.785
```

**Resultado observado:** `11/11 pruebas aprobadas`, mensaje `SMOKE INTEGRATION COMPLETADA - OK` y código de salida `0`.

Las once comprobaciones abarcan salud de MLflow, salud y disponibilidad de FastAPI, inferencia, identidad del modelo, predicción esperada, segunda inferencia, reproducibilidad, ruta inexistente (404) y tres solicitudes inválidas (422).

## 7. Problemas encontrados y medidas aplicadas

### 7.1. Memoria de Podman Machine

El primer contenedor de MLflow terminó con código 137 y se observó `OOMKilled=true`. La máquina virtual de Podman disponía inicialmente de alrededor de 3.7 GiB. Tras comprobar que el Mac tenía 16 GiB de RAM física, ampliamos Podman Machine a **8192 MiB** y mantuvimos cuatro CPU. También configuramos MLflow con **un worker** para reducir el consumo.

### 7.2. Healthchecks y puertos

Al utilizar Podman Compose en macOS, la salida de configuración mostró una sustitución incorrecta de los puertos internos por los externos. Para la validación local utilizamos un Compose independiente del sandbox y comprobamos que los healthchecks se ejecutaran contra los puertos internos `5000` y `8000`.

### 7.3. Resolución DNS en contenedores temporales

La primera ejecución del bootstrap no pudo resolver el hostname `mlflow`. Al conectar explícitamente un contenedor temporal a la red del sandbox, comprobamos que `mlflow` resolvía a `10.89.1.2` y que el endpoint interno `http://mlflow:5000/health` devolvía `OK`. A partir de ello se pudo registrar y consultar el modelo sintético.

### 7.4. Consumo de memoria observado

En una medición puntual posterior al smoke test, `podman stats --no-stream` reportó aproximadamente **1.71 GiB** para MLflow y **156.4 MiB** para FastAPI. Estas cifras son observaciones del sandbox local y no representan consumos máximos o estimaciones definitivas para GitHub Actions.

## 8. Evidencias

Las capturas se conservan en `reports/week10/evidence/cicd/`:

| Archivo | Evidencia |
|---|---|
| `01_ci_sandbox_mlflow_healthy.png` | MLflow saludable y endpoint HTTP 200. |
| `02_ci_registry_champion.png` | Modelo sintético registrado como Champion, `READY` y con firma P1. |
| `03_ci_compose_services_healthy.png` | Ambos servicios saludables. |
| `04_ci_api_health_ready.png` | Endpoints `/health` y `/ready` con HTTP 200. |
| `05_ci_api_predict.png` | Predicción HTTP 200 con salida cercana a `2.785`. |
| `06_ci_smoke_integration.png` | Smoke test 11/11 y salida `0`. |

## 9. Pendientes antes de cerrar el punto 3

1. Revisar el workflow final y sus condiciones para las cuatro modalidades manuales.
2. Integrar o referenciar en el repositorio la configuración Compose compatible con CI que corresponda; no depender de archivos temporales exclusivos del sandbox local.
3. Confirmar que el bootstrap se ejecuta en la red correcta y que el runner puede iniciar MLflow antes de lanzar FastAPI.
4. Publicar el workflow en una rama separada del PR de integración del punto 2, después de su aprobación.
5. Ejecutar desde GitHub Actions `tests`, `build`, `integration` y `full`, y recopilar logs, resultados y capturas de los jobs.
6. Documentar cualquier diferencia entre Podman local y el runtime disponible en los runners de GitHub.

## 10. Conclusión

En esta etapa construimos y validamos localmente una ruta reproducible de integración entre FastAPI y MLflow utilizando un Champion sintético que mantiene el contrato de ocho variables P1. Logramos registrar el modelo, cargarlo desde FastAPI y superar las 11 pruebas automatizadas sin modificar el Champion original. Los resultados permiten avanzar a la validación del workflow en GitHub Actions; esa ejecución remota sigue pendiente y será necesaria para considerar terminado el componente CI/CD.
