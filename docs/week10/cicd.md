# Semana 10 — Pipeline CI/CD de MiBici (Equipo 15)

## 1. Objetivo

En esta actividad implementamos y validamos **MiBici-pipeline**, un workflow de GitHub Actions para automatizar pruebas de software, construcción de imágenes y validaciones de integración entre FastAPI y MLflow. Incluimos un disparador manual (`workflow_dispatch`) para ejecutar las validaciones sin necesidad de realizar un nuevo *push* ni abrir un *pull request*.

Validamos la integración tanto en un sandbox local independiente con Podman como en un runner de GitHub Actions. **La ejecución manual #9, realizada sobre `main` con la modalidad `integration`, finalizó satisfactoriamente**, incluyendo el registro del Champion sintético, las once pruebas end-to-end y la limpieza del entorno temporal.

## 2. Archivos y componentes

| Elemento | Función |
|---|---|
| `.github/workflows/MiBici-pipeline.yml` | Define los disparadores, modalidades y jobs de CI/CD. |
| `scripts/week10/bootstrap_ci_registry.py` | Registra un modelo P1 sintético, reproducible y exclusivo para CI en un Registry temporal. |
| `scripts/week10/smoke_integration.py` | Ejecuta once verificaciones HTTP, de inferencia y reproducibilidad. |
| `docker-compose.yml` | Orquesta los servicios de FastAPI y MLflow. |
| `Containerfile` y `Containerfile.mlflow` | Definen la construcción de las imágenes de FastAPI y MLflow. |
| `docs/week10/cicd.md` | Documenta la implementación, las validaciones y las evidencias. |
| `reports/week10/evidence/cicd/` | Reúne las capturas locales y de GitHub Actions. |

## 3. Disparadores y modalidades de ejecución

El workflow responde automáticamente a los eventos `push` y `pull_request`. También ofrece el disparador manual `workflow_dispatch`, disponible desde la pestaña **Actions** del repositorio, con cuatro modalidades:

- **`tests`:** ejecuta las pruebas de Python y Pytest.
- **`build`:** construye las imágenes de FastAPI y MLflow.
- **`integration`:** inicia un Registry temporal, registra el Champion sintético, levanta los servicios y ejecuta las pruebas end-to-end.
- **`full`:** selecciona los tres jobs: pruebas, construcción e integración.

En los eventos automáticos `push` y `pull_request` se ejecutan los jobs `tests` y `build`; el job `integration` se reserva para la ejecución manual. Los jobs pueden seleccionarse sin establecer dependencias obligatorias entre ellos.

Las ejecuciones automáticas de pruebas y construcción finalizaron con éxito en GitHub Actions. La ejecución manual `integration` también se validó satisfactoriamente en la ejecución #9. La modalidad `full` figura como opción implementada; no atribuimos a dicha modalidad una ejecución específica que no se haya documentado.

## 4. Arquitectura de la validación de integración

Para la primera validación utilizamos un sandbox Git independiente, `week10-ci-sandbox`, separado del directorio principal de desarrollo. Esto permitió comprobar el funcionamiento de los servicios sin modificar el Registry real del Champion P1.

La secuencia validada fue:

1. Iniciar MLflow con una base SQLite y un repositorio de artefactos exclusivos del entorno de prueba.
2. Verificar la disponibilidad HTTP de MLflow.
3. Registrar un modelo sintético determinista con las ocho variables P1 y asignarle el alias `champion`.
4. Verificar el estado `READY` y la firma de entrada del modelo.
5. Iniciar FastAPI y comprobar que recupera el Champion desde MLflow.
6. Validar endpoints, predicción de referencia, reproducibilidad y rechazo de solicitudes inválidas.
7. Publicar logs y eliminar el Registry temporal al terminar la ejecución de CI.

### 4.1. Puertos utilizados en el sandbox local

| Servicio | Puerto publicado en macOS | Puerto interno del contenedor |
|---|---|---|
| MLflow | `127.0.0.1:15000` | `5000` |
| FastAPI | `127.0.0.1:18000` | `8000` |

FastAPI utiliza `MLFLOW_TRACKING_URI=http://mlflow:5000` dentro de la red de contenedores. Los puertos `15000` y `18000` se utilizaron para evitar conflictos con los servicios del proyecto original en el equipo local. En GitHub Actions, el job de integración emplea la configuración Compose del workflow y sus puertos de servicio.

## 5. Champion sintético exclusivo para CI

En un Registry temporal inicialmente vacío registramos el siguiente modelo:

| Propiedad | Valor |
|---|---|
| Modelo | `mibici-equipo15-p1-demanda` |
| Alias | `champion` |
| Versión del Registry de prueba | `1` |
| Estado | `READY` |
| Variables de entrada | Ocho características de tipo `double` |
| Predicción de referencia CI | `2.785` |

Las ocho características son `hour_of_day`, `weekday`, `fin_semana`, `salidas_lag_1h`, `salidas_lag_24h`, `salidas_lag_168h`, `media_estacion` y `media_estacion_dia_hora`.

**Alcance del modelo:** el Champion sintético permite comprobar de manera determinista el contrato de inferencia y la comunicación entre servicios. No sustituye al modelo entrenado del proyecto MiBici ni representa una medida de calidad predictiva. El Champion real utilizado en la actividad anterior produjo una predicción de referencia diferente (`3.694163560787005`).

## 6. Validaciones y resultados

### 6.1. Validación local con Podman

Confirmamos que MLflow y FastAPI quedaron en estado `healthy`, que sus endpoints respondieron HTTP 200 y que la inferencia produjo una predicción próxima a `2.785`. Ejecutamos:

```bash
python3 scripts/week10/smoke_integration.py \
  --api-url http://127.0.0.1:18000 \
  --mlflow-url http://127.0.0.1:15000 \
  --expected 2.785
```

**Resultado:** `11/11 pruebas aprobadas`, mensaje `SMOKE INTEGRATION COMPLETADA - OK` y código de salida `0`.

### 6.2. Pruebas y construcción en GitHub Actions

Las ejecuciones automáticas asociadas a los eventos `push` y `pull_request` validaron satisfactoriamente los jobs **Pruebas y validación** y **Construcción de imágenes Podman**. Estos resultados acreditan la ejecución de tests y la construcción de imágenes en un runner remoto, además de las pruebas locales.

### 6.3. Integración end-to-end en GitHub Actions: ejecución #9

Ejecutamos manualmente `MiBici-pipeline` desde la rama `main`, seleccionando `execution_type=integration`. La ejecución **#9 terminó en estado `Success`**, con una duración observada de **2 minutos y 22 segundos**.

Completamos las operaciones de inicio de MLflow, registro del Champion sintético, inicio de FastAPI, inspección del Registry, smoke test, publicación de evidencias y limpieza final. El log del smoke test confirmó:

```text
RESULTADO: 11/11 pruebas aprobadas
SMOKE INTEGRATION COMPLETADA - OK
```

Las once pruebas verificaron:

| # | Comprobación | Resultado observado |
|---:|---|---|
| 01 | Salud de MLflow | HTTP 200 |
| 02 | Salud de FastAPI | HTTP 200 |
| 03 | Disponibilidad `/ready` | HTTP 200 |
| 04 | Inferencia con el Champion P1 | HTTP 200; modelo y predicción correctos |
| 05 | Predicción de referencia | Aproximadamente `2.785` |
| 06 | Segunda inferencia | HTTP 200 |
| 07 | Reproducibilidad | Mismo resultado en dos ejecuciones |
| 08 | Ruta inexistente | HTTP 404 |
| 09 | JSON vacío | HTTP 422 |
| 10 | Hora fuera de rango | HTTP 422 |
| 11 | Característica faltante | HTTP 422 |

La evidencia de esta ejecución se consulta en [MiBici-pipeline #9](https://github.com/A01797899/TC5061-Proyecto-Equipo15/actions/runs/44780494623).

## 7. Problemas encontrados y soluciones aplicadas

### 7.1. Consumo de memoria de Podman Machine

El primer contenedor de MLflow terminó con código `137` y el indicador `OOMKilled=true`. La máquina virtual disponía inicialmente de aproximadamente 3.7 GiB. Tras comprobar que el Mac tenía 16 GiB de RAM física, ampliamos Podman Machine a **8192 MiB**, mantuvimos cuatro CPU y configuramos MLflow con **un solo worker** para reducir su consumo.

### 7.2. Healthchecks y puertos del sandbox

La configuración combinada de Podman Compose en macOS produjo una interpretación incorrecta de puertos internos y publicados. Para las pruebas locales utilizamos un Compose independiente y verificamos los healthchecks contra los puertos internos `5000` y `8000`. En GitHub Actions se empleó Docker Compose con una configuración adicional de CI que fija MLflow a un worker.

### 7.3. Resolución DNS entre contenedores

El primer intento de registro del Champion no pudo resolver `mlflow`. Conectamos explícitamente el contenedor temporal a la red del sandbox y validamos que el nombre resolviera a `10.89.1.2` y que `http://mlflow:5000/health` devolviera `OK`. La comunicación entre contenedores quedó validada.

### 7.4. Memoria observada

Después del smoke test local, una medición puntual mediante `podman stats --no-stream` reportó aproximadamente **1.71 GiB** para MLflow y **156.4 MiB** para FastAPI. Estas mediciones corresponden al sandbox local y no representan máximos de consumo de los runners remotos.

### 7.5. Limpieza del Registry temporal en GitHub Actions — PR #39

En la ejecución manual **#6**, el registro del Champion, FastAPI y el smoke test finalizaron correctamente, pero la limpieza falló con `Permission denied` al intentar eliminar archivos de `runtime/mlflow/mlartifacts/` mediante `rm -rf`. Estos archivos habían sido generados por procesos dentro de los contenedores.

Mediante el [PR #39](https://github.com/A01797899/TC5061-Proyecto-Equipo15/pull/39), aprobado e integrado a `main`, corregimos exclusivamente el paso final del workflow:

```yaml
- name: Detener contenedores y eliminar Registry temporal
  if: always()
  run: |
    docker compose -f docker-compose.yml -f docker-compose.ci.yml down --remove-orphans || true
    if [ -d runtime/mlflow ]; then
      sudo rm -rf -- runtime/mlflow
    fi
```

La ejecución **#9** confirmó que la limpieza terminó correctamente y que el job completo alcanzó el estado `Success`.

## 8. Evidencias

Las ocho capturas corresponden a las validaciones del punto 3 y se organizan en `reports/week10/evidence/cicd/`:

| Archivo | Evidencia |
|---|---|
| `01_ci_sandbox_mlflow_healthy.png` | MLflow local saludable y endpoint HTTP 200. |
| `02_ci_registry_champion.png` | Champion sintético registrado en estado `READY` y con firma P1. |
| `03_ci_compose_services_healthy.png` | FastAPI y MLflow locales en estado saludable. |
| `04_ci_api_health_ready.png` | Endpoints `/health` y `/ready` con HTTP 200. |
| `05_ci_api_predict.png` | Predicción HTTP 200 próxima a `2.785`. |
| `06_ci_smoke_integration.png` | Validación local del smoke test: 11/11 y salida `0`. |
| **`07_ci_github_actions_integration.png`** | **Ejecución #9 en GitHub Actions: job de integración exitoso y limpieza completada.** |
| **`08_ci_github_actions_smoke_test.png`** | **Log de la ejecución #9: once comprobaciones PASS y mensaje de éxito del smoke test.** |

Las capturas 07 y 08 documentan la validación remota, que complementa las seis evidencias obtenidas en el sandbox local.

## 9. Estado final de la implementación CI/CD

La implementación del punto 3 quedó integrada en `main` mediante el PR #38 y se incorporó la corrección de permisos mediante el PR #39. Verificamos el funcionamiento de las modalidades automáticas para pruebas y construcción, así como la modalidad manual de integración en GitHub Actions.

El workflow está preparado para ofrecer cuatro opciones manuales (`tests`, `build`, `integration` y `full`). La evidencia documentada acredita las ejecuciones automáticas de pruebas y construcción y la ejecución manual `integration` #9; no atribuye una ejecución independiente a `full`.

**Estado del punto 3: implementación integrada y validada en GitHub Actions.**

## 10. Conclusión

En esta actividad implementamos un pipeline CI/CD que conecta las pruebas automatizadas, la construcción de imágenes y la integración reproducible de FastAPI con MLflow. Validamos el proceso tanto en nuestro entorno local como en GitHub Actions, donde registramos un Champion sintético con el contrato de ocho características P1 y completamos satisfactoriamente las once pruebas end-to-end.

También identificamos y resolvimos problemas de memoria, conectividad entre contenedores y permisos de archivos temporales. La corrección del PR #39 permitió que la ejecución manual #9 finalizara por completo, incluida la limpieza del Registry. Con las evidencias obtenidas documentamos una solución CI/CD funcional, repetible y separada del Champion real del proyecto MiBici.
