# Semana 10 - Orquestación de FastAPI y MLflow con Podman Compose

## Objetivo

Levantar con un solo comando la API de inferencia (Champion P1) y el servidor MLflow (Tracking + Model Registry), comunicados por una red privada y con el Registry en almacenamiento persistente. Es la base para la integración y el CI/CD (José Manuel), las pruebas de carga (Óscar) y el análisis FinOps (José Luis).

## Arquitectura

```
Cliente HTTP / Locust ──► 127.0.0.1:8000 ──► api (FastAPI: /health, /ready, /predict)
                                                │  http://mlflow:5000 (red mibici-net)
Navegador ─────────────► 127.0.0.1:5000 ──► mlflow (Tracking + Model Registry)
                                                │  volumen ./runtime/mlflow:/mlflow
                                                ▼
                                    mlflow.db + mlartifacts/ (fuera de Git)
```

## Archivos

| Archivo | Contenido |
|---|---|
| `docker-compose.yml` | Servicios `mlflow` y `api`, red `mibici-net`, volumen, variables, healthchecks, `depends_on` y reinicio |
| `Containerfile.mlflow` | Imagen del servidor MLflow (`python:3.12-slim` + `mlflow==3.16.1`), sin datos |
| `Containerfile` | Imagen de la API de la semana 8, sin cambios |
| `.env.example` | `API_PORT` y `MLFLOW_PORT` para Compose; conserva `MLFLOW_TRACKING_URI` local |
| `.gitignore` | Ignora `/runtime/` |
| `.containerignore`, `.dockerignore` | Excluyen `runtime/` para que la base y los artefactos no entren a la imagen de la API |

## Configuración de los servicios

| | `mlflow` | `api` |
|---|---|---|
| Imagen | `mibici-mlflow:week10` (958 MB) | `mibici-api:week10` (1.5 GB) |
| Puerto publicado | `127.0.0.1:${MLFLOW_PORT:-5000}` → 5000 | `127.0.0.1:${API_PORT:-8000}` → 8000 |
| Variables | `MLFLOW_DISABLE_TELEMETRY` | `MLFLOW_TRACKING_URI=http://mlflow:5000`, `MLFLOW_DISABLE_TELEMETRY` |
| Volumen | `./runtime/mlflow:/mlflow` | Ninguno |
| Healthcheck | `GET /health` cada 10 s, 12 reintentos, 30 s de gracia | `GET /ready` cada 10 s, 12 reintentos, 60 s de gracia |
| Arranque | Primero | Espera a que `mlflow` esté sano (`condition: service_healthy`) |
| Reinicio | `unless-stopped` | `unless-stopped` |

Decisiones:

- **`--serve-artifacts`:** los modelos se registraron con el esquema `mlflow-artifacts:/`, así que el servidor debe servir los artefactos.
- **`--allowed-hosts`:** lleva `mlflow:5000`, `localhost:5000` y `127.0.0.1:5000`. MLflow 3.16 responde `403 Invalid Host header` a cualquier otro `Host`, incluido el nombre del servicio, si no está en la lista.
- **Tracking URI de la API:** dentro de la red, la API usa `http://mlflow:5000`, porque `localhost` sería el propio contenedor de la API.
- **Healthcheck de la API:** usa `/ready` y no `/health`, porque solo responde 200 con el Champion cargado.
- **Arranque en orden:** la API carga el modelo una sola vez al arrancar y no reintenta. Por eso espera con `depends_on: service_healthy`, que podman-compose respeta desde la versión 1.1.
- **Puertos en `127.0.0.1`:** no se expone MLflow fuera del equipo, porque no tiene autenticación.

## Requisitos

Probado en macOS (Apple Silicon) con:

| Herramienta | Versión |
|---|---|
| Podman | 6.2.0 |
| podman-compose | 1.6.0 |
| Máquina de Podman | `libkrun`, 4 CPU, 4 GiB, 100 GiB |

```bash
brew install podman podman-compose
podman machine init --cpus 4 --memory 4096
podman machine start
```

## Datos de MLflow

`mlflow.db` y `mlartifacts/` no están en Git. Se copian una sola vez del respaldo del Registry original (el de José Manuel, con el Champion P1) a `runtime/mlflow/`:

```bash
mkdir -p runtime/mlflow
cp /ruta/al/respaldo/mlflow.db runtime/mlflow/mlflow.db
cp -R /ruta/al/respaldo/mlartifacts runtime/mlflow/
git check-ignore runtime/mlflow/mlflow.db
```

La base debe tener `mibici-equipo15-p1-demanda` con el alias `champion` en la versión 1, y `mlartifacts/1/models/m-04774b8f25ff468b933a18756c24b068/` debe existir. Conviene conservar el respaldo original aparte, porque el servidor escribe en la copia de `runtime/`.

## Operación

```bash
podman-compose -f docker-compose.yml up -d --build
podman-compose -f docker-compose.yml ps
podman-compose -f docker-compose.yml logs api
podman-compose -f docker-compose.yml logs mlflow
podman-compose -f docker-compose.yml down
```

Con `up`, los dos contenedores quedan `(healthy)` en menos de un minuto. `down` borra los contenedores y la red, pero no `runtime/mlflow`. No hay que usar `podman volume prune` ni `podman system prune --volumes`.

```bash
curl -i http://127.0.0.1:8000/health
curl -i http://127.0.0.1:8000/ready
curl -i -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" --data-binary @postman/examples/p1-valid.json
```

## Validaciones realizadas

| Validación | Resultado |
|---|---|
| `podman-compose config` | Servicios `api` y `mlflow`, red, volumen, variables, puertos y healthchecks interpretados sin errores |
| `build` | `mibici-api:week10` y `mibici-mlflow:week10` construidas |
| `up -d --build` | `mibici-equipo15_mlflow_1` y `mibici-equipo15_api_1` en `(healthy)` |
| MLflow | Interfaz en `127.0.0.1:5000`; `mibici-equipo15-p1-demanda` v1 `READY` con `champion` |
| `/health` | 200, `{"status":"ok"}` |
| `/ready` | 200, `{"ready":true}` |
| `/predict` con `p1-valid.json` | 200, `{"prediction":3.694163560787005,"model":"mibici-equipo15-p1-demanda"}`, el mismo valor de la semana 8 |
| Red interna | Desde `api`: `MLFLOW_TRACKING_URI = http://mlflow:5000`, `mlflow` resuelve a `10.89.0.2` y `http://mlflow:5000/version` responde `3.16.1` |
| Persistencia | Tras `down`, la base y los 155 archivos de artefactos siguen en `runtime/mlflow`; tras `up`, `/ready` y `/predict` responden igual |
| Reinicio de MLflow | `podman restart mibici-equipo15_mlflow_1`: vuelve a `(healthy)` y la API sigue respondiendo sin reiniciarse |
| Recursos (`podman stats`) | `mlflow`: 2.25 GB de memoria, 132 procesos; `api`: 227 MB, 16 procesos (en reposo) |

## Limitaciones y notas para las siguientes actividades

- **CI/CD (José Manuel).** GitHub Actions corre en una máquina limpia sin `runtime/mlflow`, así que el pipeline necesita una forma de proveer la base y los artefactos del Champion (un artefacto privado o un respaldo descargable) o limitarse a construir y validar `/health` sin el modelo. `docker-compose.yml` sigue la especificación de Compose y funciona igual con `docker compose`.
- **Recuperación de la API.** Si la API arranca sin MLflow disponible, queda en `/ready` 503 y no vuelve a intentar cargar el modelo; hay que reiniciarla (`podman restart mibici-equipo15_api_1`). El reinicio de MLflow con la API ya cargada no la afecta.
- **Memoria de MLflow (Óscar y José Luis).** El servidor arranca varios procesos y usa unos 2.25 GB en reposo, más que la API. Limitar los workers del servidor es una optimización candidata para el análisis FinOps, pero no se aplicó para no cambiar la configuración validada.
- **Contenido del Registry.** El respaldo usado solo trae los artefactos del modelo P1; P2 y P3 aparecen registrados, pero no se pueden cargar. Tampoco incluye el Challenger de la semana 9. La API solo usa P1 `champion`, así que no la afecta.
- **SQLite.** Admite un solo escritor, así que es adecuado para este entorno local y no para varias réplicas de MLflow.
- **Usuario de MLflow.** El contenedor de MLflow corre como root para poder escribir en la base montada desde macOS; la API corre como `appuser`.

## Problemas frecuentes

| Síntoma | Qué revisar |
|---|---|
| `Cannot connect to Podman` | `podman machine list` y `podman machine start` |
| Puerto 5000 u 8000 ocupado | Que no siga corriendo el MLflow local o la API de la semana 8: `lsof -nP -iTCP:5000 -sTCP:LISTEN` |
| `/ready` 503 | `podman-compose logs api`; base o artefactos incompletos en `runtime/mlflow` |
| `403 Invalid Host header` | El `Host` usado no está en `--allowed-hosts` |
| No aparece el Champion | `runtime/mlflow/mlflow.db` es otra base o está vacía |

## Evidencias

En `reports/week10/evidence/compose/`:

| Archivo | Evidencia |
|---|---|
| `01_entorno_podman.png` | Versiones de Podman y podman-compose, máquina en ejecución y `hello-world` |
| `02_rama_github.png` | Rama al día con `main` |
| `03_compose_config.png` | Configuración de Compose interpretada |
| `04_build_images.png` | Construcción de las dos imágenes |
| `05_compose_up.png` | Arranque con un solo comando |
| `06_contenedores_running.png` | Ambos contenedores en `(healthy)` |
| `07_mlflow_ui.png` | Interfaz de MLflow con los modelos registrados |
| `08_champion_registry.png` | P1 versión 1 `Ready` con alias `champion` |
| `09_api_health.png` | `/health` 200 |
| `10_api_ready.png` | `/ready` 200 |
| `11_api_predict.png` | Predicción real con el Champion |
| `12_red_interna.png` | Variable, DNS y HTTP de `api` hacia `mlflow` |
| `13_persistencia.png` | Datos y predicción después de `down` y `up` |
| `14_stats.png` | Recursos por contenedor |
| `15_validacion_final.png` | Reinicio de MLflow y sistema completo funcionando |
