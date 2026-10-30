# Hallazgos de seguridad — Semana 7

Revisión del 30 de octubre de 2026 en `feature/A01732182-week7-Security`.

Entorno: `TC5061-Proyecto-Equipo15/.venv`, Python 3.12.13. Ahí están las dependencias del proyecto y las herramientas (`bandit==1.9.4`, `pip-audit==2.10.1`). Comprobación: `sys.prefix` apunta a esa carpeta `.venv`.

Pasos, desde la raíz del repositorio: crear `.venv` con Python 3.12, `pip install -r requirements.txt`, `pip install -r requirements-security.txt`, Bandit sobre `mibici/`, pip-audit sobre `requirements.txt` y `requirements-dev.txt`. El detalle de credenciales está en `diagnostico.md`. Los avisos de dependencias se listan aquí completos.

## Credenciales

Git no sigue `.env`, `mlflow.db` ni archivos `.pem`, `.key`, `.p12` o `.pfx`. El código Python no asigna contraseñas, secretos, API keys ni tokens. La única variable de entorno es `MLFLOW_TRACKING_URI`, con valor por defecto `http://127.0.0.1:5000`.

Correcciones de configuración:

| Cambio | Motivo |
|---|---|
| `.gitignore`: `*.pem`, `*.key`, `*.p12`, `*.pfx` | Cubrir los nombres de claves y certificados de la revisión |
| `.env.example` | Plantilla con la URI local, sin secretos. Un `.env` real sigue ignorado |

## Bandit (`mibici/`)

El primer análisis está en `bandit_report.json` y en `reports/week7/evidence/security/01_bandit.txt`: **19 hallazgos** (1 medio, 18 bajos), todos con confianza alta. Tras las correcciones, `bandit_report_despues.json` y `02_bandit_despues.txt` quedan en **0 hallazgos**; las 19 líneas anotadas siguen documentadas abajo.

| Id | Severidad | Dónde | Decisión |
|---|---|---|---|
| B310 | Media | `mibici/dataset.py`, `urlopen` | Se aceptan solo los esquemas `http`, `https` y `file`. `file` lo usan las pruebas con un CSV local (`Path.as_uri()`). La ingesta real usa las URL `https` de `config.py`. La llamada queda con `# nosec B310` porque Bandit marca `urlopen` aunque el esquema ya se comprobó. Prueba nueva: `test_descarga_rechaza_esquema_no_permitido`. |
| B404 | Baja | `import subprocess` en `train.py`, `search.py`, `registry.py`, `p1_baseline.py`, `p2_baseline.py`, `p3_baseline.py` | El módulo solo arma listas de argumentos, sin `shell=True`. `# nosec B404`. |
| B603 | Baja | `subprocess.run` en esos mismos módulos | Los comandos son `git rev-parse HEAD` con literales, o el intérprete del proyecto (`sys.executable -m`) y un módulo de `BASELINE_MODULES` / `SEARCH_MODULES`. `# nosec B603`. |
| B607 | Baja | Las llamadas a `git` | El ejecutable es el nombre `git` resuelto en `PATH`, con argumentos fijos. `# nosec B607`. |

Esas llamadas no cambian de comportamiento. El entrenamiento, el tracking y el registro siguen igual.

## pip-audit

Salidas de texto: `reports/week7/evidence/security/03_pip_audit_requirements.txt` y `04_pip_audit_requirements_dev.txt`. JSON: `pip_audit_requirements.json` y `pip_audit_requirements_dev.json`.

`-e .` instala el paquete local `mibici` 0.0.1. pip-audit lo omite con el mensaje *Dependency not found on PyPI and could not be audited: mibici (0.0.1)*. El resto de cada archivo sí se auditó. La línea `-e .` de `requirements.txt` se mantiene.

### `requirements.txt` — mlflow 3.16.1

Tres avisos. La corrección publicada más alta es **3.17.2**. La versión del proyecto permanece en **3.16.1**, la de la reproducción documentada de las semanas 6 y 7 (`requirements.txt` y `python_env.yaml`).

| Aviso | Corrección | Qué describe el aviso |
|---|---|---|
| CVE-2026-53995 (GHSA-wqmh-qccw-pqwf) | 3.17.2 | Recorrido de rutas en el backend FileStore: un trace id con `..` en `POST /api/3.0/mlflow/traces` puede escribir archivos fuera del directorio de trazas. Afecta a MLflow 3.17.1 y anteriores. |
| CVE-2026-115886 (GHSA-2wqp-q3c6-ccj5) | 3.17.0 | `POST /ajax-api/3.0/mlflow/scorer/invoke` no exige autenticación e instancia una clase indicada por el cliente. Con los paquetes opcionales `trulens` y `trulens-providers-litellm` (no están en `requirements.txt`) el servidor puede hacer una petición HTTP saliente. |
| CVE-2026-116691 (GHSA-j7qq-vr8c-xvw3) | 3.17.0 | `mlflow.crewai.autolog()` puede copiar la API key del LLM de CrewAI a la telemetría de la traza. Este repositorio no llama a ese autolog. |

El servidor de las instrucciones es `http://127.0.0.1:5000`. Subir el pin a 3.17.2 queda como actualización aparte del stack de entrenamiento, cuando el equipo vuelva a reproducir baselines, búsquedas y el registro.

### `requirements-dev.txt` — setuptools 80.10.2

Además de los tres avisos de MLflow (el archivo incluye `-r requirements.txt`):

| Aviso | Corrección | Qué describe el aviso |
|---|---|---|
| PYSEC-2026-3447, también CVE-2026-59890 y GHSA-h35f-9h28-mq5c | 83.0.0 | Al armar un sdist, `FileList` compara `MANIFEST.in` sin normalizar Unicode. En APFS o HFS+ un nombre NFD puede saltarse una exclusión NFC. pip-audit repite este aviso con dos redacciones. |
| GHSA-xmj3-9gf7-h52w | 84.1.0 | En Windows, `namespace_packages.txt` de un wheel puede hacer que la instalación legacy cree un `__init__.py` fuera del directorio pedido. El contenido de ese archivo no lo controla el atacante. |

`setuptools==80.10.2` permanece: DeepChecks 0.19.1 importa `pkg_resources`, que setuptools 81 y posteriores ya no incluyen. Es dependencia de desarrollo. Esta revisión se ejecutó en Linux.

Ningún otro paquete de `requirements.txt` o `requirements-dev.txt` tuvo avisos en esta corrida.
