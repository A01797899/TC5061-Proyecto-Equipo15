# MiBici — Proyecto MLOps | Equipo 15 | Semanas 5 y 6

[![Cookiecutter Data Science](https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter)](https://cookiecutter-data-science.drivendata.org/)

**Repositorio:** https://github.com/A01797899/TC5061-Proyecto-Equipo15
**Curso:** TC5061 — Operaciones de Aprendizaje Automático
**Experimento:** `mibici-equipo15`
**Datos:** viajes de MiBici, enero–diciembre de 2024
**Versiones utilizadas en la prueba:** Python 3.12.15 y MLflow 3.16.1.

> Este README integra la base de la Semana 5 con la implementación **Parte A — MLflow Project y reproducibilidad** de la Semana 6. Los pasos de ingesta y los tres baselines se validaron con `--env-manager local`; además, la ingesta y P1 se validaron con `--env-manager virtualenv`. La integración de modelos candidatos, búsquedas avanzadas y Model Registry corresponde a otras actividades y **no debe darse por finalizada** con esta guía.

## 1. Descripción, objetivos y alcance

El proyecto utiliza datos abiertos de MiBici para construir una base reproducible de aprendizaje automático: descarga, limpieza, control de calidad, ingeniería de características, experimentación y seguimiento con MLflow. En la Semana 6, se añaden un archivo `MLproject`, un entorno declarado en `python_env.yaml`, un servidor MLflow Tracking con SQLite y el punto de entrada `train` parametrizable.

| Pregunta | Objetivo | Problema |
|---|---|---|
| **P1 — Demanda** | Estimar el número de viajes iniciados en una estación durante una hora. | Regresión |
| **P2 — Riesgo proxy** | Estimar presión de desequilibrio aproximada a partir de llegadas, salidas y capacidad. | Clasificación |
| **P3 — Perfiles de uso** | Agrupar estaciones con patrones horarios semejantes. | Clustering |

**Limitaciones:** la etiqueta de P2 **no** indica disponibilidad real de bicicletas ni anclajes; depende de una presión proxy `abs(llegadas - salidas) / capacidad` con umbral 0.25. La nomenclatura de estaciones usada es de septiembre de 2026, posterior a los viajes de 2024. El clustering de P3 tiene carácter exploratorio.

## 2. Requisitos previos

- **Sistemas operativos:** macOS (`zsh`), Linux (`bash`/`zsh`) y Windows 10/11 (PowerShell). El procedimiento de ejecución validado se realizó en macOS; las instrucciones Linux y Windows son orientativas y deben comprobarse en sus plataformas respectivas.
- Git, conexión a Internet para la primera descarga, acceso a los archivos públicos de MiBici y espacio suficiente para más de 4.6 millones de viajes.
- **Python 3.12.15** recomendado para reproducir exactamente el entorno probado. `pyproject.toml` requiere Python `~=3.12.0`, por lo que Python 3.13 no es compatible con esa restricción.
- Dependencias fijadas en `requirements.txt`; en las pruebas se utilizó **MLflow 3.16.1**.
- **Servidor MLflow local activo** para los comandos de Tracking y Model Registry.

**Todos los comandos de proyecto se ejecutan desde la raíz** `TC5061-Proyecto-Equipo15`, salvo donde se indique lo contrario. Los comandos posteriores con `.venv/bin/...` corresponden a **macOS/Linux**; en **Windows PowerShell** se reemplazan por `.venv\Scripts\python.exe` y `.venv\Scripts\mlflow.exe`. Para evitar el alias de `python` a Python 3.13 que puede existir en `zsh`, se usan rutas explícitas al intérprete del entorno virtual.

## 3. Clonar el proyecto o preparar la rama de trabajo

Clon limpio:

```bash
git clone https://github.com/A01797899/TC5061-Proyecto-Equipo15.git
cd TC5061-Proyecto-Equipo15
git status
```

Para desarrollar la Parte A desde un clon ya existente (realizar esto solamente si no se ha creado la rama):

```bash
git status --short            # Comprobar que no hay cambios pendientes
git checkout main
git pull origin main
git checkout -b feature/A01797899-mlflow-project
git push -u origin feature/A01797899-mlflow-project
```

Si la rama ya existe, utilizar `git switch feature/A01797899-mlflow-project`; no repetir `git checkout -b`.

> En un clon limpio de `main` anterior a la integración del PR de Semana 6, los archivos `MLproject` y `python_env.yaml` todavía no estarán disponibles. Para reproducir la Parte A, utilizar la rama que contenga esos cambios o `main` una vez fusionado el PR.

## 4. Preparación de Python y virtualenv local en macOS

Comprobar la versión:

```bash
python3.12 --version
```

Resultado usado en las pruebas: `Python 3.12.15`.

Si Python 3.12 no está instalado y se usa Homebrew:

```bash
brew install python@3.12
"$(brew --prefix python@3.12)/bin/python3.12" --version
```

Crear y activar el entorno de desarrollo:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Si `python3.12` no está en el `PATH`:

```bash
"$(brew --prefix python@3.12)/bin/python3.12" -m venv .venv
source .venv/bin/activate
```

Verificar **directamente el Python del entorno**:

```bash
.venv/bin/python --version
.venv/bin/python -m pip --version
```

### 4.1. Solucionar el alias `python=python3.13` de zsh

En la sesión de desarrollo se observó que `which python` mostraba un alias a Python 3.13, aunque `.venv` estuviera activado. Revisar:

```bash
alias python
command -v python
.venv/bin/python --version
```

Si el alias existe, desactivarlo **solo para la terminal actual**:

```bash
unalias python
python --version
which python
```

La ruta de `which python` debe terminar en `.venv/bin/python` y la versión debe ser 3.12.15. También puede evitarse completamente el alias usando `.venv/bin/python`.

### 4.2. Instalar dependencias

```bash
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python -c 'import mlflow; print(mlflow.__version__)'
.venv/bin/mlflow --version
```

Resultados esperados: `No broken requirements found` y MLflow `3.16.1`.

El archivo `requirements.txt` incluye el paquete local mediante `-e .`. El archivo `python_env.yaml` usa dependencias fijadas y no incluye `-e .`; MLflow ejecuta los módulos `mibici` desde la raíz del proyecto. Si se actualizan librerías, sincronizar **ambos** archivos.

### 4.3. Preparación del entorno local en Linux (Ubuntu/Debian)

Instalar herramientas de compilación y bibliotecas necesarias para construir Python con `pyenv`:

```bash
sudo apt update
sudo apt install -y git curl build-essential make libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev libffi-dev liblzma-dev \
  xz-utils tk-dev ca-certificates
```

Después de instalar y configurar `pyenv` en la sección siguiente, crear el entorno local con el intérprete administrado por `pyenv`:

```bash
~/.pyenv/versions/3.12.15/bin/python -m venv .venv
source .venv/bin/activate
.venv/bin/python --version
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
```

En otras distribuciones, instalar los paquetes equivalentes con su gestor. La disponibilidad de compiladores y bibliotecas puede modificar el procedimiento.

### 4.4. Preparación del entorno local en Windows (PowerShell)

Instalar **Python 3.12.15** con `pyenv-win`, conforme a la sección 5.4. Abrir PowerShell desde la raíz del proyecto y crear un entorno virtual usando la ruta del intérprete instalado:

```powershell
$python312 = Join-Path $env:USERPROFILE '.pyenv\pyenv-win\versions\3.12.15\python.exe'
& $python312 --version
& $python312 -m venv .venv
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import mlflow; print(mlflow.__version__)"
```

Si PowerShell bloquea la activación por la política de ejecución, puede omitirse la activación y ejecutar directamente `.\.venv\Scripts\python.exe` y `.\.venv\Scripts\mlflow.exe`; no es necesario cambiar una política de seguridad global. La ruta anterior corresponde a la estructura predeterminada de `pyenv-win`; comprobarla si se instaló en otra ubicación.

Para ejecutar los entry points desde Windows, abrir otra consola PowerShell para el servidor y establecer `$env:MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000'` en la consola del cliente. Los comandos Unix de las secciones 7–11 deben adaptarse a PowerShell. **No se ha validado en Windows la creación automática del entorno mediante `--env-manager virtualenv`;** empezar con `--env-manager local` y comprobar posteriormente la compatibilidad del gestor de entornos de MLflow.

## 5. Instalar y configurar pyenv según el sistema operativo

`--env-manager virtualenv` permite a MLflow crear su propio entorno usando la versión de Python declarada en `python_env.yaml`; en el equipo macOS probado se requirió `pyenv`.

### 5.1. Instalación en macOS

```bash
brew install pyenv
pyenv --version
```

La prueba documentada utilizó `pyenv 2.8.13`.

Configurar zsh **de forma permanente**: comprobar primero si existe la línea de inicialización:

```bash
grep -n 'pyenv init' ~/.zshrc
```

Si **no** aparece, añadir **una sola vez** al final de `~/.zshrc`:

```bash
printf '\n# pyenv for MLflow virtualenv\neval "$(pyenv init - zsh)"\n' >> ~/.zshrc
```

Recargar y verificar:

```bash
source ~/.zshrc
pyenv --version
```

Para la terminal actual, también es válido ejecutar `eval "$(pyenv init - zsh)"`.

### 5.2. Instalar Python 3.12.15 con pyenv

```bash
pyenv versions
pyenv install --list | grep 3.12.15
pyenv install 3.12.15
pyenv versions
~/.pyenv/versions/3.12.15/bin/python --version
```

La instalación no reemplaza el Python instalado con Homebrew ni `.venv`. No es necesario ejecutar `pyenv global` ni `pyenv local`. En el equipo probado, se verificó Python 3.12.15 tanto en `.venv` como en `~/.pyenv/versions/3.12.15/bin/python`.

### 5.3. Instalar y configurar pyenv en Linux

La instalación oficial para Linux/Unix utiliza el instalador de `pyenv`. Ejecutar en una terminal con `git` y `curl` disponibles y con las dependencias de compilación de la sección 4.3:

```bash
curl -fsSL https://pyenv.run | bash
```

En **bash**, añadir la configuración una sola vez a `~/.bashrc` (si todavía no está presente):

```bash
cat >> ~/.bashrc <<'EOF'

# pyenv
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - bash)"
EOF
source ~/.bashrc
```

Si usas **zsh**, colocar las declaraciones equivalentes en `~/.zshrc` y utilizar `eval "$(pyenv init - zsh)"` en lugar de la variante para bash. Verificar e instalar:

```bash
pyenv --version
pyenv install 3.12.15
pyenv versions
~/.pyenv/versions/3.12.15/bin/python --version
```

No es necesario establecer esta versión como `pyenv global`. Repetir la prueba de `mlflow run` con el entorno declarado conforme a la sección posterior.

### 5.4. Instalar pyenv-win en Windows (PowerShell)

En Windows nativo se utiliza **`pyenv-win`**, proyecto diferente a `pyenv` de macOS/Linux. Según su documentación oficial, descargar el instalador PowerShell, revisarlo si así lo requieren las políticas del equipo y ejecutarlo:

```powershell
Invoke-WebRequest -UseBasicParsing -Uri 'https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1' -OutFile '.\install-pyenv-win.ps1'
.\install-pyenv-win.ps1
```

Cerrar y volver a abrir PowerShell para cargar las variables de entorno. Si `pyenv` no se reconoce, consultar el procedimiento de variables `PYENV`, `PYENV_HOME`, `PYENV_ROOT` y `Path` en la [guía oficial de instalación](https://github.com/pyenv-win/pyenv-win/blob/master/docs/installation.md).

```powershell
pyenv --version
pyenv install -l
pyenv install 3.12.15
pyenv versions
```

Comprobar que la versión quedó instalada (sin necesidad de modificar la versión global):

```powershell
$python312 = Join-Path $env:USERPROFILE '.pyenv\pyenv-win\versions\3.12.15\python.exe'
& $python312 --version
```

Si Windows redirige `python` a Microsoft Store, revisar **Settings > Apps > Advanced app settings > App execution aliases** y desactivar los alias de instalador que interfieran. Para este proyecto puede utilizarse siempre la ruta completa del ejecutable.

**Compatibilidad:** la validación de MLflow `--env-manager virtualenv` con `pyenv` se realizó en macOS. `pyenv-win` no es el mismo programa y no debe darse por garantizado que MLflow lo use automáticamente de la misma forma. En Windows, la alternativa documentada es crear `.venv` con Python 3.12.15, instalar `requirements.txt` y ejecutar inicialmente con `--env-manager local`; registrar separadamente cualquier validación del entorno aislado.

## 6. Arquitectura y archivos del MLflow Project

```text
TC5061-Proyecto-Equipo15/
├── MLproject                         # NUEVO: puntos de entrada ingest/train
├── python_env.yaml                   # NUEVO: entorno Python fijado
├── requirements.txt
├── pyproject.toml
├── README.md
├── .gitignore
├── data/
│   ├── raw/                         # CSV de 2024, NO versionados
│   ├── processed/                   # Parquet, NO versionados
│   └── manifiesto.json               # Metadatos y hashes SHA-256
├── mibici/
│   ├── config.py                     # Fechas, rutas, semilla y Tracking URI
│   ├── dataset.py                    # Ingesta
│   ├── features.py                   # Transformaciones reutilizables
│   └── modeling/
│       ├── train.py                  # NUEVO orquestador parametrizado
│       ├── p1_baseline.py
│       ├── p2_baseline.py
│       └── p3_baseline.py
├── tests/
├── notebooks/
├── references/
├── reports/figures/
└── models/
```

`MLproject` declara `ingest`, que ejecuta `mibici.dataset`, y `train`, que recibe `question`, `model` y `n_estimators`. En la implementación de la Parte A, `model=baseline` está conectado a las tres implementaciones existentes; los valores nuevos de `model` (como `rf`) **no se consideran disponibles** hasta que se integren las ramas de experimentación. El valor `n_estimators=0` indica al orquestador que resuelva el valor predeterminado desde `mibici/config.py` y no se utiliza en el entrenamiento baseline.

`config.py` mantiene el experimento `mibici-equipo15`, la semilla `42`, rutas de datos y las fechas de partición. Para P1/P2: entrenamiento enero–agosto 2024, validación septiembre–octubre, prueba noviembre–diciembre.

## 7. Tracking Server SQLite: respaldar, iniciar y validar

### 7.1. Proteger corridas existentes

Si ya existe `mlflow.db`, **detener primero cualquier servidor que la esté modificando** y realizar un respaldo **fuera** de Git:

```bash
mkdir -p ../mlflow-backup
cp -p mlflow.db ../mlflow-backup/mlflow-semana5.db
ls -lh mlflow.db ../mlflow-backup/mlflow-semana5.db
shasum -a 256 mlflow.db ../mlflow-backup/mlflow-semana5.db
cmp mlflow.db ../mlflow-backup/mlflow-semana5.db
sqlite3 ../mlflow-backup/mlflow-semana5.db 'PRAGMA integrity_check;'
```

`cmp` sin salida indica archivos idénticos; la integridad SQLite debe devolver `ok`. Este procedimiento respalda la **base SQLite**, no todos los artefactos externos: conservar también las carpetas utilizadas por los artefactos de las corridas.

**Importante:** durante la Semana 5 también se hizo una ejecución integral en un clon separado llamado `TC5061-MiBici-MLflow-Runs`. Su `mlflow.db` es **distinta** a la del clon de desarrollo. El mismo nombre `mibici-equipo15` no significa que ambas bases compartan corridas. Elegir una fuente oficial de tracking y verificar sus run IDs, métricas y `artifact_uri` antes de comparar modelos o mover bases. No sustituir una base por otra sin planificar la migración de artefactos.

### 7.2. Arrancar MLflow Server (terminal A)

Abrir una terminal en la raíz del proyecto:

```bash
source .venv/bin/activate
.venv/bin/mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --host 127.0.0.1 \
  --port 5000
```

Mantener abierta la terminal mientras se ejecutan los experimentos. El directorio de trabajo importa para resolver `sqlite:///mlflow.db`: ejecutar siempre desde la raíz del repositorio. Si el archivo no existe, MLflow puede crear una base nueva y los baselines previos no aparecerán.

Abrir http://127.0.0.1:5000 en el navegador.

### 7.3. Configurar cliente (terminal B)

En una segunda terminal, desde la misma raíz:

```bash
source .venv/bin/activate
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
echo "$MLFLOW_TRACKING_URI"
.venv/bin/python -c 'import mlflow; print(mlflow.get_tracking_uri())'
curl -I http://127.0.0.1:5000
```

Se verificó respuesta `HTTP/1.1 200 OK` del servidor. Mostrar la URI con Python solo confirma la configuración de cliente; la llamada `curl` comprueba la accesibilidad HTTP.

### 7.4. Comprobar experimento y corridas

```bash
.venv/bin/python - <<'PY'
from mlflow import MlflowClient
client = MlflowClient()
for exp in client.search_experiments():
    print(exp.experiment_id, exp.name)
exp = client.get_experiment_by_name('mibici-equipo15')
if exp is None:
    raise SystemExit('No se encontró mibici-equipo15: revisar backend SQLite del servidor')
for run in client.search_runs([exp.experiment_id], max_results=100):
    print(run.info.run_id, run.info.status, run.data.tags.get('mlflow.runName'))
PY
```

La base utilizada durante las validaciones incluyó los cinco nombres de baselines de la Semana 5 y algunas corridas repetidas de reproducción. Todos los baselines mostrados estaban en estado `FINISHED`.

## 8. Ejecutar la ingesta con MLflow Project

Con el servidor activo y `MLFLOW_TRACKING_URI` exportada en terminal B:

```bash
.venv/bin/mlflow run . -e ingest --env-manager local
```

Resultado validado el 23 de octubre de 2026:

- 12 CSV mensuales reutilizados; se utiliza también `nomenclatura_2026_09.csv`.
- 4,645,921 viajes limpios; 0 registros descartados y 0 duplicados en la reproducción.
- Manifiesto y Parquet regenerados.
- **Versión de datos:** `7e1470ac0632`.
- MLflow Project finalizó con `Run (...) succeeded`.

Verificar:

```bash
ls -lh data/manifiesto.json
ls -lh data/processed/
.venv/bin/python -c 'from mibici.dataset import version_datos; print(version_datos())'
```

La versión debe coincidir con la de la Semana 5 para poder comparar resultados. Si cambia, examinar los hashes fuente del manifiesto antes de entrenar.

### 8.1. Ingesta con entorno aislado declarado

Tras completar la configuración de `pyenv` de la sección 6:

```bash
.venv/bin/mlflow run . -e ingest --env-manager virtualenv
```

**Validado:** MLflow creó/utilizó un entorno propio bajo `~/.mlflow/envs/`, instaló dependencias y ejecutó `mibici.dataset`. Se confirmó nuevamente `7e1470ac0632` y `Run (...) succeeded`. La carpeta `~/.mlflow/envs/` es local; no se versiona.

## 9. Ejecutar los baselines P1, P2 y P3

Los comandos siguientes se validaron usando `--env-manager local` desde la raíz con el servidor activo:

### P1 — Regresión

```bash
.venv/bin/mlflow run . -e train \
  -P question=p1 \
  -P model=baseline \
  --env-manager local
```

Ejecuta los baselines `station_mean` y `station_weekday_hour_mean`.

| Modelo | MAE de test | RMSE de test |
|---|---:|---:|
| Media histórica por estación | 1.3621 | 2.1185 |
| Media por estación, día y hora | **0.8556** | **1.4635** |

### P2 — Clasificación de riesgo proxy

```bash
.venv/bin/mlflow run . -e train \
  -P question=p2 \
  -P model=baseline \
  --env-manager local
```

| Modelo | Precision | Recall | F1 | Average Precision |
|---|---:|---:|---:|---:|
| Clase mayoritaria | 0.0000 | 0.0000 | 0.0000 | 0.0471 |
| Probabilidad histórica | 0.2851 | 0.4563 | 0.3509 | 0.3052 |

- Panel de entrenamiento: 2,119,872 observaciones estación-hora.
- Validación: 529,968; prueba: 529,968.
- Umbral de presión proxy: 0.25; umbral de probabilidad seleccionado mediante validación: 0.21.

### P3 — Clustering

```bash
.venv/bin/mlflow run . -e train \
  -P question=p3 \
  -P model=baseline \
  --env-manager local
```

| Indicador | Resultado de la reproducción |
|---|---:|
| Viajes procesados | 4,645,921 |
| Estaciones observadas / agrupadas | 370 / 368 |
| Características | 49 |
| K seleccionado | 2 |
| Silhouette Score | 0.127636 |
| Davies–Bouldin | 2.369866 |
| Tamaños de grupos | 177 y 191 |

Se evaluaron K=2 a K=8. Los gráficos locales se generan en `reports/figures/p3/`. Las etiquetas numéricas de K-Means pueden permutarse entre ejecuciones: comparar perfiles y métricas, no el identificador arbitrario del grupo.

### 9.1. Entrenamiento reproducido con virtualenv independiente

Además de las pruebas locales, se validó **P1** mediante:

```bash
.venv/bin/mlflow run . -e train \
  -P question=p1 \
  -P model=baseline \
  --env-manager virtualenv
```

**Resultado observado:** `Run (...) succeeded`, MAE/RMSE idénticos a la tabla de P1. P2 y P3 **no se han documentado como validados bajo `--env-manager virtualenv`**; se validaron bajo `local`.

### 9.2. Modelos candidatos P1 y P2

`mibici/modeling/p1_p2_candidates.py` entrena los candidatos supervisados con la misma partición temporal, las mismas estaciones y las mismas métricas de los baselines. `train` acepta los siguientes modelos:

| Pregunta | `model` | Algoritmo |
|---|---|---|
| P1 | `rf` | `RandomForestRegressor` |
| P1 | `hgb` | `HistGradientBoostingRegressor` |
| P2 | `logreg` | `StandardScaler` + `LogisticRegression(class_weight="balanced")` |
| P2 | `rf` | `RandomForestClassifier` |
| P2 | `hgb` | `HistGradientBoostingClassifier` |

Los hiperparámetros por defecto están en `HIPERPARAMETROS_CANDIDATOS` de `config.py`. Los parámetros `n_estimators`, `max_depth`, `learning_rate` y `max_iter` del MLproject los reemplazan cuando son distintos de 0.

Características (solo información anterior a la hora objetivo):

- P1: `hour_of_day`, `weekday`, `fin_semana`, rezagos de salidas a 1, 24 y 168 horas, y promedios de train por estación y por estación-día-hora.
- P2: las de calendario, `capacity`, rezagos de salidas, llegadas y presión a 1, 24 y 168 horas, y probabilidad histórica de train por estación y por estación-día-hora.
- Las primeras 168 horas de train se excluyen por falta de historia para los rezagos.
- En P2, el umbral de probabilidad se elige con el mejor F1 en validación, igual que en el baseline.

Comandos ejecutados (servidor activo y `MLFLOW_TRACKING_URI` exportada):

```bash
.venv/bin/mlflow run . -e train -P question=p1 -P model=hgb --env-manager local
.venv/bin/mlflow run . -e train -P question=p1 -P model=rf --env-manager local
.venv/bin/mlflow run . -e train -P question=p1 -P model=rf -P n_estimators=150 -P max_depth=8 --env-manager local
.venv/bin/mlflow run . -e train -P question=p2 -P model=logreg --env-manager local
.venv/bin/mlflow run . -e train -P question=p2 -P model=hgb --env-manager local
.venv/bin/mlflow run . -e train -P question=p2 -P model=rf --env-manager local
```

Cada corrida candidata registra:

- Modelo con `log_model`, firma e `input_example`.
- `data/manifiesto.json`, `data/csv_manifest.json` y `mlflow.log_input` de train, validación y prueba con la versión de datos en el nombre.
- `plots/`: real contra predicho y residuales (P1); matriz de confusión y curva precisión-recall (P2).
- `config/`: `MLproject`, `python_env.yaml`, `config.py` y `run_params.json`.
- `metrics/metrics.json` y las métricas `baseline_test_*` y `test_*_mejora_pct`, calculadas sobre el mismo conjunto de prueba.
- Etiquetas `team`, `question`, `model_family`, `stage_of_work=candidate`, `git_commit`, `data_version` y `dataset`.

Resultados en prueba (noviembre–diciembre 2024, versión de datos `7e1470ac0632`):

**P1 — Regresión**

| Modelo | Hiperparámetros | MAE val | MAE test | RMSE test | Mejora MAE | Mejora RMSE |
|---|---|---|---|---|---|---|
| Baseline estación-día-hora | — | 0.8544 | 0.8556 | 1.4635 | — | — |
| `hgb` | `max_iter=300`, `learning_rate=0.1` | 0.8396 | **0.8279** | **1.3930** | 3.23 % | 4.82 % |
| `rf` | `n_estimators=300`, `max_depth=12` | 0.8405 | 0.8297 | 1.4003 | 3.02 % | 4.32 % |
| `rf` | `n_estimators=150`, `max_depth=8` | 0.8450 | 0.8376 | 1.4142 | 2.10 % | 3.37 % |

**P2 — Clasificación de riesgo proxy**

| Modelo | Hiperparámetros | Umbral | F1 val | Precision test | Recall test | F1 test | AP test | Mejora F1 | Mejora AP |
|---|---|---|---|---|---|---|---|---|---|
| Baseline histórico | — | 0.21 | 0.3861 | 0.2851 | 0.4563 | 0.3509 | 0.3052 | — | — |
| `rf` | `n_estimators=300`, `max_depth=12` | 0.21 | 0.3977 | 0.3013 | 0.4801 | **0.3702** | **0.3510** | 5.49 % | 14.99 % |
| `hgb` | `max_iter=300`, `learning_rate=0.1` | 0.22 | 0.3977 | 0.3053 | 0.4590 | 0.3667 | 0.3425 | 4.49 % | 12.22 % |
| `logreg` | `C=1.0` | 0.82 | 0.3934 | 0.2945 | 0.4853 | 0.3666 | 0.3399 | 4.46 % | 11.36 % |

## 10. Ingeniería de características y control de fuga temporal

`mibici/features.py` centraliza funciones reutilizables:

| Función | Uso |
|---|---|
| `hourly_departures()` | Conteos de salida por estación/hora |
| `complete_hourly_panel()` | Horas sin viajes incluidas como ceros |
| `hourly_arrivals()` | Llegadas por estación/hora |
| `hourly_flow_balance()` | Llegadas menos salidas |
| `proxy_risk_labels()` | Etiqueta de P2 |
| `calendar_features()` | Hora, día de semana y fin de semana |
| `hourly_usage_profiles()` | Perfil horario de estaciones para P3 |
| `lag_features()` | Rezagos que utilizan historia pasada |
| `trip_quality_flags()` | Indicadores diagnósticos de calidad |

P1 y P2 deben usar información disponible **antes** de la hora objetivo; no utilizar como predictoras las salidas/llegadas de esa misma hora. Se conservan las particiones cronológicas de Semana 5. P3 describe estaciones sobre 2024 y no sigue una separación supervisada convencional.

## 11. Pruebas automatizadas y validación final

Ejecutar desde la raíz con el entorno de desarrollo:

```bash
.venv/bin/python -m pip check
.venv/bin/python -m pytest tests/ -v
.venv/bin/python -m compileall -q mibici
.venv/bin/python -m mibici.modeling.train --help
```

**Referencia de Semana 5:** `21 passed`. La repetición de pytest **en la rama de Semana 6 queda pendiente de constatar con su salida**, por lo que no debe declararse completada antes de ejecutarla.

## 12. Protección de artefactos locales y datos

**No subir a Git** los CSV/Parquet, `mlflow.db`, `mlruns/`, `mlartifacts/`, entornos virtuales, credenciales ni modelos serializados. El archivo `.gitignore` ya excluye los principales elementos; mantener esas reglas y confirmar:

```bash
git check-ignore -v mlflow.db
git check-ignore -v data/raw/datos_abiertos_2024_01.csv
git check-ignore -v models/test_model.pkl
git status --short
git diff --check
```

Para comprobar archivos ya rastreados usar `git ls-files`; las reglas de `.gitignore` no eliminan archivos previamente añadidos al índice. Revisar `reports/figures/p3/` y cualquier archivo de evidencia generado antes del commit. El manifiesto `data/manifiesto.json` es una excepción documental expresamente versionada y no contiene los CSV fuente.

## 13. Integración y Pull Request de Parte A

Con las pruebas exitosas y el README revisado:

```bash
git branch --show-current
git status --short
git diff --stat
git diff --check

git add MLproject python_env.yaml mibici/config.py mibici/modeling/train.py README.md
# Incluir .gitignore solo si efectivamente cambió y se revisó.
git diff --cached --stat
git commit -m 'feat(mlflow): implement reproducible MLflow project'
git push origin feature/A01797899-mlflow-project
```

Crear un Pull Request contra `main` desde https://github.com/A01797899/TC5061-Proyecto-Equipo15/pulls y solicitar revisión a otro integrante.

La selección y registro de modelos de P1/P2/P3, los candidatos adicionales y las nested runs deben integrarse después con los módulos de José Luis, Jorge y Óscar. Las nuevas variantes de `model`, sus hiperparámetros y los comandos finales de ganadores se añadirán al README una vez disponibles y comprobados. No etiquetar versiones `@champion` hasta que haya selección fundada y validación de carga en Model Registry.

## 14. Referencias y documentación

- [Repositorio de GitHub del Equipo 15](https://github.com/A01797899/TC5061-Proyecto-Equipo15)
- [MiBici — Datos abiertos](https://www.mibici.net/es/datos-abiertos/)
- [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/)
- [MLflow Projects](https://mlflow.org/docs/latest/ml/projects/)
- [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/)
- [MLflow Model Registry](https://mlflow.org/docs/latest/ml/model-registry/)
- [pyenv](https://github.com/pyenv/pyenv)
- Documentación interna en `references/`, pruebas en `tests/` y notebooks de análisis en `notebooks/`.

---

**Última validación documentada de la Parte A:** 23 de octubre de 2026. Este README describe resultados realmente observados y señala explícitamente las validaciones e integraciones pendientes.
