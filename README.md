# MiBici — Proyecto MLOps | Equipo 15 | Semanas 5 y 6

[![Cookiecutter Data Science](https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter)](https://cookiecutter-data-science.drivendata.org/)

**Repositorio:** https://github.com/A01797899/TC5061-Proyecto-Equipo15
**Curso:** TC5061 — Operaciones de Aprendizaje Automático
**Experimento:** `mibici-equipo15`
**Datos:** viajes de MiBici, enero–diciembre de 2024
**Versiones utilizadas en la prueba:** Python 3.12.15 y MLflow 3.16.1.

>Esta guía documenta la ejecución reproducible del proyecto MiBici para la Semana 6, desde la preparación del entorno y la ingesta hasta las búsquedas anidadas, la selección de modelos y su registro en MLflow Model Registry. Incluye las pruebas automatizadas, la validación de firmas y la inferencia mediante el alias champion.

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
├── MLproject                       # Entradas ingest y train de MLflow
├── python_env.yaml                 # Configuración del entorno Python
├── requirements.txt                # Dependencias del proyecto
├── pyproject.toml                   # Configuración y herramientas de pruebas
├── README.md                       # Documentación y ejecución reproducible
├── .gitignore                      # Exclusiones de Git
│
├── data/
│   ├── raw/                        # Archivos CSV originales de MiBici 2024
│   ├── processed/                  # Datos procesados en formato Parquet
│   └── manifiesto.json             # Metadatos y hashes SHA-256
│
├── mibici/
│   ├── config.py                   # Configuración, rutas y tracking URI
│   ├── dataset.py                  # Descarga e ingesta de datos
│   ├── features.py                 # Ingeniería de características
│   │
│   └── modeling/
│       ├── train.py                # Orquestador parametrizado P1/P2/P3
│       ├── p1_baseline.py          # Baseline de predicción de demanda
│       ├── p2_baseline.py          # Baseline de clasificación de riesgo
│       ├── p3_baseline.py          # Baseline de agrupamiento de estaciones
│       ├── p1_p2_candidates.py     # Entrenamiento de candidatos P1/P2
│       ├── p1_p2_search.py         # Búsquedas anidadas P1/P2
│       ├── p3_search.py            # Búsquedas y candidatos P3
│       ├── search.py               # Funciones auxiliares de búsqueda
│       ├── registry.py             # Operaciones de Model Registry
│       └── validate_registry.py    # Validación de modelos registrados
│
├── tests/
│   ├── test_dataset.py             # Pruebas de ingesta
│   ├── test_features.py            # Pruebas de ingeniería de características
│   ├── test_p1_p2_search.py        # Pruebas de nested runs P1/P2
│   └── test_registry.py            # Pruebas de Model Registry
│
├── notebooks/                      # Análisis exploratorio
├── references/                     # Referencias y documentación
├── reports/
│   ├── figures/                    # Figuras y resultados
│   └── evidence/                   # Capturas para el reporte
│
└── models/                          # Artefactos locales de modelos
```

El archivo `MLproject` define los puntos de entrada `ingest` y `train`. El primero ejecuta la ingesta de datos mediante `mibici.dataset`, mientras que el segundo utiliza `mibici.modeling.train` como orquestador parametrizado de los experimentos.

Para P1 y P2, el proyecto permite ejecutar baselines, entrenar modelos candidatos y realizar búsquedas de hiperparámetros utilizando corridas anidadas (*nested runs*) en MLflow. Para P3 se implementan búsquedas de configuraciones de K-Means, agrupamiento jerárquico y Gaussian Mixture Model.

Los módulos `registry.py` y `validate_registry.py` proporcionan funcionalidades relacionadas con el registro y la validación de modelos en MLflow Model Registry.

Los resultados se registran en el experimento `mibici-equipo15`, permitiendo consultar métricas, parámetros, artefactos, corridas padre e hijas y modelos registrados.

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

Abrir una terminal desde la raíz del proyecto y activar el entorno virtual:

```bash
source .venv/bin/activate
```

Iniciar el servidor de MLflow:

```bash
.venv/bin/mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --serve-artifacts \
  --artifacts-destination ./mlflow-artifacts \
  --host 127.0.0.1 \
  --port 5000
```

Este comando inicia un servidor MLflow local que utiliza:

- **Backend Store:** `mlflow.db`, una base de datos SQLite para almacenar experimentos, corridas, métricas, parámetros y metadatos del Model Registry.
- **Artifact Store:** `mlflow-artifacts/`, directorio para almacenar modelos y otros artefactos generados durante las ejecuciones.
- **Host:** `127.0.0.1`, accesible únicamente desde la computadora local.
- **Puerto:** `5000`, utilizado por el servidor de MLflow.

La base de datos y los artefactos se mantienen dentro del directorio del proyecto. Si no existen, MLflow los inicializará al comenzar a trabajar.

**Importante:** ejecutar el servidor desde la raíz del repositorio para conservar las rutas relativas y mantener abierta esta terminal durante las ejecuciones.

Acceder a la interfaz gráfica desde:

http://127.0.0.1:5000

### 7.3. Configurar cliente MLflow (terminal B)

Abrir una segunda terminal desde la raíz del mismo proyecto.

Activar el entorno virtual:

```bash
source .venv/bin/activate
```

Configurar la dirección del servidor MLflow:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

Verificar la configuración:

```bash
echo "$MLFLOW_TRACKING_URI"

.venv/bin/python -c "import mlflow; print(mlflow.get_tracking_uri())"

curl -I http://127.0.0.1:5000
```

**Resultado esperado:**

```text
http://127.0.0.1:5000
http://127.0.0.1:5000
HTTP/1.1 200 OK
```

Esta configuración permite que los comandos de MLflow Projects y los scripts de Python registren sus experimentos en el servidor local.

Para ejecutar las actividades de Semana 6 se utilizará `--env-manager virtualenv`, de manera que MLflow gestione un entorno aislado conforme a la configuración del proyecto.

Mantener la terminal A ejecutando el servidor mientras se realizan los experimentos desde la terminal B.

### 7.4. Comprobar experimento y corridas

Orden de ejecución: este script se utiliza después de completar las búsquedas de modelos documentadas en el apartado 9.3. No debe ejecutarse inmediatamente después de iniciar el servidor MLflow en una instalación nueva.

Después de ejecutar la ingesta, los baselines y las búsquedas de modelos, verificar que los resultados estén registrados correctamente en el servidor MLflow.

Desde la terminal B, ejecutar:

```bash
.venv/bin/python - <<'PY'
from mlflow import MlflowClient

client = MlflowClient()

experiment = client.get_experiment_by_name("mibici-equipo15")

if experiment is None:
    raise SystemExit(
        "No se encontró el experimento mibici-equipo15. "
        "Ejecutar primero los entrenamientos."
    )

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    max_results=5000,
)

print("=" * 65)
print("VALIDACION DE EXPERIMENTOS Y CORRIDAS - MIBICI")
print("=" * 65)

print("Experimento:", experiment.name)
print("Experiment ID:", experiment.experiment_id)
print("Total de corridas:", len(runs))

for question in ("P1", "P2"):
    parent_name = f"{question}_search_candidates"

    parents = [
        r for r in runs
        if r.data.tags.get("mlflow.runName") == parent_name
    ]

    assert parents, f"No se encontró {parent_name}"

    parent = max(parents, key=lambda r: r.info.start_time)

    children = [
        r for r in runs
        if r.data.tags.get("mlflow.parentRunId")
        == parent.info.run_id
    ]

    print(f"\n=== {question} ===")
    print("Padre:", parent_name)
    print("Estado:", parent.info.status)
    print("Corridas hijas:", len(children))

    for child in children:
        print(
            child.data.tags.get("mlflow.runName"),
            "|", child.info.status,
        )

    assert parent.info.status == "FINISHED"
    assert len(children) == 3
    assert all(c.info.status == "FINISHED" for c in children)

print("\nVALIDACION DE NESTED RUNS P1/P2 COMPLETADA")
PY
```

**Resultado esperado:**

- Experimento `mibici-equipo15` disponible.
- Corrida padre `P1_search_candidates` con tres corridas hijas.
- Corrida padre `P2_search_candidates` con tres corridas hijas.
- Todas las corridas verificadas en estado `FINISHED`.

Las búsquedas de P3 también generan corridas anidadas para K-Means, agrupamiento jerárquico y GMM. Su verificación se documenta en el apartado correspondiente a la búsqueda de hiperparámetros de P3.

Esta comprobación debe realizarse **después de ejecutar los entrenamientos y las búsquedas**, ya que en una instalación nueva inicialmente no existirán corridas registradas.

Los resultados pueden consultarse visualmente desde la interfaz de MLflow en:

http://127.0.0.1:5000

### 7.5. Validar las corridas anidadas de P3

Orden de ejecución: este script se utiliza después de completar las búsquedas de modelos documentadas en el apartado 9.3. No debe ejecutarse inmediatamente después de iniciar el servidor MLflow en una instalación nueva.

Para P3 (agrupamiento de estaciones), MLflow registra las búsquedas de hiperparámetros de tres familias de algoritmos:

- **K-Means + PCA:** 21 combinaciones.
- **Agrupamiento jerárquico:** 21 combinaciones.
- **Gaussian Mixture Model (GMM) + PCA:** 21 combinaciones.

En total se ejecutan **63 combinaciones**, organizadas en tres corridas padre y sus respectivas corridas hijas (*nested runs*).

Después de ejecutar la búsqueda de P3, verificar la estructura de las corridas mediante la API de MLflow:

```bash
.venv/bin/python - <<'PY'
from mlflow import MlflowClient

client = MlflowClient()

experiment = client.get_experiment_by_name("mibici-equipo15")

assert experiment is not None, (
    "No se encontró el experimento mibici-equipo15"
)

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    max_results=5000,
)

families = {
    "K-Means + PCA": "P3_search_kmeans_pca",
    "Jerárquico": "P3_search_jerarquico",
    "GMM + PCA": "P3_search_gmm_pca",
}

print("=" * 70)
print("VALIDACION DE NESTED RUNS P3 - MIBICI EQUIPO 15")
print("=" * 70)

total_children = 0

for family, parent_name in families.items():

    parents = [
        run for run in runs
        if run.data.tags.get("mlflow.runName") == parent_name
    ]

    assert parents, (
        f"No se encontró la corrida padre: {parent_name}"
    )

    parent = max(
        parents,
        key=lambda run: run.info.start_time
    )

    children = [
        run for run in runs
        if run.data.tags.get("mlflow.parentRunId")
        == parent.info.run_id
    ]

    print(f"\n=== {family} ===")
    print("Padre:", parent_name)
    print("Run ID:", parent.info.run_id)
    print("Estado:", parent.info.status)
    print("Corridas hijas:", len(children))

    for child in children:
        print(
            child.data.tags.get("mlflow.runName"),
            "|", child.info.status,
        )

    assert parent.info.status == "FINISHED", (
        f"{family}: corrida padre no finalizada"
    )

    assert len(children) == 21, (
        f"{family}: se esperaban 21 corridas hijas"
    )

    assert all(
        child.info.status == "FINISHED"
        for child in children
    ), f"{family}: existen corridas hijas no finalizadas"

    total_children += len(children)

print("\n" + "=" * 70)
print("RESUMEN DE VALIDACION P3")
print("=" * 70)

print("Familias verificadas:", len(families))
print("Total de corridas hijas:", total_children)

assert total_children == 63, (
    "El total de corridas hijas debe ser 63"
)

print("\nVALIDACION DE NESTED RUNS P3 COMPLETADA")
PY
```

**Resultado esperado:**

| Familia | Corridas padre | Corridas hijas | Estado esperado |
|---|---:|---:|---|
| K-Means + PCA | 1 | 21 | FINISHED |
| Jerárquico | 1 | 21 | FINISHED |
| GMM + PCA | 1 | 21 | FINISHED |
| **Total** | **3** | **63** | **FINISHED** |

Al terminar correctamente, el script mostrará:

```text
RESUMEN DE VALIDACION P3
======================================================================
Familias verificadas: 3
Total de corridas hijas: 63

VALIDACION DE NESTED RUNS P3 COMPLETADA
```

**Nota:** esta comprobación verifica la ejecución y la organización de las búsquedas anidadas, no la calidad de los agrupamientos. La comparación de Silhouette, Davies–Bouldin y estabilidad se documenta posteriormente en el apartado de selección del modelo P3.

El script consulta únicamente el servidor MLflow; no modifica corridas, modelos ni artefactos.

Las evidencias también pueden visualizarse desde http://127.0.0.1:5000, entrando al experimento `mibici-equipo15` y desplegando las tres corridas padre correspondientes a P3.


## 8. Ejecutar la ingesta con MLflow Project

Con el servidor MLflow activo en la terminal A y la variable
`MLFLOW_TRACKING_URI` configurada en la terminal B, ejecutar
la ingesta inicial desde la raíz del repositorio:

```bash
.venv/bin/mlflow run . -e ingest \
  --env-manager virtualenv
```

El parámetro `--env-manager virtualenv` permite que MLflow gestione
un entorno Python aislado utilizando las dependencias declaradas
en el proyecto.

Durante la ejecución, el proceso:

1. Descarga o reutiliza los 12 archivos CSV mensuales de MiBici 2024.
2. Utiliza el archivo de nomenclatura para identificar las estaciones.
3. Realiza la limpieza y validación de los registros.
4. Genera los archivos procesados en formato Parquet.
5. Actualiza el manifiesto de datos con sus hashes SHA-256.
6. Registra la ejecución en MLflow.

### 8.1. Resultados de la ingesta

En la reproducción realizada el 25 de octubre de 2026,
se obtuvieron los siguientes resultados:

| Indicador | Resultado |
|---|---:|
| Periodo procesado | Enero–diciembre de 2024 |
| Archivos CSV | 12 |
| Viajes limpios | 4,645,921 |
| Registros descartados | 0 |
| Duplicados | 0 |
| Versión de datos | `7e1470ac0632` |
| Estado MLflow Project | `FINISHED` |

El resultado esperado en la terminal es:

```text
Run (...) succeeded
```

### 8.2. Verificar los archivos generados

Una vez finalizada la ingesta, comprobar que se generaron
los archivos procesados y el manifiesto de datos:

```bash
ls -lh data/manifiesto.json
ls -lh data/processed/
```

Consultar la versión de datos:

```bash
.venv/bin/python -c \
  'from mibici.dataset import version_datos; print(version_datos())'
```

Para reproducir exactamente los resultados documentados, la
versión obtenida debe ser:

```text
7e1470ac0632
```

Si la versión es diferente, los datos de origen pueden haber
cambiado y las métricas posteriores podrían variar.

**Importante:** la ingesta solamente necesita ejecutarse una vez
al preparar los datos del proyecto. Los entrenamientos de P1,
P2 y P3, las búsquedas anidadas y las operaciones de Model Registry
utilizarán los datos procesados existentes.

Después de validar la ingesta, continuar con el punto 9:
ejecución de baselines y búsquedas de modelos.

## 9. Ejecución integral desde un clon limpio

**Antes de comenzar:** seguir las secciones 1–8 una sola vez, en un clon reciente de `main` (con el PR #18 ya integrado). Abrir el servidor MLflow de la sección 7 en terminal A y exportar `MLFLOW_TRACKING_URI=http://127.0.0.1:5000` en terminal B. Usar `.venv` y Python 3.12.15, siempre desde la raíz del repositorio. Los comandos de este apartado son para macOS/Linux. La explicación histórica de la Parte A en las secciones 3 y 6 queda conservada, pero para ejecutar las búsquedas es imprescindible la versión integrada de `main` que contenga `p1_p2_search.py`.

### 9.1. Ingesta

La sección 8 documenta la ingesta. En una nueva instalación sin datos procesados, ejecutarla una sola vez (no repetirla si ya terminó correctamente en el mismo clon):

```bash
.venv/bin/mlflow run . -e ingest --env-manager virtualenv
```

En la ejecución de referencia se procesaron los doce CSV de 2024, con 4,645,921 viajes limpios y versión de datos `7e1470ac0632`. Conservar `data/manifiesto.json` para verificar la versión en el reporte.

### 9.2. Entrenar los baselines

```bash
.venv/bin/mlflow run . -e train -P question=p1 -P model=baseline --env-manager virtualenv
.venv/bin/mlflow run . -e train -P question=p2 -P model=baseline --env-manager virtualenv
.venv/bin/mlflow run . -e train -P question=p3 -P model=baseline --env-manager virtualenv
```

En P1 y P2 se respetan las particiones temporales: entrenamiento enero–agosto de 2024, validación septiembre–octubre y prueba noviembre–diciembre. P3 utiliza perfiles anuales de estaciones.

| Pregunta | Baseline de referencia | Resultado |
|---|---|---:|
| P1 | Media por estación, día y hora | MAE test = 0.8556 |
| P2 | Riesgo proxy con probabilidad histórica | Average Precision test = 0.3052 |
| P3 | K-Means, K=2 | Silhouette = 0.127636 |

### 9.3. Ejecutar las búsquedas de candidatos con nested runs

```bash
.venv/bin/mlflow run . -e train -P question=p1 -P model=search --env-manager virtualenv
.venv/bin/mlflow run . -e train -P question=p2 -P model=search --env-manager virtualenv
.venv/bin/mlflow run . -e train -P question=p3 -P model=search --env-manager virtualenv
```

P1 y P2 generan **una corrida padre y tres hijas por pregunta**. P3 registra **tres familias con 21 corridas hijas cada una (63 combinaciones)**. En la ejecución documentada 34 combinaciones de P3 no fueron admisibles porque generaban grupos menores a 5 % de las estaciones.

### 9.4. Verificar las relaciones padre/hijo

```bash
.venv/bin/python - <<'PY'
from mlflow import MlflowClient

client = MlflowClient()
exp = client.get_experiment_by_name('mibici-equipo15')
assert exp is not None
runs = client.search_runs([exp.experiment_id], max_results=5000)
for parent_name in ('P1_search_candidates', 'P2_search_candidates'):
    parents = [r for r in runs if r.data.tags.get('mlflow.runName') == parent_name]
    assert parents, f'No se encontró {parent_name}'
    parent = max(parents, key=lambda r: r.info.start_time)
    children = [r for r in runs if r.data.tags.get('mlflow.parentRunId') == parent.info.run_id]
    assert parent.info.status == 'FINISHED'
    assert len(children) == 3
    assert all(r.info.status == 'FINISHED' for r in children)
    print(parent_name, '| padre FINISHED | hijas:', len(children))
print('Validación P1/P2 completada; verificar en la interfaz los tres padres y 63 hijos de P3.')
PY
```

Para P3, abrir **Experiments → mibici-equipo15 → Runs**, localizar las tres búsquedas de las familias K-Means, jerárquica y GMM, desplegar sus hijas y conservar una captura. El criterio es una corrida padre por búsqueda y una hija por configuración.

## 10. Comparación de candidatos y selección justificada

**Los modelos se seleccionan con métricas de validación; las métricas de test solo cuantifican el resultado una vez elegido el candidato.** La tabla siguiente reproduce los resultados obtenidos el 25 de octubre de 2026 y sirve de evidencia y criterio de contraste para una nueva corrida.

### 10.1. P1 — Demanda (minimizar MAE de validación)

| Candidato | Hiperparámetros | `val_mae` ↓ |
|---|---|---:|
| Random Forest | 300 árboles, profundidad 12 | 0.840485 |
| **HistGradientBoosting** | **300 iteraciones, learning rate 0.1** | **0.839559** |
| HistGradientBoosting | 200 iteraciones, learning rate 0.05 | 0.841175 |

Se eligió **HistGradientBoostingRegressor** por obtener el menor `val_mae`. En test alcanzó MAE ≈ **0.8279**, contra **0.8556** del baseline (≈ **3.23 %** de mejora). Nombre de la corrida: `P1_candidate_hgb_max_iter=300_learning_rate=0.1_max_depth=None`.

### 10.2. P2 — Riesgo proxy (maximizar Average Precision de validación)

| Candidato | Hiperparámetros | `val_average_precision` ↑ |
|---|---|---:|
| Logistic Regression | C=1.0, 1000 iteraciones | 0.364140 |
| **Random Forest** | **300 árboles, profundidad 12** | **0.376480** |
| HistGradientBoosting | 300 iteraciones, learning rate 0.1 | 0.370612 |

Se eligió **RandomForestClassifier** por lograr la mejor Average Precision de validación, pertinente en clasificación con positivos poco frecuentes. En test alcanzó Average Precision ≈ **0.3510**, contra **0.3052** del baseline (≈ **14.99 %** de mejora), y F1 ≈ **0.3702**. Corrida: `P2_candidate_rf_n_estimators=300_max_depth=12`.

### 10.3. P3 — Grupos de estaciones (admisibilidad y estabilidad)

| Familia | Configuración | Silhouette ↑ | Davies–Bouldin ↓ |
|---|---|---:|---:|
| **K-Means + PCA** | **k=2, 5 componentes** | 0.127636 | **2.369866** |
| Jerárquico | k=2, Ward | 0.106341 | 2.613176 |
| GMM + PCA | k=2, 10 componentes, full | **0.132514** | 3.646119 |

**GMM alcanzó un Silhouette puntual mayor que K-Means.** No se debe afirmar lo contrario. Se seleccionó **K-Means + PCA** considerando el menor Davies–Bouldin, solo 5 componentes y la estabilidad reportada en cinco semillas: GMM promedió Silhouette **0.126419 ± 0.012190** y modificó las agrupaciones, mientras K-Means mantuvo desviación cero y reprodujo la partición de referencia (ARI=1.0, grupos de 177 y 191 estaciones). Esto permite interpretar de modo más consistente los perfiles horarios. Corrida seleccionada: `P3_candidate_kmeans_pca`. Conservar la comparación multisemilla y perfiles como evidencia en el PDF.

## 11. Pruebas automatizadas

```bash
.venv/bin/python -m pip check
.venv/bin/python -m compileall -q mibici
.venv/bin/python -m pytest tests/ -v
```

**Resultado obtenido en la reproducción:** `40 passed` (incluye siete pruebas de las búsquedas anidadas de P1 y P2). Una nueva ejecución debe terminar satisfactoriamente antes de registrar modelos.

### 11.1. Validación de datos con DeepChecks

`mibici/data_validation.py` valida la calidad y la integridad de los datos de P1 y P2. Construye los paneles con `datos_p1()` y `datos_p2()` de `p1_p2_candidates.py`, de modo que usa el mismo procesamiento, las mismas características y la misma partición cronológica que los modelos candidatos; no repite la limpieza de `dataset.py`.

**Instalación.** DeepChecks es una dependencia de desarrollo: está en `requirements-dev.txt` y no modifica `requirements.txt` ni `python_env.yaml`, por lo que el entorno de MLflow no cambia.

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
```

**Ejecución.**

```bash
# Validación de P1 y P2 con los datos reales (requiere data/raw)
.venv/bin/python -m mibici.data_validation

# Solo una pregunta o con otra muestra
.venv/bin/python -m mibici.data_validation --preguntas p2 --muestra 100000

# Pruebas del módulo (datos sintéticos, no requieren data/raw)
.venv/bin/python -m pytest tests/test_data_validation.py -v
```

| Opción | Valor por defecto | Uso |
|---|---|---|
| `--preguntas` | `p1 p2` | Preguntas a validar |
| `--muestra` | `200000` | Filas por split para DeepChecks, con `RANDOM_SEED` |
| `--salida` | `reports/week7/deepchecks` | Carpeta de reportes |
| `--estricto` | desactivado | Termina con código 1 si algún check queda como problema |

**Qué se valida.** Antes de DeepChecks, el módulo rechaza paneles vacíos, sin columnas requeridas, sin splits `train` y `test`, o que usen como predictor una columna de la hora objetivo (`pressure`, `salidas`, `llegadas`). También cuenta sobre el panel completo lo que el muestreo no garantiza: llave estación-hora duplicada, filas repetidas, nulos por columna y rango de fechas por split.

| Suite | Se ejecuta sobre | Checks |
|---|---|---|
| Integridad | train y test por separado | Tipos mezclados, nulos (`MixedNulls`, `PercentOfNulls` ≤ 5 %), duplicados, columnas constantes, caracteres especiales, correlación característica-etiqueta (fuga), correlación entre características; en P2 además desbalance de clases y etiquetas en conflicto |
| Train contra test | ambos | Drift de características y de etiqueta, fuga temporal por fecha, categorías y clases nuevas, comparación de tamaños, cambio de poder predictivo |

**Reportes.** En `reports/week7/deepchecks/` se generan, por pregunta, los HTML `{p}_integridad_train.html`, `{p}_integridad_test.html` y `{p}_train_test.html`, el resumen `{p}_resumen.json` y un `resumen.md` con todas las preguntas. Cada check queda clasificado como aprobado, advertencia o problema, con una recomendación para los dos últimos. Los HTML son estáticos: muestran todas las secciones sin clics y cargan una sola copia de plotly.js desde jsDelivr, por lo que pesan entre 40 y 120 KB y necesitan internet para dibujar las gráficas.

**Resultados** (versión de datos `7e1470ac0632`, 200,000 filas por split, semilla 42):

| Indicador | P1 | P2 |
|---|---|---|
| Filas train / val / test | 2,119,872 / 529,968 / 529,968 | 2,119,872 / 529,968 / 529,968 |
| Llave estación-hora duplicada | 0 | 0 |
| Filas de train con nulos en rezagos | 60,816 | 60,816 |
| Nulos en test | 0 | 0 |
| Correlación máxima entre características | 0.79 | 0.67 |
| Drift máximo de características | 0.03 (`weekday`) | 0.03 (`weekday`) |
| Drift de la etiqueta | 0.01 (Kolmogorov-Smirnov) | 0.01 (Cramér's V) |
| Fechas compartidas entre train y test | ninguna | ninguna |
| Checks aprobados / advertencias / problemas | 30 / 0 / 0 | 31 / 2 / 2 |

**Hallazgos.**

- **Nulos en los rezagos.** Solo están al inicio de train, donde todavía no hay historia: 362, 8,688 y 60,816 filas en los rezagos de 1, 24 y 168 h, que son 362 estaciones × 1, 24 y 168 h. Representan 2.87 % en el peor caso, por debajo del umbral de 5 %, y `run_candidate` los descarta con `dropna`. Test no tiene nulos.
- **Filas repetidas en características.** El conteo exacto encuentra 817,067 filas (38.5 %) en P1 y 657,852 (31.0 %) en P2 con las mismas características y etiqueta. No son registros duplicados, porque la llave estación-hora no se repite: son horas distintas con los mismos valores, como horas sin viajes. DeepChecks reporta 0.03 % y 0.34 % porque compara el registro completo, incluida la fecha.
- **Desbalance en P2 (advertencia).** La clase positiva es cerca de 5 % de la negativa. Es propio de la etiqueta proxy, y los candidatos lo manejan con `class_weight="balanced"`, F1 y Average Precision, y el umbral elegido en validación.
- **Etiquetas en conflicto en P2 (problema).** El 0.02 % de las filas tiene las mismas características y etiqueta distinta. Es esperable: la etiqueta depende de la presión de la hora objetivo, que a propósito no se usa como predictor.
- **Drift.** Las distribuciones casi no cambian entre train (enero–agosto) y test (noviembre–diciembre): ningún check de drift supera su umbral.

**Compatibilidad.** `deepchecks==0.19.1`, la última versión publicada (diciembre de 2024), es anterior a NumPy 2.5, pandas 3 y scikit-learn 1.9 del proyecto. `cargar_deepchecks()` aplica las correcciones antes de importarla:

| Problema | Corrección |
|---|---|
| Importa `pkg_resources`, que setuptools 81+ ya no incluye | `setuptools==80.10.2` en `requirements-dev.txt` |
| Usa `np.Inf` y `np.NaN`, eliminados en NumPy 2 | Alias antes de importar |
| Pide el scorer `max_error`, renombrado a `neg_max_error` | Alias en el registro de scorers |
| Su HTML interactivo no dibuja las gráficas con plotly 7, y su HTML estático incrusta plotly.js (~5 MB) en cada gráfica | `guardar_html()` genera un HTML estático con una sola copia de plotly.js desde CDN |
| La condición de `FeatureFeatureCorrelation` usa `stack()`, que en pandas 3 conserva NaN y marca todos los pares | Condición propia con `stack().dropna()` |
| La condición de `ClassImbalance` avisa cuando las clases están balanceadas | Condición propia que avisa si la minoritaria es menor al 10 % |

Se excluyen `OutlierSampleDetection`, `TrainTestSamplesMix` y `MultivariateDrift`, que fallan con pandas 3 y scikit-learn 1.9, e `IndexTrainTestLeakage`, que no aplica porque la misma estación aparece en train y test por diseño. Los motivos quedan registrados en cada `{p}_resumen.json`.

**Pruebas.** `tests/test_data_validation.py` agrega 25 pruebas sobre datos sintéticos: casos límite del panel, rechazo de columnas de la hora objetivo, conteos exactos, muestreo reproducible, las dos condiciones corregidas, detección de drift y que una segunda corrida reemplace los reportes. Sin DeepChecks instalado, 11 se omiten y las demás pasan. Con estas pruebas la suite completa pasa de 40 a **65 passed**.

**Evidencia** en `reports/week7/evidence/deepchecks/`:

| Archivo | Contenido |
|---|---|
| `01_terminal_p1.png` | Ejecución de `mibici.data_validation`, resultados de P1 |
| `02_terminal_p2.png` | Resultados de P2 y ubicación de los reportes |
| `03_html_integridad.png` | Reporte de integridad de P2 (train) |
| `04_html_train_test.png` | Reporte de drift train contra test de P1 |
| `05_pytest_data_validation.png` | `25 passed` de `tests/test_data_validation.py` |
| `06_pytest_suite_completa.png` | `65 passed` de la suite completa |

### 11.2. Pruebas de entrenamiento y reproducibilidad con PyTest — Semana 7

Como parte de la actividad **Construye: Tests + Chequeos de seguridad básicos integrados al repositorio**, incorporamos 35 casos de prueba automatizados para validar el entrenamiento, la evaluación y la reproducibilidad de modelos del proyecto MiBici. Las pruebas aprovechan las funciones reales de `mibici/modeling/p1_p2_candidates.py` y `mibici/modeling/train.py` y complementan las pruebas de ingesta, características, búsqueda, registro y DeepChecks ya integradas.

**Archivos de pruebas añadidos:**

- `tests/test_model_training.py`: **28 pruebas aprobadas**, que incluyen hiperparámetros, construcción de estimadores, entrenamiento con datos sintéticos, predicciones, métricas y selección de módulos de ejecución.
- `tests/test_model_reproducibility.py`: **7 pruebas aprobadas**, que comparan entrenamientos independientes de P1 y P2 con los mismos datos, parámetros y semilla fija.

#### 11.2.1. Funciones y casos evaluados

| Función o componente | Validación | Tipo de caso |
|---|---|---|
| `resolver_hiperparametros()` | Valores predeterminados, personalización y mantenimiento del valor por defecto cuando se recibe cero | Normal y límite |
| `resolver_hiperparametros()` | Identificador sin configuración | Error esperado |
| `crear_modelo()` | Estimadores `RandomForestRegressor`, `RandomForestClassifier`, `HistGradientBoostingRegressor`, `HistGradientBoostingClassifier` y `LogisticRegression` | Normal |
| `crear_modelo()` | Combinaciones de pregunta/modelo no admitidas | Error esperado |
| Entrenamiento de Random Forest | Ajuste y predicción para P1 (regresión) y P2 (clasificación), dimensiones y predicciones finitas | Normal |
| `mejor_umbral()` | Elección del umbral que maximiza F1 | Normal |
| `mejora_pct()` | Mejoras positivas, negativas, métricas de maximización y baseline cero | Normal y límite |
| `run_baseline()` | Selección de los módulos baseline P1, P2 y P3, argumentos y variable `MLFLOW_TRACKING_URI` | Normal con `Mock` |
| `run_search()` | Selección de módulos de búsqueda y rechazo de una pregunta inválida | Normal y error esperado |

Las pruebas de orquestación usan `unittest.mock` y `monkeypatch` para verificar las llamadas esperadas sin ejecutar subprocesos ni requerir un servidor de MLflow.

#### 11.2.2. Reproducibilidad del entrenamiento

Se utilizó `RANDOM_SEED = 42`, definido en `mibici/config.py`, junto con datos sintéticos deterministas y los mismos hiperparámetros. Para cada modelo se crean **dos estimadores independientes**, se entrenan con el mismo conjunto de entrenamiento y se comparan los resultados sobre el mismo conjunto de prueba.

| Problema | Modelo | Comparaciones |
|---|---|---|
| P1 — Demanda (regresión) | Random Forest | Predicciones, MAE y R² |
| P1 — Demanda (regresión) | HistGradientBoosting | Predicciones, MAE y R² |
| P2 — Riesgo proxy (clasificación) | Random Forest | Clases predichas, exactitud y probabilidades |
| P2 — Riesgo proxy (clasificación) | HistGradientBoosting | Clases predichas, exactitud y probabilidades |
| P2 — Riesgo proxy (clasificación) | LogisticRegression | Clases predichas, exactitud y probabilidades |

Además, se comprueba que la misma semilla genera los mismos datos sintéticos y que los estimadores utilizan la semilla configurada. Para predicciones y probabilidades continuas se usa `numpy.testing.assert_allclose` con tolerancias de `1e-10`, y para etiquetas discretas se usa igualdad exacta.

**Alcance:** estas pruebas verifican la reproducibilidad de los estimadores con datos pequeños y controlados; no son una ejecución integral del pipeline de ingesta, entrenamiento, seguimiento y registro de modelos en MLflow sobre todos los CSV de 2024.

#### 11.2.3. Comandos para ejecutar las pruebas

Ejecutar desde la raíz del repositorio, con las dependencias del proyecto instaladas:

```bash
# Pruebas de entrenamiento, parámetros, métricas y orquestación
.venv/bin/python -m pytest tests/test_model_training.py -v

# Pruebas de reproducibilidad de P1 y P2
.venv/bin/python -m pytest tests/test_model_reproducibility.py -v

# Toda la suite del repositorio, incluida la validación de datos con DeepChecks
.venv/bin/python -m pytest tests/ -v
```

Para guardar de nuevo las salidas textuales y visualizarlas simultáneamente en macOS/Linux:

```bash
mkdir -p reports/week7/evidence/training
set -o pipefail

.venv/bin/python -m pytest tests/test_model_training.py -v \
  | tee reports/week7/evidence/training/01_training_tests.txt

.venv/bin/python -m pytest tests/test_model_reproducibility.py -v \
  | tee reports/week7/evidence/training/02_reproducibility_tests.txt

.venv/bin/python -m pytest tests/ -v \
  | tee reports/week7/evidence/training/03_full_test_suite.txt
```

#### 11.2.4. Resultados obtenidos y evidencias

| Conjunto de pruebas | Resultado | Tiempo mostrado en la captura |
|---|---:|---:|
| Entrenamiento y evaluación | **28 passed** | 4.83 s |
| Reproducibilidad | **7 passed** | 5.09 s |
| Suite previa a esta contribución (incluido DeepChecks) | **65 passed** | 17.25 s |
| **Suite completa después de incorporar las 35 nuevas pruebas** | **100 passed** | **13.07 s** (última ejecución registrada) |

La suite completa finalizó sin pruebas fallidas ni omitidas en la ejecución registrada. Los tiempos son observaciones de las ejecuciones locales y pueden variar entre equipos.

Las evidencias se almacenan en `reports/week7/evidence/training/`, en formato PNG para revisión visual y TXT para consulta de la salida de PyTest:

| Archivo | Evidencia |
|---|---|
| `01_training_tests.png` / `01_training_tests.txt` | Pruebas de entrenamiento y evaluación |
| `02_reproducibility_tests.png` / `02_reproducibility_tests.txt` | Pruebas de reproducibilidad |
| `03_full_test_suite.png` / `03_full_test_suite.txt` | Ejecución integral de la suite |

**Evidencia 1 — Pruebas de entrenamiento (28 passed):**

![Pruebas de entrenamiento de MiBici: 28 aprobadas](reports/week7/evidence/training/01_training_tests.png)

**Evidencia 2 — Pruebas de reproducibilidad (7 passed):**

![Reproducibilidad de modelos MiBici: 7 aprobadas](reports/week7/evidence/training/02_reproducibility_tests.png)

**Evidencia 3 — Suite completa del proyecto (100 passed):**

![Suite PyTest completa de MiBici: 100 aprobadas](reports/week7/evidence/training/03_full_test_suite.png)

Estas pruebas aportan cobertura de funciones clave, casos normales y límite, y reproducibilidad de modelos, manteniendo compatibles las funcionalidades previamente validadas del repositorio.

### 11.3. Chequeos de seguridad — Semana 7

Revisión de credenciales, Bandit y pip-audit. Todo se ejecutó en el entorno virtual del repositorio, `TC5061-Proyecto-Equipo15/.venv` (Python 3.12.13). Bandit 1.9.4, pip-audit 2.10.1, MLflow 3.16.1, NumPy, pandas y scikit-learn quedaron en ese mismo `site-packages`. `.venv/` no se versiona.

**Pasos en la raíz del proyecto:**

```bash
# 1. Entorno del repositorio con Python 3.12 (pyproject.toml exige ~=3.12.0)
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

# 2. Herramientas de seguridad, en el mismo .venv
.venv/bin/python -m pip install -r requirements-security.txt
.venv/bin/python -c 'import sys; print(sys.prefix)'
.venv/bin/python -m bandit --version
.venv/bin/python -m pip_audit --version
```

`sys.prefix` debe imprimir la ruta `.../TC5061-Proyecto-Equipo15/.venv`. `requirements-security.txt` fija `bandit==1.9.4` y `pip-audit==2.10.1`. Esas dos herramientas quedan fuera del entorno de MLflow (`requirements.txt` y `python_env.yaml`) y de DeepChecks (`requirements-dev.txt`).

**Credenciales.** `.gitignore` ya excluía `.env`, los entornos virtuales, `mlruns/`, `mlartifacts/` y `mlflow.db`. Esta revisión añade `*.pem`, `*.key`, `*.p12` y `*.pfx`. Git no sigue ninguno de esos archivos. En el código Python no hay contraseñas, secretos, API keys ni tokens. La única variable de entorno es `MLFLOW_TRACKING_URI`: `mibici/config.py` la lee con `os.getenv` después de `load_dotenv()`, y el valor por defecto es `http://127.0.0.1:5000`. `.env.example` documenta ese valor; un `.env` real no se versiona. El diagnóstico está en `reports/week7/security/diagnostico.md`.

**Bandit** revisa `mibici/` (no existe una carpeta `src/`):

```bash
mkdir -p reports/week7/security reports/week7/evidence/security
.venv/bin/python -m bandit -r mibici/
.venv/bin/python -m bandit -r mibici/ -f json -o reports/week7/security/bandit_report.json
```

Un código de salida distinto de cero significa que hubo hallazgos; el JSON igual se escribe. La primera corrida (2 870 líneas) reportó **19 hallazgos**: 1 de severidad media y 18 bajos, todos con confianza alta.

| Id | Dónde | Decisión |
|---|---|---|
| B310, media | `urlopen` en `mibici/dataset.py` | La descarga acepta solo `http`, `https` y `file`. Las URL de MiBici son `https`. `file` queda para las pruebas que leen un CSV local. Un esquema distinto termina con `ValueError`. |
| B404, B603, B607, bajas | `subprocess` en entrenamiento, búsqueda, registro y baselines | Las llamadas usan una lista de argumentos y no pasan por el shell: `git rev-parse HEAD`, o el intérprete del proyecto y un módulo ya definido. Cada caso lleva `# nosec` y un comentario con el motivo. |

La corrida posterior a esas anotaciones está en `reports/week7/security/bandit_report_despues.json`: **0 hallazgos**. El informe inicial se conserva para que los 19 casos sigan visibles.

**pip-audit** lee los requisitos sin modificarlos. `-e .` hace que el paquete local `mibici` 0.0.1 se omita (no está en PyPI); el resto de cada archivo sí se revisa.

```bash
.venv/bin/python -m pip_audit -r requirements.txt
.venv/bin/python -m pip_audit -r requirements.txt -f json \
  -o reports/week7/security/pip_audit_requirements.json

.venv/bin/python -m pip_audit -r requirements-dev.txt
.venv/bin/python -m pip_audit -r requirements-dev.txt -f json \
  -o reports/week7/security/pip_audit_requirements_dev.json
```

| Paquete | Versión | Avisos | Corrección publicada | Decisión |
|---|---|---|---|---|
| mlflow | 3.16.1 | CVE-2026-53995, CVE-2026-115886, CVE-2026-116691 | 3.17.2, 3.17.0 y 3.17.0 | El pin permanece en 3.16.1, la versión de la reproducción de las semanas 6 y 7 |
| setuptools | 80.10.2 | PYSEC-2026-3447 (CVE-2026-59890) y GHSA-xmj3-9gf7-h52w | 83.0.0 y 84.1.0 | El pin permanece: DeepChecks 0.19.1 necesita `pkg_resources`, ausente desde setuptools 81 |

Los tres avisos de MLflow: recorrido de rutas en FileStore al crear una traza (`CVE-2026-53995`), instanciación sin autenticación en `scorer/invoke` si están instalados paquetes opcionales de TruLens que este proyecto no declara (`CVE-2026-115886`) y filtración de una API key de CrewAI si se usa `mlflow.crewai.autolog()`, que este código no llama (`CVE-2026-116691`). El servidor de las instrucciones escucha en `http://127.0.0.1:5000`. Los dos avisos de setuptools aplican al armado de un sdist en APFS/HFS+ y a la instalación legacy de wheels en Windows; esta corrida fue en Linux y setuptools es solo dependencia de desarrollo. El detalle, con alias GHSA, está en `reports/week7/security/hallazgos.md`.

**Evidencia** en `reports/week7/evidence/security/`:

| Archivo | Contenido |
|---|---|
| `01_bandit.txt` | Primera corrida de Bandit: 19 hallazgos |
| `02_bandit_despues.txt` | Bandit después de las anotaciones: 0 hallazgos |
| `03_pip_audit_requirements.txt` | Tres avisos de MLflow en `requirements.txt` |
| `04_pip_audit_requirements_dev.txt` | Esos tres avisos más setuptools |

## 12. Registrar modelos y validar Model Registry

### 12.1. Registro automatizado

Una vez elegidos los ganadores, registrar los modelos desde la corrida de origen. **Los Logged Model IDs son diferentes en cada backend**: el script los localiza a través de la última corrida `FINISHED` de cada ganador y evita registrar otra versión cuando ya existe la misma fuente. Si un nombre ya tiene versiones de otra fuente, requiere revisión antes de volver a registrar. También añade **la descripción exigida por la rúbrica**.

```bash
.venv/bin/python - <<'PY'
import mlflow
from mlflow import MlflowClient

assert mlflow.get_tracking_uri() == 'http://127.0.0.1:5000'
assert mlflow.get_registry_uri() == mlflow.get_tracking_uri()
client = MlflowClient()
exp = client.get_experiment_by_name('mibici-equipo15')
assert exp is not None
runs = client.search_runs([exp.experiment_id], max_results=5000)
chosen = {
    'P1': ('P1_candidate_hgb_max_iter=300_learning_rate=0.1_max_depth=None',
           'mibici-equipo15-p1-demanda',
           'Predicción de demanda por estación y hora; elegido por MAE de validación.'),
    'P2': ('P2_candidate_rf_n_estimators=300_max_depth=12',
           'mibici-equipo15-p2-riesgo',
           'Clasificación de riesgo proxy; elegido por Average Precision de validación.'),
    'P3': ('P3_candidate_kmeans_pca',
           'mibici-equipo15-p3-grupos',
           'Perfiles horarios agrupados con K-Means y PCA; elección por estabilidad e interpretación.'),
}
for question, (run_name, name, description) in chosen.items():
    found = [r for r in runs if r.info.status == 'FINISHED'
             and r.data.tags.get('mlflow.runName') == run_name]
    assert found, f'{question}: no se encontró una corrida candidata FINISHED'
    run = max(found, key=lambda r: r.info.start_time)
    logged = client.search_logged_models(
        experiment_ids=[exp.experiment_id],
        filter_string=f"source_run_id = '{run.info.run_id}'",
    )
    assert len(logged) == 1, f'{question}: se esperaba 1 Logged Model; hay {len(logged)}'
    source = f'models:/{logged[0].model_id}'
    existing = client.search_model_versions(f"name = '{name}'")
    same = [v for v in existing if v.source == source]
    if same:
        version = max(same, key=lambda v: int(v.version))
    else:
        assert not existing, f'{name}: ya existen otras versiones; revisar antes de registrar'
        version = mlflow.register_model(model_uri=source, name=name)
    assert version.status == 'READY'
    client.update_registered_model(name=name, description=description)
    print(question, name, 'versión', version.version, '| READY |', source)
print('REGISTRO COMPLETADO')
PY
```

En la ejecución de referencia se registraron estos tres modelos, cada uno con **versión 1 y estado `READY`**:

| Pregunta | Nombre registrado | Logged Model ID de referencia |
|---|---|---|
| P1 | `mibici-equipo15-p1-demanda` | `m-04774b8f25ff468b933a18756c24b068` |
| P2 | `mibici-equipo15-p2-riesgo` | `m-165c943b3eec4915aabc8e801feb9979` |
| P3 | `mibici-equipo15-p3-grupos` | `m-041658517cae46178fa30b582bd44b22` |

Estos identificadores son **solo históricos**, no deben copiarse para una nueva ejecución.

### 12.2. Validación de versiones, estado y firmas

```bash
.venv/bin/python - <<'PY'
from mlflow import MlflowClient
from mlflow.models import Model
client = MlflowClient()
names = {'P1': 'mibici-equipo15-p1-demanda',
         'P2': 'mibici-equipo15-p2-riesgo',
         'P3': 'mibici-equipo15-p3-grupos'}
for question, name in names.items():
    versions = client.search_model_versions(f"name = '{name}'")
    assert versions, name
    version = max(versions, key=lambda v: int(v.version))
    assert version.status == 'READY'
    model = Model.load(f'models:/{name}/{version.version}')
    assert model.signature and model.signature.inputs and model.signature.outputs
    columns = [c.name for c in model.signature.inputs.inputs]
    print(question, name, 'v'+version.version, 'READY', len(columns), 'columnas')
    print('Source:', version.source, '| Firma:', model.signature)
    print('Validación: OK')
print('VALIDACION COMPLETADA: 3 MODELOS REGISTRADOS')
PY
```

Firmas observadas: **P1 = 8 entradas**, **P2 = 15 entradas**, **P3 = 49 entradas**. La revisión de firma no hace predicciones. No se utiliza `mlflow.models.load_serving_example()` (no disponible en la instalación usada).

### 12.3. Confirmar los artefactos de los candidatos

Desde MLflow, entrar en **Experiments → mibici-equipo15 → Runs** y comprobar para cada candidato: modelo con firma y ejemplo de entrada, manifiesto de versión de datos, artefactos `config/`, `data/` y `plots/`, etiquetas de pregunta/familia/commit y corrida fuente. Esto constituye evidencia solicitada en la rúbrica y no requiere ejecutar nuevas predicciones reales.

## 13. Promoción con alias `champion` y validación final

### 13.1. Asignar y verificar los alias

Se utiliza `champion` en lugar de las antiguas etapas de Model Registry, pues resuelve un modelo por nombre estable y permite cambiar de versión en el futuro sin modificar el código cliente.

```bash
.venv/bin/python - <<'PY'
from mlflow import MlflowClient
client = MlflowClient()
names = {'P1': 'mibici-equipo15-p1-demanda',
         'P2': 'mibici-equipo15-p2-riesgo',
         'P3': 'mibici-equipo15-p3-grupos'}
for question, name in names.items():
    versions = client.search_model_versions(f"name = '{name}'")
    assert versions, name
    version = max(versions, key=lambda v: int(v.version))
    assert version.status == 'READY'
    client.set_registered_model_alias(name=name, alias='champion', version=version.version)
    champion = client.get_model_version_by_alias(name=name, alias='champion')
    assert champion.version == version.version and champion.status == 'READY'
    print(question, name, 'champion -> v'+champion.version)
print('ASIGNACION Y VERIFICACION COMPLETADAS')
PY
```

### 13.2. Inferencia técnica por alias (datos sintéticos)

El siguiente comando **sí ejecuta** una predicción para comprobar la carga desde Model Registry; usa una fila sintética de unos y **no evalúa la calidad de una predicción con datos reales**. No se requieren nuevas pruebas con datos reales para documentar este requisito.

```bash
.venv/bin/python - <<'PY'
import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
client = MlflowClient()
names = {'P1': 'mibici-equipo15-p1-demanda',
         'P2': 'mibici-equipo15-p2-riesgo',
         'P3': 'mibici-equipo15-p3-grupos'}
for question, name in names.items():
    version = client.get_model_version_by_alias(name=name, alias='champion')
    assert version.status == 'READY'
    uri = f'models:/{name}@champion'
    model = mlflow.pyfunc.load_model(uri)
    signature = model.metadata.signature
    assert signature and signature.inputs
    cols = [col.name for col in signature.inputs.inputs]
    X = pd.DataFrame(np.ones((1, len(cols)), dtype=np.float64), columns=cols)
    result = model.predict(X)
    assert len(result) == 1
    print(question, 'v'+version.version, len(cols), 'columnas', uri,
          '| predicción sintética:', result, '| Inferencia: OK')
print('VALIDACION CHAMPION COMPLETADA: 3 MODELOS')
PY
```

En la ejecución del equipo: P1 → `[1.0482985]`; P2 → `[1]`; P3 → `[0]`. Son únicamente pruebas de interfaz y funcionamiento, no pronósticos o clasificaciones de situaciones reales.

## 14. Evidencia para el reporte PDF y el repositorio

| Evidencia | Captura o salida |
|---|---|
| Reproducción | Ingesta, baselines, búsquedas y `40 passed` |
| Nested runs | Padres desplegados con hijas `FINISHED` de P1/P2/P3 |
| Comparación | Métricas de validación y test frente al baseline para las tres preguntas |
| Elección de P3 | Tabla entre familias + análisis de estabilidad multisemilla |
| Artefactos | Modelo, firma, `input_example`, manifiesto, gráficos y configuración de al menos una candidata |
| Registry | Captura de **Registered Models** con tres modelos y `@champion` |
| Linaje | Logged Model → corrida fuente → commit → versión de datos |
| Código de validación | Salida `VALIDACION CHAMPION COMPLETADA: 3 MODELOS` |
| Trabajo colaborativo | Commits, Pull Requests y revisiones de los cuatro integrantes |

El reporte debe incluir enlace al repositorio, comandos de reproducción, tablas de baselines/candidatos, justificación de selección, explicación de alias como esquema de ciclo de vida, capturas y referencias APA 7. **No** subir a Git `mlflow.db`, `mlruns/`, `mlartifacts/`, CSV o modelos serializados. La rúbrica exige además descripciones de los modelos registrados y ejemplos de entrada de las corridas candidatas.

## 15. Referencias

- [Repositorio de GitHub del Equipo 15](https://github.com/A01797899/TC5061-Proyecto-Equipo15)
- [MiBici — Datos abiertos](https://www.mibici.net/es/datos-abiertos/)
- [MLflow Projects](https://mlflow.org/docs/latest/ml/projects/)
- [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/)
- [MLflow Model Registry](https://mlflow.org/docs/latest/ml/model-registry/)
- [MLflow Model Registry workflows](https://mlflow.org/docs/latest/ml/model-registry/workflow/)
- [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/)
- [pyenv](https://github.com/pyenv/pyenv)
- [Bandit](https://bandit.readthedocs.io/)
- [pip-audit](https://pypi.org/project/pip-audit/)

**Resultados verificados, 25 de octubre de 2026:** suite de 40 tests aprobada, modelos P1/P2/P3 registrados (v1, `READY`), tres alias `champion` asignados e inferencia sintética por alias sin errores.
