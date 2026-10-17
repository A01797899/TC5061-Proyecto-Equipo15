# MiBici — Proyecto MLOps, Semana 5 | Equipo 15

[![Cookiecutter Data Science](https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter)](https://cookiecutter-data-science.drivendata.org/)

Repositorio: [TC5061-Proyecto-Equipo15](https://github.com/A01797899/TC5061-Proyecto-Equipo15)

## 1. Descripción y objetivos

Proyecto académico de operaciones de aprendizaje automático (MLOps) que utiliza los datos abiertos de viajes de **MiBici** correspondientes a **2024**. El objetivo es establecer una base de experimentación reproducible: ingesta de datos, validaciones, generación de características, ejecución de modelos *baseline* y seguimiento de experimentos mediante MLflow.

Las tres preguntas de análisis son:

| Pregunta | Objetivo | Enfoque |
|---|---|---|
| **P1 — Demanda** | Estimar la cantidad de viajes que salen de cada estación por hora. | Baselines de media histórica por estación y por estación–día de semana–hora. |
| **P2 — Riesgo proxy** | Estimar el riesgo aproximado de desequilibrio en una estación a partir del flujo neto y su capacidad. | Clasificación con etiqueta proxy construida a partir de salidas, llegadas y capacidad. |
| **P3 — Perfiles de uso** | Identificar estaciones con patrones horarios de utilización similares. | Clustering no supervisado con K-Means. |

> **Limitación de P2:** MiBici no proporciona en estos archivos el inventario histórico de bicicletas y espacios libres por estación. La etiqueta `riesgo_proxy` representa la presión relativa del balance horario sobre la capacidad, **no una observación real de estaciones vacías o llenas**. La nomenclatura de estaciones utilizada es de **septiembre de 2026**, posterior a los viajes de 2024; este desfase debe considerarse al interpretar las capacidades y los resultados.

## 2. Integrantes del equipo

| Integrante | Matrícula |
|---|---|
| José Manuel Martínez Romero | A01797899 |
| Jorge Alcázar | A01252855 |
| José Luis Mendoza Mora | A01732182 |
| Óscar Gutiérrez | A00832992 |

## 3. Requisitos

- **Python 3.12.x**, de acuerdo con `requires-python = "~=3.12.0"` en `pyproject.toml`. Python 3.13 no satisface esa restricción.
- Git y acceso a Internet para descargar los datos abiertos cuando no estén disponibles localmente.
- Espacio en disco y memoria suficientes para procesar aproximadamente **4,645,921 viajes** y almacenar los CSV, los Parquet y los resultados de los modelos.
- Dependencias fijadas en `requirements.txt`. Los comandos se ejecutan desde la **raíz del repositorio**.

## 4. Clonar e instalar el proyecto

```bash
git clone https://github.com/A01797899/TC5061-Proyecto-Equipo15.git
cd TC5061-Proyecto-Equipo15

# Comprobar que Python es 3.12.x
python3.12 --version

# Crear y activar un entorno virtual (macOS/Linux)
python3.12 -m venv .venv
source .venv/bin/activate

# Instalar dependencias y el proyecto en modo editable (-e .)
python -m pip install -r requirements.txt
python -m pip check
```

En Windows, si Python 3.12 está instalado y disponible mediante el lanzador `py`, utilizar:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip check
```

**Resultado esperado de `pip check`:** `No broken requirements found.`

> **macOS con Homebrew:** si Python 3.12 está instalado mediante Homebrew pero `python3.12` no aparece en el `PATH`, se puede crear el entorno con:
>
> ```bash
> "$(brew --prefix python@3.12)/bin/python3.12" -m venv .venv
> ```
>
> Después, activar el entorno e instalar las dependencias con los comandos anteriores.

> **Nota:** en el proyecto se utilizó también el nombre `.venv-repro` para un entorno aislado de auditoría. Si se utiliza un nombre distinto de `.venv`, debe excluirse de Git y mantenerse fuera del control de versiones.

## 5. Ingesta de datos reproducible

El módulo `mibici/dataset.py` descarga y procesa los **12 CSV mensuales de viajes de 2024** y un archivo de nomenclatura de estaciones. Normaliza encabezados, tipos de datos, codificaciones y fechas; elimina registros sin inicio u origen y viajes duplicados; genera Parquet y un manifiesto de trazabilidad.

```bash
python -m mibici.dataset
```

Si un CSV ya está presente en `data/raw/`, se reutiliza. Para forzar una nueva descarga de las fuentes definidas en `mibici/config.py`:

```bash
python -m mibici.dataset --forzar-descarga
```

**Archivos de salida:**

```text
data/
├── raw/
│   ├── datos_abiertos_2024_01.csv
│   ├── ...
│   ├── datos_abiertos_2024_12.csv
│   └── nomenclatura_2026_09.csv
├── processed/
│   ├── viajes_2024.parquet
│   └── estaciones.parquet
└── manifiesto.json
```

El archivo `data/manifiesto.json` registra para cada fuente su ruta, URL, tamaño, hash **SHA-256**, número de registros y, cuando corresponde, codificación y formato de fecha. El identificador `version_datos` se calcula a partir de las huellas de las fuentes. El manifiesto está versionado en Git; los CSV y Parquet pueden regenerarse mediante la ingesta.

### Referencia de los datos utilizados

| Indicador | Valor verificado |
|---|---:|
| Archivos de viajes | 12 |
| Archivos de nomenclatura | 1 |
| Total de viajes procesados | 4,645,921 |
| Estaciones en la nomenclatura | 484 |
| Registros descartados en la reproducción realizada | 0 |
| `version_datos` | `7e1470ac0632` |

El valor de `version_datos` anterior corresponde a las fuentes verificadas durante esta práctica; puede cambiar si los archivos originales publicados por MiBici cambian.

Para inspeccionar las salidas:

```bash
python - <<'PY'
import pandas as pd
from mibici.config import VIAJES_PARQUET, ESTACIONES_PARQUET
from mibici.dataset import version_datos

print('Viajes:', len(pd.read_parquet(VIAJES_PARQUET)))
print('Estaciones:', len(pd.read_parquet(ESTACIONES_PARQUET)))
print('Versión de datos:', version_datos())
PY
```

## 6. Generación de características (`mibici/features.py`)

Las funciones de *feature engineering* son reutilizables, reciben DataFrames y permiten separar el procesamiento de datos de la lógica de modelado:

| Función | Uso |
|---|---|
| `hourly_departures()` | Salidas por estación y hora; base de P1. |
| `complete_hourly_panel()` | Panel estación–hora con horas de cero viajes. |
| `hourly_arrivals()` | Llegadas por estación y hora; base de P2. |
| `hourly_flow_balance()` | Balance `llegadas - salidas` para P2. |
| `proxy_risk_labels()` | Etiqueta proxy con umbral sobre `abs(balance) / capacidad`. |
| `calendar_features()` | Día de semana, hora y fin de semana. |
| `hourly_usage_profiles()` | Perfiles de salidas según tipo de día y hora para P3. |
| `lag_features()` | Rezagos temporales basados en observaciones anteriores. |
| `trip_quality_flags()` | Indicadores de calidad, como duraciones y edades atípicas. |

Los módulos `mibici/modeling/p1_baseline.py`, `p2_baseline.py` y `p3_baseline.py` **importan y reutilizan funciones de `mibici.features`**, evitando duplicar transformaciones entre experimentos.

### Precaución contra fuga de información

Los conteos y balances de la hora objetivo se utilizan para construir objetivos o etiquetas, no como variables predictoras de esa misma hora. Las características de calendario y los valores históricos o rezagados permiten trabajar con información disponible antes del momento de predicción.

## 7. Ejecutar los tres baselines

Los comandos siguientes se validaron con Python **3.12.15**, las dependencias del repositorio y los datos de 2024. Ejecutarlos desde la raíz del proyecto, después de la ingesta. Los baselines utilizan los CSV en `data/raw/`, aunque `dataset.py` también produce Parquet normalizados.

### P1. Predicción de salidas por estación y hora

```bash
python -m mibici.modeling.p1_baseline \
  --raw-dir data/raw \
  --chunk-size 100000 \
  --tracking-uri sqlite:///mlflow.db
```

P1 compara una media histórica por estación con una media por estación, día de la semana y hora. Las estaciones de entrenamiento son 362 y se utiliza partición cronológica:

- **Entrenamiento:** enero–agosto de 2024.
- **Validación:** septiembre–octubre de 2024.
- **Prueba:** noviembre–diciembre de 2024.

### P2. Clasificación de riesgo proxy

```bash
python -m mibici.modeling.p2_baseline \
  --tracking-uri sqlite:///mlflow.db \
  --threshold 0.25 \
  --chunk-size 100000
```

La presión proxy se define como `abs(llegadas - salidas) / capacidad`, con etiqueta positiva cuando es mayor o igual a `0.25`. La clasificación se evalúa con precision, recall, F1-score y average precision. El umbral de probabilidad del baseline histórico se selecciona utilizando el conjunto de validación.

### P3. Agrupamiento de perfiles de estaciones

```bash
python -m mibici.modeling.p3_baseline \
  --data-dir data/raw \
  --tracking-uri sqlite:///mlflow.db \
  --experiment mibici-equipo15 \
  --min-departures 100 \
  --chunk-size 100000 \
  --k-min 2 \
  --k-max 8 \
  --seed 42 \
  --output-dir reports/figures/p3
```

P3 forma perfiles horarios de días laborales y fines de semana, evalúa valores de K entre 2 y 8 y reporta, entre otras métricas, Silhouette y Davies–Bouldin. Las etiquetas numéricas de K-Means pueden intercambiarse entre ejecuciones; es preferible comparar perfiles y métricas antes que los identificadores de grupo.

> **Artefactos locales:** antes de realizar commits, revisar `git status --short`. Los archivos CSV y JSON que se generen en `reports/figures/p3/` son resultados de ejecución; no deben incorporarse accidentalmente a un PR. Versionarlos solo si el equipo decide incluirlos expresamente como entregables.

## 8. Resultados de referencia reproducidos

### P1 — Predicción de demanda (conjunto de prueba)

| Modelo | MAE | RMSE |
|---|---:|---:|
| Media histórica por estación | 1.3621 | 2.1185 |
| Media histórica por estación–día–hora | **0.8556** | **1.4635** |

### P2 — Riesgo proxy (baseline histórico, conjunto de prueba)

| Indicador | Valor |
|---|---:|
| Precision | 0.285109 |
| Recall | 0.456266 |
| F1-score | 0.350931 |
| Average Precision | 0.305219 |
| Umbral de probabilidad seleccionado | 0.21 |

### P3 — K-Means

| Indicador | Valor |
|---|---:|
| Viajes procesados | 4,645,921 |
| Estaciones observadas | 370 |
| Estaciones agrupadas | 368 |
| Características del modelo | 49 |
| Mejor K identificado | 2 |
| Silhouette Score | 0.127636 |
| Davies–Bouldin | 2.369866 |
| Tamaño de los grupos | 177 y 191 |

Los resultados anteriores corresponden a ejecuciones reproducidas en un clon limpio de `main`; sirven como referencia para verificar el comportamiento del proyecto. Los resultados de P3 deben interpretarse de forma exploratoria debido a la separación moderada/baja entre grupos.

## 9. Seguimiento de experimentos con MLflow

Los baselines registran parámetros y métricas en el experimento **`mibici-equipo15`**, con seguimiento local SQLite en `mlflow.db` cuando se utiliza `--tracking-uri sqlite:///mlflow.db`.

Desde la raíz del proyecto, iniciar la interfaz de MLflow:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Abrir [http://127.0.0.1:5000](http://127.0.0.1:5000) en el navegador. Los identificadores de los runs cambian entre ejecuciones. La base `mlflow.db` y los artefactos locales no deben subirse al repositorio.

## 10. Pruebas automatizadas

Después de instalar las dependencias:

```bash
python -m pip check
python -m pytest tests/ -v
```

Los tests de `tests/test_dataset.py` y `tests/test_features.py` verifican, entre otros aspectos, encabezados y fechas, codificaciones, descargas, hashes, salidas y llegadas, balance de flujos, etiquetas proxy, perfiles horarios, paneles completos, rezagos y controles de calidad.

**Resultado de referencia:** `21 passed` con Python 3.12.15 y cero errores. El archivo de prueba de plantilla `tests/test_data.py` fue eliminado de `main` mediante el PR #7; por tanto, el comando anterior ejecuta la suite completa sin el fallo artificial de `assert False`.

## 11. Organización del repositorio

El proyecto sigue la estructura de **Cookiecutter Data Science**, adaptada a MiBici:

```text
TC5061-Proyecto-Equipo15/
├── README.md
├── requirements.txt
├── pyproject.toml
├── Makefile
├── data/
│   ├── raw/                  # CSV originales descargados
│   ├── interim/              # Datos intermedios, cuando aplique
│   ├── processed/            # Parquet de viajes y estaciones
│   └── manifiesto.json       # Versión y huellas de las fuentes
├── notebooks/                # Exploración y análisis
├── references/               # Documentación P1, P2 y P3
├── reports/
│   └── figures/              # Figuras y resultados de los experimentos
├── models/                   # Salidas de modelos, cuando aplique
├── tests/
│   ├── test_dataset.py
│   └── test_features.py
└── mibici/
    ├── __init__.py
    ├── config.py             # Rutas, fuentes, formatos y parámetros
    ├── dataset.py            # Descarga, normalización y manifiesto
    ├── features.py           # Características reutilizables
    ├── plots.py
    └── modeling/
        ├── __init__.py
        ├── p1_baseline.py
        ├── p2_baseline.py
        ├── p3_baseline.py
        ├── train.py
        └── predict.py
```

Algunas carpetas y artefactos se generan al ejecutar los comandos. Los datos crudos, Parquet, entornos virtuales y bases locales de MLflow deben mantenerse fuera de los commits, salvo el manifiesto versionado expresamente.

## 12. Notas de reproducibilidad y limitaciones

- Se utiliza una semilla fija (`42`) en P3, además de versiones de librerías fijadas en `requirements.txt`.
- La separación de P1 y P2 es **temporal** para evitar evaluar con observaciones anteriores mezcladas aleatoriamente con posteriores.
- Los hashes SHA-256 de las fuentes permiten comprobar si se reproduce exactamente la versión documentada.
- El criterio de P2 es una **aproximación de riesgo**, no una etiqueta operativa basada en sensores o inventario real.
- La nomenclatura es de septiembre de 2026 y puede no representar las mismas capacidades y estaciones existentes en 2024.
- K-Means es exploratorio: los clusters no deben considerarse categorías oficiales de MiBici.
- El README documenta los experimentos *baseline* y su reproducción, no un servicio de inferencia desplegado en producción.

## 13. Fuentes y documentación

- [Repositorio oficial del equipo](https://github.com/A01797899/TC5061-Proyecto-Equipo15).
- [MiBici — sitio oficial y datos abiertos](https://www.mibici.net/).
- [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/).
- [Documentación de MLflow](https://mlflow.org/docs/latest/).
- Documentación interna del repositorio en `references/` y análisis en `notebooks/`.
