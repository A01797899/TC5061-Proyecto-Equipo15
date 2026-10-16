# P2 — Clasificación de presión operativa proxy (Equipo 15)

## Objetivo

**Pregunta de negocio:** ¿Una estación estará en riesgo de quedarse sin bicicletas o sin anclajes disponibles en un periodo futuro?

**Tarea experimental:** clasificar para cada **estación-hora** si la presión de flujos en **la siguiente hora** supera un umbral relativo a su capacidad. Este resultado es **un indicador proxy**, no una constatación de inventario insuficiente: los datos abiertos no proporcionan inventario inicial, bicicletas disponibles ni rebalanceos.

## Datos

- Los 12 CSV `datos_abiertos_2024_01.csv` a `datos_abiertos_2024_12.csv` en `data/raw/` (4,645,921 viajes según el EDA).
- La nomenclatura `nomenclatura_2026_09.csv` con las columnas `id` y `dpcapacity`.
- 484 estaciones en el catálogo; 480 con capacidad válida; 370 observadas en los viajes de 2024, todas con capacidad válida según el notebook P2.
- **Advertencia:** se utilizan capacidades publicadas en 2026 para analizar 2024; su equivalencia histórica no está verificada.

## Etiqueta proxy

Para cada estación s y hora t:

- Salidas: cantidad de viajes cuya fecha/hora de inicio está dentro de t y cuya estación de origen es s.
- Llegadas: cantidad de viajes cuya fecha/hora de fin está dentro de t y cuya estación de destino es s.
- Balance: `llegadas - salidas`.
- Presión: `abs(balance) / dpcapacity`.
- Clase `1` (presión alta): `presión >= 0.25`; clase `0`: caso contrario.

El umbral **0.25 es provisional**, elegido después de analizar prevalencia y estabilidad de alertas en el notebook. No equivale a una probabilidad ni a una regla operativa validada por MiBici. El balance negativo sugiere presión hacia escasez de bicicletas; el positivo, hacia escasez de anclajes. En esta primera versión ambas direcciones comparten una etiqueta binaria.

Las horas sin entradas ni salidas se incorporan con flujo cero para el universo de estaciones presentes en **entrenamiento**. Es una suposición de modelado: no demuestra operación ininterrumpida. Las estaciones sin historial en enero–agosto quedan fuera de la evaluación principal y se reporta cuántas son.

## División cronológica y prevención de fuga

| Conjunto | Periodo |
|---|---|
| Entrenamiento | Enero–agosto de 2024 |
| Validación | Septiembre–octubre de 2024 |
| Prueba | Noviembre–diciembre de 2024 |

La etiqueta se genera a partir del balance **real de la hora a predecir**, pero ninguna variable predictora del baseline contiene ese balance ni las salidas/llegadas de esa hora. Las variables usadas por el baseline histórico son solo identificador de estación, día de semana y hora objetivo, y sus probabilidades se estiman **únicamente** con el conjunto de entrenamiento.

## Baselines

1. `P2_baseline_dummy_most_frequent`: siempre predice la clase mayoritaria (`0`) detectada en entrenamiento. Referencia mínima para una tarea desbalanceada.
2. `P2_baseline_historical_station_weekday_hour`: predice la frecuencia histórica de clase positiva para cada combinación estación/día/hora. Si no hay combinación usa el promedio de la estación y, finalmente, el promedio global de entrenamiento. Ajusta **solo con validación** un umbral de probabilidad en una rejilla de 0.01 a 0.50 que maximiza F1.

**Métricas:** precision, recall, F1, average precision (área bajo la curva precision-recall) y tasas de clase positiva, para train, val y test. Se generan matrices de confusión para test. Si la clase positiva es escasa, accuracy no es adecuada como único indicador.

## Archivos e instalación

Copiar:

```text
mibici/modeling/p2_baseline.py
references/P2_INSTRUCCIONES.md
```

Conservar en la raíz de proyecto `data/raw/` los 13 CSV locales. Requiere `pandas`, `numpy`, `matplotlib`, `scikit-learn` y `mlflow`, además de las dependencias generales del proyecto (incluyendo `loguru` por el paquete `mibici`). Emplear el entorno `mlops` o instalar las dependencias de `requirements.txt`.

Desde la raíz de `TC5061-Proyecto-Equipo15`, con el entorno activado:

```bash
python -m mibici.modeling.p2_baseline --tracking-uri sqlite:///mlflow.db
```

Opcional: explorar otro umbral proxy con `--threshold 0.50`; **ese ensayo no sustituye al baseline seleccionado**.

```bash
python -m mibici.modeling.p2_baseline --tracking-uri sqlite:///mlflow.db --threshold 0.50
```

En otra terminal:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Abrir `http://127.0.0.1:5000` y seleccionar el experimento `mibici-equipo15`.

## Seguimiento en MLflow

Se crean **dos runs**: uno para cada baseline. Cada run registra métricas con prefijos `train_`, `val_`, `test_`, parámetros (periodos, umbrales, cantidad de estaciones, tamaño de bloques), etiquetas (`team`, `question`, `git_commit`, limitación del proxy) y artefactos (`data_manifest.json`, `metrics.json`, `test_confusion_matrix.png`).

## Resultados obtenidos y validados (15 de octubre de 2026)

Los dos runs finalizaron correctamente en **MLflow 3.16.1** dentro del experimento `mibici-equipo15`. Los resultados se corroboraron con capturas de métricas, parámetros, archivos `metrics.json` y matrices de confusión.

**Particiones y distribución de la clase positiva:**

| Conjunto | Estación-horas | Etiquetas positivas | Tasa positiva |
|---|---:|---:|---:|
| Train (ene–ago) | 2,119,872 | 110,792 | 5.226 % |
| Validación (sep–oct) | 529,968 | 28,520 | 5.381 % |
| Test (nov–dic) | 529,968 | 24,935 | 4.705 % |

Se trabajó con **362 estaciones presentes en entrenamiento**; **8 estaciones observadas en 2024** quedaron fuera del universo evaluado por no contar con historial de entrenamiento. La etiqueta positiva indica `abs(llegadas-salidas)/dpcapacity >= 0.25`.

### Comparación cuantitativa de los baselines

| Métrica | Dummy train | Dummy val | Dummy test | Histórico train | Histórico val | Histórico test |
|---|---:|---:|---:|---:|---:|---:|
| Precision | 0.0000 | 0.0000 | 0.0000 | 0.3738 | 0.3287 | **0.2851** |
| Recall | 0.0000 | 0.0000 | 0.0000 | 0.5457 | 0.4680 | **0.4563** |
| F1-score | 0.0000 | 0.0000 | 0.0000 | 0.4437 | 0.3861 | **0.3509** |
| Average Precision | 0.0523 | 0.0538 | 0.0471 | 0.4304 | 0.3413 | **0.3052** |

La tasa de predicciones positivas del modelo histórico fue **7.63 % en train**, **7.66 % en validación** y **7.53 % en test**; el Dummy predijo siempre la clase `0`.

- **Umbral proxy de presión:** `0.25` (criterio definido en el EDA).
- **Umbral de probabilidad del histórico:** `0.21`, seleccionado por mayor F1 en validación mediante una cuadrícula de `0.01` a `0.50`.
- **Importante:** el umbral de probabilidad no se optimizó contra los datos de prueba.

### Matrices de confusión en test

Filas: **etiqueta real proxy**; columnas: **predicción**. La clase `1` indica presión relativa alta en la siguiente hora.

**DummyClassifier — clase mayoritaria**

| Real / predicho | 0 | 1 |
|---|---:|---:|
| 0 | 505,033 | 0 |
| 1 | 24,935 | 0 |

El modelo clasifica todos los registros como negativos y no detecta ninguna etiqueta positiva: `recall = 0`. Aunque su accuracy sería aproximadamente **95.29 %**, esta métrica sería engañosa sin recall/F1.

**Histórico estación–día de semana–hora**

| Real / predicho | 0 | 1 |
|---|---:|---:|
| 0 | 476,506 | 28,527 |
| 1 | 13,558 | 11,377 |

Esto corresponde a **11,377 verdaderos positivos**, **28,527 falsos positivos** y **13,558 falsos negativos**. Detecta **45.63 %** de las horas etiquetadas como presión alta, mientras que **28.51 %** de sus alertas corresponden a positivos de la etiqueta proxy.

El histórico supera con claridad al Dummy en F1 y Average Precision. El F1 cae de `0.4437` (train) a `0.3861` (validación) y `0.3509` (test); esta disminución sugiere variabilidad temporal y debe analizarse, pero no demuestra por sí sola sobreajuste.

### Trazabilidad en MLflow

- **Run histórico:** `8bd68fd098de4097aa363d9cb4977d5d`.
- **Run Dummy:** `b2498d43bba34728ab6b1fbc1e7c2260`.
- **Estado:** ambos `Finished`.
- **Registro por run:** 18 métricas (`train_*`, `val_*`, `test_*`), 14 parámetros, tags de proyecto y tres artefactos bajo `p2/`: `data_manifest.json` (incluye SHA-256 de archivos de entrada), `metrics.json` y `test_confusion_matrix.png`.
- **Origen Git mostrado durante ejecución:** rama `feature/A01797899-josemanuel-martinez`, commit `ad2c95f`. Este identificador corresponde al HEAD observado entonces; **no acredita por sí solo que el código P2 estuviera incluido en ese commit**. El commit que incorpore `p2_baseline.py` deberá quedar documentado también.

### Interpretación y limitaciones

Estos resultados prueban la capacidad de anticipar **la etiqueta proxy**, no un agotamiento real de bicicletas o anclajes. No disponemos de inventario ni registros de rebalanceo; además, `dpcapacity` procede de 2026 y podría ser diferente de 2024. Las horas con flujo cero se incorporan por hipótesis de modelado, sin verificar operación ininterrumpida de cada estación.

**Conclusión de P2:** el baseline histórico mejora de forma sustancial al clasificador mayoritario, pero su precision de **28.51 %** y su recall de **45.63 %** evidencian que todavía hay numerosas falsas alertas y eventos proxy no detectados. Estos valores se consideran una **referencia reproducible inicial**, no una solución lista para operación real.

## Protección de datos y reproducibilidad

Los CSV, `mlflow.db`, `mlruns/` y `mlartifacts/` deben seguir excluidos por `.gitignore`. Antes del commit comprobar `git status --short` y `git check-ignore -v mlflow.db`.

Es importante registrar los hashes de los archivos de origen y el commit que contiene el código del experimento para reconstruir los runs. El registro de un commit anterior al script no constituye trazabilidad completa del código ejecutado.

## Próximo paso

Guardar la documentación y el script P2 en la rama personal mediante commit y pull request, dejando CSV, SQLite, artefactos MLflow y cachés de Python fuera de Git. Como experimentación posterior, contrastar umbrales proxy o modelos alternativos **solo con entrenamiento y validación**, reservando test como evaluación final; no modificar retrospectivamente el baseline registrado.
