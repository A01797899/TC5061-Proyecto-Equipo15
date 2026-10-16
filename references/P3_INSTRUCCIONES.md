# P3 — Patrones de uso y agrupamiento de estaciones MiBici 2024

## 1. Pregunta y propósito

**¿Qué grupos de estaciones comparten patrones de uso y dónde se ubican en la ciudad?**

Se utiliza **clustering no supervisado** con K-Means para caracterizar las estaciones según sus salidas de bicicletas durante 2024. Las coordenadas se usan **solo en el mapa**, no como variables de entrenamiento.

## 2. Archivos y entorno

- Script: `mibici/modeling/p3_baseline.py`.
- Notebook EDA validado: `notebooks/3.0-analisis-patrones-estaciones-mibici-p3.ipynb`.
- Datos locales: `data/raw/datos_abiertos_2024_01.csv` a `datos_abiertos_2024_12.csv`.
- Catálogo: `data/raw/nomenclatura_2026_09.csv` (campos `id`, `name`, `latitude`, `longitude`, `dpcapacity`).
- Entorno `mlops`: Python 3.13, pandas, numpy, scikit-learn, matplotlib y MLflow.

Los CSV originales, la base de MLflow y sus artefactos locales **no deben incluirse en Git**. Los resultados generados en `reports/figures/p3/` pueden revisarse antes de decidir qué versionar.

## 3. Características y metodología

- Se procesan los 12 CSV por bloques de 100,000 filas, aplicando los formatos de fecha observados en el EDA (septiembre y octubre con `%d/%m/%Y %H:%M`).
- Se agregan salidas por estación, día, hora y tipo de día.
- Se incluyen estaciones con **al menos 100 salidas** y coordenadas completas.
- Se construyen **49 variables**: 24 proporciones horarias en laborables, 24 proporciones en fin de semana y `log1p(salidas_2024 / 366)`.
- Se estandarizan las variables con `StandardScaler`.
- Se entrena K-Means con `random_state=42`, `n_init=20`, para **K=2 a 8**.
- Se comparan **Silhouette (mayor es mejor)**, **Davies–Bouldin (menor es mejor)**, inercia y tamaños de grupos. El K exploratorio se selecciona según Silhouette.
- Se generan gráficas de puntuación por K, patrones horarios y un mapa de puntos por coordenadas.

**Nota:** este procedimiento es *no supervisado* y descriptivo; no aplica la división train/validation/test de P1 o P2. El análisis de estabilidad y de características alternativas puede realizarse como experimento adicional.

## 4. Ejecución desde la raíz del repositorio

```bash
python -m mibici.modeling.p3_baseline --tracking-uri sqlite:///mlflow.db
```

Opciones de sensibilidad:

```bash
python -m mibici.modeling.p3_baseline --tracking-uri sqlite:///mlflow.db --min-departures 1000
```

Para consultar los runs:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Abrir `http://127.0.0.1:5000`, experimento **`mibici-equipo15`**, run **`P3_baseline_kmeans_k2_to_k8`**.

## 5. Evidencias registradas en MLflow

- Parámetros: K inicial/final, `seed`, `n_init`, mínimo de salidas, escalador, definición de variables y fuente del catálogo.
- Métricas: Silhouette, Davies–Bouldin e inercia por K; mejor K y su puntuación; cantidad de estaciones incluidas y excluidas.
- Etiquetas: equipo, pregunta P3, dataset, commit Git y limitación del catálogo geográfico.
- Artefactos en `p3/`: `p3_data_manifest.json`, `p3_metrics.json`, `p3_comparacion_k.csv`, `p3_resumen_grupos.csv`, `p3_estaciones_grupos.csv`, `p3_silhouette_por_k.png`, `p3_perfiles_horarios.png` y `p3_mapa_clusters.png`.

## 6. Resultados reales de ejecución en MLflow

El **15 de octubre de 2026** se ejecutó satisfactoriamente `p3_baseline.py` con los 12 CSV de 2024 y el catálogo de estaciones de septiembre de 2026. La interfaz de MLflow muestra el run en estado **Finished**.

| Elemento | Valor confirmado |
|---|---|
| Experimento | `mibici-equipo15` |
| Run | `P3_baseline_kmeans_k2_to_k8` |
| Run ID | `4bd37aa47cdf4c6787e2957d02fe2945` |
| Commit detectado al ejecutar | `0e24243` |
| Viajes procesados | **4,645,921** |
| Estaciones observadas | **370** |
| Estaciones agrupadas | **368** |
| Estaciones excluidas por el criterio de actividad | **2** |
| Variables del modelo | **49** |
| Valores evaluados | **K=2 a K=8** |
| Mejor K según Silhouette | **2** |
| Mejor Silhouette | **0.12763635001979978** |
| Davies–Bouldin de K=2 | **2.3698664533515115** |
| Tamaños de los grupos seleccionados | **177 y 191 estaciones** |

### 6.1. Comparación de configuraciones K-Means

Las métricas que muestra `p3_comparacion_k.csv` en MLflow son:

| K | Silhouette ↑ | Davies–Bouldin ↓ | Inercia | Grupo menor | Grupo mayor |
|---:|---:|---:|---:|---:|---:|
| **2** | **0.127636** | 2.369866 | 13689.811821 | 177 | 191 |
| 3 | 0.112191 | 2.407129 | 12900.462966 | 64 | 157 |
| 4 | 0.082353 | 2.476469 | 12257.560061 | 35 | 141 |
| 5 | 0.073328 | 2.378216 | 11843.284708 | 29 | 128 |
| 6 | 0.060343 | 2.478017 | 11468.528303 | 28 | 105 |
| 7 | 0.064895 | 2.279183 | 11141.187068 | 5 | 88 |
| 8 | 0.073785 | **2.198064** | 10897.123692 | 4 | 89 |

**Selección:** K=2 ofrece el mayor Silhouette entre los valores probados. K=8 obtiene el menor Davies–Bouldin, pero origina un grupo de solo **cuatro estaciones**; por ello no se eligió como segmentación exploratoria principal. La inercia disminuye al aumentar K, como es esperable, y no debe tomarse por sí sola como razón para elegir muchos grupos.

**Interpretación:** Silhouette ≈ 0.128 es bajo y evidencia **separación débil**. Los resultados identifican perfiles útiles para explorar, pero no validan la existencia de segmentos naturales perfectamente diferenciados. No afirmar una agrupación concluyente ni un desempeño predictivo: P3 es un análisis descriptivo no supervisado.

### 6.2. Interpretación de los grupos seleccionados (K=2)

El archivo `p3_resumen_grupos.csv` y las gráficas permiten describir dos perfiles relativos:

| Indicador del notebook exploratorio (medianas o fracciones por grupo) | Grupo 0 | Grupo 1 |
|---|---:|---:|
| Número de estaciones | **177** | **191** |
| Mediana de salidas anuales | 11,828 | 7,787 |
| Mediana de salidas diarias (sobre 366 días) | 32.32 | 21.28 |
| Fracción matutina en laborables | 20.06 % | 31.94 % |
| Fracción vespertina en laborables | 34.22 % | 24.84 % |
| Fracción de actividad en fin de semana | 19.30 % | 20.63 % |

- **Grupo 0 — perfil relativamente vespertino:** mayor intensidad mediana de salidas y pico laboral visible alrededor de las 18:00.
- **Grupo 1 — perfil relativamente matutino:** menor intensidad mediana y pico laboral visible alrededor de las 8:00.

Estas etiquetas son interpretaciones de los **perfiles agregados**, no características universales de cada estación. Los valores de las medianas y porcentajes proceden del **notebook P3 ejecutado**; las capturas de MLflow confirman los tamaños de grupo y las formas de los perfiles horarios.

**Advertencia de etiquetas:** los números `0` y `1` que asigna K-Means carecen de significado intrínseco y pueden intercambiarse entre ejecuciones o versiones. Para comparar resultados deben utilizarse los perfiles, las estaciones asignadas y, de ser necesario, una métrica invariante al nombre de los grupos (por ejemplo, Adjusted Rand Index), no el número de cluster por sí solo.

### 6.3. Distribución geográfica

El artefacto `p3_mapa_clusters.png` sitúa las **368 estaciones** en un plano de longitud y latitud, con puntos de ambos grupos distribuidos y parcialmente solapados en Guadalajara. Como **las coordenadas no participan en el entrenamiento**, el mapa sirve para analizar dónde se encuentran los perfiles de uso detectados; no demuestra una separación espacial ni que la cercanía geográfica haya generado los grupos. Las coordenadas corresponden al catálogo de 2026.

### 6.4. Evidencia de MLflow y reproducibilidad

En el run se verificaron **29 métricas**, parámetros tales como `seed=42`, `n_init=20`, `min_departures=100`, `scaler=StandardScaler`, `k_min=2`, `k_max=8` y la exclusión de coordenadas del modelo. También constan las etiquetas `team=15`, `question=P3`, `methodology=unsupervised_clustering`, `data_version=sha256_manifest` y la limitación geográfica.

Se observaron estos **ocho artefactos** en la carpeta `p3/` del run:

1. `p3_comparacion_k.csv`: puntuaciones por K y tamaños mínimo/máximo de grupos.
2. `p3_data_manifest.json`: hashes SHA-256 de los 12 CSV y del catálogo de nomenclatura.
3. `p3_estaciones_grupos.csv`: correspondencia entre estaciones y cluster.
4. `p3_mapa_clusters.png`: localización de las estaciones por grupo.
5. `p3_metrics.json`: resumen de métricas y selección de K.
6. `p3_perfiles_horarios.png`: curvas horarias de días laborables y fines de semana.
7. `p3_resumen_grupos.csv`: características resumidas por agrupamiento.
8. `p3_silhouette_por_k.png`: comparación gráfica del Silhouette.

Los valores principales del run **coinciden con el notebook exploratorio ejecutado**, incluyendo 4,645,921 viajes, 370 estaciones observadas, 368 agrupadas, 49 variables, K=2, Silhouette 0.127636 y tamaños de 177/191 estaciones. El `git_commit` es el estado del repositorio al ejecutar el script; **no garantiza por sí mismo que `p3_baseline.py` ya estuviera confirmado en ese commit**.

## 7. Limitaciones y trabajo posterior

- Las coordenadas y capacidad corresponden al catálogo de **2026**, no certifican ubicación o características históricas durante 2024.
- La media diaria se calcula sobre **366 días**: puede subestimar el uso durante los días efectivamente activos de estaciones incorporadas más tarde o con periodos de inactividad.
- Se excluyeron **dos estaciones** por no alcanzar las 100 salidas. Conviene repetir como análisis de sensibilidad con 1,000 salidas y comparar cobertura y estabilidad.
- Las 49 variables estandarizadas influyen simultáneamente en distancias euclidianas; considerar un análisis con menos características para valorar la robustez de los perfiles.
- Los valores bajos de Silhouette no respaldan conclusiones sobre clusters naturales bien separados; K-Means siempre asigna cada observación a un grupo incluso si hay solapamiento.
- Los IDs de cluster son arbitrarios: para comprobar estabilidad entre corridas se puede utilizar **Adjusted Rand Index (ARI)**, sin confundir el cambio de etiquetas con un cambio de pertenencia.
- No se aplican particiones train/validation/test porque el objetivo de P3 es exploratorio y no supervisado. Tampoco hay métricas de clasificación o regresión apropiadas para esta pregunta.
- Los datos CSV y artefactos locales de MLflow deben permanecer excluidos del control de versiones. Revisar `git status` y `git diff --cached --name-only` antes del commit y del Pull Request de P3.

## 8. Conclusión final

La implementación de **P3 con K-Means quedó ejecutada y registrada exitosamente en MLflow**, y sus resultados reproducen los obtenidos previamente en el notebook. Para K=2 se observan dos tendencias agregadas de uso —**vespertina (177 estaciones)** y **matutina (191 estaciones)**— útiles para caracterización y visualización territorial. Sin embargo, la puntuación Silhouette **0.1276** obliga a presentar la solución como **segmentación exploratoria de separación débil**, no como clasificación concluyente de estaciones. Como mejoras futuras se recomiendan pruebas de estabilidad (ARI) y sensibilidad al filtrado y al conjunto de características.
