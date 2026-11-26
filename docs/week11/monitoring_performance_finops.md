# Semana 11 - Monitoreo, A/B testing, rendimiento y FinOps

Resumen del monitoreo y la operación del Champion P1 (`mibici-equipo15-p1-demanda`) para la entrega final. Los análisis completos, los datos y las gráficas están en los documentos de las semanas 9 y 10 que se enlazan en cada sección.

## 1. Drift con Evidently

Detalle: [`docs/week9/drift_monitoring.md`](../week9/drift_monitoring.md). Reportes: `reports/week9/drift/evidently_drift_principal.html` y `evidently_drift_estacional.html`.

Se comparó el entrenamiento del Champion (enero a agosto de 2024) contra enero a marzo de 2025, con distancia de Jensen-Shannon para las variables de calendario y Wasserstein normalizada para las numéricas, umbral 0.1.

- Ninguna de las 8 variables de entrada ni el objetivo `y` supera el umbral en ningún escenario. El puntaje más alto es 0.084 (`salidas_lag_168h`, enero de 2025, por la semana de Navidad).
- El riesgo está por estación: 44 de 362 estaciones cambian su demanda más de 20 % y 11 más de 50 %. Las medias históricas del modelo quedaron congeladas con 2024 y dejan de representar esas estaciones.

## 2. A/B testing Champion contra Challenger

Detalle: [`docs/week9/ab_testing_analysis.md`](../week9/ab_testing_analysis.md). Datos y gráficas: `reports/week9/ab_testing/`.

El Challenger (versión 2) se reentrenó con enero y febrero de 2025. Ambos modelos se evaluaron sobre las mismas 269,328 observaciones de marzo de 2025.

| Métrica | Champion v1 | Challenger v2 | Mejora |
|---|---:|---:|---:|
| MAE | 0.8407 | 0.8243 | 1.94 % |
| RMSE | 1.4370 | 1.4046 | 2.25 % |

El Challenger tuvo menor MAE en 28 de 31 días y en 269 de 362 estaciones, y empeoró en 93. La evaluación es offline y de un solo mes, por eso el alias `champion` sigue en la versión 1 y el Challenger queda como candidato.

## 3. Rendimiento con Locust

Detalle: [`docs/week10/load_testing_analysis.md`](../week10/load_testing_analysis.md). Código: `load_tests/`. Resultados: `reports/week10/`.

Pruebas contra el Champion en Compose, con 4 CPU y 3.8 GiB, de 5 a 300 usuarios concurrentes y 49,348 solicitudes en total.

- 0 errores y 0 reinicios de contenedores en todos los escenarios.
- Hasta 150 usuarios (96.9 req/s) el servicio atiende toda la carga con P95 de 76 ms. El máximo es de unas 145 req/s; por encima, la latencia sube hasta 540 ms de mediana sin generar errores.
- El límite está en el proceso único de la API, que se detiene en cerca de 2.4 de 4 núcleos. MLflow usa 2.17 GiB de memoria y no cambia con la carga.

## 4. FinOps

Detalle: [`docs/week10/finops.md`](../week10/finops.md). Cálculo: `reports/week10/finops/cost_by_scenario.csv`.

Con la tarifa de referencia de 0.145 USD por hora y el punto de operación de 150 usuarios (96.9 req/s, P95 76 ms), el costo es de 0.4157 USD por millón de inferencias exitosas. Con 300 usuarios baja a 0.2738 USD, pero con P95 de 740 ms. MLflow es el mayor costo fijo, con cerca del 87 % de la memoria de los dos contenedores.

## 5. Recomendaciones de monitoreo y operación

1. Mantener Evidently para el drift global y agregar una vista por estación con MAE y volumen de demanda.
2. Recalcular periódicamente las medias históricas, que son la parte del modelo que envejece.
3. Reentrenar por evidencia (drift de 0.1 o más en alguna variable o caída del MAE) y promover un Challenger solo con criterios de aceptación y reversión definidos.
4. Vigilar P95 de `/predict` como señal de saturación, porque la saturación no produjo errores.
5. Separar MLflow del servicio de inferencia y probar `OMP_NUM_THREADS=1` o `--workers 2` antes de reducir el tamaño de la instancia; ninguno de estos cambios se ha medido todavía.

## 6. Información para el Model Card

| Tema | Valor | Fuente |
|---|---|---|
| Desempeño en marzo de 2025 | MAE 0.8407, RMSE 1.4370 (Champion v1) | `ab_testing_analysis.md` |
| Candidato | Challenger v2, MAE 0.8243, no promovido | `ab_testing_analysis.md` |
| Drift | Sin drift global; 44 estaciones con cambio de demanda mayor a 20 % | `drift_monitoring.md` |
| Limitación conocida | Medias por estación congeladas con 2024; estaciones 104 y 310 sin salidas en 2025, a las que el modelo seguirá prediciendo demanda positiva | `drift_monitoring.md` |
| Capacidad | Unas 100 req/s con P95 menor a 80 ms; máximo cerca de 145 req/s (4 CPU, 3.8 GiB) | `load_testing_analysis.md` |
| Costo | 0.4157 USD por millón de inferencias en el punto de operación | `finops.md` |
