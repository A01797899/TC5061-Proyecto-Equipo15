# Estrategia del Challenger P1

Actividad 3.3. Rama `feature/week9-challenger-JLMM`. José Luis Mendoza Mora.

El Challenger es el mismo `HistGradientBoostingRegressor` del Champion: `max_iter=300`, `learning_rate=0.1`, `max_depth=None`, `early_stopping=False` y `random_state=42`. El cambio es la ventana de entrenamiento, no la familia ni los hiperparámetros.

El ajuste usa solo `data/processed/week9/train_alternative.parquet` (1 ene – 28 feb 2025, 512,592 filas). Las ocho columnas de `P1_FEATURES` ya traen `media_estacion` y `media_estacion_dia_hora` congeladas con el entrenamiento de 2024. No se recalculan, para que la firma siga siendo la del Champion.

`data/processed/week9/evaluation.parquet` (marzo 2025, 269,328 filas) no se abre durante el entrenamiento. Marzo no entra al `fit`.

La corrida se registra en el experimento `mibici-equipo15`. El alias `champion` permanece en la versión 1. Este carril no lo mueve.
