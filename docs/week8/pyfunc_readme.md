# Semana 8 - Servicio de inferencia MLflow pyfunc

## Objetivo

`mibici/api/model_service.py` recupera el modelo P1 del Model Registry, lo mantiene
en memoria y ejecuta predicciones con MLflow pyfunc. La API FastAPI
(`mibici/api/main.py`) lo usa para responder `POST /predict`. El servicio no expone
HTTP ni modifica el Registry.

## Modelo utilizado

| Dato | Valor |
|---|---|
| Modelo registrado | `mibici-equipo15-p1-demanda` |
| Alias | `champion` |
| Versión | `1` |
| Run ID | `6e3429f222a0407aa159dd9f46699365` |
| Algoritmo | `HistGradientBoostingRegressor`, serializado con `skops` |
| Salida | Viajes iniciados en una estación durante una hora |

El contrato completo está en `docs/week8/model_registry_contract.md`.

## Recuperación desde el Model Registry

`load_model()` se ejecuta una sola vez, al iniciar FastAPI:

1. Configura `MLFLOW_TRACKING_URI` (por defecto `http://127.0.0.1:5000`, desde `mibici/config.py`).
2. Resuelve el alias `champion` a un número de versión con `MlflowClient.get_model_version_by_alias`.
3. Carga esa versión exacta con `mlflow.pyfunc.load_model("models:/mibici-equipo15-p1-demanda/<versión>")`.
4. Lee los nombres de las columnas desde la firma del modelo.

Se carga la versión resuelta y no el alias directamente para que `model_version` sea
siempre la versión que está en memoria, aunque el alias cambie después.
Si el Registry no responde o el modelo no tiene firma, `load_model()` lanza
`ModelNotReadyError` con el error original como causa.

El servidor MLflow debe levantarse con `--serve-artifacts`, porque los artefactos se
registraron con el esquema `mlflow-artifacts:/`.

## Variables de entrada

`predict()` recibe una lista de diccionarios con las 8 variables de la firma:
`hour_of_day`, `weekday`, `fin_semana`, `salidas_lag_1h`, `salidas_lag_24h`,
`salidas_lag_168h`, `media_estacion` y `media_estacion_dia_hora`. Sus rangos están en
el contrato del Registry.

Antes de llamar al modelo, el servicio:

- Rechaza registros con columnas faltantes o no esperadas. El modelo ignora en silencio las columnas extra, así que se validan aquí.
- Rechaza valores no numéricos, `None`, booleanos, `NaN` e infinitos. El modelo predice con `NaN` sin error, pero el entrenamiento descartó las filas con nulos.
- Convierte todos los valores a `float64` y ordena las columnas como en la firma. La firma exige `double` y MLflow no convierte `int64` automáticamente.

## Formato de salida

`predict()` devuelve una lista de `float` de Python, una predicción por registro y en
el mismo orden. Ejemplo con la entrada documentada en el contrato:

```python
service.predict([{
    "hour_of_day": 8, "weekday": 2, "fin_semana": 0,
    "salidas_lag_1h": 3, "salidas_lag_24h": 4, "salidas_lag_168h": 5,
    "media_estacion": 2.1, "media_estacion_dia_hora": 3.4,
}])
# [3.694163560787005]
```

Las predicciones negativas se recortan a 0, igual que en la evaluación del
entrenamiento (`np.clip(pred, 0, None)` en `p1_p2_candidates.py`), porque la demanda
no puede ser negativa. Así la salida es consistente con el MAE y RMSE reportados.
El servicio también valida que la salida tenga una predicción por registro y que no
contenga `NaN` ni infinitos.

El formato JSON de la respuesta lo define la API, no el servicio.

## Manejo de errores

| Excepción | Hereda de | Cuándo ocurre | Código HTTP sugerido |
|---|---|---|---|
| `ModelNotReadyError` | `RuntimeError` | El Registry no responde, el modelo no tiene firma o se llama a `predict()` sin modelo cargado | 503 |
| `ModelInputError` | `ValueError` | Lista vacía, columnas faltantes o extra, valores no numéricos o no finitos | 422 |
| `ModelPredictionError` | `RuntimeError` | Falla `model.predict()` o la salida no es válida | 500 |

Cuando la falla viene de MLflow o del modelo, la excepción conserva el error original
en `__cause__`. `ModelInputError` hereda de `ValueError` para que el `main.py` actual
la capture; para responder 422 en lugar de 500, la API debe capturarla antes que
`ValueError`.

## Pruebas automatizadas

`tests/test_model_service.py` contiene 26 pruebas que no requieren el servidor MLflow:

- 25 usan un modelo simulado para cubrir la carga por alias y versión, el Registry no disponible, el modelo sin firma, la predicción válida y en lote, el orden y tipo de las columnas, el recorte a 0, la salida en tipos de Python y cada caso de error de la sección anterior.
- 1 registra un modelo `LinearRegression` en un Registry SQLite temporal, le asigna el alias `champion` y lo carga con el servicio. Comprueba el flujo real de MLflow pyfunc, incluida la conversión de enteros a `float64`.

```bash
python -m pytest tests/test_model_service.py -v
```

Resultado: 26 passed (`reports/week8/evidence/pyfunc/01_unit_tests.png`).

## Prueba con el modelo real

`scripts/week8/test_pyfunc_inference.py` carga el modelo `champion` del Registry del
equipo, predice la entrada documentada en el contrato y verifica que el resultado
coincida con el valor registrado por Jorge (`3.69416356`). También ejecuta la
predicción dos veces para comprobar que es reproducible, predice un lote de 3
registros y comprueba que se rechace un registro incompleto.

Con el servidor MLflow levantado:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
python scripts/week8/test_pyfunc_inference.py
```

Resultado: versión 1, run `6e3429f222a0407aa159dd9f46699365`, predicción
`3.694163560787005`, igual a la del contrato
(`reports/week8/evidence/pyfunc/04_real_inference.png`).
