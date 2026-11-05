"""Inspección del modelo P1 registrado en MLflow."""

import os

import mlflow
from mlflow import MlflowClient
from mlflow.models import Model

MODEL_NAME = "mibici-equipo15-p1-demanda"
MODEL_ALIAS = "champion"

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_registry_uri(TRACKING_URI)

client = MlflowClient()

print("MLflow Tracking URI:", TRACKING_URI)
print("Registered Model:", MODEL_NAME)

# Consulto todas las versiones registradas del modelo.
versions = client.search_model_versions(f"name = '{MODEL_NAME}'")

if not versions:
    raise RuntimeError(f"No se encontraron versiones de {MODEL_NAME}")

print("\nVERSIONES REGISTRADAS")

for version in versions:
    print("Version:", version.version, "| Status:", version.status, "| Run ID:", version.run_id)

# Recupero la versión que apunta el alias champion.
champion = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)

if champion.status != "READY":
    raise RuntimeError(f"El modelo champion no está READY: {champion.status}")

model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

print("\nMODELO CHAMPION")
print("Name:", champion.name)
print("Version:", champion.version)
print("Status:", champion.status)
print("Run ID:", champion.run_id)
print("Source:", champion.source)
print("Model URI:", model_uri)

# Leo solo el MLmodel para obtener la firma sin cargar el modelo.
model_metadata = Model.load(model_uri)
signature = model_metadata.signature

if signature is None:
    raise RuntimeError("El modelo registrado no tiene firma.")

if signature.inputs is None or signature.outputs is None:
    raise RuntimeError("La firma no contiene entradas o salidas.")

print("\nFIRMA DEL MODELO")
print(signature)

print("\nVARIABLES DE ENTRADA")

for index, column in enumerate(signature.inputs.inputs, start=1):
    print(index, column.name, column.type, "Required:", column.required)

print("\nVALIDACIÓN DEL REGISTRY COMPLETADA")
