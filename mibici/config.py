"""Configuración compartida de MiBici; no descargar/importar datos al importar."""

import os

from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RANDOM_SEED = 42
ANIO_DATOS = 2024
MESES = tuple(range(1, 13))
TRAIN_END = "2024-09-01"  # exclusivo; train enero-agosto
VAL_END = "2024-11-01"  # exclusivo; val septiembre-octubre
TEST_END = "2025-01-01"  # exclusivo; test noviembre-diciembre
FECHA_INICIO = "2024-01-01"
FECHA_FIN = "2024-12-31"
FRECUENCIA_AGREGACION = "1h"
UMBRAL_RIESGO_CAPACIDAD = 0.25  # proxy P2, no inventario real
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000"
)
MLFLOW_EXPERIMENT = "mibici-equipo15"
URL_BASE = "https://www.mibici.net/site/assets/files"
# Referencias propuestas por Jorge (PR #1). Verificar accesibilidad/identidad de cada URL.
IDS_VIAJES = {
    1: 61450,
    2: 77251,
    3: 77252,
    4: 80028,
    5: 81663,
    6: 83901,
    7: 85500,
    8: 87665,
    9: 89736,
    10: 91758,
    11: 93516,
    12: 94962,
}
URLS_VIAJES = {
    m: f"{URL_BASE}/{ident}/datos_abiertos_{ANIO_DATOS}_{m:02d}.csv"
    for m, ident in IDS_VIAJES.items()
}
URL_NOMENCLATURA = f"{URL_BASE}/1118/nomenclatura_2026_09.csv"
COLUMNS_VIAJES = (
    "viaje_id",
    "usuario_id",
    "genero",
    "anio_nacimiento",
    "inicio_viaje",
    "fin_viaje",
    "origen_id",
    "destino_id",
)
RENOMBRAR_COLUMNAS = {
    "Viaje_Id": "viaje_id",
    "Usuario_Id": "usuario_id",
    "Genero": "genero",
    "Año_de_nacimiento": "anio_nacimiento",
    "Inicio_del_viaje": "inicio_viaje",
    "Fin_del_viaje": "fin_viaje",
    "Origen_Id": "origen_id",
    "Destino_Id": "destino_id",
}
RENOMBRAR_ESTACIONES = {
    "id": "estacion_id",
    "name": "nombre",
    "latitude": "latitud",
    "longitude": "longitud",
    "dpcapacity": "capacidad",
}
FORMATO_FECHA_POR_MES = {
    m: ("%d/%m/%Y %H:%M" if m in (9, 10) else "%Y-%m-%d %H:%M:%S") for m in MESES
}
ENCODING_POR_MES = {m: ("utf-8" if m == 7 else "latin1") for m in MESES}
VIAJES_PARQUET = PROCESSED_DATA_DIR / "viajes_2024.parquet"
ESTACIONES_PARQUET = PROCESSED_DATA_DIR / "estaciones.parquet"
MANIFIESTO_DATOS = DATA_DIR / "manifiesto.json"
# Los umbrales de calidad son diagnósticos: NO se filtran viajes automáticamente.
DURACION_MIN_MINUTOS = 1
DURACION_MAX_MINUTOS = 180
EDAD_MIN, EDAD_MAX = 10, 90

# Configuración para MLflow Project
DEFAULT_N_ESTIMATORS = 300
DEFAULT_MAX_DEPTH = None

# Parámetros de P3
P3_MIN_DEPARTURES = 100
P3_K_MIN = 2
P3_K_MAX = 8
