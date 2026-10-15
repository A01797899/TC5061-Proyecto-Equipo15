from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Dejo la semilla aqui para que todos la importen y no se repita en el codigo.
RANDOM_SEED = 42
TEST_SIZE = 0.2


# Periodo de datos

# Uso 2024 completo para tener la estacionalidad entera; falta confirmarlo.
ANIO_DATOS = 2024
MESES = list(range(1, 13))
FECHA_INICIO = "2024-01-01"
FECHA_FIN = "2024-12-31"

# Formato de los archivos

# Intento UTF-8 primero porque es estricto; latin-1 nunca falla y taparia el error.
ENCODINGS_CANDIDATOS = ["utf-8", "iso-8859-1"]

# Dejo lo que observe el 15/oct como referencia; la lectura real lo detecta sola.
ENCODINGS_OBSERVADOS_2024 = {7: "utf-8"}

FORMATO_FECHA = "%Y-%m-%d %H:%M:%S"


# Fuentes de datos
URL_BASE = "https://www.mibici.net/site/assets/files"

# Guardo el id de carpeta de cada mes porque MiBici no sigue ningun patron.
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

# El nombre del archivo si es predecible, asi que lo armo con el id.
URLS_VIAJES = {
    mes: f"{URL_BASE}/{id_archivo}/datos_abiertos_{ANIO_DATOS}_{mes:02d}.csv"
    for mes, id_archivo in IDS_VIAJES.items()
}

URL_NOMENCLATURA = f"{URL_BASE}/1118/nomenclatura_2026_09.csv"

# Columnas de los viajes

COL_VIAJE_ID = "Viaje_Id"
COL_USUARIO_ID = "Usuario_Id"
COL_GENERO = "Genero"
COL_ANIO_NACIMIENTO = "Año_de_nacimiento"
COL_INICIO = "Inicio_del_viaje"
COL_FIN = "Fin_del_viaje"
COL_ORIGEN_ID = "Origen_Id"
COL_DESTINO_ID = "Destino_Id"

COLUMNAS_VIAJES = [
    COL_VIAJE_ID,
    COL_USUARIO_ID,
    COL_GENERO,
    COL_ANIO_NACIMIENTO,
    COL_INICIO,
    COL_FIN,
    COL_ORIGEN_ID,
    COL_DESTINO_ID,
]

# Normalizo a snake_case para no depender de como venga el encabezado original.
RENOMBRAR_COLUMNAS = {
    COL_VIAJE_ID: "viaje_id",
    COL_USUARIO_ID: "usuario_id",
    COL_GENERO: "genero",
    COL_ANIO_NACIMIENTO: "anio_nacimiento",
    COL_INICIO: "inicio_viaje",
    COL_FIN: "fin_viaje",
    COL_ORIGEN_ID: "origen_id",
    COL_DESTINO_ID: "destino_id",
}

COLUMNAS_FECHA = ["inicio_viaje", "fin_viaje"]


# Estas vienen en ingles, a diferencia de las de viajes.
COL_EST_ID = "id"
COL_EST_NOMBRE = "name"
COL_EST_LAT = "latitude"
COL_EST_LON = "longitude"
COL_EST_CAPACIDAD = "dpcapacity"

RENOMBRAR_ESTACIONES = {
    COL_EST_ID: "estacion_id",
    COL_EST_NOMBRE: "nombre",
    COL_EST_LAT: "latitud",
    COL_EST_LON: "longitud",
    COL_EST_CAPACIDAD: "capacidad",
}

# Archivos que genera la ingesta
VIAJES_PARQUET = PROCESSED_DATA_DIR / f"viajes_{ANIO_DATOS}.parquet"
ESTACIONES_PARQUET = PROCESSED_DATA_DIR / "estaciones.parquet"

# Versiono este manifiesto en Git con el conteo y el hash de cada archivo; los datos no
MANIFIESTO_DATOS = DATA_DIR / "manifiesto.json"

# Reglas de limpieza

# Descarto los viajes de menos de un minuto porque son falsos arranques
DURACION_MIN_MINUTOS = 1

# Descarto los de mas de tres horas porque son bicis no devueltas
DURACION_MAX_MINUTOS = 180

# Acoto la edad para quitar años de nacimiento capturados mal
EDAD_MIN = 10
EDAD_MAX = 90

# MLflow
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

# Nombro el experimento con el numero de equipo, como lo pide la actividad
MLFLOW_EXPERIMENT = "mibici-equipo15"

# Agregacion para P1 y P2 (propuesta)

# Propongo contar los viajes por estacion cada hora; falta acordarlo
FRECUENCIA_AGREGACION = "1h"

# Aproximo el riesgo de P2 con el flujo neto contra la capacidad, no hay inventario
UMBRAL_RIESGO_CAPACIDAD = 0.9

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
