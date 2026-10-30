"""Ingesta reproducible de los viajes de MiBici.

Descarga los CSV mensuales y la nomenclatura de estaciones, normaliza encabezados,
codificacion, formatos de fecha y tipos, y guarda el resultado en Parquet. Al final
escribe un manifiesto con el conteo y el hash SHA-256 de cada archivo; ese manifiesto
si se versiona en Git y su ``version_datos`` es la que se registra en MLflow.

Uso desde la raiz del repositorio::

    python -m mibici.dataset
    python -m mibici.dataset --forzar-descarga
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from urllib.parse import urlparse
import urllib.request

from loguru import logger
import pandas as pd
from tqdm import tqdm
import typer

from mibici.config import (
    ANIO_DATOS,
    ENCODING_POR_MES,
    ESTACIONES_PARQUET,
    FORMATO_FECHA_POR_MES,
    MANIFIESTO_DATOS,
    PROJ_ROOT,
    RAW_DATA_DIR,
    RENOMBRAR_COLUMNAS,
    RENOMBRAR_ESTACIONES,
    URL_NOMENCLATURA,
    URLS_VIAJES,
    VIAJES_PARQUET,
)

app = typer.Typer()

# Se intenta UTF-8 primero porque es estricto; latin1 nunca falla y taparia el error.
ENCODINGS_CANDIDATOS = ("utf-8", "latin1")

COLUMNAS_FECHA = ("inicio_viaje", "fin_viaje")

# Tipos finales; se usan enteros nullable porque genero y anio traen vacios.
TIPOS_VIAJES = {
    "viaje_id": "Int64",
    "usuario_id": "Int64",
    "anio_nacimiento": "Int16",
    "origen_id": "Int32",
    "destino_id": "Int32",
}

TIPOS_ESTACIONES = {
    "estacion_id": "Int32",
    "latitud": "float64",
    "longitud": "float64",
    "capacidad": "Int16",
}


# Descarga y huellas


def descargar(url: str, destino: Path, forzar: bool = False) -> Path:
    """Descarga ``url`` en ``destino`` si no existe (o si ``forzar``).

    Escribe primero a un temporal y luego lo mueve, para que una descarga
    interrumpida no deje un CSV a medias que parezca valido.
    """
    if destino.exists() and not forzar:
        logger.info(f"Ya existe {destino.name}, se omite la descarga")
        return destino

    # https en la ingesta; file solo para los CSV locales de las pruebas.
    esquema = urlparse(url).scheme
    if esquema not in {"http", "https", "file"}:
        raise ValueError(f"Esquema de descarga no permitido: {esquema or 'vacío'}")

    destino.parent.mkdir(parents=True, exist_ok=True)
    peticion = urllib.request.Request(url, headers={"User-Agent": "mibici-equipo15"})
    with tempfile.NamedTemporaryFile(dir=destino.parent, delete=False) as tmp:
        temporal = Path(tmp.name)
    try:
        with (
            urllib.request.urlopen(peticion, timeout=120) as respuesta,  # nosec B310
            temporal.open("wb") as f,
        ):
            shutil.copyfileobj(respuesta, f)
        if temporal.stat().st_size == 0:
            raise ValueError(f"La descarga de {url} llego vacia")
        temporal.replace(destino)
    finally:
        # Si algo fallo, se borra el temporal para no dejar basura en data/raw.
        temporal.unlink(missing_ok=True)
    logger.info(f"Descargado {destino.name}")
    return destino


def sha256_archivo(ruta: Path, bloque: int = 1024 * 1024) -> str:
    """Hash SHA-256 del contenido de ``ruta``, leido por bloques."""
    h = hashlib.sha256()
    with ruta.open("rb") as f:
        while datos := f.read(bloque):
            h.update(datos)
    return h.hexdigest()


def detectar_encoding(ruta: Path) -> str:
    """Primer encoding de ``ENCODINGS_CANDIDATOS`` que decodifica todo el archivo."""
    contenido = ruta.read_bytes()
    for encoding in ENCODINGS_CANDIDATOS:
        try:
            contenido.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    raise ValueError(f"{ruta.name} no se pudo decodificar con {ENCODINGS_CANDIDATOS}")


# Normalizacion


def normalizar_encabezados(columnas: list[str]) -> list[str]:
    """Devuelve los nombres en snake_case validando el esquema de viajes."""
    columnas = [c.strip().strip('"') for c in columnas]
    esperadas = list(RENOMBRAR_COLUMNAS)
    if len(columnas) != len(esperadas):
        raise ValueError(f"Se esperaban {len(esperadas)} columnas y llegaron {columnas}")

    nombres = []
    for recibida, esperada in zip(columnas, esperadas):
        nombre = RENOMBRAR_COLUMNAS[esperada]
        coincide = recibida == esperada or (
            nombre == "anio_nacimiento" and recibida.lower().endswith("_nacimiento")
        )
        if not coincide:
            raise ValueError(f"Columna inesperada {recibida!r}; se esperaba {esperada!r}")
        nombres.append(nombre)
    return nombres


def parsear_fechas(serie: pd.Series, formato: str) -> pd.Series:
    """Convierte texto a datetime con el formato del mes (``FORMATO_FECHA_POR_MES``).

    Se usan formatos explicitos en lugar de que pandas adivine, porque ``01/09/2024``
    se confundiria con 9 de enero. Lo que no coincide queda como NaT y se reporta.
    """
    return pd.to_datetime(serie, format=formato, errors="coerce")


def leer_viajes(ruta: Path, encoding: str, formato_fecha: str) -> pd.DataFrame:
    """Lee un CSV mensual y lo devuelve con nombres, fechas y tipos normalizados."""
    df = pd.read_csv(ruta, encoding=encoding, dtype=str)
    df.columns = normalizar_encabezados(list(df.columns))

    for columna in COLUMNAS_FECHA:
        df[columna] = parsear_fechas(df[columna], formato_fecha)
    for columna, tipo in TIPOS_VIAJES.items():
        df[columna] = pd.to_numeric(df[columna], errors="coerce").astype(tipo)
    df["genero"] = df["genero"].astype("category")
    return df


def leer_estaciones(ruta: Path) -> pd.DataFrame:
    """Lee la nomenclatura de estaciones con nombres en espanol y tipos numericos."""
    df = pd.read_csv(ruta, encoding=detectar_encoding(ruta), dtype=str)
    df = df.rename(columns=RENOMBRAR_ESTACIONES)[list(RENOMBRAR_ESTACIONES.values())]
    for columna, tipo in TIPOS_ESTACIONES.items():
        df[columna] = pd.to_numeric(df[columna], errors="coerce").astype(tipo)
    return df


# Manifiesto


def ruta_relativa(ruta: Path) -> str:
    """Ruta relativa a la raiz del proyecto, para que el manifiesto no dependa de la maquina."""
    return ruta.resolve().relative_to(PROJ_ROOT).as_posix()


def calcular_version(fuentes: list[dict]) -> str:
    """Version de los datos: hash de los hashes de los archivos de origen.

    Solo depende de los CSV descargados, asi que es igual en cualquier computadora
    que haya bajado los mismos archivos.
    """
    huellas = "\n".join(f"{f['archivo']}:{f['sha256']}" for f in fuentes)
    return hashlib.sha256(huellas.encode()).hexdigest()[:12]


def cargar_manifiesto(ruta: Path = MANIFIESTO_DATOS) -> dict:
    """Lee el manifiesto versionado en Git."""
    return json.loads(ruta.read_text(encoding="utf-8"))


def version_datos(ruta: Path = MANIFIESTO_DATOS) -> str:
    """Version de datos para registrar como parametro o etiqueta en MLflow."""
    return cargar_manifiesto(ruta)["version_datos"]


def advertir_cambios(fuentes: list[dict], ruta: Path = MANIFIESTO_DATOS) -> None:
    """Avisa si algun CSV cambio respecto al manifiesto que esta en Git."""
    if not ruta.exists():
        return
    anteriores = {f["archivo"]: f["sha256"] for f in cargar_manifiesto(ruta)["fuentes"]}
    for fuente in fuentes:
        anterior = anteriores.get(fuente["archivo"])
        if anterior and anterior != fuente["sha256"]:
            logger.warning(f"{fuente['archivo']} cambio respecto al manifiesto versionado")


# Proceso completo


@app.command()
def main(
    forzar_descarga: bool = typer.Option(False, help="Vuelve a descargar aunque ya existan."),
):
    """Descarga, normaliza y guarda los viajes y estaciones, y escribe el manifiesto."""
    fuentes = []
    viajes = []

    for mes, url in tqdm(URLS_VIAJES.items(), desc="Meses"):
        ruta = descargar(url, RAW_DATA_DIR / url.rsplit("/", 1)[-1], forzar_descarga)
        encoding = detectar_encoding(ruta)
        if encoding != ENCODING_POR_MES[mes]:
            logger.warning(f"{ruta.name}: config dice {ENCODING_POR_MES[mes]}, se usa {encoding}")
        formato_fecha = FORMATO_FECHA_POR_MES[mes]
        df = leer_viajes(ruta, encoding, formato_fecha)

        sin_fecha = int(df["inicio_viaje"].isna().sum())
        # Si falla mas del 1% casi seguro el formato del mes en config esta mal
        if sin_fecha > 0.01 * len(df):
            raise ValueError(f"{ruta.name}: {sin_fecha} fechas no coinciden con {formato_fecha}")
        fuera_de_mes = int(
            ((df["inicio_viaje"].dt.year != ANIO_DATOS) | (df["inicio_viaje"].dt.month != mes))
            .fillna(False)
            .sum()
        )
        fuentes.append(
            {
                "archivo": ruta_relativa(ruta),
                "url": url,
                "bytes": ruta.stat().st_size,
                "sha256": sha256_archivo(ruta),
                "encoding": encoding,
                "formato_fecha": formato_fecha,
                "registros": len(df),
                "fechas_invalidas": sin_fecha,
                "inicios_fuera_del_mes": fuera_de_mes,
            }
        )
        viajes.append(df)

    advertir_cambios(fuentes)

    viajes = pd.concat(viajes, ignore_index=True)
    sin_inicio = viajes["inicio_viaje"].isna() | viajes["origen_id"].isna()
    duplicados = viajes["viaje_id"].duplicated()
    viajes = viajes[~sin_inicio & ~duplicados].sort_values("inicio_viaje", kind="stable")
    viajes = viajes.reset_index(drop=True)
    logger.info(
        f"{len(viajes):,} viajes limpios; descartados {int(sin_inicio.sum())} sin inicio u "
        f"origen y {int(duplicados.sum())} duplicados"
    )

    ruta_nomenclatura = descargar(
        URL_NOMENCLATURA, RAW_DATA_DIR / URL_NOMENCLATURA.rsplit("/", 1)[-1], forzar_descarga
    )
    estaciones = leer_estaciones(ruta_nomenclatura)
    fuentes.append(
        {
            "archivo": ruta_relativa(ruta_nomenclatura),
            "url": URL_NOMENCLATURA,
            "bytes": ruta_nomenclatura.stat().st_size,
            "sha256": sha256_archivo(ruta_nomenclatura),
            "encoding": detectar_encoding(ruta_nomenclatura),
            "registros": len(estaciones),
        }
    )

    VIAJES_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    viajes.to_parquet(VIAJES_PARQUET, index=False)
    estaciones.to_parquet(ESTACIONES_PARQUET, index=False)

    manifiesto = {
        "anio": ANIO_DATOS,
        "version_datos": calcular_version(fuentes),
        "fuentes": fuentes,
        "salidas": [
            {"archivo": ruta_relativa(VIAJES_PARQUET), "registros": len(viajes)},
            {"archivo": ruta_relativa(ESTACIONES_PARQUET), "registros": len(estaciones)},
        ],
        "descartados": {
            "sin_inicio_u_origen": int(sin_inicio.sum()),
            "viaje_id_duplicado": int(duplicados.sum()),
        },
    }
    MANIFIESTO_DATOS.write_text(
        json.dumps(manifiesto, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.success(f"Ingesta completa; version de datos {manifiesto['version_datos']}")


if __name__ == "__main__":
    app()
