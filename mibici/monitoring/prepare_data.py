"""Preparo los viajes y las variables de P1 de referencia (2024) y actuales (enero-marzo 2025).

Los CSV de 2025 se leen con las mismas funciones de ``mibici.dataset``, pero el encoding y el
formato de fecha se detectan por archivo en lugar de usar las reglas fijas de 2024. Las 8
variables de P1 se construyen con las mismas funciones del entrenamiento del Champion. La
ingesta de 2024 y ``data/manifiesto.json`` no se modifican.

Uso desde la raíz del repositorio::

    python -m mibici.monitoring.prepare_data
    python -m mibici.monitoring.prepare_data --etapa p1
    python -m mibici.monitoring.prepare_data --etapa particiones
"""

from __future__ import annotations

import argparse
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mibici.api.schemas import P1_FEATURES
from mibici.config import (
    EXTERNAL_DATA_DIR,
    FECHA_INICIO,
    MODEL_PRODUCTION_ALIAS,
    PROCESSED_DATA_DIR,
    PROJ_ROOT,
    REGISTERED_MODELS,
    RENOMBRAR_COLUMNAS,
    REPORTS_DIR,
    TRAIN_END,
    VIAJES_PARQUET,
)
from mibici.dataset import (
    detectar_encoding,
    leer_viajes,
    ruta_relativa,
    sha256_archivo,
    version_datos,
)
from mibici.modeling import p1_baseline
from mibici.modeling.p1_p2_candidates import agregar_historicos, agregar_rezagos, asignar_split

ANIO_ACTUAL = 2025
MESES_ACTUALES = (1, 2, 3)
URLS_2025 = {
    1: "https://mibici.net/site/assets/files/97653/datos_abiertos_2025_01.csv",
    2: "https://mibici.net/site/assets/files/99879/datos_abiertos_2025_02.csv",
    3: "https://mibici.net/site/assets/files/101986/datos_abiertos_2025_03.csv",
}
DIR_2025 = EXTERNAL_DATA_DIR / "week9_2025"
SALIDA = PROCESSED_DATA_DIR / "week9"
VIAJES_2025 = "viajes_2025_q1.parquet"
REFERENCE_P1 = "reference_p1.parquet"
CURRENT_P1 = "current_p1.parquet"
TRAIN_ALTERNATIVE = "train_alternative.parquet"
EVALUATION = "evaluation.parquet"
INICIO_ACTUAL = "2025-01-01"
# Enero-febrero entrena al Challenger y marzo queda reservado para comparar ambos modelos.
INICIO_EVALUACION = "2025-03-01"
FIN_ACTUAL = "2025-04-01"
FORMATOS_FECHA = ("%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M")
COLUMNAS_2024 = list(RENOMBRAR_COLUMNAS)
# Mismo umbral que mibici.dataset: más del 1% indica formato o mes equivocado.
MAX_INVALIDAS = 0.01
FEATURES_P1 = list(P1_FEATURES)
COLUMNAS_P1 = ["station", "hour", "y", *FEATURES_P1]
NO_NEGATIVAS = FEATURES_P1[3:]
MANIFIESTO = REPORTS_DIR / "week9" / "data_manifest.json"
CHAMPION = {
    "modelo": REGISTERED_MODELS["p1"],
    "alias": MODEL_PRODUCTION_ALIAS,
    "version": "1",
    "run_id": "6e3429f222a0407aa159dd9f46699365",
    "uri": f"models:/{REGISTERED_MODELS['p1']}@{MODEL_PRODUCTION_ALIAS}",
}
REGLAS = [
    (
        "Se descartan los viajes sin inicio u origen y los que empiezan fuera de su mes; las "
        "demás anomalías se reportan en data_quality.json y se conservan."
    ),
    (
        "El panel usa las estaciones con viajes en el entrenamiento del Champion; las nuevas de "
        "2025 se reportan aparte y las horas sin viajes valen 0."
    ),
    (
        "Los rezagos de 1, 24 y 168 horas salen de un panel continuo desde 2024, así que enero "
        "de 2025 usa diciembre de 2024."
    ),
    "media_estacion y media_estacion_dia_hora se calculan solo con el entrenamiento de 2024.",
    "La referencia descarta las primeras 168 horas sin rezagos completos, igual que el Champion.",
    "Las particiones de 2025 son por fecha: enero-febrero para entrenar y marzo para evaluar.",
]


def nombre_archivo(ruta: Path) -> str:
    """Uso la ruta relativa al proyecto y, si el archivo está fuera, solo su nombre."""
    try:
        return ruta_relativa(ruta)
    except ValueError:
        return ruta.name


def detectar_formato_fecha(ruta: Path, encoding: str, filas: int = 1000) -> str:
    """Elijo el primer formato que interpreta todas las fechas de inicio de la muestra."""
    muestra = pd.read_csv(ruta, encoding=encoding, dtype=str, nrows=filas).iloc[:, 4].dropna()
    for formato in FORMATOS_FECHA:
        if pd.to_datetime(muestra, format=formato, errors="coerce").notna().all():
            return formato
    raise ValueError(f"{ruta.name}: ninguna fecha coincide con {FORMATOS_FECHA}")


def leer_mes(ruta: Path, mes: int, anio: int = ANIO_ACTUAL) -> tuple[pd.DataFrame, dict]:
    """Leo un CSV mensual con leer_viajes() y valido que sus viajes sean de ese mes."""
    encoding = detectar_encoding(ruta)
    formato = detectar_formato_fecha(ruta, encoding)
    encabezados = pd.read_csv(ruta, encoding=encoding, nrows=0).columns
    encabezados = [c.strip().strip('"') for c in encabezados]
    viajes = leer_viajes(ruta, encoding, formato)

    inicio = viajes["inicio_viaje"]
    invalidas = int(inicio.isna().sum())
    fuera = int((inicio.notna() & ((inicio.dt.year != anio) | (inicio.dt.month != mes))).sum())
    if max(invalidas, fuera) > MAX_INVALIDAS * len(viajes):
        raise ValueError(
            f"{ruta.name} no parece de {anio}-{mes:02d}: "
            f"{invalidas} fechas inválidas y {fuera} inicios fuera del mes"
        )
    info = {
        "archivo": nombre_archivo(ruta),
        "url": URLS_2025.get(mes),
        "bytes": ruta.stat().st_size,
        "sha256": sha256_archivo(ruta),
        "encoding": encoding,
        "formato_fecha": formato,
        "columnas_originales": encabezados,
        "coincide_esquema_2024": encabezados == COLUMNAS_2024,
        "registros": len(viajes),
        "fechas_invalidas": invalidas,
        "inicios_fuera_del_mes": fuera,
    }
    return viajes, info


def cargar_actual(
    directorio: Path = DIR_2025, meses: tuple[int, ...] = MESES_ACTUALES
) -> tuple[pd.DataFrame, list[dict]]:
    """Uno los meses de 2025 en orden cronológico y conservo la huella de cada archivo."""
    partes, fuentes = [], []
    for mes in meses:
        ruta = directorio / f"datos_abiertos_{ANIO_ACTUAL}_{mes:02d}.csv"
        if not ruta.exists():
            raise FileNotFoundError(f"Falta {ruta}; descárgalo de {URLS_2025.get(mes)}")
        viajes, info = leer_mes(ruta, mes)
        partes.append(viajes)
        fuentes.append(info)

    viajes = pd.concat(partes, ignore_index=True)
    viajes["genero"] = viajes["genero"].astype("category")
    viajes = viajes.sort_values(["inicio_viaje", "viaje_id"], kind="stable")
    return viajes.reset_index(drop=True), fuentes


def cargar_referencia(ruta: Path = VIAJES_PARQUET) -> pd.DataFrame:
    """Leo los viajes de 2024 que ya normalizó mibici.dataset, sin volver a escribirlos."""
    if not ruta.exists():
        raise FileNotFoundError(
            f"Falta {ruta}; genera la ingesta de 2024 con python -m mibici.dataset"
        )
    return pd.read_parquet(ruta)


def salidas_por_hora(viajes: pd.DataFrame) -> pd.DataFrame:
    """Cuento salidas por estación y hora con la regla de P1: inicio y origen válidos."""
    validos = viajes.dropna(subset=["inicio_viaje", "origen_id"])
    conteo = validos.assign(
        station=validos["origen_id"].astype("int64"),
        hour=validos["inicio_viaje"].dt.floor("h"),
    )
    return conteo.groupby(["station", "hour"]).size().rename("trips").reset_index()


def construir_panel_p1(
    referencia: pd.DataFrame, actual: pd.DataFrame, fin: str = FIN_ACTUAL
) -> pd.DataFrame:
    """Construyo el panel estación-hora de 2024 a ``fin`` con las 8 variables de P1."""
    columnas = ["inicio_viaje", "origen_id"]
    viajes = pd.concat([referencia[columnas], actual[columnas]], ignore_index=True)
    observados = salidas_por_hora(viajes)
    hora = observados["hour"]
    en_train = (hora >= pd.Timestamp(FECHA_INICIO)) & (hora < pd.Timestamp(TRAIN_END))
    estaciones = pd.Index(sorted(observados.loc[en_train, "station"].unique()), name="station")

    # Uso un solo panel continuo para que los rezagos de enero de 2025 lean diciembre de 2024.
    panel = asignar_split(p1_baseline.panel(observados, estaciones, FECHA_INICIO, fin))
    panel.loc[panel["hour"] >= pd.Timestamp(INICIO_ACTUAL), "split"] = "actual"
    panel = agregar_rezagos(panel.rename(columns={"y": "salidas"}), ["salidas"])
    # Las medias salen solo de train 2024, así que quedan congeladas como en el Champion.
    panel = agregar_historicos(panel.rename(columns={"salidas": "y"}), "media")
    panel[FEATURES_P1] = panel[FEATURES_P1].astype("float64")
    return panel


def validar_p1(df: pd.DataFrame, nombre: str) -> None:
    """Verifico que un dataset cumpla el contrato de entrada del Champion P1."""
    if list(df.columns) != COLUMNAS_P1:
        raise ValueError(f"{nombre}: columnas {list(df.columns)}, se esperaban {COLUMNAS_P1}")
    x = df[FEATURES_P1]
    if not (x.dtypes == "float64").all():
        raise ValueError(f"{nombre}: las 8 variables deben ser float64")
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError(f"{nombre}: hay variables nulas o infinitas")
    if df.duplicated(["station", "hour"]).any():
        raise ValueError(f"{nombre}: hay llaves estación-hora repetidas")
    reglas = {
        "hour_of_day": (x["hour_of_day"] == df["hour"].dt.hour).all(),
        "weekday": (x["weekday"] == df["hour"].dt.dayofweek).all(),
        "fin_semana": (x["fin_semana"] == (x["weekday"] >= 5)).all(),
        "no_negativas": (x[NO_NEGATIVAS] >= 0).all().all() and (df["y"] >= 0).all(),
    }
    fallas = [regla for regla, cumple in reglas.items() if not cumple]
    if fallas:
        raise ValueError(f"{nombre}: no cumple {fallas}")


def separar_p1(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separo la referencia (train 2024 sin nulos, como el Champion) y el periodo actual."""
    referencia = panel[panel["split"] == "train"].dropna(subset=FEATURES_P1)
    actual = panel[panel["split"] == "actual"]
    referencia = referencia[COLUMNAS_P1].reset_index(drop=True)
    actual = actual[COLUMNAS_P1].reset_index(drop=True)
    validar_p1(referencia, REFERENCE_P1)
    validar_p1(actual, CURRENT_P1)
    return referencia, actual


def particionar(actual: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separo por fecha el entrenamiento alternativo y la evaluación, sin mezclar horas."""
    corte = pd.Timestamp(INICIO_EVALUACION)
    entrenamiento = actual[actual["hour"] < corte].reset_index(drop=True)
    evaluacion = actual[actual["hour"] >= corte].reset_index(drop=True)
    if entrenamiento.empty or evaluacion.empty:
        raise ValueError(f"Alguna partición quedó vacía con el corte {INICIO_EVALUACION}")
    if entrenamiento["hour"].max() >= evaluacion["hour"].min():
        raise ValueError("Las particiones se superponen en el tiempo")
    validar_p1(entrenamiento, TRAIN_ALTERNATIVE)
    validar_p1(evaluacion, EVALUATION)
    return entrenamiento, evaluacion


def escribir_manifiesto(
    fuentes: list[dict],
    actual: pd.DataFrame,
    datasets: dict[str, pd.DataFrame],
    carpeta: Path,
    ruta: Path = MANIFIESTO,
) -> dict:
    """Guardo de dónde salen los datos, cuántas filas tiene cada dataset y con qué reglas."""
    try:
        version_proyecto = version("mibici")
    except PackageNotFoundError:
        version_proyecto = "sin instalar"
    origenes = {int(e) for e in actual["origen_id"].dropna().unique()}
    modelo = {int(e) for e in datasets[REFERENCE_P1]["station"].unique()}
    manifiesto = {
        "proyecto": {"nombre": "mibici", "version": version_proyecto},
        "commit": p1_baseline.git_sha(PROJ_ROOT),
        "fecha_ejecucion": datetime.now().isoformat(timespec="seconds"),
        "champion": CHAMPION,
        "referencia": {
            "periodo": {"inicio": FECHA_INICIO, "fin_exclusivo": TRAIN_END},
            "viajes": nombre_archivo(VIAJES_PARQUET),
            "manifiesto_2024": "data/manifiesto.json",
            "version_datos": version_datos(),
        },
        "actual": {
            "periodo": {"inicio": INICIO_ACTUAL, "fin_exclusivo": FIN_ACTUAL},
            "registros": len(actual),
            "archivos": fuentes,
        },
        "estaciones": {
            "modelo": len(modelo),
            "con_viajes_2025": len(origenes),
            "nuevas_2025": sorted(origenes - modelo),
            "del_modelo_sin_viajes_2025": sorted(modelo - origenes),
        },
        "columnas_entrada_p1": FEATURES_P1,
        "columnas_datasets": COLUMNAS_P1,
        "datasets": {
            nombre: {
                "ruta": nombre_archivo(carpeta / nombre),
                "filas": len(df),
                "estaciones": int(df["station"].nunique()),
                "hora_inicio": str(df["hour"].min()),
                "hora_fin": str(df["hour"].max()),
            }
            for nombre, df in datasets.items()
        },
        "reglas": REGLAS,
    }
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(manifiesto, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifiesto


def imprimir_periodo(nombre: str, viajes: pd.DataFrame) -> None:
    """Muestro registros, rango de fechas y columnas de un periodo."""
    inicio = viajes["inicio_viaje"]
    print(f"\n{nombre}")
    print(f"  Registros: {len(viajes):,}")
    print(f"  Inicio mínimo: {inicio.min()}  |  Inicio máximo: {inicio.max()}")
    print(f"  Columnas: {list(viajes.columns)}")


def imprimir_viajes(referencia: pd.DataFrame, actual: pd.DataFrame, fuentes: list[dict]) -> None:
    """Muestro los archivos de 2025 y los viajes de ambos periodos."""
    entrenamiento = referencia["inicio_viaje"].between(
        pd.Timestamp(FECHA_INICIO), pd.Timestamp(TRAIN_END), inclusive="left"
    )
    print("ARCHIVOS 2025")
    for f in fuentes:
        print(
            f"  {Path(f['archivo']).name} | {f['encoding']} | {f['formato_fecha']} | "
            f"{f['registros']:,} registros | fuera del mes: {f['inicios_fuera_del_mes']} | "
            f"esquema igual a 2024: {f['coincide_esquema_2024']}"
        )
    imprimir_periodo(
        "REFERENCIA (entrenamiento del Champion, ene-ago 2024)", referencia[entrenamiento]
    )
    imprimir_periodo("ACTUAL (ene-mar 2025)", actual)
    print(f"\nTipos:\n{actual.dtypes.to_string()}")


def imprimir_p1(referencia: pd.DataFrame, actual: pd.DataFrame) -> None:
    """Muestro tamaño, periodo y medias de las variables de P1 en ambos datasets."""
    print("\nVARIABLES P1 (contrato del Champion validado)")
    for nombre, df in ((REFERENCE_P1, referencia), (CURRENT_P1, actual)):
        print(
            f"  {nombre}: {len(df):,} filas | {df['station'].nunique()} estaciones | "
            f"{df['hour'].min()} a {df['hour'].max()} | nulos: {int(df.isna().sum().sum())}"
        )
    print(f"  Columnas: {COLUMNAS_P1}")
    print(f"  Tipos de las 8 variables: {sorted(set(map(str, referencia[FEATURES_P1].dtypes)))}")
    medias = pd.DataFrame(
        {
            "referencia": referencia[["y", *FEATURES_P1]].mean(),
            "actual": actual[["y", *FEATURES_P1]].mean(),
        }
    )
    print(f"\nMedias por variable:\n{medias.round(3).to_string()}")


def imprimir_particiones(entrenamiento: pd.DataFrame, evaluacion: pd.DataFrame) -> None:
    """Muestro el tamaño y las fechas de cada partición y la ausencia de superposición."""
    print("\nPARTICIONES TEMPORALES (contrato del Champion validado)")
    for nombre, df in ((TRAIN_ALTERNATIVE, entrenamiento), (EVALUATION, evaluacion)):
        print(
            f"  {nombre}: {len(df):,} filas | {df['station'].nunique()} estaciones | "
            f"{df['hour'].min()} a {df['hour'].max()} | media de y: {df['y'].mean():.3f}"
        )
    llaves = pd.concat([entrenamiento, evaluacion])[["station", "hour"]]
    print(
        f"  Última hora de entrenamiento < primera de evaluación: "
        f"{entrenamiento['hour'].max() < evaluacion['hour'].min()}"
    )
    print(f"  Llaves estación-hora repetidas entre particiones: {int(llaves.duplicated().sum())}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--directorio", type=Path, default=DIR_2025)
    parser.add_argument("--carpeta", type=Path, default=SALIDA)
    parser.add_argument("--manifiesto", type=Path, default=MANIFIESTO)
    parser.add_argument("--etapa", choices=("viajes", "p1", "particiones", "todo"), default="todo")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.carpeta.mkdir(parents=True, exist_ok=True)
    if args.etapa != "particiones":
        referencia = cargar_referencia()
        actual, fuentes = cargar_actual(args.directorio)
        actual.to_parquet(args.carpeta / VIAJES_2025, index=False)

    if args.etapa in ("viajes", "todo"):
        imprimir_viajes(referencia, actual, fuentes)
        print(f"\nGuardado: {nombre_archivo(args.carpeta / VIAJES_2025)}")
    if args.etapa in ("p1", "todo"):
        reference_p1, current_p1 = separar_p1(construir_panel_p1(referencia, actual))
        reference_p1.to_parquet(args.carpeta / REFERENCE_P1, index=False)
        current_p1.to_parquet(args.carpeta / CURRENT_P1, index=False)
        imprimir_p1(reference_p1, current_p1)
        for archivo in (REFERENCE_P1, CURRENT_P1):
            print(f"Guardado: {nombre_archivo(args.carpeta / archivo)}")
    if args.etapa in ("particiones", "todo"):
        if args.etapa == "particiones":
            ruta = args.carpeta / CURRENT_P1
            if not ruta.exists():
                raise FileNotFoundError(f"Falta {ruta}; corre antes la etapa p1")
            current_p1 = pd.read_parquet(ruta)
        entrenamiento, evaluacion = particionar(current_p1)
        entrenamiento.to_parquet(args.carpeta / TRAIN_ALTERNATIVE, index=False)
        evaluacion.to_parquet(args.carpeta / EVALUATION, index=False)
        imprimir_particiones(entrenamiento, evaluacion)
        for archivo in (TRAIN_ALTERNATIVE, EVALUATION):
            print(f"Guardado: {nombre_archivo(args.carpeta / archivo)}")
    if args.etapa == "todo":
        datasets = {
            REFERENCE_P1: reference_p1,
            CURRENT_P1: current_p1,
            TRAIN_ALTERNATIVE: entrenamiento,
            EVALUATION: evaluacion,
        }
        escribir_manifiesto(fuentes, actual, datasets, args.carpeta, args.manifiesto)
        print(f"\nManifiesto: {nombre_archivo(args.manifiesto)}")


if __name__ == "__main__":
    main()
