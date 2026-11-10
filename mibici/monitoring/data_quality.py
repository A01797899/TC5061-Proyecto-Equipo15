"""Valido la calidad de los viajes de referencia (2024) y actuales (enero-marzo 2025).

Distingue los registros descartados (la misma regla de la ingesta de P1) de las anomalías que
solo se reportan y se conservan. El resultado queda en ``reports/week9/data_quality.json``.

Uso desde la raíz del repositorio::

    python -m mibici.monitoring.data_quality
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from mibici.config import (
    DURACION_MAX_MINUTOS,
    DURACION_MIN_MINUTOS,
    EDAD_MAX,
    EDAD_MIN,
    FECHA_INICIO,
    REPORTS_DIR,
    TRAIN_END,
)
from mibici.features import trip_quality_flags
from mibici.monitoring.prepare_data import (
    DIR_2025,
    FIN_ACTUAL,
    INICIO_ACTUAL,
    cargar_actual,
    cargar_referencia,
    nombre_archivo,
)

REPORTE = REPORTS_DIR / "week9" / "data_quality.json"
REGLAS = {
    "descartados": (
        "Viajes sin fecha de inicio u origen, o con inicio fuera del periodo; es la misma "
        "regla con la que la ingesta de P1 cuenta salidas."
    ),
    "anomalias_conservadas": (
        "Género o año de nacimiento vacíos, fin antes del inicio, duración fuera de "
        f"[{DURACION_MIN_MINUTOS}, {DURACION_MAX_MINUTOS}] min, edad fuera de [{EDAD_MIN}, "
        f"{EDAD_MAX}] y viajes de estaciones nuevas se reportan y se conservan."
    ),
    "estaciones": (
        "El panel de P1 usa las estaciones con viajes en el entrenamiento del Champion; las "
        "nuevas se reportan aparte y las que no tienen viajes quedan con salidas en 0."
    ),
}


def estaciones_modelo(referencia: pd.DataFrame) -> list[int]:
    """Estaciones con al menos un viaje en el entrenamiento del Champion (ene-ago 2024)."""
    inicio = referencia["inicio_viaje"]
    en_train = (inicio >= pd.Timestamp(FECHA_INICIO)) & (inicio < pd.Timestamp(TRAIN_END))
    return sorted(int(e) for e in referencia.loc[en_train, "origen_id"].dropna().unique())


def evaluar_periodo(viajes: pd.DataFrame, inicio: str, fin: str, estaciones: list[int]) -> dict:
    """Cuento nulos, duplicados, fechas, descartes, anomalías y estaciones de un periodo."""
    fechas = viajes["inicio_viaje"]
    sin_inicio_u_origen = fechas.isna() | viajes["origen_id"].isna()
    fuera = ~sin_inicio_u_origen & (
        (fechas < pd.Timestamp(inicio)) | (fechas >= pd.Timestamp(fin))
    )
    validos = viajes[~(sin_inicio_u_origen | fuera)]

    marcas = trip_quality_flags(validos)
    duracion = marcas["duracion_min"]
    # Calculo la edad con el año del viaje porque trip_quality_flags usa 2024 fijo.
    edad = validos["inicio_viaje"].dt.year - validos["anio_nacimiento"].astype("Float64")
    edad_atipica = ((edad < EDAD_MIN) | (edad > EDAD_MAX)).fillna(False)

    origenes = {int(e) for e in validos["origen_id"].unique()}
    nuevas = sorted(origenes - set(estaciones))
    total = max(len(viajes), 1)
    return {
        "periodo": {"inicio": inicio, "fin_exclusivo": fin},
        "registros": len(viajes),
        "columnas": list(viajes.columns),
        "nulos": {c: int(n) for c, n in viajes.isna().sum().items()},
        "nulos_pct": {c: round(100 * int(n) / total, 3) for c, n in viajes.isna().sum().items()},
        "duplicados": {
            "viaje_id": int(viajes["viaje_id"].dropna().duplicated().sum()),
            "filas_completas": int(viajes.duplicated().sum()),
        },
        "inicio_min": str(fechas.min()),
        "inicio_max": str(fechas.max()),
        "descartados": {
            "sin_inicio_u_origen": int(sin_inicio_u_origen.sum()),
            "fuera_del_periodo": int(fuera.sum()),
            "total": int((sin_inicio_u_origen | fuera).sum()),
        },
        "registros_validos": len(validos),
        "anomalias_conservadas": {
            "genero_vacio": int(validos["genero"].isna().sum()),
            "anio_nacimiento_vacio": int(validos["anio_nacimiento"].isna().sum()),
            "fin_antes_de_inicio": int((duracion < 0).sum()),
            "duracion_atipica": int(marcas["duracion_atipica"].sum()),
            "edad_atipica": int(edad_atipica.sum()),
            "viajes_de_estaciones_nuevas": int(validos["origen_id"].isin(nuevas).sum()),
        },
        "duracion_min": {
            "p50": round(float(duracion.median()), 2),
            "p99": round(float(duracion.quantile(0.99)), 2),
            "max": round(float(duracion.max()), 2),
        },
        "estaciones": {
            "con_viajes": len(origenes),
            "del_modelo_con_viajes": len(origenes & set(estaciones)),
            "nuevas": nuevas,
            "del_modelo_sin_viajes": sorted(set(estaciones) - origenes),
        },
    }


def generar_reporte(
    referencia: pd.DataFrame, actual: pd.DataFrame, fuentes: list[dict], salida: Path = REPORTE
) -> dict:
    """Evalúo la referencia y el periodo actual y guardo el reporte en JSON."""
    estaciones = estaciones_modelo(referencia)
    fechas = referencia["inicio_viaje"]
    en_train = (fechas >= pd.Timestamp(FECHA_INICIO)) & (fechas < pd.Timestamp(TRAIN_END))
    reporte = {
        "reglas": REGLAS,
        "estaciones_modelo": len(estaciones),
        "esquema_2025": fuentes,
        "referencia": evaluar_periodo(referencia[en_train], FECHA_INICIO, TRAIN_END, estaciones),
        "actual": evaluar_periodo(actual, INICIO_ACTUAL, FIN_ACTUAL, estaciones),
    }
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps(reporte, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return reporte


def imprimir(reporte: dict) -> None:
    """Resumo el reporte en la terminal."""
    print(f"Estaciones del modelo: {reporte['estaciones_modelo']}")
    print(
        "Esquema 2025 igual a 2024:",
        all(f["coincide_esquema_2024"] for f in reporte["esquema_2025"]),
    )
    for nombre in ("referencia", "actual"):
        r = reporte[nombre]
        a = r["anomalias_conservadas"]
        e = r["estaciones"]
        print(
            f"\n{nombre.upper()} {r['periodo']['inicio']} a {r['periodo']['fin_exclusivo']} (exclusivo)"
        )
        print(
            f"  Registros: {r['registros']:,} | válidos: {r['registros_validos']:,} | "
            f"descartados: {r['descartados']['total']:,}"
        )
        print(f"  Inicio: {r['inicio_min']} a {r['inicio_max']}")
        print(
            f"  Duplicados viaje_id: {r['duplicados']['viaje_id']} | "
            f"filas: {r['duplicados']['filas_completas']}"
        )
        print(f"  Nulos: { {c: n for c, n in r['nulos'].items() if n} }")
        print(f"  Anomalías conservadas: {a}")
        print(f"  Duración (min) p50/p99/max: {r['duracion_min']}")
        print(
            f"  Estaciones con viajes: {e['con_viajes']} | nuevas: {e['nuevas']} | "
            f"del modelo sin viajes: {e['del_modelo_sin_viajes']}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--directorio", type=Path, default=DIR_2025)
    parser.add_argument("--salida", type=Path, default=REPORTE)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    actual, fuentes = cargar_actual(args.directorio)
    reporte = generar_reporte(cargar_referencia(), actual, fuentes, args.salida)
    imprimir(reporte)
    print(f"\nReporte: {nombre_archivo(args.salida)}")


if __name__ == "__main__":
    main()
