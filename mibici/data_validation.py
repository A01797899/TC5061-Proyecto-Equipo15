"""Validación de calidad e integridad de los datos de P1 y P2 con DeepChecks.

Usa el mismo procesamiento, las mismas características y la misma partición cronológica
que los candidatos de mibici/modeling/p1_p2_candidates.py; no repite la limpieza de dataset.py.

Ejecutar desde la raíz del repositorio:
    python -m mibici.data_validation
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import re
from types import SimpleNamespace
import warnings

import pandas as pd

from mibici.config import RANDOM_SEED, RAW_DATA_DIR, REPORTS_DIR

SALIDA_REPORTES = REPORTS_DIR / "week7" / "deepchecks"
PLOTLY_CDN = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@{version}/plotly.min.js"
PLOTLY_INCRUSTADO = re.compile(
    r"<script[^>]*>\s*/\*\*\s*\*\s*plotly\.js v[\d.]+.*?</script>", re.DOTALL
)
PREGUNTAS = ("p1", "p2")
CATEGORICAS = ["hour_of_day", "weekday", "fin_semana"]
TIPO_ETIQUETA = {"p1": "regression", "p2": "binary"}
# Describen la hora objetivo: solo pueden formar la etiqueta, nunca ser predictores.
COLUMNAS_HORA_OBJETIVO = {"y", "salidas", "llegadas", "pressure"}

CHECKS_INCOMPATIBLES = {
    "OutlierSampleDetection": "llama quantile(interpolation=...), argumento que NumPy 2 eliminó",
    "TrainTestSamplesMix": "usa DataFrame.applymap, que pandas 3 eliminó",
    "MultivariateDrift": "su modelo interno no implementa __sklearn_tags__ (scikit-learn 1.6+)",
}
CHECKS_NO_APLICAN = {
    "IndexTrainTestLeakage": "la misma estación aparece en train y test por diseño; la fuga se mide por fecha",
}
CHECKS_SOLO_CLASIFICACION = {"ConflictingLabels", "NewLabelTrainTest"}
CONDICIONES_PROPIAS = {
    "FeatureFeatureCorrelation": (
        "la condición original usa stack(), que en pandas 3 ya no descarta NaN y marca todos "
        "los pares como correlacionados; la evalúo con stack().dropna()"
    ),
    "ClassImbalance": (
        "la condición original avisa cuando las clases están balanceadas; la invierto para "
        "avisar cuando la clase minoritaria es menor al 10 % de la mayoritaria"
    ),
}

NOTAS_INTERPRETACION = (
    (
        "`DataDuplicates` compara el registro completo, incluida la fecha; el conteo exacto "
        "compara solo características y etiqueta. Filas distintas con los mismos valores, como "
        "horas sin viajes, no son registros repetidos: la llave estación-hora no se duplica."
    ),
    (
        "Los nulos están solo en los rezagos y al inicio de train, donde todavía no hay historia; "
        "el pipeline de candidatos los descarta con `dropna`."
    ),
)

RECOMENDACIONES = {
    "IsSingleValue": "Eliminar la columna constante: no aporta información al modelo.",
    "SpecialCharacters": "Limpiar caracteres especiales en las columnas de texto.",
    "MixedNulls": "Unificar la representación de nulos antes de entrenar.",
    "PercentOfNulls": "Revisar el origen de los nulos; en los rezagos se esperan al inicio de la serie.",
    "MixedDataTypes": "Forzar un solo tipo por columna al construir el panel.",
    "StringMismatch": "Normalizar mayúsculas y espacios en las categorías.",
    "StringLengthOutOfBounds": "Revisar textos con longitud atípica.",
    "DataDuplicates": "Distinguir registros repetidos de filas distintas con las mismas características.",
    "ConflictingLabels": "Revisar filas con las mismas características y etiquetas distintas.",
    "ClassImbalance": "Mantener métricas robustas al desbalance (F1, AP) y el umbral elegido en validación.",
    "FeatureLabelCorrelation": "Una característica que sola predice la etiqueta sugiere fuga; revisar cómo se construye.",
    "FeatureFeatureCorrelation": "Valorar si conviene quitar características redundantes.",
    "IdentifierLabelCorrelation": "Confirmar que la fecha no se use como predictor directo.",
    "DatasetsSizeComparison": "Verificar la proporción entre entrenamiento y prueba.",
    "NewLabelTrainTest": "Revisar clases que aparecen solo en prueba.",
    "NewCategoryTrainTest": "Revisar categorías que aparecen solo en prueba.",
    "StringMismatchComparison": "Normalizar categorías entre entrenamiento y prueba.",
    "DateTrainTestLeakageDuplicates": "Eliminar fechas repetidas entre entrenamiento y prueba.",
    "DateTrainTestLeakageOverlap": "Corregir la partición: prueba debe ser posterior a entrenamiento.",
    "FeatureLabelCorrelationChange": "Revisar características cuyo poder predictivo cambia entre periodos.",
    "FeatureDrift": "Documentar el cambio; con partición cronológica parte del drift es estacional.",
    "LabelDrift": "Monitorear el desempeño en prueba: la etiqueta cambia entre periodos.",
}


def cargar_deepchecks() -> SimpleNamespace:
    import numpy as np
    from sklearn.metrics import _scorer

    for viejo, nuevo in {"Inf": np.inf, "NaN": np.nan}.items():
        if not hasattr(np, viejo):
            setattr(np, viejo, nuevo)
    if "max_error" not in _scorer._SCORERS:
        _scorer._SCORERS["max_error"] = _scorer._SCORERS["neg_max_error"]
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

    from deepchecks.core import ConditionCategory, ConditionResult
    from deepchecks.core.errors import DeepchecksBaseError
    from deepchecks.tabular import Dataset
    from deepchecks.tabular.checks import (
        ClassImbalance,
        FeatureFeatureCorrelation,
        PercentOfNulls,
    )
    from deepchecks.tabular.suites import data_integrity, train_test_validation

    return SimpleNamespace(
        Dataset=Dataset,
        ConditionCategory=ConditionCategory,
        ConditionResult=ConditionResult,
        FeatureFeatureCorrelation=FeatureFeatureCorrelation,
        DeepchecksBaseError=DeepchecksBaseError,
        PercentOfNulls=PercentOfNulls,
        ClassImbalance=ClassImbalance,
        data_integrity=data_integrity,
        train_test_validation=train_test_validation,
    )


def cargar_panel(pregunta: str, raw: Path, chunk_size: int) -> tuple[pd.DataFrame, list[str]]:
    """Construyo el panel con las mismas funciones que usan los candidatos P1 y P2."""
    from mibici.modeling.p1_p2_candidates import datos_p1, datos_p2

    cargar = datos_p1 if pregunta == "p1" else datos_p2
    panel, features, _ = cargar(raw, chunk_size)
    return panel, features


def validar_columnas(panel: pd.DataFrame, features: list[str]) -> None:
    if panel.empty:
        raise ValueError("El panel está vacío")
    faltantes = {"y", "split", "hour", "station", *features} - set(panel.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas en el panel: {sorted(faltantes)}")
    fuga = COLUMNAS_HORA_OBJETIVO & set(features)
    if fuga:
        raise ValueError(f"Columnas de la hora objetivo usadas como predictor: {sorted(fuga)}")
    if not {"train", "test"} <= set(panel["split"].unique()):
        raise ValueError("El panel debe incluir los splits train y test")


def resumen_exacto(panel: pd.DataFrame, features: list[str]) -> dict:
    """Cuento sobre el panel completo lo que el muestreo de deepchecks no puede garantizar."""
    train = panel[panel["split"] == "train"]
    nulos = train[features].isna().sum()
    return {
        "filas_por_split": {
            k: int(v) for k, v in panel["split"].value_counts().sort_index().items()
        },
        "llave_estacion_hora_duplicada": int(panel.duplicated(["station", "hour"]).sum()),
        "filas_duplicadas_en_caracteristicas_train": int(train.duplicated([*features, "y"]).sum()),
        "nulos_train_por_columna": {k: int(v) for k, v in nulos[nulos > 0].items()},
        "filas_train_descartadas_por_nulos": int(train[features].isna().any(axis=1).sum()),
        "nulos_test": int(panel.loc[panel["split"] == "test", features].isna().sum().sum()),
        "rango_fechas": {
            n: [str(g["hour"].min()), str(g["hour"].max())]
            for n, g in panel.groupby("split", observed=True)
        },
    }


def dividir(
    panel: pd.DataFrame, features: list[str], muestra: int, semilla: int = RANDOM_SEED
) -> dict:
    """Separo train y test sin quitar nulos y muestreo con semilla fija."""
    columnas = ["hour", *features, "y"]
    partes = {}
    for nombre in ("train", "test"):
        df = panel.loc[panel["split"] == nombre, columnas]
        if muestra and len(df) > muestra:
            df = df.sample(n=muestra, random_state=semilla)
        partes[nombre] = df.sort_values("hour").reset_index(drop=True)
    return partes


def guardar_html(resultado, ruta: Path) -> None:
    """Guardo el reporte como HTML estático con una sola copia de plotly.js desde CDN.

    El HTML de widgets de deepchecks 0.19.1 no dibuja las gráficas con plotly 7, y su HTML
    estático incrusta plotly.js (~5 MB) en cada gráfica.
    """
    from deepchecks.core.serialization.suite_result.html import SuiteResultSerializer
    import plotly.offline

    cuerpo = SuiteResultSerializer(resultado).serialize(
        full_html=False, include_requirejs=False, include_plotlyjs=False
    )
    cuerpo = PLOTLY_INCRUSTADO.sub("", cuerpo)
    plotly_js = PLOTLY_CDN.format(version=plotly.offline.get_plotlyjs_version())
    ruta.write_text(
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        f"<title>{html.escape(resultado.name)}</title>"
        f'<script src="{plotly_js}"></script></head>'
        f"<body>{cuerpo}</body></html>",
        encoding="utf-8",
    )


def construir_dataset(df: pd.DataFrame, features: list[str], pregunta: str, dc: SimpleNamespace):
    return dc.Dataset(
        df,
        label="y",
        features=features,
        cat_features=[c for c in CATEGORICAS if c in features],
        datetime_name="hour",
        label_type=TIPO_ETIQUETA[pregunta],
    )


def quitar_checks(suite, nombres: set[str]) -> list[str]:
    """Quito de la suite los checks indicados y regreso los que encontré."""
    quitados = []
    for indice, check in list(suite.checks.items()):
        if type(check).__name__ in nombres:
            suite.remove(indice)
            quitados.append(type(check).__name__)
    return quitados


def condicion_pares_correlacionados(dc: SimpleNamespace, umbral: float = 0.9) -> tuple:
    """Reemplazo la condición original, que con pandas 3 marca todos los pares."""

    def evaluar(matriz: pd.DataFrame):
        valores = matriz.stack().dropna()
        fuera_diagonal = valores[[i != j for i, j in valores.index]]
        pares = [(i, j) for (i, j), v in fuera_diagonal.items() if i < j and v > umbral]
        maximo = float(fuera_diagonal.max()) if len(fuera_diagonal) else 0.0
        if pares:
            return dc.ConditionResult(
                dc.ConditionCategory.FAIL,
                f"Correlación mayor a {umbral} en los pares {pares}",
            )
        return dc.ConditionResult(
            dc.ConditionCategory.PASS,
            f"Ningún par supera {umbral}; máximo {maximo:.2f}",
        )

    return f"Ningún par de características con correlación mayor a {umbral}", evaluar


def condicion_desbalance(dc: SimpleNamespace, umbral: float = 0.1) -> tuple:
    """Aviso cuando la clase minoritaria es menor al umbral respecto de la mayoritaria."""

    def evaluar(proporciones: dict):
        valores = list(proporciones.values())
        razon = min(valores) / max(valores)
        detalle = f"La clase minoritaria es {razon:.2f} veces la mayoritaria"
        categoria = dc.ConditionCategory.WARN if razon < umbral else dc.ConditionCategory.PASS
        return dc.ConditionResult(categoria, detalle)

    return f"La clase minoritaria es al menos {umbral:.0%} de la mayoritaria", evaluar


def preparar_suites(pregunta: str, dc: SimpleNamespace) -> tuple:
    excluir = set(CHECKS_INCOMPATIBLES) | set(CHECKS_NO_APLICAN)
    if TIPO_ETIQUETA[pregunta] == "regression":
        excluir |= CHECKS_SOLO_CLASIFICACION

    integridad = dc.data_integrity()
    quitar_checks(integridad, excluir | {"FeatureFeatureCorrelation"})
    integridad.add(
        dc.FeatureFeatureCorrelation().add_condition(*condicion_pares_correlacionados(dc))
    )
    integridad.add(dc.PercentOfNulls().add_condition_percent_of_nulls_not_greater_than(0.05))
    if TIPO_ETIQUETA[pregunta] == "binary":
        integridad.add(dc.ClassImbalance().add_condition(*condicion_desbalance(dc)))

    train_test = dc.train_test_validation()
    quitar_checks(train_test, excluir)
    return integridad, train_test


def clasificar(resultado, dc: SimpleNamespace) -> list[dict]:
    """Clasifico cada check en aprobado, advertencia, problema, sin condición o no ejecutado."""
    filas = []
    for r in resultado.results:
        nombre = type(r.check).__name__
        if isinstance(r, Exception) or not hasattr(r, "conditions_results"):
            filas.append(
                {
                    "check": nombre,
                    "estado": "no_ejecutado",
                    "detalle": str(getattr(r, "exception", r))[:300],
                }
            )
            continue
        categorias = [c.category for c in r.conditions_results]
        if not categorias:
            estado = "sin_condicion"
        elif dc.ConditionCategory.FAIL in categorias or dc.ConditionCategory.ERROR in categorias:
            estado = "problema"
        elif dc.ConditionCategory.WARN in categorias:
            estado = "advertencia"
        else:
            estado = "aprobado"
        detalle = " | ".join(c.details for c in r.conditions_results if c.details)
        fila = {"check": nombre, "estado": estado, "detalle": detalle[:300]}
        if estado in ("problema", "advertencia"):
            fila["recomendacion"] = RECOMENDACIONES.get(
                nombre, "Revisar el detalle en el reporte HTML."
            )
        filas.append(fila)
    return filas


def validar_pregunta(pregunta: str, args, dc: SimpleNamespace) -> dict:
    print(f"\n=== {pregunta.upper()}: construyendo panel ===", flush=True)
    panel, features = cargar_panel(pregunta, args.raw, args.chunk_size)
    validar_columnas(panel, features)
    exacto = resumen_exacto(panel, features)
    partes = dividir(panel, features, args.muestra)
    train = construir_dataset(partes["train"], features, pregunta, dc)
    test = construir_dataset(partes["test"], features, pregunta, dc)
    integridad, train_test = preparar_suites(pregunta, dc)

    args.salida.mkdir(parents=True, exist_ok=True)
    corridas = {
        "integridad_train": integridad.run(train),
        "integridad_test": integridad.run(test),
        "train_test": train_test.run(train, test),
    }
    resumen = {
        "pregunta": pregunta,
        "tipo_etiqueta": TIPO_ETIQUETA[pregunta],
        "caracteristicas": features,
        "muestra_por_split": args.muestra,
        "semilla": RANDOM_SEED,
        "conteos_panel_completo": exacto,
        "checks_excluidos": {**CHECKS_INCOMPATIBLES, **CHECKS_NO_APLICAN},
        "condiciones_propias": CONDICIONES_PROPIAS,
        "suites": {},
    }
    for nombre, resultado in corridas.items():
        guardar_html(resultado, args.salida / f"{pregunta}_{nombre}.html")
        resumen["suites"][nombre] = clasificar(resultado, dc)

    (args.salida / f"{pregunta}_resumen.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    imprimir(resumen)
    return resumen


def imprimir(resumen: dict) -> None:
    conteo = resumen["conteos_panel_completo"]
    print(f"Filas por split: {conteo['filas_por_split']}")
    print(f"Llave estación-hora duplicada: {conteo['llave_estacion_hora_duplicada']}")
    print(
        f"Filas de train repetidas en características y etiqueta: {conteo['filas_duplicadas_en_caracteristicas_train']:,}"
    )
    print(
        f"Filas de train con nulos (el pipeline las descarta): {conteo['filas_train_descartadas_por_nulos']:,}"
    )
    for suite, filas in resumen["suites"].items():
        print(f"\n  [{suite}]")
        for f in filas:
            print(f"    {f['estado']:<13} {f['check']}")


def escribir_markdown(resumenes: list[dict], salida: Path) -> Path:
    lineas = ["# Validación de datos con DeepChecks", ""]
    for r in resumenes:
        c = r["conteos_panel_completo"]
        lineas += [
            f"## {r['pregunta'].upper()} ({r['tipo_etiqueta']})",
            "",
            f"- Filas por split: {c['filas_por_split']}",
            f"- Llave estación-hora duplicada: {c['llave_estacion_hora_duplicada']}",
            f"- Filas de train repetidas en características y etiqueta: {c['filas_duplicadas_en_caracteristicas_train']:,}",
            f"- Filas de train con nulos en rezagos: {c['filas_train_descartadas_por_nulos']:,}",
            f"- Muestra por split para DeepChecks: {r['muestra_por_split']:,} (semilla {r['semilla']})",
            "",
            "| Suite | Check | Estado | Recomendación |",
            "|---|---|---|---|",
        ]
        for suite, filas in r["suites"].items():
            for f in filas:
                lineas.append(
                    f"| {suite} | {f['check']} | {f['estado']} | {f.get('recomendacion', '')} |"
                )
        lineas.append("")
    lineas += ["## Checks excluidos", ""]
    for nombre, motivo in {**CHECKS_INCOMPATIBLES, **CHECKS_NO_APLICAN}.items():
        lineas.append(f"- `{nombre}`: {motivo}.")
    lineas += ["", "## Condiciones reemplazadas", ""]
    for nombre, motivo in CONDICIONES_PROPIAS.items():
        lineas.append(f"- `{nombre}`: {motivo}.")
    lineas += [
        "",
        "## Notas de interpretación",
        "",
        *(f"- {nota}" for nota in NOTAS_INTERPRETACION),
    ]
    ruta = salida / "resumen.md"
    ruta.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    return ruta


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Validación de datos P1 y P2 con DeepChecks")
    p.add_argument("--preguntas", nargs="+", choices=PREGUNTAS, default=list(PREGUNTAS))
    p.add_argument("--raw", type=Path, default=RAW_DATA_DIR)
    p.add_argument("--salida", type=Path, default=SALIDA_REPORTES)
    p.add_argument("--muestra", type=int, default=200_000)
    p.add_argument("--chunk-size", type=int, default=100_000)
    p.add_argument("--estricto", action="store_true", help="termina con error si hay problemas")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    dc = cargar_deepchecks()
    resumenes = [validar_pregunta(p, args, dc) for p in args.preguntas]
    ruta = escribir_markdown(resumenes, args.salida)
    print(f"\nReportes guardados en {args.salida}\nResumen: {ruta}")
    problemas = sum(
        f["estado"] == "problema"
        for r in resumenes
        for filas in r["suites"].values()
        for f in filas
    )
    return 1 if args.estricto and problemas else 0


if __name__ == "__main__":
    raise SystemExit(main())
