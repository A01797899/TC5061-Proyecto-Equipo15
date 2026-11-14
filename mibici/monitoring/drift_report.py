"""Reporte de drift del Champion P1 con Evidently.

Uso: python -m mibici.monitoring.drift_report [--muestra N] [--sin-html]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Any

import evidently
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mibici.config import PROCESSED_DATA_DIR, REPORTS_DIR
from mibici.monitoring.prepare_data import (
    CHAMPION,
    COLUMNAS_P1,
    CURRENT_P1,
    FEATURES_P1,
    REFERENCE_P1,
)

DATOS = PROCESSED_DATA_DIR / "week9"
SALIDA = REPORTS_DIR / "week9" / "drift"
OBJETIVO = "y"

CATEGORICAS = ["hour_of_day", "weekday", "fin_semana"]
NUMERICAS = [c for c in FEATURES_P1 if c not in CATEGORICAS]
COLUMNAS_MONITOREO = [*FEATURES_P1, OBJETIVO]

METODO_NUMERICO = "wasserstein"
METODO_CATEGORICO = "jensenshannon"
UMBRAL_DRIFT = 0.1
DRIFT_SHARE = 0.5

INICIO_ESTACIONAL = "2024-01-01"
FIN_ESTACIONAL = "2024-04-01"
SEMILLA = 42

COLOR_REFERENCIA = "#2a78d6"
COLOR_ACTUAL = "#eb6834"
TINTA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
TINTA_TENUE = "#898781"
REJILLA = "#e1e0d9"


@dataclass(frozen=True)
class Escenario:
    """Comparación entre referencia y actual."""

    nombre: str
    descripcion: str
    referencia: pd.DataFrame
    actual: pd.DataFrame


def validar_columnas(df: pd.DataFrame, nombre: str) -> None:
    """Valida columnas y nulos del contrato de P1."""
    faltantes = [c for c in COLUMNAS_P1 if c not in df.columns]
    if faltantes:
        raise ValueError(f"{nombre}: faltan columnas del contrato de P1: {faltantes}")
    nulos = df[COLUMNAS_MONITOREO].isna().sum()
    if nulos.any():
        raise ValueError(f"{nombre}: hay nulos en {nulos[nulos > 0].to_dict()}")


def cargar_conjuntos(
    datos: Path = DATOS, muestra: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga los parquets de referencia y actual, con muestreo opcional."""
    rutas = {"referencia": datos / REFERENCE_P1, "actual": datos / CURRENT_P1}
    for nombre, ruta in rutas.items():
        if not ruta.exists():
            raise FileNotFoundError(
                f"No existe {ruta} ({nombre}). Genera los datos con "
                "`python -m mibici.monitoring.prepare_data`."
            )
    referencia = pd.read_parquet(rutas["referencia"])
    actual = pd.read_parquet(rutas["actual"])
    validar_columnas(referencia, "referencia")
    validar_columnas(actual, "actual")
    if muestra:
        referencia = referencia.sample(min(muestra, len(referencia)), random_state=SEMILLA)
        actual = actual.sample(min(muestra, len(actual)), random_state=SEMILLA)
    return referencia.sort_values(["hour", "station"]), actual.sort_values(["hour", "station"])


def para_evidently(df: pd.DataFrame) -> pd.DataFrame:
    """Copia con las columnas monitoreadas y las categóricas como enteros."""
    salida = df[["station", "hour", *COLUMNAS_MONITOREO]].copy()
    salida[["station", *CATEGORICAS]] = salida[["station", *CATEGORICAS]].astype("int64")
    salida["hour"] = pd.to_datetime(salida["hour"])
    return salida


def definicion_datos() -> DataDefinition:
    """Definición de columnas para Evidently."""
    return DataDefinition(
        id_column="station",
        timestamp="hour",
        numerical_columns=[*NUMERICAS, OBJETIVO],
        categorical_columns=list(CATEGORICAS),
    )


def armar_escenarios(referencia: pd.DataFrame, actual: pd.DataFrame) -> list[Escenario]:
    """Escenarios principal, estacional y uno por mes del periodo actual."""
    hora_ref = pd.to_datetime(referencia["hour"])
    misma_temporada = referencia[
        (hora_ref >= pd.Timestamp(INICIO_ESTACIONAL)) & (hora_ref < pd.Timestamp(FIN_ESTACIONAL))
    ]
    escenarios = [
        Escenario(
            "principal",
            "Entrenamiento del Champion (ene-ago 2024) contra ene-mar 2025",
            referencia,
            actual,
        ),
        Escenario(
            "estacional",
            "Mismos meses un año antes (ene-mar 2024) contra ene-mar 2025",
            misma_temporada,
            actual,
        ),
    ]
    hora_act = pd.to_datetime(actual["hour"])
    for periodo in sorted(hora_act.dt.to_period("M").unique()):
        escenarios.append(
            Escenario(
                f"mes_{periodo}",
                f"Entrenamiento del Champion contra {periodo}",
                referencia,
                actual[hora_act.dt.to_period("M") == periodo],
            )
        )
    return escenarios


def construir_reporte(con_calidad: bool = True) -> Report:
    """Reporte de drift y, opcionalmente, resumen de calidad."""
    metricas: list[Any] = [
        DataDriftPreset(
            columns=COLUMNAS_MONITOREO,
            num_method=METODO_NUMERICO,
            cat_method=METODO_CATEGORICO,
            threshold=UMBRAL_DRIFT,
            drift_share=DRIFT_SHARE,
        )
    ]
    if con_calidad:
        metricas.append(DataSummaryPreset(columns=COLUMNAS_MONITOREO))
    return Report(
        metricas,
        include_tests=True,
        metadata={"modelo": CHAMPION["modelo"], "alias": CHAMPION["alias"]},
        tags=["week9", "drift", "p1"],
    )


def ejecutar(escenario: Escenario, con_calidad: bool = True):
    """Ejecuta Evidently para un escenario."""
    definicion = definicion_datos()
    actual = Dataset.from_pandas(para_evidently(escenario.actual), data_definition=definicion)
    referencia = Dataset.from_pandas(
        para_evidently(escenario.referencia), data_definition=definicion
    )
    return construir_reporte(con_calidad).run(actual, referencia)


def hay_drift(metodo: str, puntaje: float, umbral: float) -> bool:
    """Indica si hay drift según el método y el umbral."""
    if "p_value" in metodo.lower().replace("-", "_") or "test" in metodo.lower():
        return puntaje < umbral
    return puntaje >= umbral


def resultados_drift(snapshot_dict: dict) -> pd.DataFrame:
    """Resultados de drift por variable a partir del snapshot."""
    filas = []
    for metrica in snapshot_dict["metrics"]:
        config = metrica.get("config", {})
        if not str(config.get("type", "")).endswith(":ValueDrift"):
            continue
        columna = config["column"]
        metodo = str(config.get("method", ""))
        umbral = float(config.get("threshold", UMBRAL_DRIFT))
        puntaje = float(metrica["value"])
        filas.append(
            {
                "variable": columna,
                "tipo": "categórica" if columna in CATEGORICAS else "numérica",
                "rol": "objetivo" if columna == OBJETIVO else "entrada",
                "metodo": metodo,
                "umbral": umbral,
                "puntaje": puntaje,
                "drift": hay_drift(metodo, puntaje, umbral),
            }
        )
    if not filas:
        raise ValueError("El snapshot no trae métricas ValueDrift; revisa el reporte.")
    orden = {c: i for i, c in enumerate(COLUMNAS_MONITOREO)}
    return pd.DataFrame(filas).sort_values(
        "variable", key=lambda s: s.map(orden), ignore_index=True
    )


def resumen_pruebas(snapshot_dict: dict) -> dict[str, Any]:
    """Conteo de pruebas de Evidently por estado y lista de las que no pasaron."""
    conteo: dict[str, int] = {}
    no_pasaron = []
    for prueba in snapshot_dict.get("tests", []):
        estado = str(prueba.get("status", "DESCONOCIDO")).rsplit(".", 1)[-1]
        conteo[estado] = conteo.get(estado, 0) + 1
        if estado != "SUCCESS":
            no_pasaron.append(f"{estado}: {prueba.get('name', '')}")
    return {"conteo": conteo, "no_pasaron": no_pasaron}


def estadisticas(referencia: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    """Estadísticas descriptivas por variable en ambos conjuntos."""
    filas = []
    for col in COLUMNAS_MONITOREO:
        r, a = referencia[col], actual[col]
        media_r, media_a = float(r.mean()), float(a.mean())
        filas.append(
            {
                "variable": col,
                "media_ref": media_r,
                "media_act": media_a,
                "cambio_media_pct": (media_a - media_r) / media_r * 100 if media_r else np.nan,
                "std_ref": float(r.std()),
                "std_act": float(a.std()),
                "p50_ref": float(r.quantile(0.5)),
                "p50_act": float(a.quantile(0.5)),
                "p90_ref": float(r.quantile(0.9)),
                "p90_act": float(a.quantile(0.9)),
                "ceros_ref_pct": float((r == 0).mean() * 100),
                "ceros_act_pct": float((a == 0).mean() * 100),
            }
        )
    return pd.DataFrame(filas)


def calidad(df: pd.DataFrame) -> dict[str, Any]:
    """Revisiones de calidad sobre el contrato de P1."""
    rangos = {
        "hour_of_day": (0, 23),
        "weekday": (0, 6),
        "fin_semana": (0, 1),
    }
    fuera_de_rango = {
        col: int(((df[col] < lo) | (df[col] > hi)).sum()) for col, (lo, hi) in rangos.items()
    }
    no_negativas = [c for c in [*NUMERICAS, OBJETIVO] if c not in rangos]
    negativos = {col: int((df[col] < 0).sum()) for col in no_negativas}
    por_estacion = df.groupby("station")[OBJETIVO].sum()
    hora = pd.to_datetime(df["hour"])
    return {
        "filas": len(df),
        "estaciones": int(df["station"].nunique()),
        "desde": str(hora.min()),
        "hasta": str(hora.max()),
        "nulos": int(df[COLUMNAS_MONITOREO].isna().sum().sum()),
        "duplicados_estacion_hora": int(df.duplicated(["station", "hour"]).sum()),
        "fuera_de_rango": fuera_de_rango,
        "negativos": negativos,
        "tipos_float64": bool((df[FEATURES_P1].dtypes == "float64").all()),
        "estaciones_sin_salidas": sorted(int(e) for e in por_estacion[por_estacion == 0].index),
    }


def demanda_por_estacion(referencia: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    """Media congelada contra demanda observada por estación."""
    ref = referencia.groupby("station").agg(
        media_congelada=("media_estacion", "first"), demanda_ref=(OBJETIVO, "mean")
    )
    act = actual.groupby("station").agg(demanda_act=(OBJETIVO, "mean"))
    tabla = ref.join(act, how="outer")
    tabla["cambio_abs"] = tabla["demanda_act"] - tabla["demanda_ref"]
    tabla["cambio_pct"] = np.where(
        tabla["demanda_ref"] > 0, tabla["cambio_abs"] / tabla["demanda_ref"] * 100, np.nan
    )
    tabla["brecha_congelada"] = tabla["demanda_act"] - tabla["media_congelada"]
    return tabla.reset_index().sort_values("cambio_abs", ignore_index=True)


def _estilo(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c3c2b7")
    ax.tick_params(colors=TINTA_SECUNDARIA, labelsize=9)
    ax.grid(color=REJILLA, linewidth=0.6)
    ax.set_axisbelow(True)


def grafica_puntajes(drift: pd.DataFrame, ruta: Path) -> Path:
    """Gráfica del puntaje de drift por variable."""
    tabla = drift.pivot(index="variable", columns="escenario", values="puntaje")
    tabla = tabla.reindex([c for c in COLUMNAS_MONITOREO if c in tabla.index])[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=200)
    y = np.arange(len(tabla))
    alto = 0.38
    series = [
        ("principal", "Principal: ene-ago 2024 vs ene-mar 2025", COLOR_REFERENCIA),
        ("estacional", "Control estacional: ene-mar 2024 vs ene-mar 2025", COLOR_ACTUAL),
    ]
    for i, (col, etiqueta, color) in enumerate(series):
        if col not in tabla:
            continue
        valores = tabla[col].to_numpy()
        ax.barh(y + (0.5 - i) * alto, valores, height=alto - 0.04, color=color, label=etiqueta)
        for yy, v in zip(y + (0.5 - i) * alto, valores):
            ax.text(v + 0.004, yy, f"{v:.3f}", va="center", fontsize=7.5, color=TINTA_SECUNDARIA)
    ax.axvline(UMBRAL_DRIFT, color=TINTA, linestyle="--", linewidth=1)
    ax.text(
        UMBRAL_DRIFT,
        len(tabla) - 0.4,
        f" umbral {UMBRAL_DRIFT}",
        fontsize=8,
        color=TINTA,
        va="bottom",
    )
    ax.set_yticks(y, tabla.index)
    ax.set_xlabel(
        "Puntaje de drift (Wasserstein normalizada / Jensen-Shannon)",
        fontsize=9,
        color=TINTA_SECUNDARIA,
    )
    fig.suptitle("Drift por variable del Champion P1", x=0.01, ha="left", fontsize=11, color=TINTA)
    ax.set_xlim(0, max(float(tabla.max().max()), UMBRAL_DRIFT) * 1.18)
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=1)
    _estilo(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)
    return ruta


def grafica_perfil_horario(referencia: pd.DataFrame, actual: pd.DataFrame, ruta: Path) -> Path:
    """Gráfica de demanda media por hora del día."""
    fig, ejes = plt.subplots(1, 2, figsize=(8.5, 3.4), dpi=200, sharey=True)
    for ax, (finde, titulo) in zip(ejes, [(0.0, "Entre semana"), (1.0, "Fin de semana")]):
        for df, etiqueta, color in [
            (referencia, "Referencia (ene-ago 2024)", COLOR_REFERENCIA),
            (actual, "Actual (ene-mar 2025)", COLOR_ACTUAL),
        ]:
            perfil = df[df["fin_semana"] == finde].groupby("hour_of_day")[OBJETIVO].mean()
            ax.plot(perfil.index, perfil.to_numpy(), color=color, linewidth=2, label=etiqueta)
        ax.set_title(titulo, loc="left", fontsize=10, color=TINTA)
        ax.set_xlabel("Hora del día", fontsize=9, color=TINTA_SECUNDARIA)
        ax.set_xticks(range(0, 24, 3))
        _estilo(ax)
    ejes[0].set_ylabel("Salidas medias por estación-hora", fontsize=9, color=TINTA_SECUNDARIA)
    ejes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)
    return ruta


def grafica_distribuciones(referencia: pd.DataFrame, actual: pd.DataFrame, ruta: Path) -> Path:
    """Histogramas de las variables numéricas y del objetivo."""
    columnas = [*NUMERICAS, OBJETIVO]
    fig, ejes = plt.subplots(2, 3, figsize=(9, 5), dpi=200)
    for ax, col in zip(ejes.flat, columnas):
        tope = float(np.nanquantile(np.concatenate([referencia[col], actual[col]]), 0.99))
        discreta = col.startswith("salidas") or col == OBJETIVO
        bins = np.arange(0, np.floor(tope) + 2) - 0.5 if discreta else np.linspace(0, tope, 40)
        for df, etiqueta, color, estilo, grosor in [
            (referencia, "Referencia", COLOR_REFERENCIA, "-", 2.5),
            (actual, "Actual", COLOR_ACTUAL, "--", 1.6),
        ]:
            ax.hist(
                df[col].clip(upper=tope),
                bins=bins,
                density=True,
                histtype="step",
                linewidth=grosor,
                linestyle=estilo,
                color=color,
                label=etiqueta,
            )
        ax.set_title(col, loc="left", fontsize=9, color=TINTA)
        _estilo(ax)
    ejes.flat[0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Distribuciones: referencia (2024) contra actual (2025)",
        x=0.01,
        ha="left",
        fontsize=11,
        color=TINTA,
    )
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)
    return ruta


def grafica_por_mes(por_mes: pd.DataFrame, ruta: Path) -> Path:
    """Mapa de calor del drift por variable y mes."""
    tabla = por_mes.pivot(index="variable", columns="mes", values="puntaje")
    tabla = tabla.reindex([c for c in COLUMNAS_MONITOREO if c in tabla.index])
    fig, ax = plt.subplots(figsize=(5.8, 4.2), dpi=200)
    imagen = ax.imshow(
        tabla.to_numpy(),
        cmap="Blues",
        vmin=0,
        aspect="auto",
        vmax=max(float(tabla.max().max()), UMBRAL_DRIFT * 2),
    )
    for (i, j), v in np.ndenumerate(tabla.to_numpy()):
        claro = v > imagen.norm.vmax * 0.6
        marca = " *" if v >= UMBRAL_DRIFT else ""
        ax.text(
            j,
            i,
            f"{v:.3f}{marca}",
            ha="center",
            va="center",
            fontsize=8,
            color="#ffffff" if claro else TINTA,
        )
    ax.set_xticks(range(len(tabla.columns)), tabla.columns)
    ax.set_yticks(range(len(tabla.index)), tabla.index)
    ax.tick_params(colors=TINTA_SECUNDARIA, labelsize=8, length=0)
    ax.set_title(
        "Drift por mes contra la referencia (* = supera el umbral)",
        loc="left",
        fontsize=10,
        color=TINTA,
    )
    for borde in ax.spines.values():
        borde.set_visible(False)
    fig.colorbar(imagen, ax=ax, shrink=0.8).ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)
    return ruta


def grafica_estaciones(estaciones: pd.DataFrame, ruta: Path, etiquetas: int = 2) -> Path:
    """Gráfica de media congelada contra demanda observada por estación."""
    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=200)
    tope = float(max(estaciones["media_congelada"].max(), estaciones["demanda_act"].max())) * 1.05
    ax.plot([0, tope], [0, tope], color=TINTA_TENUE, linestyle="--", linewidth=1)
    ax.scatter(
        estaciones["media_congelada"],
        estaciones["demanda_act"],
        s=18,
        color=COLOR_REFERENCIA,
        edgecolor="#fcfcfb",
        linewidth=0.6,
        label="Estación",
    )
    extremos = pd.concat(
        [
            estaciones.nsmallest(etiquetas, "brecha_congelada"),
            estaciones.nlargest(etiquetas, "brecha_congelada"),
            estaciones[estaciones["demanda_act"] == 0],
        ]
    ).drop_duplicates("station")
    ax.scatter(
        extremos["media_congelada"],
        extremos["demanda_act"],
        s=30,
        color=COLOR_ACTUAL,
        edgecolor="#fcfcfb",
        linewidth=0.6,
        label="Mayor brecha o sin salidas en 2025",
    )
    sin_salidas = extremos["demanda_act"] == 0
    for _, fila in extremos[~sin_salidas].iterrows():
        ax.annotate(
            f"{int(fila['station'])}",
            (fila["media_congelada"], fila["demanda_act"]),
            xytext=(5, 3),
            textcoords="offset points",
            fontsize=7.5,
            color=TINTA_SECUNDARIA,
        )
    if sin_salidas.any():
        cero = extremos[sin_salidas]
        nombres = ", ".join(str(int(e)) for e in sorted(cero["station"]))
        ax.annotate(
            f"{nombres}: sin salidas en 2025",
            (float(cero["media_congelada"].max()), 0),
            xytext=(0.45, 0.04),
            textcoords="axes fraction",
            fontsize=7.5,
            color=TINTA_SECUNDARIA,
            arrowprops={"arrowstyle": "-", "color": TINTA_TENUE, "linewidth": 0.8},
        )
    ax.set_xlim(0, tope)
    ax.set_ylim(0, tope)
    ax.set_xlabel("media_estacion congelada (ene-ago 2024)", fontsize=9, color=TINTA_SECUNDARIA)
    ax.set_ylabel(
        "Salidas medias por hora observadas (ene-mar 2025)", fontsize=9, color=TINTA_SECUNDARIA
    )
    ax.set_title(
        "Media congelada contra demanda observada por estación",
        loc="left",
        fontsize=10,
        color=TINTA,
    )
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _estilo(ax)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)
    return ruta


def commit_actual() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def generar(
    referencia: pd.DataFrame,
    actual: pd.DataFrame,
    salida: Path = SALIDA,
    html: bool = True,
    graficas: bool = True,
) -> dict[str, Any]:
    """Ejecuta todos los escenarios y escribe los artefactos."""
    salida.mkdir(parents=True, exist_ok=True)
    figuras = salida / "figures"
    escenarios = armar_escenarios(referencia, actual)

    drift_frames, mensual, resumen_escenarios = [], [], {}
    for esc in escenarios:
        mensual_esc = esc.nombre.startswith("mes_")
        print(f"Evidently: {esc.nombre} ({len(esc.referencia):,} vs {len(esc.actual):,} filas)")
        snapshot = ejecutar(esc, con_calidad=not mensual_esc)
        snap = snapshot.dict()
        drift = resultados_drift(snap)
        drift.insert(0, "escenario", esc.nombre)
        if mensual_esc:
            mensual.append(drift.assign(mes=esc.nombre.removeprefix("mes_")))
            continue
        drift_frames.append(drift)
        if html:
            snapshot.save_html(str(salida / f"evidently_drift_{esc.nombre}.html"))
        snapshot.save_json(str(salida / f"evidently_drift_{esc.nombre}.json"))
        derivadas = drift.loc[drift["drift"], "variable"].tolist()
        resumen_escenarios[esc.nombre] = {
            "descripcion": esc.descripcion,
            "filas_referencia": len(esc.referencia),
            "filas_actual": len(esc.actual),
            "variables_con_drift": derivadas,
            "proporcion_drift": len(derivadas) / len(drift),
            "drift_del_conjunto": len(derivadas) / len(drift) >= DRIFT_SHARE,
            "pruebas_evidently": resumen_pruebas(snap),
        }

    drift_todo = pd.concat(drift_frames, ignore_index=True)
    por_mes = pd.concat(mensual, ignore_index=True) if mensual else pd.DataFrame()
    stats = estadisticas(referencia, actual)
    estaciones = demanda_por_estacion(referencia, actual)

    principal = drift_todo[drift_todo["escenario"] == "principal"].merge(stats, on="variable")
    principal.to_csv(salida / "drift_por_variable.csv", index=False)
    drift_todo.to_csv(salida / "drift_escenarios.csv", index=False)
    if not por_mes.empty:
        por_mes.to_csv(salida / "drift_por_mes.csv", index=False)
    estaciones.to_csv(salida / "demanda_por_estacion.csv", index=False)

    resumen = {
        "fecha_ejecucion": pd.Timestamp.now().isoformat(timespec="seconds"),
        "commit": commit_actual(),
        "evidently": evidently.__version__,
        "champion": CHAMPION,
        "configuracion": {
            "columnas": COLUMNAS_MONITOREO,
            "categoricas": CATEGORICAS,
            "numericas": [*NUMERICAS, OBJETIVO],
            "metodo_numerico": METODO_NUMERICO,
            "metodo_categorico": METODO_CATEGORICO,
            "umbral": UMBRAL_DRIFT,
            "drift_share": DRIFT_SHARE,
        },
        "escenarios": resumen_escenarios,
        "calidad": {"referencia": calidad(referencia), "actual": calidad(actual)},
        "estaciones": {
            "brecha_media_abs_congelada": float(estaciones["brecha_congelada"].abs().mean()),
            "con_caida_mayor_50pct": int((estaciones["cambio_pct"] <= -50).sum()),
            "con_alza_mayor_50pct": int((estaciones["cambio_pct"] >= 50).sum()),
        },
    }
    if not por_mes.empty:
        resumen["por_mes"] = {
            mes: grupo.loc[grupo["drift"], "variable"].tolist()
            for mes, grupo in por_mes.groupby("mes")
        }
    (salida / "drift_resumen.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if graficas:
        figuras.mkdir(exist_ok=True)
        grafica_puntajes(drift_todo, figuras / "01_drift_por_variable.png")
        grafica_perfil_horario(referencia, actual, figuras / "02_perfil_horario.png")
        grafica_distribuciones(referencia, actual, figuras / "03_distribuciones.png")
        if not por_mes.empty:
            grafica_por_mes(por_mes, figuras / "04_drift_por_mes.png")
        grafica_estaciones(estaciones, figuras / "05_estaciones_media_congelada.png")
    return resumen


def imprimir(resumen: dict[str, Any], salida: Path) -> None:
    for nombre, esc in resumen["escenarios"].items():
        print(f"\n[{nombre}] {esc['descripcion']}")
        print(f"  Variables con drift: {esc['variables_con_drift'] or 'ninguna'}")
        print(
            f"  Proporción: {esc['proporcion_drift']:.2f}; drift del conjunto: "
            f"{esc['drift_del_conjunto']}"
        )
        print(f"  Pruebas de Evidently: {esc['pruebas_evidently']['conteo']}")
        for prueba in esc["pruebas_evidently"]["no_pasaron"]:
            print(f"    - {prueba}")
    if "por_mes" in resumen:
        print(f"\nDrift por mes: {resumen['por_mes']}")
    print(f"\nArtefactos en {salida}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datos", type=Path, default=DATOS, help="Carpeta con los parquets.")
    parser.add_argument("--salida", type=Path, default=SALIDA, help="Carpeta de resultados.")
    parser.add_argument(
        "--muestra",
        type=int,
        default=None,
        help="Filas al azar por conjunto (solo para pruebas rápidas).",
    )
    parser.add_argument("--sin-html", action="store_true", help="No escribir los HTML.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    referencia, actual = cargar_conjuntos(args.datos, args.muestra)
    resumen = generar(referencia, actual, args.salida, html=not args.sin_html)
    imprimir(resumen, args.salida)


if __name__ == "__main__":
    main()
