"""Semana 9: gráficas reproducibles de evaluación A/B Champion vs Challenger.

Entradas (generadas por compare_models.py y el análisis A/B + drift):
  comparison_metrics.json
  daily_metrics.csv
  station_comparison_metrics.csv
  station_ab_drift_analysis.csv

Ejemplo:
  python -m scripts.week9.plot_ab_final \
      --input-dir reports/week9/ab_testing \
      --output-dir reports/week9/ab_testing

No usa MLflow, no modifica el Registry y no recalcula predicciones.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


def leer_csv(ruta: Path, columnas: set[str]) -> pd.DataFrame:
    """Carga un CSV y comprueba que tenga los campos necesarios."""
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe el archivo requerido: {ruta}")
    datos = pd.read_csv(ruta)
    faltantes = columnas - set(datos.columns)
    if faltantes:
        raise ValueError(f"Columnas faltantes en {ruta.name}: {sorted(faltantes)}")
    if datos.empty:
        raise ValueError(f"El archivo {ruta.name} está vacío")
    return datos


def guardar(fig: plt.Figure, ruta: Path) -> None:
    fig.savefig(ruta, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Gráfica generada: {ruta}")


def grafica_metricas_globales(metricas: dict, ruta: Path) -> None:
    """MAE/RMSE offline pareados de los dos modelos."""
    champion = metricas["champion"]
    challenger = metricas["challenger"]
    columnas = ["MAE", "RMSE"]
    valores_champion = [float(champion["mae"]), float(champion["rmse"])]
    valores_challenger = [float(challenger["mae"]), float(challenger["rmse"])]
    x = np.arange(len(columnas))
    ancho = 0.34

    fig, ax = plt.subplots(figsize=(9, 5.2))
    a = ax.bar(x - ancho / 2, valores_champion, width=ancho, label="Champion v1")
    b = ax.bar(x + ancho / 2, valores_challenger, width=ancho, label="Challenger v2")
    ax.bar_label(a, fmt="%.4f", padding=4)
    ax.bar_label(b, fmt="%.4f", padding=4)
    ax.set_xticks(x, columnas)
    ax.set_ylabel("Error promedio (menor es mejor)")
    ax.set_ylim(0, max(valores_champion + valores_challenger) * 1.22)
    ax.set_title("MiBici P1: comparación global sobre marzo de 2025")
    ax.legend()
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    guardar(fig, ruta)


def grafica_mae_diario(diario: pd.DataFrame, ruta: Path) -> None:
    """Comparación día a día sin asumir homogeneidad de mejoras."""
    datos = diario.copy()
    datos["date"] = pd.to_datetime(datos["date"], errors="raise")
    datos = datos.sort_values("date")
    if datos["date"].duplicated().any():
        raise ValueError("El CSV diario contiene fechas duplicadas")

    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.plot(datos["date"], datos["champion_mae"], marker="o", markersize=3.5, label="Champion v1")
    ax.plot(datos["date"], datos["challenger_mae"], marker="o", markersize=3.5, label="Challenger v2")
    ax.set_xlabel("Fecha de evaluación (marzo 2025)")
    ax.set_ylabel("MAE diario (menor es mejor)")
    ax.set_title("MiBici P1: MAE diario de Champion y Challenger")
    # Limitar estrictamente al periodo observado: evita la marca de abril.
    inicio = datos["date"].min()
    fin = datos["date"].max()
    ax.set_xlim(inicio, fin)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%mar"))
    ax.grid(alpha=0.22)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    guardar(fig, ruta)


def grafica_estaciones(estaciones: pd.DataFrame, ruta: Path) -> None:
    """Muestra el rango principal sin ocultar estaciones con mejoras extremas."""
    datos = pd.to_numeric(estaciones["mae_improvement_pct"], errors="coerce")
    validos = datos.replace([np.inf, -np.inf], np.nan).dropna()
    if validos.empty:
        raise ValueError("No hay porcentajes de mejora válidos por estación")

    mejora = int((estaciones["challenger_mae"] < estaciones["champion_mae"]).sum())
    deterioro = int((estaciones["challenger_mae"] > estaciones["champion_mae"]).sum())
    sin_cambio = int(len(estaciones) - mejora - deterioro)

    # Rango ampliado: la estación 180 (~74 %) no comprime el histograma.
    limite_izq, limite_der = -15.0, 20.0
    centro = validos.between(limite_izq, limite_der, inclusive="both")
    fuera_izq = int((validos < limite_izq).sum())
    fuera_der = int((validos > limite_der).sum())
    extremos = estaciones.loc[
        pd.to_numeric(estaciones["mae_improvement_pct"], errors="coerce") > limite_der,
        ["station", "mae_improvement_pct"],
    ].sort_values("mae_improvement_pct", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.hist(
        validos[centro].to_numpy(),
        bins=np.linspace(limite_izq, limite_der, 29),
        edgecolor="white", linewidth=0.6,
    )
    ax.axvline(0, linestyle="--", linewidth=1.6, label="Sin cambio (0 %)")
    ax.set_xlim(limite_izq, limite_der)
    ax.set_xlabel("Mejora porcentual de MAE por estación (positivo = mejora)")
    ax.set_ylabel("Número de estaciones")
    ax.set_title(f"MiBici P1: mejora por estación — {mejora} mejoran, {deterioro} empeoran")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.22)

    # Los valores fuera del rango no se descartan del cómputo estadístico.
    aviso = (
        f"Vista ampliada [{limite_izq:.0f} %, +{limite_der:.0f} %]. "
        f"Mostradas: {int(centro.sum())}/{int(len(validos))}; "
        f"fuera del rango: {fuera_izq} por debajo, {fuera_der} por encima. "
        f"Sin cambio: {sin_cambio}."
    )
    if not extremos.empty:
        principal = extremos.iloc[0]
        aviso += (
            f"  Extremo: estación {int(principal['station'])} "
            f"(+{principal['mae_improvement_pct']:.2f} %)."
        )
    fig.text(0.5, 0.025, aviso, ha="center", va="center", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    guardar(fig, ruta)


def grafica_drift_vs_mejora(integrado: pd.DataFrame, ruta: Path) -> None:
    """Asociación descriptiva entre cambio de demanda y mejora del MAE."""
    datos = integrado[["station", "cambio_pct", "mae_improvement_pct"]].copy()
    datos["cambio_pct"] = pd.to_numeric(datos["cambio_pct"], errors="coerce")
    datos["mae_improvement_pct"] = pd.to_numeric(datos["mae_improvement_pct"], errors="coerce")
    datos["cambio_demanda_abs_pct"] = datos["cambio_pct"].abs()
    datos = datos.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["cambio_demanda_abs_pct", "mae_improvement_pct"]
    )
    if datos.empty:
        raise ValueError("No hay pares válidos para comparar drift y mejora de MAE")

    correlacion = datos["cambio_demanda_abs_pct"].corr(datos["mae_improvement_pct"])
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.scatter(datos["cambio_demanda_abs_pct"], datos["mae_improvement_pct"], alpha=0.58, s=27)
    ax.axhline(0, linestyle="--", linewidth=1.4, label="Sin mejora")
    ax.axvline(20, linestyle=":", linewidth=1.2, label="Cambio de demanda de 20 %")
    ax.set_xlabel("Magnitud del cambio de demanda ref. vs. actual (%)")
    ax.set_ylabel("Mejora del MAE del Challenger (%)")
    texto = f"r = {correlacion:.3f}" if np.isfinite(correlacion) else "r no definido"
    ax.set_title(f"MiBici P1: demanda vs. mejora del Challenger ({texto})")
    ax.legend()
    ax.grid(alpha=0.22)
    fig.text(0.5, 0.01, "Relación descriptiva; no representa causalidad ni drift global de Evidently.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    guardar(fig, ruta)


def generar(input_dir: Path, output_dir: Path) -> list[Path]:
    """Valida entradas y produce cuatro gráficas sin cambiar los datos originales."""
    ruta_metricas = input_dir / "comparison_metrics.json"
    if not ruta_metricas.is_file():
        raise FileNotFoundError(f"No existe: {ruta_metricas}")
    with ruta_metricas.open(encoding="utf-8") as archivo:
        metricas = json.load(archivo)
    for modelo in ("champion", "challenger"):
        if modelo not in metricas or not {"mae", "rmse"} <= set(metricas[modelo]):
            raise ValueError(f"Faltan MAE/RMSE del modelo {modelo} en comparison_metrics.json")

    diario = leer_csv(input_dir / "daily_metrics.csv", {"date", "champion_mae", "challenger_mae"})
    estaciones = leer_csv(
        input_dir / "station_comparison_metrics.csv",
        {"station", "observations", "champion_mae", "challenger_mae", "mae_improvement_pct"},
    )
    integrado = leer_csv(
        input_dir / "station_ab_drift_analysis.csv",
        {"station", "cambio_pct", "mae_improvement_pct"},
    )
    if estaciones["station"].duplicated().any() or integrado["station"].duplicated().any():
        raise ValueError("Hay estaciones duplicadas en los archivos por estación")
    if set(estaciones["station"]) != set(integrado["station"]):
        raise ValueError("Las estaciones de A/B y Drift no coinciden")
    filas = metricas.get("evaluation_rows")
    if filas is not None and int(estaciones["observations"].sum()) != int(filas):
        raise ValueError("La suma de observaciones por estación no coincide con el JSON global")

    output_dir.mkdir(parents=True, exist_ok=True)
    rutas = [
        output_dir / "ab_final_global_mae_rmse.png",
        output_dir / "ab_final_mae_diario.png",
        output_dir / "ab_final_mejora_por_estacion.png",
        output_dir / "ab_final_drift_vs_mejora_mae.png",
    ]
    grafica_metricas_globales(metricas, rutas[0])
    grafica_mae_diario(diario, rutas[1])
    grafica_estaciones(estaciones, rutas[2])
    grafica_drift_vs_mejora(integrado, rutas[3])
    print(f"\nVISUALIZACIONES COMPLETADAS: {len(rutas)} archivos PNG")
    return rutas


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera cuatro gráficas finales A/B y drift de MiBici")
    parser.add_argument("--input-dir", type=Path, default=Path("reports/week9/ab_testing"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/week9/ab_testing"))
    args = parser.parse_args()
    generar(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
