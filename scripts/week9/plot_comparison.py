
"""Visualizaciones del Champion P1 - Semana 9."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def load_daily_metrics(path: Path) -> pd.DataFrame:
    """Carga y valida las métricas diarias."""

    if not path.is_file():
        raise FileNotFoundError(path)

    df = pd.read_csv(
        path,
        parse_dates=["date"],
    )

    required = {
        "date",
        "actual",
        "champion_pred",
        "champion_mae",
        "champion_rmse",
        "observations",
    }

    if df.empty or not required.issubset(df.columns):
        raise ValueError(
            "CSV diario vacío o con columnas faltantes"
        )

    return df.sort_values("date")


def load_station_metrics(path: Path) -> pd.DataFrame:
    """Carga y valida las métricas por estación."""

    if not path.is_file():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = {
        "station",
        "champion_mae",
        "champion_rmse",
        "observations",
    }

    if df.empty or not required.issubset(df.columns):
        raise ValueError(
            "CSV de estaciones vacío o con columnas faltantes"
        )

    return df


def plot_daily_demand(
    daily: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Genera la demanda real vs. predicha por día."""

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        daily["date"],
        daily["actual"],
        label="Demanda real",
        linewidth=2,
    )

    ax.plot(
        daily["date"],
        daily["champion_pred"],
        label="Champion P1",
        linewidth=2,
        linestyle="--",
    )

    ax.set_title(
        "MiBici - Demanda diaria real vs. Champion P1\n"
        "Marzo de 2025"
    )

    ax.set_xlabel("Fecha")
    ax.set_ylabel("Total de salidas diarias")

    ax.legend()
    ax.grid(alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()

    output = (
        output_dir / "champion_actual_vs_predicted.png"
    )

    fig.savefig(output, dpi=160)
    plt.close(fig)

    return output


def plot_daily_error(
    daily: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Genera la gráfica del MAE diario del Champion."""

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        daily["date"],
        daily["champion_mae"],
        marker="o",
        markersize=3,
        label="MAE diario - Champion",
    )

    global_mae = (
        daily["champion_mae"]
        .mul(daily["observations"])
        .sum()
        / daily["observations"].sum()
    )

    ax.axhline(
        global_mae,
        linestyle="--",
        label=f"MAE global = {global_mae:.4f}",
    )

    ax.set_title(
        "MiBici - Error absoluto medio diario\n"
        "Champion P1 - Marzo de 2025"
    )

    ax.set_xlabel("Fecha")
    ax.set_ylabel(
        "MAE (salidas por estación-hora)"
    )

    ax.legend()
    ax.grid(alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()

    output = output_dir / "champion_daily_mae.png"

    fig.savefig(output, dpi=160)
    plt.close(fig)

    return output


def plot_station_mae(
    station: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Genera un histograma de MAE por estación."""

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(
        station["champion_mae"],
        bins=30,
        edgecolor="white",
    )

    ax.set_title(
        "MiBici - Distribución del MAE por estación\n"
        "Champion P1 - Marzo de 2025"
    )

    ax.set_xlabel(
        "MAE de cada estación"
    )

    ax.set_ylabel(
        "Número de estaciones"
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    output = output_dir / "champion_station_mae.png"

    fig.savefig(output, dpi=160)
    plt.close(fig)

    return output


def generate_plots(output_dir: Path):
    """Genera las tres visualizaciones del Champion."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    daily = load_daily_metrics(
        output_dir / "champion_daily_metrics.csv"
    )

    station = load_station_metrics(
        output_dir / "champion_station_metrics.csv"
    )

    figures = [
        plot_daily_demand(daily, output_dir),
        plot_daily_error(daily, output_dir),
        plot_station_mae(station, output_dir),
    ]

    print("\n=== GRÁFICAS GENERADAS ===")

    for path in figures:
        print(path)

    return figures


def main():
    parser = argparse.ArgumentParser(
        description="Gráficas Champion P1 - Semana 9"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "reports/week9/ab_testing"
        ),
    )

    args = parser.parse_args()

    generate_plots(args.output)


if __name__ == "__main__":
    main()
