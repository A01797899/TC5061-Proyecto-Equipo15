"""Genera tablas y gráficas comparativas de las pruebas de carga.

Entradas:
    reports/week10/locust/S*/locust_stats.csv
    reports/week10/locust/S5_capacity/locust_stats_history.csv
    reports/week10/metrics/*_resources.csv
    reports/week10/metrics/resource_summary.csv
Salidas:
    reports/week10/metrics/load_summary.csv
    reports/week10/metrics/load_capacity_steps.csv
    reports/week10/figures/*.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
LOCUST = ROOT / "reports/week10/locust"
METRICS = ROOT / "reports/week10/metrics"
FIGURES = ROOT / "reports/week10/figures"

SCENARIOS = {
    "S1": ("S1_low", 5),
    "S2": ("S2_medium", 20),
    "S3": ("S3_high", 50),
    "S4": ("S4_stress", 100),
    "S5": ("S5_capacity", None),
}
MEAN_WAIT_S = 1.5
STEP_SECONDS = 60
WARMUP_SECONDS = 15

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
PCT_COLORS = {"P50": "#9ec5f4", "P95": "#3987e5", "P99": "#1c5cab"}
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
SERVICE_COLORS = {"api": BLUE, "mlflow": ORANGE}
SERVICE_LABELS = {"api": "API (FastAPI)", "mlflow": "MLflow"}

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 10,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlecolor": TEXT,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)


def aggregated(path):
    """Devuelve la fila Aggregated de un archivo locust_stats.csv."""
    data = pd.read_csv(path)
    return data[data["Name"] == "Aggregated"].iloc[0]


def load_summary():
    """Construye la tabla comparativa de los cinco escenarios."""
    rows = []
    for scenario, (folder, users) in SCENARIOS.items():
        row = aggregated(LOCUST / folder / "locust_stats.csv")
        total = int(row["Request Count"])
        failures = int(row["Failure Count"])
        rows.append(
            {
                "escenario": scenario,
                "usuarios": users if users is not None else "50-300",
                "solicitudes": total,
                "fallos": failures,
                "error_pct": round(100 * failures / total, 3),
                "media_ms": round(row["Average Response Time"], 1),
                "p50_ms": row["50%"],
                "p95_ms": row["95%"],
                "p99_ms": row["99%"],
                "max_ms": round(row["Max Response Time"], 1),
                "rps": round(row["Requests/s"], 2),
                "rps_ofrecido": round(users / MEAN_WAIT_S, 1) if users else None,
            }
        )
    return pd.DataFrame(rows)


def s5_history():
    """Lee el historial agregado de S5 con tiempo relativo y escalón."""
    data = pd.read_csv(LOCUST / "S5_capacity/locust_stats_history.csv")
    data = data[data["Name"] == "Aggregated"].copy()
    data["t"] = data["Timestamp"] - data["Timestamp"].min()
    data["escalon"] = (data["t"] // STEP_SECONDS).astype(int)
    return data


def capacity_steps(history):
    """Resume cada escalón de S5 sin los primeros segundos de incorporación."""
    steady = history[history["t"] % STEP_SECONDS >= WARMUP_SECONDS]
    table = steady.groupby("escalon").agg(
        usuarios=("User Count", "max"),
        rps=("Requests/s", "mean"),
        fallos_s=("Failures/s", "mean"),
        p50_ms=("50%", "median"),
        p95_ms=("95%", "median"),
        p99_ms=("99%", "median"),
    )
    table["rps_ofrecido"] = table["usuarios"] / MEAN_WAIT_S
    table["rps_vs_ofrecido_pct"] = 100 * table["rps"] / table["rps_ofrecido"]
    return table.round(1).reset_index(drop=True)


def finish(fig, name):
    """Guarda y cierra una figura."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, facecolor="white")
    plt.close(fig)


def plot_latency(summary):
    """Barras agrupadas de P50, P95 y P99 para S1-S4."""
    data = summary[summary["escenario"] != "S5"]
    labels = [f"{s}\n{u} usuarios" for s, u in zip(data["escenario"], data["usuarios"])]
    width = 0.26
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for i, (key, column) in enumerate([("P50", "p50_ms"), ("P95", "p95_ms"), ("P99", "p99_ms")]):
        positions = [x + (i - 1) * width for x in range(len(data))]
        bars = ax.bar(positions, data[column], width - 0.03, color=PCT_COLORS[key], label=key)
        if key == "P99":
            ax.bar_label(bars, fmt="%.0f", padding=2, color=MUTED, fontsize=8)
    ax.set_xticks(range(len(data)), labels)
    ax.set_ylabel("Tiempo de respuesta (ms)")
    ax.set_title("Latencia de POST /predict por escenario")
    ax.grid(axis="x", visible=False)
    ax.legend(ncols=3, loc="upper left")
    finish(fig, "01_latency.png")


def plot_throughput(summary, steps):
    """RPS medido frente a usuarios, con la carga ofrecida como referencia."""
    data = summary[summary["escenario"] != "S5"]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    users = range(0, 310, 10)
    ax.plot(users, [u / MEAN_WAIT_S for u in users], color=MUTED, linestyle="--",
            linewidth=1.2, label="Carga ofrecida (usuarios / 1.5 s)")
    ax.plot(steps["usuarios"], steps["rps"], color=ORANGE, linewidth=2, marker="o",
            markersize=6, label="S5 por escalones (estable)")
    ax.plot(data["usuarios"], data["rps"], color=BLUE, linewidth=0, marker="o",
            markersize=8, label="S1-S4")
    for _, row in data.iterrows():
        ax.annotate(row["escenario"], (row["usuarios"], row["rps"]),
                    textcoords="offset points", xytext=(-6, 8), ha="right", color=TEXT, fontsize=8)
    last = steps.iloc[-1]
    ax.annotate(f"{last['rps']:.0f} req/s", (last["usuarios"], last["rps"]),
                textcoords="offset points", xytext=(0, -16), ha="center", color=TEXT, fontsize=8)
    ax.set_xlabel("Usuarios concurrentes")
    ax.set_ylabel("Peticiones por segundo (req/s)")
    ax.set_title("Throughput medido frente a la carga ofrecida")
    ax.set_xlim(0, 310)
    ax.set_ylim(0, None)
    ax.legend(loc="upper left")
    finish(fig, "02_throughput.png")


def plot_errors(summary):
    """Tasa de error por escenario con el total de solicitudes."""
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    x = range(len(summary))
    ax.bar(x, summary["error_pct"], 0.6, color=BLUE)
    for i, row in summary.iterrows():
        ax.annotate(f"{row['error_pct']:.2f} %\n{row['fallos']} de {row['solicitudes']:,}",
                    (i, row["error_pct"]), textcoords="offset points", xytext=(0, 4),
                    ha="center", va="bottom", color=TEXT, fontsize=8)
    labels = [f"{s}\n{u} usuarios" for s, u in zip(summary["escenario"], summary["usuarios"])]
    ax.set_xticks(list(x), labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Tasa de error (%)")
    ax.set_title("Tasa de error de POST /predict por escenario")
    ax.grid(axis="x", visible=False)
    finish(fig, "03_error_rate.png")


def plot_capacity(history):
    """Paneles de S5 con usuarios, throughput, latencia y CPU en el tiempo."""
    resources = pd.read_csv(METRICS / "S5_resources.csv")
    start = pd.to_datetime(history["Timestamp"].min(), unit="s", utc=True)
    resources["t"] = (pd.to_datetime(resources["timestamp"]) - start).dt.total_seconds()
    resources = resources[(resources["t"] >= 0) & (resources["t"] <= history["t"].max())]
    api = resources[resources["container"].str.contains("-api-")]
    mlflow = resources[resources["container"].str.contains("-mlflow-")]

    fig, axes = plt.subplots(4, 1, figsize=(7.5, 8.5), sharex=True)
    axes[0].plot(history["t"], history["User Count"], color=MUTED, linewidth=2)
    axes[0].set_ylabel("Usuarios")
    axes[0].set_title("S5: capacidad por escalones de 60 s")

    axes[1].plot(history["t"], history["Requests/s"], color=BLUE, linewidth=2)
    axes[1].set_ylabel("req/s")

    axes[2].plot(history["t"], history["50%"], color=PCT_COLORS["P50"], linewidth=2, label="P50")
    axes[2].plot(history["t"], history["95%"], color=PCT_COLORS["P99"], linewidth=2, label="P95")
    axes[2].set_ylabel("Latencia (ms)")
    axes[2].legend(loc="upper left", ncols=2)

    axes[3].plot(api["t"], api["cpu_percent"], color=BLUE, linewidth=2, label=SERVICE_LABELS["api"])
    axes[3].plot(mlflow["t"], mlflow["cpu_percent"], color=ORANGE, linewidth=2,
                 label=SERVICE_LABELS["mlflow"])
    axes[3].axhline(400, color=MUTED, linestyle="--", linewidth=1)
    axes[3].annotate("Límite de la VM: 4 CPU (400 %)", (5, 400), textcoords="offset points",
                     xytext=(0, -12), color=MUTED, fontsize=8)
    axes[3].set_ylabel("CPU (%)")
    axes[3].set_ylim(0, 430)
    axes[3].set_xlabel("Tiempo desde el inicio (s)")
    axes[3].legend(loc="upper left", ncols=2, bbox_to_anchor=(0, 0.85))

    for ax in axes:
        for edge in range(STEP_SECONDS, int(history["t"].max()) + 1, STEP_SECONDS):
            ax.axvline(edge, color=GRID, linewidth=1)
    fig.align_ylabels(axes)
    finish(fig, "04_capacity_s5.png")


def plot_resources():
    """CPU y memoria media por escenario para cada contenedor."""
    summary = pd.read_csv(METRICS / "resource_summary.csv")
    summary = summary[summary["escenario"] != "S5"]
    scenarios = list(dict.fromkeys(summary["escenario"]))
    width = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for i, service in enumerate(["api", "mlflow"]):
        data = summary[summary["servicio"] == service].set_index("escenario").loc[scenarios]
        positions = [x + (i - 0.5) * width for x in range(len(scenarios))]
        cpu = axes[0].bar(positions, data["cpu_media_pct"], width - 0.03,
                          color=SERVICE_COLORS[service], label=SERVICE_LABELS[service])
        axes[0].bar_label(cpu, fmt="%.0f", padding=2, color=MUTED, fontsize=8)
        mem = axes[1].bar(positions, data["mem_media_mib"], width - 0.03,
                          color=SERVICE_COLORS[service], label=SERVICE_LABELS[service])
        axes[1].bar_label(mem, fmt="%.0f", padding=2, color=MUTED, fontsize=8)
    for ax in axes:
        ax.set_xticks(range(len(scenarios)), scenarios)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("CPU media (%)")
    axes[0].set_title("CPU media por escenario")
    axes[1].set_ylabel("Memoria media (MiB)")
    axes[1].set_title("Memoria media por escenario")
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    finish(fig, "05_resources.png")


def main():
    summary = load_summary()
    history = s5_history()
    steps = capacity_steps(history)

    METRICS.mkdir(parents=True, exist_ok=True)
    summary.to_csv(METRICS / "load_summary.csv", index=False)
    steps.to_csv(METRICS / "load_capacity_steps.csv", index=False)

    plot_latency(summary)
    plot_throughput(summary, steps)
    plot_errors(summary)
    plot_capacity(history)
    plot_resources()

    print(summary.to_string(index=False))
    print()
    print(steps.to_string(index=False))


if __name__ == "__main__":
    main()
