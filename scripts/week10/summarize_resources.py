"""Resume CPU y memoria por escenario y contenedor.

Entradas:
    reports/week10/metrics/{baseline,S1..S5}_resources.csv
    reports/week10/locust/S5_capacity/locust_stats_history.csv
Salidas:
    reports/week10/metrics/resource_summary.csv
    reports/week10/metrics/resource_capacity_steps.csv

En S1-S5 solo se usan las muestras dentro del periodo de ejecución de Locust.
Los valores faltantes se conservan como faltantes y no se reemplazan por cero.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "reports/week10/metrics"
S5_HISTORY = ROOT / "reports/week10/locust/S5_capacity/locust_stats_history.csv"

FILES = {
    "Base": "baseline_resources.csv",
    "S1": "S1_resources.csv",
    "S2": "S2_resources.csv",
    "S3": "S3_resources.csv",
    "S4": "S4_resources.csv",
    "S5": "S5_resources.csv",
}
LOCUST_FOLDERS = {
    "S1": "S1_low",
    "S2": "S2_medium",
    "S3": "S3_high",
    "S4": "S4_stress",
    "S5": "S5_capacity",
}
STEP_SECONDS = 60
WARMUP_SECONDS = 15


def load(name):
    """Lee un CSV de recursos y agrega el nombre corto del servicio."""
    data = pd.read_csv(METRICS / name)
    data["servicio"] = data["container"].str.extract(r"-(api|mlflow)-")[0]
    return data


def locust_window(data, scenario):
    """Conserva solo las muestras tomadas mientras Locust generaba carga."""
    folder = LOCUST_FOLDERS.get(scenario)
    if folder is None:
        return data
    history = pd.read_csv(ROOT / "reports/week10/locust" / folder / "locust_stats_history.csv")
    start = pd.to_datetime(history["Timestamp"].min(), unit="s", utc=True)
    end = pd.to_datetime(history["Timestamp"].max(), unit="s", utc=True)
    moment = pd.to_datetime(data["timestamp"])
    return data[(moment >= start) & (moment <= end)]


def summarize(data):
    """Calcula media y máximo de CPU y memoria por servicio."""
    return data.groupby("servicio").agg(
        muestras=("cpu_percent", "size"),
        cpu_faltantes=("cpu_percent", lambda s: int(s.isna().sum())),
        cpu_media_pct=("cpu_percent", "mean"),
        cpu_max_pct=("cpu_percent", "max"),
        mem_media_mib=("mem_usage_mib", "mean"),
        mem_max_mib=("mem_usage_mib", "max"),
        mem_limite_mib=("mem_limit_mib", "max"),
    )


def capacity_steps(data):
    """Asigna cada muestra de S5 a su escalón, sin los primeros segundos de cada uno."""
    history = pd.read_csv(S5_HISTORY)
    users = history[history["Name"] == "Aggregated"][["Timestamp", "User Count"]]
    start = pd.to_datetime(users["Timestamp"].min(), unit="s", utc=True)

    data = data.copy()
    data["t"] = (pd.to_datetime(data["timestamp"]) - start).dt.total_seconds()
    data = data[(data["t"] >= 0) & (data["t"] % STEP_SECONDS >= WARMUP_SECONDS)]
    data["escalon"] = (data["t"] // STEP_SECONDS).astype(int)

    step_users = users.assign(
        escalon=((users["Timestamp"] - users["Timestamp"].min()) // STEP_SECONDS).astype(int)
    ).groupby("escalon")["User Count"].max()

    data = data[data["escalon"].isin(step_users.index)]
    data["usuarios"] = data["escalon"].map(step_users)
    return data


def main():
    rows = []
    for scenario, name in FILES.items():
        table = summarize(locust_window(load(name), scenario)).reset_index()
        table.insert(0, "escenario", scenario)
        rows.append(table)
    summary = pd.concat(rows).round(2)
    summary.to_csv(METRICS / "resource_summary.csv", index=False)

    steps = capacity_steps(load(FILES["S5"]))
    step_table = (
        steps.groupby(["usuarios", "servicio"])
        .agg(
            muestras=("cpu_percent", "size"),
            cpu_media_pct=("cpu_percent", "mean"),
            cpu_max_pct=("cpu_percent", "max"),
            mem_media_mib=("mem_usage_mib", "mean"),
            mem_max_mib=("mem_usage_mib", "max"),
        )
        .round(2)
        .reset_index()
    )
    step_table.to_csv(METRICS / "resource_capacity_steps.csv", index=False)

    print(summary.to_string(index=False))
    print()
    print(step_table.to_string(index=False))


if __name__ == "__main__":
    main()
