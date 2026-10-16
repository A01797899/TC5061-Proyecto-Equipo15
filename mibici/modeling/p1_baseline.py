"""P1 MiBici: baseline de demanda horaria por estación, con evaluación cronológica.

Ejecutar desde la raíz del repositorio:
    python -m mibici.modeling.p1_baseline

No genera ni versiona archivos de datos. Los runs quedan en MLflow local.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mibici.features import complete_hourly_panel

TEAM_EXPERIMENT = "mibici-equipo15"
COLS = ["viaje_id", "usuario_id", "genero", "anio_nacimiento", "inicio", "fin", "origen", "destino"]
SPLITS = {"train": ("2024-01-01", "2024-09-01"), "val": ("2024-09-01", "2024-11-01"), "test": ("2024-11-01", "2025-01-01")}


def root_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def git_sha(root: Path) -> str:
    res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False)
    return res.stdout.strip() if res.returncode == 0 else "uncommitted"


def aggregate_monthly(raw: Path, chunk_size: int) -> tuple[pd.DataFrame, dict]:
    files = sorted(raw.glob("datos_abiertos_2024_*.csv"))
    if len(files) != 12 or [p.stem[-2:] for p in files] != [f"{i:02d}" for i in range(1,13)]:
        raise ValueError("Se requieren exactamente los 12 CSV datos_abiertos_2024_01.csv ... _12.csv")
    count_parts = []
    manifest = {"files": [], "valid_start_records": 0, "invalid_start_records": 0, "missing_origin_records": 0}
    for path in files:
        month = int(path.stem[-2:])
        encoding = "utf-8" if month == 7 else "latin1"
        date_format = "%d/%m/%Y %H:%M" if month in (9,10) else "%Y-%m-%d %H:%M:%S"
        month_parts = []
        file_rows = 0
        for chunk in pd.read_csv(path, encoding=encoding, dtype=str, chunksize=chunk_size):
            if len(chunk.columns) != 8:
                raise ValueError(f"Esquema inesperado en {path.name}: {len(chunk.columns)} columnas")
            chunk.columns = COLS
            file_rows += len(chunk)
            starts = pd.to_datetime(chunk["inicio"], format=date_format, errors="coerce")
            origins = chunk["origen"].astype("string").str.strip()
            good = starts.notna() & origins.notna() & origins.ne("") & (starts.dt.year == 2024) & (starts.dt.month == month)
            manifest["invalid_start_records"] += int(starts.isna().sum())
            manifest["missing_origin_records"] += int((origins.isna() | origins.eq("")).sum())
            manifest["valid_start_records"] += int(good.sum())
            if good.any():
                part = pd.DataFrame({"station": origins.loc[good].astype(str).to_numpy(),
                                     "hour": starts.loc[good].dt.floor("h").to_numpy()})
                month_parts.append(part.groupby(["station", "hour"], observed=True).size().rename("trips"))
        if month_parts:
            m = pd.concat(month_parts).groupby(level=[0,1]).sum().reset_index()
            count_parts.append(m)
        manifest["files"].append({"name": path.name, "rows": file_rows, "sha256": sha256_file(path)})
        print(f"{path.name}: {file_rows:,} registros", flush=True)
    observed = pd.concat(count_parts, ignore_index=True)
    observed = observed.groupby(["station", "hour"], as_index=False)["trips"].sum()
    manifest["total_rows"] = sum(f["rows"] for f in manifest["files"])
    return observed, manifest


def panel(
    observed: pd.DataFrame,
    stations: pd.Index,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Construye el panel horario completo reutilizando
    el módulo de feature engineering.

    Conserva la interfaz y las columnas del baseline P1.
    """

    departures = observed.rename(
        columns={
            "station": "estacion_id",
            "hour": "hora",
            "trips": "salidas",
        }
    )

    result = complete_hourly_panel(
        departures=departures,
        stations=stations,
        start=start,
        end=end,
    )

    return result.rename(
        columns={
            "estacion_id": "station",
            "hora": "hour",
            "salidas": "y",
            "dia_semana": "weekday",
            "hora_dia": "hour_of_day",
        }
    ).drop(columns=["fin_semana"])


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    delta = y - pred
    return {"mae": float(np.mean(np.abs(delta))), "rmse": float(np.sqrt(np.mean(delta ** 2)))}


def evaluate(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, mode: str) -> tuple[dict, pd.DataFrame]:
    global_mean = float(train["y"].mean())
    group_cols = ["station"] if mode == "station_mean" else ["station", "weekday", "hour_of_day"]
    means = train.groupby(group_cols, observed=True)["y"].mean().rename("pred")
    results = {}
    test_output = None
    for name, df in (("train", train), ("val", val), ("test", test)):
        # Mismos promedios de TRAIN para todos los splits: ningún dato futuro alimenta el baseline.
        pred = pd.MultiIndex.from_frame(df[group_cols]).map(means).to_numpy(dtype=float) if len(group_cols) > 1 else df["station"].map(means).to_numpy(dtype=float)
        pred = np.where(np.isnan(pred), global_mean, pred)
        pred = np.clip(pred, 0, None)
        y = df["y"].to_numpy(dtype=float)
        for k, v in metrics(y, pred).items():
            results[f"{name}_{k}"] = v
        if name == "test":
            test_output = pd.DataFrame({"hour": df["hour"], "actual": y, "pred": pred})
    return results, test_output


def write_artifacts(folder: Path, mode: str, out: pd.DataFrame, manifest: dict, results: dict) -> None:
    with (folder / "data_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    with (folder / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    by_day = out.assign(date=out["hour"].dt.date).groupby("date", as_index=False)[["actual", "pred"]].sum()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(by_day["date"], by_day["actual"], label="Viajes reales", linewidth=1)
    ax.plot(by_day["date"], by_day["pred"], label="Predicción", linewidth=1)
    ax.set_title(f"P1: viajes diarios agregados en test — {mode}")
    ax.set_ylabel("Salidas por día")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(folder / "p1_test_real_vs_predicho.png", dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="P1: baselines MiBici estación/hora en MLflow")
    parser.add_argument("--raw-dir", type=Path, default=root_dir() / "data" / "raw")
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--tracking-uri", default=None, help="Ej. sqlite:///mlflow.db")
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size debe ser positivo")
    root = root_dir()
    raw = args.raw_dir.resolve()
    if not raw.is_dir():
        parser.error(f"No existe data/raw: {raw}")
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    elif not os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(
            f"sqlite:///{root / 'mlflow.db'}"
        )
    observed, manifest = aggregate_monthly(raw, args.chunk_size)
    # Solo estaciones observadas en entrenamiento; las incorporaciones futuras se reportan aparte.
    train_start, train_end = SPLITS["train"]
    train_obs = observed[(observed["hour"] >= train_start) & (observed["hour"] < train_end)]
    stations = pd.Index(sorted(train_obs["station"].unique()), name="station")
    unseen = sorted(set(observed["station"]) - set(stations))
    manifest["training_stations"] = len(stations)
    manifest["stations_absent_in_training"] = unseen
    print(f"Estaciones en train: {len(stations)}; ausentes de train: {len(unseen)}", flush=True)
    frames = {}
    for split, (start, end) in SPLITS.items():
        subset = observed[(observed["hour"] >= start) & (observed["hour"] < end)]
        frames[split] = panel(subset, stations, start, end)
        print(f"{split}: {len(frames[split]):,} filas estación-hora", flush=True)
    mlflow.set_experiment(TEAM_EXPERIMENT)
    commit = git_sha(root)
    for mode in ("station_mean", "station_weekday_hour_mean"):
        results, output = evaluate(frames["train"], frames["val"], frames["test"], mode)
        print(f"{mode}: test_mae={results['test_mae']:.4f}; test_rmse={results['test_rmse']:.4f}", flush=True)
        with mlflow.start_run(run_name=f"P1_baseline_{mode}"):
            mlflow.set_tags({"team": "15", "question": "P1", "git_commit": commit,
                             "dataset": "MiBici 2024", "data_version": "sha256_manifest",
                             "target": "trips_started_per_station_hour", "prediction_horizon": "next_hour"})
            mlflow.log_params({"model": mode, "train_period": "2024-01_to_2024-08",
                               "val_period": "2024-09_to_2024-10", "test_period": "2024-11_to_2024-12",
                               "chunk_size": args.chunk_size, "train_stations": len(stations),
                               "seed": 42, "missing_hour_policy": "zero", "unseen_station_policy": "excluded_and_reported"})
            mlflow.log_metrics(results)
            with tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                write_artifacts(folder, mode, output, manifest, results)
                mlflow.log_artifacts(str(folder), artifact_path="p1")
    print("Experimento registrado:", TEAM_EXPERIMENT)


if __name__ == "__main__":
    main()
