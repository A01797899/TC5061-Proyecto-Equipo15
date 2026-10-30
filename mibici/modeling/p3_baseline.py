"""P3 MiBici: clustering reproducible de estaciones y registro MLflow.

Ejecutar desde la raíz del repositorio:
 python -m mibici.modeling.p3_baseline --tracking-uri sqlite:///mlflow.db
"""
from __future__ import annotations

import argparse
import hashlib
import json
# Solo listas de argumentos, sin shell.
import subprocess  # nosec B404
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from mibici.features import hourly_usage_profiles


def parse_args():
    p = argparse.ArgumentParser(description="Baseline P3: agrupamiento de estaciones MiBici")
    p.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--catalog", type=Path, default=None)
    p.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    p.add_argument("--experiment", default="mibici-equipo15")
    p.add_argument("--min-departures", type=int, default=100)
    p.add_argument("--chunk-size", type=int, default=100_000)
    p.add_argument("--k-min", type=int, default=2)
    p.add_argument("--k-max", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=Path("reports/figures/p3"))
    return p.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def current_commit() -> str:
    # git con argumentos fijos; el ejecutable sale de PATH y no hay shell.
    p = subprocess.run(  # nosec B603 B607
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return p.stdout.strip() if p.returncode == 0 else "unknown"


def build_profiles(files: list[Path], chunk_size: int) -> tuple[pd.DataFrame, list[dict]]:
    # Mismo método que el notebook P3: agregación por estación, fecha, hora y tipo de día.
    parts, months = [], []
    for file in files:
        month = int(file.stem[-2:])
        encoding = "utf-8" if month == 7 else "latin1"
        total = invalid = empty = 0
        for chunk in pd.read_csv(file, dtype=str, chunksize=chunk_size, encoding=encoding):
            if chunk.shape[1] != 8:
                raise ValueError(f"{file.name}: se esperaban 8 columnas, recibidas {chunk.shape[1]}")
            chunk.columns = ["viaje_id", "usuario_id", "genero", "anio_nacimiento", "inicio", "fin", "origen", "destino"]
            date = pd.to_datetime(chunk["inicio"], format="%d/%m/%Y %H:%M" if month in (9, 10) else "%Y-%m-%d %H:%M:%S", errors="coerce")
            ids = chunk["origen"].astype("string").str.strip()
            total += len(chunk)
            invalid += int(date.isna().sum())
            empty += int(ids.isna().sum() + ids.eq("").fillna(False).sum())
            data = pd.DataFrame({"estacion": ids, "fecha": date.dt.normalize(), "hora": date.dt.hour, "fin_de_semana": date.dt.dayofweek.ge(5)})
            data = data.dropna(subset=["estacion", "fecha", "hora", "fin_de_semana"])
            data = data.loc[data["estacion"].ne("")]
            parts.append(data.groupby(["estacion", "fecha", "hora", "fin_de_semana"], observed=True).size().rename("salidas").reset_index())
        months.append({"mes": month, "viajes": total, "fechas_invalidas": invalid, "origen_vacio": empty})
        print(f"{file.name}: {total:,} viajes")
    agg = pd.concat(parts, ignore_index=True).groupby(["estacion", "fecha", "hora", "fin_de_semana"], as_index=False)["salidas"].sum()
    expected = sum(x["viajes"] - x["fechas_invalidas"] - x["origen_vacio"] for x in months)
    if int(agg["salidas"].sum()) != expected:
        raise AssertionError("La suma de salidas agregadas no corresponde a los viajes válidos")
    # Reutilizar la función común de feature engineering
    aggregated_trips = agg.rename(
        columns={
            "estacion": "origen_id",
            "fecha": "inicio_viaje",
        }
    ).copy()

    aggregated_trips["inicio_viaje"] = (
        pd.to_datetime(aggregated_trips["inicio_viaje"])
        + pd.to_timedelta(aggregated_trips["hora"], unit="h")
    )

    hourly_profiles = hourly_usage_profiles(
        aggregated_trips,
        count_column="salidas",
    )

    hourly_profiles["tipo_dia"] = hourly_profiles[
        "tipo_dia"
    ].map({
        "laboral": "lv",
        "fin_semana": "fs",
    })

    hourly_profiles["columna"] = (
        hourly_profiles["tipo_dia"]
        + "_"
        + hourly_profiles["hora_dia"].astype(str).str.zfill(2)
    )

    by_hour = hourly_profiles.pivot_table(
        index="estacion_id",
        columns="columna",
        values="salidas",
        aggfunc="sum",
        fill_value=0,
    )

    columnas = [
        f"{tipo}_{hora:02d}"
        for tipo in ("lv", "fs")
        for hora in range(24)
    ]

    by_hour = by_hour.reindex(columns=columnas, fill_value=0)

    # Conservar nombres e índice esperados por P3
    by_hour = by_hour.rename(
        columns={
            f"lv_{hora:02d}": f"lv_{hora:02d}"
            for hora in range(24)
        }
    )

    by_hour.index.name = "estacion"
    profiles = by_hour.join(agg.groupby("estacion")["salidas"].sum().rename("salidas_2024"))
    profiles = profiles.join(agg.groupby("estacion")["fecha"].nunique().rename("dias_con_salidas"))
    profiles["media_diaria_calendario"] = profiles["salidas_2024"] / 366
    profiles["fraccion_fin_semana"] = profiles[[f"fs_{h:02d}" for h in range(24)]].sum(axis=1) / profiles["salidas_2024"]
    profiles["fraccion_manana_laboral"] = profiles[[f"lv_{h:02d}" for h in range(6,12)]].sum(axis=1) / profiles["salidas_2024"]
    profiles["fraccion_tarde_laboral"] = profiles[[f"lv_{h:02d}" for h in range(16,21)]].sum(axis=1) / profiles["salidas_2024"]
    return profiles, months


def create_features(stations: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=stations.index)
    for columns, prefix in [([f"lv_{h:02d}" for h in range(24)], "laboral"), ([f"fs_{h:02d}" for h in range(24)], "finde")]:
        denom = stations[columns].sum(axis=1).replace(0, np.nan)
        for hour, column in enumerate(columns):
            result[f"{prefix}_{hour:02d}"] = (stations[column] / denom).fillna(0.0)
    result["log_media_diaria"] = np.log1p(stations["media_diaria_calendario"])
    if result.shape[1] != 49 or not np.isfinite(result.to_numpy()).all():
        raise ValueError("La matriz de características P3 no es válida")
    return result


def save_plots(data: pd.DataFrame, features: pd.DataFrame, comparison: pd.DataFrame, labels: np.ndarray, chosen_k: int, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(comparison["k"], comparison["silhouette"], marker="o")
    ax.set(title="P3: Silhouette por K", xlabel="Número de grupos (K)", ylabel="Silhouette Score")
    ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(output / "p3_silhouette_por_k.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for group in sorted(np.unique(labels)):
        ids = data.index[labels == group]
        for a, prefix in zip(axes, ["laboral", "finde"]):
            vals = features.loc[ids, [f"{prefix}_{h:02d}" for h in range(24)]].mean(axis=0).to_numpy()
            a.plot(range(24), vals, label=f"Grupo {group}")
    for ax, name in zip(axes, ["Días laborables", "Fines de semana"]):
        ax.set(title=name, xlabel="Hora", ylabel="Fracción media de salidas por hora")
        ax.legend(); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(output / "p3_perfiles_horarios.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    for group in sorted(np.unique(labels)):
        subset = data.iloc[np.where(labels == group)[0]]
        ax.scatter(subset["longitud"], subset["latitud"], s=22, alpha=.75, label=f"Grupo {group} (n={len(subset)})")
    ax.set(title=f"MiBici 2024: estaciones por cluster (K={chosen_k})", xlabel="Longitud", ylabel="Latitud")
    ax.legend(); ax.grid(alpha=.15)
    fig.tight_layout(); fig.savefig(output / "p3_mapa_clusters.png", dpi=150); plt.close(fig)


def main():
    args = parse_args()
    data_dir = args.data_dir.resolve()
    files = sorted(data_dir.glob("datos_abiertos_2024_*.csv"))
    catalog_file = (args.catalog or data_dir / "nomenclatura_2026_09.csv").resolve()
    if len(files) != 12 or not catalog_file.exists():
        raise FileNotFoundError("Se requieren los 12 CSV de 2024 y nomenclatura_2026_09.csv en data/raw")
    catalog = pd.read_csv(catalog_file, dtype=str, encoding="utf-8-sig")
    catalog.columns = catalog.columns.str.strip()
    required = {"id", "name", "latitude", "longitude", "dpcapacity"}
    if not required.issubset(catalog.columns):
        raise ValueError(f"Faltan campos de catálogo: {required - set(catalog.columns)}")
    stations = catalog[["id", "name", "latitude", "longitude", "dpcapacity"]].copy()
    stations.columns = ["estacion", "nombre", "latitud", "longitud", "capacidad"]
    stations["estacion"] = stations["estacion"].str.strip()
    if stations["estacion"].duplicated().any():
        raise ValueError("IDs duplicados en catálogo")
    for c in ["latitud", "longitud", "capacidad"]:
        stations[c] = pd.to_numeric(stations[c], errors="coerce")
    profiles, monthly = build_profiles(files, args.chunk_size)
    coverage = profiles.join(stations.set_index("estacion"), how="left")
    included = coverage.loc[(coverage["salidas_2024"] >= args.min_departures) & coverage[["latitud", "longitud"]].notna().all(axis=1)].copy()
    if len(included) < 3:
        raise ValueError("Insuficientes estaciones para clustering")
    features = create_features(included)
    X = StandardScaler().fit_transform(features)
    k_values = list(range(args.k_min, min(args.k_max, len(included)-1)+1))
    if not k_values:
        raise ValueError("Rango K inválido")
    results, fitted = [], {}
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=args.seed, n_init=20)
        labels = model.fit_predict(X)
        sil = float(silhouette_score(X, labels))
        db = float(davies_bouldin_score(X, labels))
        sizes = pd.Series(labels).value_counts()
        results.append({"k": k, "silhouette": sil, "davies_bouldin": db, "inercia": float(model.inertia_), "min_estaciones_grupo": int(sizes.min()), "max_estaciones_grupo": int(sizes.max())})
        fitted[k] = labels
        print(f"K={k}: silhouette={sil:.6f}, Davies-Bouldin={db:.6f}, tamaños={sizes.sort_index().to_dict()}")
    comparison = pd.DataFrame(results)
    winner = int(comparison.loc[comparison["silhouette"].idxmax(), "k"])
    labels = fitted[winner]
    grouped = included.assign(grupo=labels)
    summary = grouped.groupby("grupo").agg(estaciones=("salidas_2024", "size"), salidas_mediana=("salidas_2024", "median"), media_diaria_mediana=("media_diaria_calendario", "median"), fraccion_fin_semana_mediana=("fraccion_fin_semana", "median"), fraccion_manana_laboral_mediana=("fraccion_manana_laboral", "median"), fraccion_tarde_laboral_mediana=("fraccion_tarde_laboral", "median"))
    output = args.output_dir.resolve()
    save_plots(included, features, comparison, labels, winner, output)
    comparison.to_csv(output / "p3_comparacion_k.csv", index=False)
    summary.to_csv(output / "p3_resumen_grupos.csv")
    grouped[["nombre", "grupo", "salidas_2024", "dias_con_salidas", "latitud", "longitud"]].to_csv(output / "p3_estaciones_grupos.csv", index_label="estacion")
    manifest = {"dataset": "MiBici 2024", "catalogo": catalog_file.name, "files": {p.name: sha256_file(p) for p in [*files, catalog_file]}, "viajes": int(sum(m["viajes"] for m in monthly)), "estaciones_observadas": len(profiles), "estaciones_incluidas": len(included)}
    (output / "p3_data_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    metrics = {"best_k": winner, "best_silhouette": float(comparison.loc[comparison["k"].eq(winner), "silhouette"].iloc[0]), "best_davies_bouldin": float(comparison.loc[comparison["k"].eq(winner), "davies_bouldin"].iloc[0]), "stations_observed": len(profiles), "stations_clustered": len(included), "stations_excluded": len(profiles)-len(included), "features": len(features.columns), "trips_processed": manifest["viajes"]}
    (output / "p3_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name="P3_baseline_kmeans_k2_to_k8"):
        mlflow.log_params({"model": "KMeans", "seed": args.seed, "n_init": 20, "k_min": args.k_min, "k_max": args.k_max, "min_departures": args.min_departures, "chunk_size": args.chunk_size, "scaler": "StandardScaler", "features": "48_proporciones_horarias_y_log_media_diaria", "cluster_selection": "max_silhouette", "geographic_coordinates_in_model": False, "capacity_reference": catalog_file.name, "year": 2024})
        mlflow.set_tags({"team": "15", "question": "P3", "dataset": "MiBici 2024", "methodology": "unsupervised_clustering", "git_commit": current_commit(), "limitation": "geography_2026_not_historical_2024", "data_version": "sha256_manifest"})
        mlflow.log_metrics(metrics)
        for row in results:
            mlflow.log_metric(f"silhouette_k{row['k']}", row["silhouette"])
            mlflow.log_metric(f"davies_bouldin_k{row['k']}", row["davies_bouldin"])
            mlflow.log_metric(f"inertia_k{row['k']}", row["inercia"])
        mlflow.log_artifacts(str(output), artifact_path="p3")
        print("MLflow run:", mlflow.active_run().info.run_id)
    print("\nResumen P3:", json.dumps(metrics, indent=2, ensure_ascii=False))
    print("Artefactos locales:", output)
    print("ADVERTENCIA: etiquetas K-Means pueden permutarse entre ejecuciones/versiones; comparar perfiles, no números de grupo.")


if __name__ == "__main__":
    main()
