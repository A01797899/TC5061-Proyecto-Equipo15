"""P3 MiBici: candidatos adicionales de agrupamiento buscados con nested runs.

Ejecutar desde la raíz del repositorio:
 python -m mibici.modeling.p3_search --tracking-uri http://127.0.0.1:5000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mibici.config import (
    INTERIM_DATA_DIR,
    MANIFIESTO_DATOS,
    MLFLOW_EXPERIMENT,
    P3_K_MAX,
    P3_K_MIN,
    P3_MIN_DEPARTURES,
    RANDOM_SEED,
    RAW_DATA_DIR,
)
from mibici.modeling.p3_baseline import build_profiles, create_features
from mibici.modeling.search import EQUIPO, buscar, commit_actual, resolver_tracking_uri

CACHE_FEATURES = INTERIM_DATA_DIR / "p3_features.parquet"
CACHE_ESTACIONES = INTERIM_DATA_DIR / "p3_estaciones.parquet"
CANDIDATOS_DIR = INTERIM_DATA_DIR / "p3_candidatos"
FAMILIAS = ("kmeans_pca", "jerarquico", "gmm_pca")


def parse_args():
    p = argparse.ArgumentParser(description="P3: búsqueda de candidatos de agrupamiento")
    p.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    p.add_argument("--catalog", type=Path, default=None)
    p.add_argument("--tracking-uri", default=None)
    p.add_argument("--experiment", default=None)
    p.add_argument("--familias", nargs="+", choices=FAMILIAS, default=list(FAMILIAS))
    p.add_argument("--min-departures", type=int, default=P3_MIN_DEPARTURES)
    p.add_argument("--chunk-size", type=int, default=100_000)
    p.add_argument("--k-min", type=int, default=P3_K_MIN)
    p.add_argument("--k-max", type=int, default=P3_K_MAX)
    p.add_argument("--min-cluster-share", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--recalcular", action="store_true")
    p.add_argument("--candidatos-dir", type=Path, default=CANDIDATOS_DIR)
    p.add_argument("--sin-promover", action="store_true")
    return p.parse_args()


def version_datos() -> str:
    if not MANIFIESTO_DATOS.exists():
        return "desconocida"
    return json.loads(MANIFIESTO_DATOS.read_text(encoding="utf-8")).get("version_datos", "desconocida")


def leer_catalogo(catalog_file: Path) -> pd.DataFrame:
    catalogo = pd.read_csv(catalog_file, dtype=str, encoding="utf-8-sig")
    catalogo.columns = catalogo.columns.str.strip()
    requeridas = {"id", "name", "latitude", "longitude", "dpcapacity"}
    if not requeridas.issubset(catalogo.columns):
        raise ValueError(f"Faltan campos de catálogo: {requeridas - set(catalogo.columns)}")
    estaciones = catalogo[["id", "name", "latitude", "longitude", "dpcapacity"]].copy()
    estaciones.columns = ["estacion", "nombre", "latitud", "longitud", "capacidad"]
    estaciones["estacion"] = estaciones["estacion"].str.strip()
    if estaciones["estacion"].duplicated().any():
        raise ValueError("IDs duplicados en catálogo")
    for columna in ["latitud", "longitud", "capacidad"]:
        estaciones[columna] = pd.to_numeric(estaciones[columna], errors="coerce")
    return estaciones


def preparar_datos(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not args.recalcular and CACHE_FEATURES.exists() and CACHE_ESTACIONES.exists():
        print(f"Reutilizando la matriz guardada en {CACHE_FEATURES}")
        return pd.read_parquet(CACHE_ESTACIONES), pd.read_parquet(CACHE_FEATURES)

    data_dir = args.data_dir.resolve()
    archivos = sorted(data_dir.glob("datos_abiertos_2024_*.csv"))
    catalog_file = (args.catalog or data_dir / "nomenclatura_2026_09.csv").resolve()
    if len(archivos) != 12 or not catalog_file.exists():
        raise FileNotFoundError("Se requieren los 12 CSV de 2024 y nomenclatura_2026_09.csv en data/raw")

    perfiles, _ = build_profiles(archivos, args.chunk_size)
    cobertura = perfiles.join(leer_catalogo(catalog_file).set_index("estacion"), how="left")
    incluidas = cobertura.loc[
        (cobertura["salidas_2024"] >= args.min_departures)
        & cobertura[["latitud", "longitud"]].notna().all(axis=1)
    ].copy()
    if len(incluidas) < 3:
        raise ValueError("Insuficientes estaciones para clustering")
    features = create_features(incluidas)

    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    incluidas.to_parquet(CACHE_ESTACIONES)
    features.to_parquet(CACHE_FEATURES)
    return incluidas, features


def evaluar(X: np.ndarray, etiquetas: np.ndarray) -> dict[str, float]:
    """Mido siempre sobre la misma matriz estandarizada para poder comparar contra el baseline."""
    grupos = np.unique(etiquetas)
    if len(grupos) < 2:
        raise ValueError("El modelo produjo un solo grupo; no se puede evaluar")
    tamanos = pd.Series(etiquetas).value_counts()
    return {
        "silhouette": float(silhouette_score(X, etiquetas)),
        "davies_bouldin": float(davies_bouldin_score(X, etiquetas)),
        "n_clusters": float(len(grupos)),
        "min_cluster_size": float(tamanos.min()),
        "max_cluster_size": float(tamanos.max()),
        "min_cluster_share": float(tamanos.min() / len(etiquetas)),
    }


def _entrenador(construir: Callable[[dict[str, Any]], Any], X: np.ndarray):
    def entrenar(parametros: dict[str, Any]):
        modelo = construir(parametros)
        etiquetas = modelo.fit_predict(X)
        return modelo, evaluar(X, etiquetas)

    return entrenar


def constructores(semilla: int) -> dict[str, Callable[[dict[str, Any]], Any]]:
    return {
        "kmeans_pca": lambda p: Pipeline(
            [
                ("pca", PCA(n_components=p["n_componentes"], random_state=semilla)),
                ("kmeans", KMeans(n_clusters=p["k"], random_state=semilla, n_init=20)),
            ]
        ),
        "jerarquico": lambda p: AgglomerativeClustering(n_clusters=p["k"], linkage=p["linkage"]),
        "gmm_pca": lambda p: Pipeline(
            [
                ("pca", PCA(n_components=p["n_componentes"], random_state=semilla)),
                (
                    "gmm",
                    GaussianMixture(
                        n_components=p["k"],
                        covariance_type=p["covarianza"],
                        random_state=semilla,
                        n_init=5,
                    ),
                ),
            ]
        ),
    }


def rejillas(args) -> dict[str, dict[str, list]]:
    ks = list(range(args.k_min, args.k_max + 1))
    return {
        "kmeans_pca": {"k": ks, "n_componentes": [5, 10, 20]},
        "jerarquico": {"k": ks, "linkage": ["ward", "average", "complete"]},
        "gmm_pca": {"k": ks, "n_componentes": [10], "covarianza": ["full", "diag", "tied"]},
    }


def graficar_candidato(
    familia: str,
    estaciones: pd.DataFrame,
    features: pd.DataFrame,
    etiquetas: np.ndarray,
    destino: Path,
) -> None:
    destino.mkdir(parents=True, exist_ok=True)
    grupos = sorted(np.unique(etiquetas))

    fig, ax = plt.subplots(figsize=(9, 7))
    for grupo in grupos:
        subconjunto = estaciones.iloc[np.where(etiquetas == grupo)[0]]
        ax.scatter(
            subconjunto["longitud"],
            subconjunto["latitud"],
            s=22,
            alpha=0.75,
            label=f"Grupo {grupo} (n={len(subconjunto)})",
        )
    ax.set(
        title=f"P3 {familia}: estaciones por grupo",
        xlabel="Longitud",
        ylabel="Latitud",
    )
    ax.legend()
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(destino / f"mapa_grupos_{familia}.png", dpi=150)
    plt.close(fig)

    fig, ejes = plt.subplots(1, 2, figsize=(13, 4))
    for grupo in grupos:
        indices = features.index[etiquetas == grupo]
        for eje, prefijo in zip(ejes, ["laboral", "finde"]):
            columnas = [f"{prefijo}_{hora:02d}" for hora in range(24)]
            eje.plot(
                range(24),
                features.loc[indices, columnas].mean(axis=0).to_numpy(),
                label=f"Grupo {grupo}",
            )
    for eje, titulo in zip(ejes, ["Días laborables", "Fines de semana"]):
        eje.set(
            title=f"{titulo} — {familia}",
            xlabel="Hora",
            ylabel="Fracción media de salidas",
        )
        eje.legend()
        eje.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(destino / f"perfiles_horarios_{familia}.png", dpi=150)
    plt.close(fig)

    tamanos = pd.Series(etiquetas).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(tamanos.index.astype(str), tamanos.to_numpy())
    ax.set(
        title=f"P3 {familia}: tamaño de cada grupo",
        xlabel="Grupo",
        ylabel="Estaciones",
    )
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(destino / f"tamanos_grupos_{familia}.png", dpi=150)
    plt.close(fig)


def _registrar_modelo(modelo, features: pd.DataFrame, etiquetas: np.ndarray) -> bool:
    """Agglomerative no tiene predict, así que el ejemplo de entrada solo va si se puede validar."""
    firma = infer_signature(features, etiquetas)
    puede_predecir = hasattr(modelo, "predict")
    ejemplo = features.head(5) if puede_predecir else None
    try:
        mlflow.sklearn.log_model(modelo, name="model", signature=firma, input_example=ejemplo)
    except TypeError:
        mlflow.sklearn.log_model(modelo, artifact_path="model", signature=firma, input_example=ejemplo)
    return puede_predecir


def registrar_candidato(
    familia: str,
    parametros: dict[str, Any],
    construir: Callable[[dict[str, Any]], Any],
    X: np.ndarray,
    features: pd.DataFrame,
    estaciones: pd.DataFrame,
    args,
    etiquetas_comunes: dict[str, str],
) -> dict[str, Any]:
    modelo = construir(parametros)
    grupos = modelo.fit_predict(X)
    metricas = evaluar(X, grupos)

    carpeta = (args.candidatos_dir / familia).resolve()
    for sub in ("plots", "data", "config"):
        (carpeta / sub).mkdir(parents=True, exist_ok=True)

    graficar_candidato(familia, estaciones, features, grupos, carpeta / "plots")

    asignacion = estaciones.assign(grupo=grupos)
    columnas = [c for c in ["nombre", "grupo", "salidas_2024", "latitud", "longitud"] if c in asignacion.columns]
    asignacion[columnas].to_csv(carpeta / "data" / f"estaciones_grupos_{familia}.csv", index_label="estacion")
    if MANIFIESTO_DATOS.exists():
        (carpeta / "data" / "manifiesto.json").write_text(
            MANIFIESTO_DATOS.read_text(encoding="utf-8"), encoding="utf-8"
        )

    configuracion = {
        "familia": familia,
        "parametros": parametros,
        "seed": args.seed,
        "min_departures": args.min_departures,
        "k_min": args.k_min,
        "k_max": args.k_max,
        "min_cluster_share": args.min_cluster_share,
        "metric_space": "standardized_49d",
        "data_version": etiquetas_comunes["data_version"],
        "git_commit": commit_actual(),
    }
    (carpeta / "config" / f"config_{familia}.json").write_text(
        json.dumps(configuracion, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    mlflow.set_tracking_uri(resolver_tracking_uri(args.tracking_uri))
    mlflow.set_experiment(args.experiment or MLFLOW_EXPERIMENT)
    etiquetas = {
        "team": EQUIPO,
        "question": "p3",
        "model_family": familia,
        "stage_of_work": "candidate",
        "git_commit": configuracion["git_commit"],
        **etiquetas_comunes,
    }

    anidado = mlflow.active_run() is not None
    with mlflow.start_run(run_name=f"P3_candidate_{familia}", nested=anidado) as corrida:
        mlflow.set_tags(etiquetas)
        mlflow.log_params({**parametros, "seed": args.seed, "scaler": "StandardScaler"})
        mlflow.log_metrics({clave: float(valor) for clave, valor in metricas.items()})
        try:
            mlflow.log_input(
                mlflow.data.from_pandas(features, name=f"p3_perfiles_horarios_{familia}"),
                context="training",
            )
        except Exception as error:
            print(f"No se pudo registrar el dataset de entrada: {error}")
        con_ejemplo = _registrar_modelo(modelo, features, grupos)
        mlflow.set_tag("supports_predict", str(con_ejemplo).lower())
        for sub in ("plots", "data", "config"):
            mlflow.log_artifacts(str(carpeta / sub), artifact_path=sub)
        run_id = corrida.info.run_id

    print(f"Candidata {familia}: run {run_id}; artefactos locales en {carpeta}")
    return {"familia": familia, "run_id": run_id, "metricas": metricas, "parametros": parametros}


def main():
    args = parse_args()
    incluidas, features = preparar_datos(args)
    X = StandardScaler().fit_transform(features)
    print(f"Estaciones: {len(incluidas)}; características: {features.shape[1]}")

    etiquetas_comunes = {
        "dataset": "MiBici 2024",
        "methodology": "unsupervised_clustering",
        "data_version": version_datos(),
        "metric_space": "standardized_49d",
    }
    construir = constructores(args.seed)
    espacios = rejillas(args)
    resumenes = {}

    def grupos_usables(metricas: dict[str, float]) -> bool:
        """Descarto particiones que aíslan estaciones sueltas: no sirven para operar."""
        return metricas["min_cluster_share"] >= args.min_cluster_share

    for familia in args.familias:
        print(f"\n=== Búsqueda {familia} ===")
        resumenes[familia] = buscar(
            pregunta="p3",
            familia_modelo=familia,
            rejilla=espacios[familia],
            entrenar=_entrenador(construir[familia], X),
            metrica_objetivo="silhouette",
            maximizar=True,
            tracking_uri=args.tracking_uri,
            experimento=args.experiment,
            semilla=args.seed,
            tags_extra=etiquetas_comunes,
            admisible=grupos_usables,
        )

    descartadas = sum(
        1 for r in resumenes.values() for h in r["history"] if not h["admisible"]
    )
    evaluadas = sum(len(r["history"]) for r in resumenes.values())
    print(
        f"\nCombinaciones evaluadas: {evaluadas}; descartadas por grupos menores "
        f"al {args.min_cluster_share:.0%}: {descartadas}"
    )
    print("\n=== Mejor admisible de cada familia ===")
    filas = [
        {
            "familia": familia,
            **resumen["best_params"],
            "silhouette": round(resumen["best_metrics"]["silhouette"], 6),
            "davies_bouldin": round(resumen["best_metrics"]["davies_bouldin"], 6),
            "min_cluster_share": round(resumen["best_metrics"]["min_cluster_share"], 4),
            "run_id": resumen["best_run_id"],
        }
        for familia, resumen in resumenes.items()
    ]
    print(pd.DataFrame(filas).to_string(index=False))

    if args.sin_promover:
        return {"busquedas": resumenes, "candidatas": []}

    print("\n=== Promoviendo la mejor admisible de cada familia a candidata ===")
    candidatas = [
        registrar_candidato(
            familia,
            resumen["best_params"],
            construir[familia],
            X,
            features,
            incluidas,
            args,
            etiquetas_comunes,
        )
        for familia, resumen in resumenes.items()
    ]
    print("\n=== Candidatas registradas ===")
    print(
        pd.DataFrame(
            [
                {
                    "familia": c["familia"],
                    "silhouette": round(c["metricas"]["silhouette"], 6),
                    "davies_bouldin": round(c["metricas"]["davies_bouldin"], 6),
                    "min_cluster_share": round(c["metricas"]["min_cluster_share"], 4),
                    "run_id": c["run_id"],
                }
                for c in candidatas
            ]
        ).to_string(index=False)
    )
    return {"busquedas": resumenes, "candidatas": candidatas}


if __name__ == "__main__":
    main()
