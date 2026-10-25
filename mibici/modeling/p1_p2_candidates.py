"""Modelos candidatos de P1 (regresión) y P2 (clasificación) registrados en MLflow.

Ejecutar desde el MLflow Project:
    mlflow run . -e train -P question=p1 -P model=hgb
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import skops.io as sio

from mibici.config import (
    CANDIDATOS,
    FECHA_INICIO,
    HIPERPARAMETROS_CANDIDATOS,
    MANIFIESTO_DATOS,
    MLFLOW_EXPERIMENT,
    PROJ_ROOT,
    RANDOM_SEED,
    RAW_DATA_DIR,
    REZAGOS_CANDIDATOS,
    TEST_END,
    TRAIN_END,
    UMBRAL_RIESGO_CAPACIDAD,
    VAL_END,
)
from mibici.dataset import version_datos
from mibici.features import lag_features
from mibici.modeling import p1_baseline as p1
from mibici.modeling import p2_baseline as p2

SPLITS = {
    "train": (FECHA_INICIO, TRAIN_END),
    "val": (TRAIN_END, VAL_END),
    "test": (VAL_END, TEST_END),
}
FAMILIAS = {
    "rf": "random_forest",
    "hgb": "hist_gradient_boosting",
    "logreg": "logistic_regression",
}
CALENDARIO = ["hour_of_day", "weekday", "fin_semana"]
UMBRALES_BASELINE = np.round(np.linspace(0.01, 0.50, 50), 4)
UMBRALES_CANDIDATO = np.round(np.linspace(0.01, 0.99, 99), 4)
ARCHIVOS_CONFIG = ["MLproject", "python_env.yaml", "mibici/config.py"]


def resolver_hiperparametros(model: str, **valores) -> dict:
    """Toma los valores de config.py y reemplaza los recibidos distintos de cero."""
    hp = dict(HIPERPARAMETROS_CANDIDATOS[model])
    for clave, valor in valores.items():
        if valor and clave in hp:
            hp[clave] = valor
    return hp


def crear_modelo(question: str, model: str, hp: dict):
    """Instancia el estimador de scikit-learn según pregunta y modelo."""
    if model == "rf":
        clase = RandomForestRegressor if question == "p1" else RandomForestClassifier
        return clase(**hp, n_jobs=-1, random_state=RANDOM_SEED)
    if model == "hgb":
        clase = HistGradientBoostingRegressor if question == "p1" else HistGradientBoostingClassifier
        return clase(**hp, early_stopping=False, random_state=RANDOM_SEED)
    if model == "logreg" and question == "p2":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(**hp, class_weight="balanced", random_state=RANDOM_SEED),
        )
    raise ValueError(f"Modelo no soportado para {question}: {model}")


def agregar_rezagos(panel: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """Agrega rezagos por estación con lag_features de features.py."""
    out = panel.rename(columns={"station": "estacion_id", "hour": "hora"})
    for columna in columnas:
        out = lag_features(out, columna, REZAGOS_CANDIDATOS)
    out = out.rename(columns={"estacion_id": "station", "hora": "hour"})
    nuevas = [f"{c}_lag_{h}h" for c in columnas for h in REZAGOS_CANDIDATOS]
    out[nuevas] = out[nuevas].astype("float32")
    return out


def agregar_historicos(panel: pd.DataFrame, prefijo: str) -> pd.DataFrame:
    """Agrega promedios de y por estación y por estación-día-hora calculados solo con train."""
    train = panel[panel["split"] == "train"]
    global_mean = float(train["y"].mean())
    por_estacion = train.groupby("station", observed=True)["y"].mean()
    por_grupo = train.groupby(["station", "weekday", "hour_of_day"], observed=True)["y"].mean()
    col_estacion = f"{prefijo}_estacion"
    col_grupo = f"{prefijo}_estacion_dia_hora"
    panel[col_estacion] = panel["station"].map(por_estacion).fillna(global_mean).astype("float32")
    idx = pd.MultiIndex.from_frame(panel[["station", "weekday", "hour_of_day"]])
    valores = por_grupo.reindex(idx).to_numpy(dtype="float64")
    panel[col_grupo] = np.where(np.isfinite(valores), valores, panel[col_estacion]).astype("float32")
    return panel


def asignar_split(panel: pd.DataFrame) -> pd.DataFrame:
    """Etiqueta cada fila con train, val o test según config.py."""
    panel["split"] = "test"
    panel.loc[panel["hour"] < pd.Timestamp(VAL_END), "split"] = "val"
    panel.loc[panel["hour"] < pd.Timestamp(TRAIN_END), "split"] = "train"
    panel["fin_semana"] = (panel["weekday"] >= 5).astype("int8")
    return panel


def datos_p1(raw: Path, chunk_size: int) -> tuple[pd.DataFrame, list[str], dict]:
    """Panel estación-hora de P1 con la misma lógica y estaciones del baseline."""
    observed, manifest = p1.aggregate_monthly(raw, chunk_size)
    inicio, fin = SPLITS["train"]
    train_obs = observed[(observed["hour"] >= inicio) & (observed["hour"] < fin)]
    stations = pd.Index(sorted(train_obs["station"].unique()), name="station")
    panel = p1.panel(observed, stations, FECHA_INICIO, TEST_END)
    panel = asignar_split(panel)
    panel = agregar_rezagos(panel.rename(columns={"y": "salidas"}), ["salidas"])
    panel = panel.rename(columns={"salidas": "y"})
    panel = agregar_historicos(panel, "media")
    rezagos = [f"salidas_lag_{h}h" for h in REZAGOS_CANDIDATOS]
    features = CALENDARIO + rezagos + ["media_estacion", "media_estacion_dia_hora"]
    manifest["training_stations"] = len(stations)
    return panel, features, manifest


def datos_p2(raw: Path, chunk_size: int) -> tuple[pd.DataFrame, list[str], dict]:
    """Panel estación-hora de P2 con la misma etiqueta proxy y estaciones del baseline."""
    capacity, cap_meta = p2.load_capacities(raw / "nomenclatura_2026_09.csv")
    flows, manifest = p2.aggregate_flows(raw, chunk_size)
    en_train = set(flows.loc[flows["hour"] < pd.Timestamp(TRAIN_END), "station"])
    stations = pd.Index(sorted(en_train & set(capacity.index)))
    panel = p2.full_panel(flows, stations, FECHA_INICIO, TEST_END, capacity)
    panel = p2.assign_label(panel, UMBRAL_RIESGO_CAPACIDAD)
    panel = asignar_split(panel)
    columnas = ["salidas", "llegadas", "pressure"]
    panel = agregar_rezagos(panel, columnas)
    panel = agregar_historicos(panel, "prob")
    rezagos = [f"{c}_lag_{h}h" for c in columnas for h in REZAGOS_CANDIDATOS]
    features = CALENDARIO + ["capacity"] + rezagos + ["prob_estacion", "prob_estacion_dia_hora"]
    manifest["nomenclature"] = {"name": "nomenclatura_2026_09.csv", **cap_meta}
    manifest["training_stations"] = len(stations)
    return panel, features, manifest


def mejor_umbral(y: np.ndarray, prob: np.ndarray, candidatos: np.ndarray) -> float:
    """Umbral de probabilidad con mejor F1 en validación."""
    f1 = [f1_score(y, (prob >= t).astype("int8"), zero_division=0) for t in candidatos]
    return float(candidatos[int(np.argmax(f1))])


def mejora_pct(baseline: float, candidato: float, menor_es_mejor: bool) -> float:
    """Mejora porcentual del candidato respecto al baseline."""
    if baseline == 0:
        return 0.0
    delta = baseline - candidato if menor_es_mejor else candidato - baseline
    return float(100 * delta / abs(baseline))


def graficas_p1(test: pd.DataFrame, pred: np.ndarray, carpeta: Path, titulo: str) -> None:
    """Real contra predicho diario y residuales en test."""
    out = pd.DataFrame({"hour": test["hour"].to_numpy(), "actual": test["y"].to_numpy(), "pred": pred})
    diario = out.assign(date=out["hour"].dt.date).groupby("date", as_index=False)[["actual", "pred"]].sum()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(diario["date"], diario["actual"], label="Viajes reales", linewidth=1)
    ax.plot(diario["date"], diario["pred"], label="Predicción", linewidth=1)
    ax.set(title=f"P1 test: viajes diarios reales contra predichos — {titulo}", ylabel="Salidas por día")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(carpeta / "p1_test_real_vs_predicho.png", dpi=140)
    plt.close(fig)

    muestra = out.sample(n=min(50_000, len(out)), random_state=RANDOM_SEED)
    residuales = muestra["actual"] - muestra["pred"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(muestra["pred"], residuales, s=2, alpha=0.2)
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set(title="Residuales contra predicción", xlabel="Predicción", ylabel="Real - predicho")
    axes[1].hist(out["actual"] - out["pred"], bins=80)
    axes[1].set(title="Distribución de residuales", xlabel="Real - predicho", ylabel="Frecuencia")
    fig.suptitle(f"P1 test — {titulo}")
    fig.tight_layout()
    fig.savefig(carpeta / "p1_test_residuales.png", dpi=140)
    plt.close(fig)


def graficas_p2(y, pred, prob, prob_base, carpeta: Path, titulo: str) -> None:
    """Matriz de confusión y curva precisión-recall en test."""
    p2.plot_cm(y, pred, carpeta / "p2_test_matriz_confusion.png", f"P2 test — {titulo}")
    fig, ax = plt.subplots(figsize=(6, 5))
    for nombre, valores in ((titulo, prob), ("baseline histórico", prob_base)):
        precision, recall, _ = precision_recall_curve(y, valores)
        ax.plot(recall, precision, label=nombre, linewidth=1)
    ax.set(title="P2 test: curva precisión-recall", xlabel="Recall", ylabel="Precisión")
    ax.legend()
    fig.tight_layout()
    fig.savefig(carpeta / "p2_test_curva_pr.png", dpi=140)
    plt.close(fig)


def evaluar_p1(model, frames: dict, features: list[str], carpeta: Path, titulo: str) -> dict:
    """Métricas de P1 por split y comparación con el baseline estación-día-hora."""
    metricas = {}
    for nombre, df in frames.items():
        pred = np.clip(model.predict(df[features]), 0, None)
        for clave, valor in p1.metrics(df["y"].to_numpy(dtype=float), pred).items():
            metricas[f"{nombre}_{clave}"] = valor
        if nombre == "test":
            graficas_p1(df, pred, carpeta, titulo)
    test = frames["test"]
    base = p1.metrics(test["y"].to_numpy(dtype=float), np.clip(test["media_estacion_dia_hora"].to_numpy(), 0, None))
    for clave, valor in base.items():
        metricas[f"baseline_test_{clave}"] = valor
        metricas[f"test_{clave}_mejora_pct"] = mejora_pct(valor, metricas[f"test_{clave}"], True)
    return metricas


def evaluar_p2(model, frames: dict, features: list[str], carpeta: Path, titulo: str) -> tuple[dict, dict]:
    """Métricas de P2 por split con umbral elegido en validación y comparación con el baseline."""
    probs = {n: model.predict_proba(df[features])[:, 1] for n, df in frames.items()}
    umbral = mejor_umbral(frames["val"]["y"].to_numpy(), probs["val"], UMBRALES_CANDIDATO)
    metricas = {}
    for nombre, df in frames.items():
        y = df["y"].to_numpy()
        pred = (probs[nombre] >= umbral).astype("int8")
        for clave, valor in p2.scores(y, pred, probs[nombre]).items():
            metricas[f"{nombre}_{clave}"] = valor
    base_val = frames["val"]["prob_estacion_dia_hora"].to_numpy()
    umbral_base = mejor_umbral(frames["val"]["y"].to_numpy(), base_val, UMBRALES_BASELINE)
    test = frames["test"]
    y_test = test["y"].to_numpy()
    prob_base = test["prob_estacion_dia_hora"].to_numpy()
    base = p2.scores(y_test, (prob_base >= umbral_base).astype("int8"), prob_base)
    for clave in ("precision", "recall", "f1", "average_precision"):
        metricas[f"baseline_test_{clave}"] = base[clave]
        metricas[f"test_{clave}_mejora_pct"] = mejora_pct(base[clave], metricas[f"test_{clave}"], False)
    pred_test = (probs["test"] >= umbral).astype("int8")
    graficas_p2(y_test, pred_test, probs["test"], prob_base, carpeta, titulo)
    return metricas, {"probability_threshold": umbral, "baseline_probability_threshold": umbral_base}


def run_candidate(
    question: str,
    model: str,
    tracking_uri: str,
    chunk_size: int = 100_000,
    nested: bool = False,
    **valores,
) -> str:
    """Entrena un candidato, registra métricas, modelo y artefactos en MLflow y devuelve el run ID."""
    if model not in CANDIDATOS[question]:
        raise ValueError(f"{model} no es candidato de {question}: {CANDIDATOS[question]}")
    hp = resolver_hiperparametros(model, **valores)
    version = version_datos()
    commit = p1.git_sha(PROJ_ROOT)

    cargar = datos_p1 if question == "p1" else datos_p2
    panel, features, csv_manifest = cargar(RAW_DATA_DIR, chunk_size)
    frames = {n: panel[panel["split"] == n].reset_index(drop=True) for n in SPLITS}
    # Las primeras horas de train no tienen historia completa para los rezagos.
    frames["train"] = frames["train"].dropna(subset=features).reset_index(drop=True)
    for df in frames.values():
        df[features] = df[features].astype("float64")
    for nombre, df in frames.items():
        print(f"{nombre}: {len(df):,} filas estación-hora", flush=True)

    estimador = crear_modelo(question, model, hp)
    estimador.fit(frames["train"][features], frames["train"]["y"])

    # Evita reutilizar el run creado por mlflow run en el experimento por defecto.
    os.environ.pop("MLFLOW_RUN_ID", None)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    titulo = model
    clave_hp = "_".join(f"{k}={v}" for k, v in hp.items() if k in ("n_estimators", "max_depth", "max_iter", "learning_rate", "C"))
    run_name = f"{question.upper()}_candidate_{model}_{clave_hp}"

    with tempfile.TemporaryDirectory(prefix=f"mibici_{question}_") as tmp, mlflow.start_run(run_name=run_name, nested=nested) as run:
        carpeta = Path(tmp)
        for sub in ("plots", "data", "config", "metrics"):
            (carpeta / sub).mkdir()

        extra = {}
        if question == "p1":
            metricas = evaluar_p1(estimador, frames, features, carpeta / "plots", titulo)
        else:
            metricas, extra = evaluar_p2(estimador, frames, features, carpeta / "plots", titulo)

        params = {
            "question": question,
            "model": model,
            **{f"hp_{k}": v for k, v in hp.items()},
            **extra,
            "features": ",".join(features),
            "lags_hours": ",".join(map(str, REZAGOS_CANDIDATOS)),
            "train_period": f"{SPLITS['train'][0]}_to_{SPLITS['train'][1]}",
            "val_period": f"{SPLITS['val'][0]}_to_{SPLITS['val'][1]}",
            "test_period": f"{SPLITS['test'][0]}_to_{SPLITS['test'][1]}",
            "train_rows": len(frames["train"]),
            "seed": RANDOM_SEED,
        }
        if question == "p2":
            params["proxy_threshold"] = UMBRAL_RIESGO_CAPACIDAD
        mlflow.log_params(params)
        mlflow.set_tags({
            "team": "15",
            "question": question.upper(),
            "model_family": FAMILIAS[model],
            "stage_of_work": "candidate",
            "git_commit": commit,
            "data_version": version,
            "dataset": "MiBici 2024",
        })
        mlflow.log_metrics(metricas)

        (carpeta / "metrics" / "metrics.json").write_text(json.dumps(metricas, indent=2), encoding="utf-8")
        (carpeta / "config" / "run_params.json").write_text(json.dumps(params, indent=2, default=str), encoding="utf-8")
        for archivo in ARCHIVOS_CONFIG:
            shutil.copy(PROJ_ROOT / archivo, carpeta / "config" / Path(archivo).name)
        shutil.copy(MANIFIESTO_DATOS, carpeta / "data" / "manifiesto.json")
        (carpeta / "data" / "csv_manifest.json").write_text(
            json.dumps(csv_manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        mlflow.log_artifacts(str(carpeta))

        contextos = {"train": "training", "val": "validation", "test": "testing"}
        for nombre, contexto in contextos.items():
            dataset = mlflow.data.from_pandas(
                frames[nombre][features + ["y"]],
                source=str(MANIFIESTO_DATOS),
                name=f"mibici_{question}_{nombre}_{version}",
                targets="y",
            )
            mlflow.log_input(dataset, context=contexto)

        ejemplo = frames["train"][features].head(5)
        salida = estimador.predict(frames["val"][features].head(1000))
        firma = infer_signature(frames["val"][features].head(1000), salida)
        # Tipos internos de scikit-learn del modelo entrenado en esta misma corrida.
        confiables = sio.get_untrusted_types(data=sio.dumps(estimador))
        mlflow.sklearn.log_model(
            estimador,
            name="model",
            signature=firma,
            input_example=ejemplo,
            skops_trusted_types=confiables,
        )

        resumen = {k: round(v, 4) for k, v in metricas.items() if k.startswith(("test_", "baseline_test_"))}
        print(f"{run_name} ({run.info.run_id}): {resumen}", flush=True)
        return run.info.run_id
