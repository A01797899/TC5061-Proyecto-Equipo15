
"""Semana 9: evaluación A/B offline de Champion y Challenger P1."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from mlflow import MlflowClient
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)

from mibici.api.schemas import P1_FEATURES
from mibici.config import REGISTERED_MODELS


FEATURES = list(P1_FEATURES)

DEFAULT_DATASET = Path(
    "data/processed/week9/evaluation.parquet"
)

DEFAULT_OUTPUT = Path(
    "reports/week9/ab_testing"
)

DEFAULT_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)


def file_sha256(path: Path) -> str:
    """Calcula la huella SHA-256 del dataset."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def load_evaluation(path: Path) -> pd.DataFrame:
    """Carga y valida marzo de 2025."""

    if not path.is_file():
        raise FileNotFoundError(
            f"No existe el dataset: {path}"
        )

    df = pd.read_parquet(path)

    if df.empty:
        raise ValueError("El dataset está vacío")

    required = set(FEATURES) | {
        "station",
        "hour",
        "y",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Faltan columnas: {sorted(missing)}"
        )

    df = df.copy()

    df["hour"] = pd.to_datetime(
        df["hour"],
        errors="raise",
    )

    if df["hour"].isna().any():
        raise ValueError("Fechas inválidas")

    if not (
        df["hour"].ge("2025-03-01").all()
        and df["hour"].lt("2025-04-01").all()
    ):
        raise ValueError(
            "El dataset debe pertenecer a marzo de 2025"
        )

    if df[["station", "hour"]].isna().any().any():
        raise ValueError(
            "Existen claves estación-hora incompletas"
        )

    if df.duplicated(["station", "hour"]).any():
        raise ValueError(
            "Existen claves estación-hora duplicadas"
        )

    for column in FEATURES + ["y"]:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

    values = df[FEATURES + ["y"]].to_numpy(
        dtype="float64"
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "El dataset contiene NaN o infinitos"
        )

    if (df["y"] < 0).any():
        raise ValueError(
            "La demanda real no puede ser negativa"
        )

    return df.sort_values(
        ["hour", "station"]
    ).reset_index(drop=True)


def calculate_metrics(y_true, y_pred) -> dict:
    """Calcula MAE y RMSE."""

    actual = np.asarray(
        y_true,
        dtype="float64",
    )

    predicted = np.asarray(
        y_pred,
        dtype="float64",
    )

    if (
        actual.ndim != 1
        or predicted.ndim != 1
        or actual.shape != predicted.shape
        or len(actual) == 0
    ):
        raise ValueError(
            "Dimensiones de evaluación incompatibles"
        )

    if not (
        np.isfinite(actual).all()
        and np.isfinite(predicted).all()
    ):
        raise ValueError(
            "Las métricas requieren valores finitos"
        )

    return {
        "mae": float(
            mean_absolute_error(actual, predicted)
        ),
        "rmse": float(
            np.sqrt(
                mean_squared_error(actual, predicted)
            )
        ),
    }


def improvement_percentage(
    champion_error: float,
    challenger_error: float,
):
    """Mejora positiva si Challenger reduce el error."""

    if champion_error == 0:
        return None

    return float(
        (champion_error - challenger_error)
        / champion_error
        * 100
    )


def predict_in_batches(
    model,
    X: pd.DataFrame,
    batch_size: int = 10000,
) -> np.ndarray:
    """Predice por lotes y recorta valores negativos a cero."""

    if batch_size <= 0:
        raise ValueError(
            "batch_size debe ser mayor que cero"
        )

    if X.empty:
        raise ValueError(
            "No hay registros para realizar predicciones"
        )

    predictions = []

    for start in range(0, len(X), batch_size):
        batch = X.iloc[
            start:start + batch_size
        ]

        values = np.asarray(
            model.predict(batch),
            dtype="float64",
        )

        if values.shape != (len(batch),):
            raise ValueError(
                "Dimensiones de predicción incorrectas"
            )

        if not np.isfinite(values).all():
            raise ValueError(
                "El modelo produjo NaN o infinitos"
            )

        predictions.append(
            np.clip(values, 0.0, None)
        )

    return np.concatenate(predictions)


def compare_predictions(
    y_true,
    champion_predictions,
    challenger_predictions,
) -> dict:
    """Compara ambos modelos sobre las mismas etiquetas."""

    champion = calculate_metrics(
        y_true,
        champion_predictions,
    )

    challenger = calculate_metrics(
        y_true,
        challenger_predictions,
    )

    return {
        "champion": champion,
        "challenger": challenger,
        "improvement_pct": {
            "mae": improvement_percentage(
                champion["mae"],
                challenger["mae"],
            ),
            "rmse": improvement_percentage(
                champion["rmse"],
                challenger["rmse"],
            ),
        },
    }


def resolve_model(
    client: MlflowClient,
    model_name: str,
    alias: str,
):
    """Resuelve un alias y carga una versión concreta."""

    version = client.get_model_version_by_alias(
        model_name,
        alias,
    )

    model_uri = (
        f"models:/{model_name}/{version.version}"
    )

    model = mlflow.pyfunc.load_model(model_uri)

    signature = model.metadata.get_input_schema()

    if signature is None:
        raise ValueError(
            f"El modelo {alias} no tiene firma"
        )

    if list(signature.input_names()) != FEATURES:
        raise ValueError(
            f"El modelo {alias} no coincide con P1"
        )

    metadata = {
        "alias": alias,
        "version": str(version.version),
        "run_id": version.run_id,
        "model_uri": model_uri,
    }

    return model, metadata


def save_json(path: Path, data: dict) -> None:
    """Guarda resultados en JSON legible."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )


def save_champion_predictions(
    df: pd.DataFrame,
    predictions: np.ndarray,
    output_dir: Path,
) -> dict:
    """Guarda predicciones reales y métricas del Champion."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions = np.asarray(
        predictions,
        dtype="float64",
    )

    if predictions.shape != (len(df),):
        raise ValueError(
            "Las predicciones no corresponden al dataset"
        )

    if not np.isfinite(predictions).all():
        raise ValueError(
            "Las predicciones contienen valores no finitos"
        )

    result = df[
        ["station", "hour", "y"]
    ].copy()

    result["pred_champion"] = predictions

    result["error_champion"] = np.abs(
        result["y"].to_numpy(dtype="float64")
        - predictions
    )

    result["squared_error_champion"] = (
        result["y"].to_numpy(dtype="float64")
        - predictions
    ) ** 2

    result["date"] = result["hour"].dt.date

    daily = result.groupby("date").agg(
        actual=("y", "sum"),
        champion_pred=("pred_champion", "sum"),
        champion_mae=("error_champion", "mean"),
        champion_mse=(
            "squared_error_champion",
            "mean",
        ),
        observations=("y", "size"),
    ).reset_index()

    daily["champion_rmse"] = np.sqrt(
        daily.pop("champion_mse")
    )

    station = result.groupby("station").agg(
        actual=("y", "sum"),
        champion_pred=("pred_champion", "sum"),
        champion_mae=("error_champion", "mean"),
        champion_mse=(
            "squared_error_champion",
            "mean",
        ),
        observations=("y", "size"),
    ).reset_index()

    station["champion_rmse"] = np.sqrt(
        station.pop("champion_mse")
    )

    result.to_parquet(
        output_dir / "champion_predictions.parquet",
        index=False,
    )

    daily.to_csv(
        output_dir / "champion_daily_metrics.csv",
        index=False,
    )

    station.to_csv(
        output_dir / "champion_station_metrics.csv",
        index=False,
    )

    return {
        "prediction_rows": int(len(result)),
        "days": int(len(daily)),
        "stations": int(len(station)),
        "daily_file": str(
            output_dir / "champion_daily_metrics.csv"
        ),
        "station_file": str(
            output_dir / "champion_station_metrics.csv"
        ),
        "predictions_file": str(
            output_dir / "champion_predictions.parquet"
        ),
    }


def run_evaluation(
    dataset_path: Path,
    output_dir: Path,
    tracking_uri: str,
    mode: str = "champion",
    batch_size: int = 10000,
) -> dict:
    """Ejecuta Champion solamente o comparación completa."""

    if mode not in ("champion", "both"):
        raise ValueError(
            "El modo debe ser champion o both"
        )

    df = load_evaluation(dataset_path)

    X = df[FEATURES].astype("float64")
    y = df["y"].to_numpy(dtype="float64")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)

    client = MlflowClient()
    model_name = REGISTERED_MODELS["p1"]

    champion_model, champion_info = resolve_model(
        client,
        model_name,
        "champion",
    )

    print("\n=== EVALUACION P1 ===")
    print("Modo:", mode)
    print("Observaciones:", len(df))
    print("Estaciones:", df["station"].nunique())
    print("Inicio:", df["hour"].min())
    print("Fin:", df["hour"].max())
    print("Champion:", champion_info["model_uri"])

    champion_predictions = predict_in_batches(
        champion_model,
        X,
        batch_size,
    )

    champion_metrics = calculate_metrics(
        y,
        champion_predictions,
    )

    common = {
        "model_name": model_name,
        "evaluation_period": (
            "2025-03-01/2025-03-31"
        ),
        "evaluation_rows": int(len(df)),
        "stations": int(df["station"].nunique()),
        "dataset_sha256": file_sha256(dataset_path),
        "batch_size": batch_size,
        "prediction_postprocessing": (
            "clip_negative_to_zero"
        ),
    }

    champion_result = {
        **champion_info,
        **champion_metrics,
    }

    print("\n=== CHAMPION ===")
    print("MAE:", champion_metrics["mae"])
    print("RMSE:", champion_metrics["rmse"])

    if mode == "champion":
        output = {
            **common,
            "champion": champion_result,
        }

        exports = save_champion_predictions(
            df=df,
            predictions=champion_predictions,
            output_dir=output_dir,
        )

        output["exported_artifacts"] = exports

        path = (
            output_dir / "champion_baseline.json"
        )

        save_json(path, output)

        print("\n=== ARCHIVOS GENERADOS ===")
        print("Baseline:", path)
        print("Predicciones:", exports["prediction_rows"])
        print("Días:", exports["days"])
        print("Estaciones:", exports["stations"])
        print("CSV diario:", exports["daily_file"])
        print("CSV estaciones:", exports["station_file"])
        print(
            "Parquet predicciones:",
            exports["predictions_file"],
        )

        return output


    # La siguiente sección se ejecuta únicamente
    # cuando el Challenger esté registrado.

    challenger_model, challenger_info = resolve_model(
        client,
        model_name,
        "challenger",
    )

    if (
        champion_info["version"]
        == challenger_info["version"]
    ):
        raise ValueError(
            "Champion y Challenger son la misma versión"
        )

    print("\nChallenger:", challenger_info["model_uri"])

    challenger_predictions = predict_in_batches(
        challenger_model,
        X,
        batch_size,
    )

    metrics = compare_predictions(
        y,
        champion_predictions,
        challenger_predictions,
    )

    output = {
        **common,
        "champion": champion_result,
        "challenger": {
            **challenger_info,
            **metrics["challenger"],
        },
        "improvement_pct": metrics["improvement_pct"],
    }

    # Resultados pareados para analizar errores.
    result = df[["station", "hour", "y"]].copy()

    result["pred_champion"] = champion_predictions
    result["pred_challenger"] = challenger_predictions

    result["error_champion"] = np.abs(
        y - champion_predictions
    )

    result["error_challenger"] = np.abs(
        y - challenger_predictions
    )

    result["date"] = result["hour"].dt.date

    daily = result.groupby("date").agg(
        actual=("y", "sum"),
        champion_pred=("pred_champion", "sum"),
        challenger_pred=("pred_challenger", "sum"),
        champion_mae=("error_champion", "mean"),
        challenger_mae=("error_challenger", "mean"),
        observations=("y", "size"),
    ).reset_index()



    # Comparación A/B por estación sobre marzo completo.
    result["squared_error_champion"] = (
        result["y"] - result["pred_champion"]
    ) ** 2

    result["squared_error_challenger"] = (
        result["y"] - result["pred_challenger"]
    ) ** 2

    station_metrics = result.groupby("station").agg(
        observations=("y", "size"),
        champion_mae=("error_champion", "mean"),
        challenger_mae=("error_challenger", "mean"),
        champion_mse=("squared_error_champion", "mean"),
        challenger_mse=("squared_error_challenger", "mean"),
    ).reset_index()

    station_metrics["champion_rmse"] = np.sqrt(
        station_metrics.pop("champion_mse")
    )

    station_metrics["challenger_rmse"] = np.sqrt(
        station_metrics.pop("challenger_mse")
    )

    station_metrics["mae_improvement_pct"] = np.where(
        station_metrics["champion_mae"] > 0,
        (
            station_metrics["champion_mae"]
            - station_metrics["challenger_mae"]
        ) / station_metrics["champion_mae"].replace(0, np.nan) * 100,
        np.nan,
    )

    station_metrics["rmse_improvement_pct"] = np.where(
        station_metrics["champion_rmse"] > 0,
        (
            station_metrics["champion_rmse"]
            - station_metrics["challenger_rmse"]
        ) / station_metrics["champion_rmse"].replace(0, np.nan) * 100,
        np.nan,
    )


    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_json(
        output_dir / "comparison_metrics.json",
        output,
    )

    daily.to_csv(
        output_dir / "daily_metrics.csv",
        index=False,
    )

    station_metrics.to_csv(
        output_dir / "station_comparison_metrics.csv",
        index=False,
    )

    # Solo una muestra para evidencia.
    result.head(1000).to_csv(
        output_dir / "predictions_sample.csv",
        index=False,
    )

    print("\n=== COMPARACION ===")
    print(json.dumps(
        output,
        indent=2,
        ensure_ascii=False,
    ))

    print("\nResultados:", output_dir)

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Comparación P1 Champion vs Challenger"
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATASET,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--tracking-uri",
        default=DEFAULT_TRACKING_URI,
    )

    parser.add_argument(
        "--mode",
        choices=("champion", "both"),
        default="champion",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
    )

    args = parser.parse_args()

    run_evaluation(
        dataset_path=args.data,
        output_dir=args.output,
        tracking_uri=args.tracking_uri,
        mode=args.mode,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
