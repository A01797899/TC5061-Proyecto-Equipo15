"""Búsquedas P1/P2 con una corrida padre y una hija por combinación.

Ejecutar a través de MLflow Project: -P model=search.
"""
from __future__ import annotations

import argparse
import json

import mlflow
from mlflow import MlflowClient

from mibici.config import MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI, RANDOM_SEED
from mibici.modeling.p1_p2_candidates import run_candidate


SEARCH_SPACE = {
    "p1": (
        ("rf", {"n_estimators": 300, "max_depth": 12}),
        ("hgb", {"max_iter": 300, "learning_rate": 0.1}),
        ("hgb", {"max_iter": 200, "learning_rate": 0.05}),
    ),
    "p2": (
        ("logreg", {}),
        ("rf", {"n_estimators": 300, "max_depth": 12}),
        ("hgb", {"max_iter": 300, "learning_rate": 0.1}),
    ),
}

SELECTION_METRIC = {"p1": "val_mae", "p2": "val_average_precision"}


def run_search(question: str, tracking_uri: str, *, candidates=None) -> dict:
    if question not in SEARCH_SPACE:
        raise ValueError(f"Pregunta no soportada: {question}")
    grid = tuple(SEARCH_SPACE[question] if candidates is None else candidates)
    if not grid:
        raise ValueError("La búsqueda debe contener al menos una combinación")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    client = MlflowClient(tracking_uri=tracking_uri)
    criterion = SELECTION_METRIC[question]
    records = []

    with mlflow.start_run(run_name=f"{question.upper()}_search_candidates") as parent:
        mlflow.set_tags({
            "team": "15", "question": question.upper(),
            "stage_of_work": "search", "search_strategy": "grid",
        })
        mlflow.log_params({
            "question": question, "n_combinations": len(grid),
            "selection_metric": criterion, "seed": RANDOM_SEED,
        })
        for index, (model, hp) in enumerate(grid, 1):
            # run_candidate conserva la lógica ya validada de entrenamiento,
            # métricas, datasets, firmas y artefactos; se ejecuta como hijo.
            run_id = run_candidate(
                question=question, model=model, tracking_uri=tracking_uri,
                nested=True, **hp,
            )
            run = client.get_run(run_id)
            if run.data.tags.get("mlflow.parentRunId") != parent.info.run_id:
                raise RuntimeError(f"La corrida {run_id} no está vinculada al padre")
            score = run.data.metrics.get(criterion)
            if score is None:
                raise RuntimeError(f"La corrida {run_id} no registró {criterion}")
            records.append({
                "index": index, "model": model, "hyperparameters": hp,
                "run_id": run_id, "metric": criterion, "score": score,
            })
            print(f"[{index}/{len(grid)}] {model} {hp}: {criterion}={score:.6f} ({run_id})", flush=True)

        best = min(records, key=lambda r: r["score"]) if question == "p1" else max(records, key=lambda r: r["score"])
        mlflow.log_metrics({"best_validation_score": best["score"], "completed_combinations": len(records)})
        mlflow.set_tags({"best_child_run_id": best["run_id"], "best_model": best["model"]})
        mlflow.log_dict({"parent_run_id": parent.info.run_id, "runs": records, "best": best}, f"search/{question}_summary.json")
        result = {"parent_run_id": parent.info.run_id, "children": len(records), "best": best}
        print(json.dumps(result, indent=2), flush=True)
        return result


def main():
    parser = argparse.ArgumentParser(description="Búsqueda anidada para P1/P2")
    parser.add_argument("--question", choices=("p1", "p2"), required=True)
    parser.add_argument("--tracking-uri", default=MLFLOW_TRACKING_URI)
    args = parser.parse_args()
    run_search(args.question, args.tracking_uri)


if __name__ == "__main__":
    main()
