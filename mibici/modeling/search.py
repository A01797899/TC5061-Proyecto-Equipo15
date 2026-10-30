"""Búsqueda de hiperparámetros con nested runs, reutilizable por P1, P2 y P3.

Arma una corrida padre por búsqueda y una corrida hija por combinación, con las
etiquetas que el equipo usa para filtrar y comparar en la interfaz de MLflow.
"""

from __future__ import annotations

import itertools
import os
# Solo listas de argumentos, sin shell.
import subprocess  # nosec B404
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import mlflow

from mibici.config import MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI, RANDOM_SEED

EQUIPO = "15"
ETAPA_BUSQUEDA = "search"


def commit_actual() -> str:
    # git con argumentos fijos; el ejecutable sale de PATH y no hay shell.
    proceso = subprocess.run(  # nosec B603 B607
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return proceso.stdout.strip() if proceso.returncode == 0 else "unknown"


def resolver_tracking_uri(uri: str | None = None) -> str:
    return uri or os.getenv("MLFLOW_TRACKING_URI") or MLFLOW_TRACKING_URI


def rejilla_a_combinaciones(
    rejilla: Mapping[str, Sequence[Any]],
) -> list[dict[str, Any]]:
    claves = list(rejilla)
    if not claves:
        raise ValueError("La rejilla de búsqueda está vacía")
    return [
        dict(zip(claves, valores))
        for valores in itertools.product(*(rejilla[clave] for clave in claves))
    ]


def _etiqueta_combinacion(parametros: Mapping[str, Any]) -> str:
    return "_".join(f"{clave}={valor}" for clave, valor in parametros.items())


def buscar(
    pregunta: str,
    familia_modelo: str,
    rejilla: Mapping[str, Sequence[Any]] | Iterable[Mapping[str, Any]],
    entrenar: Callable[[dict[str, Any]], tuple[Any, dict[str, float]]],
    metrica_objetivo: str,
    maximizar: bool = True,
    nombre_busqueda: str | None = None,
    tracking_uri: str | None = None,
    experimento: str | None = None,
    semilla: int = RANDOM_SEED,
    tags_extra: Mapping[str, str] | None = None,
    admisible: Callable[[dict[str, float]], bool] | None = None,
    al_terminar_hijo: (
        Callable[[Any, dict[str, Any], dict[str, float]], None] | None
    ) = None,
) -> dict[str, Any]:
    """Ejecuta la búsqueda y devuelve el mejor resultado junto con el historial completo.

    entrenar recibe un diccionario de parámetros y devuelve (modelo, métricas).
    admisible descarta del primer lugar las combinaciones que no cumplen una
    restricción, sin dejar de registrarlas como evidencia.
    al_terminar_hijo se invoca dentro de la corrida hija para registrar artefactos.
    """
    combinaciones = (
        rejilla_a_combinaciones(rejilla)
        if isinstance(rejilla, Mapping)
        else [dict(c) for c in rejilla]
    )
    if not combinaciones:
        raise ValueError("No hay combinaciones que evaluar")

    mlflow.set_tracking_uri(resolver_tracking_uri(tracking_uri))
    mlflow.set_experiment(experimento or MLFLOW_EXPERIMENT)

    etiquetas = {
        "team": EQUIPO,
        "question": pregunta,
        "model_family": familia_modelo,
        "stage_of_work": ETAPA_BUSQUEDA,
        "git_commit": commit_actual(),
    }
    etiquetas.update(tags_extra or {})
    nombre = nombre_busqueda or f"{pregunta.upper()}_search_{familia_modelo}"

    padre_anidado = mlflow.active_run() is not None
    historial: list[dict[str, Any]] = []
    mejor: dict[str, Any] | None = None
    mejor_global: dict[str, Any] | None = None

    with mlflow.start_run(run_name=nombre, nested=padre_anidado) as padre:
        mlflow.set_tags(etiquetas)
        mlflow.log_params(
            {
                "search_space_size": len(combinaciones),
                "target_metric": metrica_objetivo,
                "target_direction": "max" if maximizar else "min",
                "seed": semilla,
            }
        )

        for indice, parametros in enumerate(combinaciones, start=1):
            nombre_hijo = f"{nombre}__{_etiqueta_combinacion(parametros)}"
            with mlflow.start_run(run_name=nombre_hijo, nested=True) as hijo:
                mlflow.set_tags(etiquetas)
                mlflow.log_params({**parametros, "seed": semilla})
                modelo, metricas = entrenar(parametros)
                if metrica_objetivo not in metricas:
                    raise KeyError(
                        f"'{metrica_objetivo}' no viene en las métricas devueltas: {sorted(metricas)}"
                    )
                mlflow.log_metrics(
                    {clave: float(valor) for clave, valor in metricas.items()}
                )
                es_admisible = admisible is None or bool(admisible(metricas))
                mlflow.set_tag("admissible", str(es_admisible).lower())
                if al_terminar_hijo is not None:
                    al_terminar_hijo(modelo, parametros, metricas)
                resultado = {
                    "parametros": parametros,
                    "metricas": metricas,
                    "run_id": hijo.info.run_id,
                    "run_name": nombre_hijo,
                    "admisible": es_admisible,
                }
            historial.append(resultado)
            objetivo = float(metricas[metrica_objetivo])
            marca = "" if es_admisible else "  (no admisible)"
            print(
                f"[{indice}/{len(combinaciones)}] {nombre_hijo}: {metrica_objetivo}={objetivo:.6f}{marca}"
            )

            def _supera(actual: dict[str, Any] | None) -> bool:
                if actual is None:
                    return True
                referencia = float(actual["metricas"][metrica_objetivo])
                return (objetivo > referencia) if maximizar else (objetivo < referencia)

            if _supera(mejor_global):
                mejor_global = resultado
            if es_admisible and _supera(mejor):
                mejor = resultado

        admisibles = sum(1 for r in historial if r["admisible"])
        if mejor is None:
            print("Ninguna combinación cumplió la restricción; reporto el mejor global")
            mejor = mejor_global
        mlflow.log_params({"admissible_runs": admisibles})
        mlflow.set_tag("admissible_winner", str(bool(mejor["admisible"])).lower())

        # Subo el ganador al padre para poder ordenar las búsquedas entre sí desde la tabla.
        mlflow.log_params(
            {f"best_{clave}": valor for clave, valor in mejor["parametros"].items()}
        )
        mlflow.log_metrics(
            {
                f"best_{clave}": float(valor)
                for clave, valor in mejor["metricas"].items()
            }
        )
        mlflow.set_tag("best_child_run_id", mejor["run_id"])
        resumen = {
            "parent_run_id": padre.info.run_id,
            "best_run_id": mejor["run_id"],
            "best_run_name": mejor["run_name"],
            "best_params": mejor["parametros"],
            "best_metrics": mejor["metricas"],
            "best_admissible": bool(mejor["admisible"]),
            "admissible_runs": admisibles,
            "history": historial,
        }

    print(
        f"Mejor de {nombre}: {mejor['run_name']} ({metrica_objetivo}={float(mejor['metricas'][metrica_objetivo]):.6f})"
    )
    return resumen
