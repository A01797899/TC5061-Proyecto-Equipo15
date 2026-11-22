"""Estima el costo por inferencia exitosa con las mediciones de la semana 10.

La tarifa entra por --hourly-usd. El script no guarda un precio de nube.
Una inferencia del régimen estable cuesta precio_hora / (req/s * 3600).
Si el req/s es 0, el costo unitario queda vacío y se conserva el costo horario.
"""

import argparse
import csv
from decimal import Decimal
from pathlib import Path

SECONDS_PER_HOUR = Decimal(3600)
RECOMMENDED_USERS = 150
RECOMMENDED_P95_LIMIT_MS = Decimal(80)
LOAD_SCENARIOS = ("S1", "S2", "S3", "S4")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS = ROOT / "reports/week10/metrics"
DEFAULT_OUTPUT = ROOT / "reports/week10/finops/cost_by_scenario.csv"

FIELDNAMES = [
    "escenario",
    "usuarios",
    "rps",
    "errores",
    "p95_ms",
    "inferencias_por_hora",
    "costo_horario_usd",
    "costo_por_inferencia_usd",
    "costo_por_mil_usd",
    "api_cpu_media_pct",
    "api_mem_media_mib",
    "mlflow_cpu_media_pct",
    "mlflow_mem_media_mib",
    "punto_recomendado",
]


def cost_per_successful_inference(hourly_usd: Decimal, requests_per_second: Decimal) -> Decimal | None:
    """Devuelve el costo de una inferencia exitosa, o None si no hubo tráfico."""
    if requests_per_second <= 0:
        return None
    return hourly_usd / (requests_per_second * SECONDS_PER_HOUR)


def inferences_per_hour(requests_per_second: Decimal) -> Decimal:
    """Traduce req/s sostenidos a inferencias exitosas en una hora."""
    if requests_per_second <= 0:
        return Decimal(0)
    return requests_per_second * SECONDS_PER_HOUR


def is_recommended_operating_point(users: str, p95_ms: Decimal) -> bool:
    """El punto de operación es 150 usuarios con P95 por debajo de 80 ms."""
    try:
        user_count = int(users)
    except ValueError:
        return False
    return user_count == RECOMMENDED_USERS and p95_ms < RECOMMENDED_P95_LIMIT_MS


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _resources_by_scenario(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["escenario"], row["servicio"]): row for row in rows}


def _resources_by_users(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["usuarios"], row["servicio"]): row for row in rows}


def _money(value: Decimal | None) -> str:
    if value is None:
        return ""
    return format(value, "f")


def _scenario_row(
    *,
    escenario: str,
    usuarios: str,
    requests_per_second: Decimal,
    rps_text: str,
    errores: str,
    p95_ms: str,
    hourly_usd: Decimal,
    api: dict[str, str],
    mlflow: dict[str, str],
    recommended: bool,
) -> dict[str, str]:
    unit_cost = cost_per_successful_inference(hourly_usd, requests_per_second)
    per_thousand = None if unit_cost is None else unit_cost * Decimal(1000)
    return {
        "escenario": escenario,
        "usuarios": usuarios,
        "rps": rps_text,
        "errores": errores,
        "p95_ms": p95_ms,
        "inferencias_por_hora": _money(inferences_per_hour(requests_per_second)),
        "costo_horario_usd": _money(hourly_usd),
        "costo_por_inferencia_usd": _money(unit_cost),
        "costo_por_mil_usd": _money(per_thousand),
        "api_cpu_media_pct": api["cpu_media_pct"],
        "api_mem_media_mib": api["mem_media_mib"],
        "mlflow_cpu_media_pct": mlflow["cpu_media_pct"],
        "mlflow_mem_media_mib": mlflow["mem_media_mib"],
        "punto_recomendado": "si" if recommended else "no",
    }


def build_cost_table(
    load_summary: list[dict[str, str]],
    capacity_steps: list[dict[str, str]],
    resource_summary: list[dict[str, str]],
    resource_steps: list[dict[str, str]],
    hourly_usd: Decimal,
) -> list[dict[str, str]]:
    """Arma una fila de reposo, S1–S4 y cada escalón de S5."""
    by_scenario = _resources_by_scenario(resource_summary)
    by_users = _resources_by_users(resource_steps)
    rows: list[dict[str, str]] = []

    base_api = by_scenario[("Base", "api")]
    base_mlflow = by_scenario[("Base", "mlflow")]
    rows.append(
        _scenario_row(
            escenario="Base",
            usuarios="",
            requests_per_second=Decimal(0),
            rps_text="0",
            errores="0",
            p95_ms="",
            hourly_usd=hourly_usd,
            api=base_api,
            mlflow=base_mlflow,
            recommended=False,
        )
    )

    summary_by_name = {row["escenario"]: row for row in load_summary}
    for name in LOAD_SCENARIOS:
        source = summary_by_name[name]
        requests_per_second = Decimal(source["rps"])
        rows.append(
            _scenario_row(
                escenario=name,
                usuarios=source["usuarios"],
                requests_per_second=requests_per_second,
                rps_text=source["rps"],
                errores=source["error_pct"],
                p95_ms=source["p95_ms"],
                hourly_usd=hourly_usd,
                api=by_scenario[(name, "api")],
                mlflow=by_scenario[(name, "mlflow")],
                recommended=is_recommended_operating_point(
                    source["usuarios"], Decimal(source["p95_ms"])
                ),
            )
        )

    for source in capacity_steps:
        users = source["usuarios"]
        requests_per_second = Decimal(source["rps"])
        rows.append(
            _scenario_row(
                escenario=f"S5-{users}",
                usuarios=users,
                requests_per_second=requests_per_second,
                rps_text=source["rps"],
                errores=source["fallos_s"],
                p95_ms=source["p95_ms"],
                hourly_usd=hourly_usd,
                api=by_users[(users, "api")],
                mlflow=by_users[(users, "mlflow")],
                recommended=is_recommended_operating_point(users, Decimal(source["p95_ms"])),
            )
        )
    return rows


def load_errors_are_zero(rows: list[dict[str, str]]) -> bool:
    """Las filas con tráfico deben traer el indicador de error en cero."""
    for row in rows:
        if row["escenario"] == "Base":
            continue
        if Decimal(row["errores"]) != 0:
            return False
    return True


def write_cost_table(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def build_from_metrics(metrics_dir: Path, hourly_usd: Decimal) -> list[dict[str, str]]:
    return build_cost_table(
        _read_rows(metrics_dir / "load_summary.csv"),
        _read_rows(metrics_dir / "load_capacity_steps.csv"),
        _read_rows(metrics_dir / "resource_summary.csv"),
        _read_rows(metrics_dir / "resource_capacity_steps.csv"),
        hourly_usd,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estima el costo por inferencia exitosa.")
    parser.add_argument("--hourly-usd", required=True, help="Tarifa horaria de la instancia de referencia, en USD.")
    parser.add_argument("--metrics-dir", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    hourly_usd = Decimal(args.hourly_usd)
    rows = build_from_metrics(args.metrics_dir, hourly_usd)
    if not load_errors_are_zero(rows):
        print("Hay escenarios de carga con errores distintos de cero.")
        return 1
    write_cost_table(rows, args.output)
    recommended = [row["escenario"] for row in rows if row["punto_recomendado"] == "si"]
    print(f"Filas: {len(rows)}. Punto recomendado: {', '.join(recommended) or 'ninguno'}.")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
