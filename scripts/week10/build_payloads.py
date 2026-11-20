"""Genera las solicitudes de Locust a partir de registros reales de P1.

Uso:
    python scripts/week10/build_payloads.py --rows 1000 --seed 42
"""

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/processed/week9/evaluation.parquet"
OUTPUT = ROOT / "load_tests/payloads_p1.json"

FEATURES = [
    "hour_of_day",
    "weekday",
    "fin_semana",
    "salidas_lag_1h",
    "salidas_lag_24h",
    "salidas_lag_168h",
    "media_estacion",
    "media_estacion_dia_hora",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = pd.read_parquet(SOURCE, columns=FEATURES)
    sample = data.sample(n=args.rows, random_state=args.seed)

    payloads = [
        {"features": {name: float(row[name]) for name in FEATURES}}
        for _, row in sample.iterrows()
    ]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "source": str(SOURCE.relative_to(ROOT)),
                "rows": args.rows,
                "seed": args.seed,
                "payloads": payloads,
            },
            file,
            indent=1,
        )

    print(f"{len(payloads)} solicitudes guardadas en {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
