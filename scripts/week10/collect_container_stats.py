"""Registra CPU y memoria de los contenedores con docker stats o podman stats.

Uso:
    python scripts/week10/collect_container_stats.py \
        --duration 150 --interval 2 \
        --output reports/week10/metrics/S1_resources.csv
"""

import argparse
import csv
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

FIELDS = [
    "timestamp",
    "elapsed_s",
    "container",
    "cpu_percent",
    "mem_usage_mib",
    "mem_limit_mib",
    "mem_percent",
    "pids",
]

UNITS = {
    "b": 1 / 1024**2,
    "kb": 1000 / 1024**2,
    "kib": 1 / 1024,
    "mb": 1000**2 / 1024**2,
    "mib": 1,
    "gb": 1000**3 / 1024**2,
    "gib": 1024,
}


def to_mib(text):
    """Convierte un valor como '223.4MiB' a MiB; devuelve None si no se reconoce."""
    match = re.fullmatch(r"\s*([\d.]+)\s*([a-zA-Z]+)\s*", text or "")
    if not match:
        return None
    factor = UNITS.get(match.group(2).lower())
    return None if factor is None else float(match.group(1)) * factor


def to_float(text):
    """Convierte un porcentaje o número a float; devuelve None si no se reconoce."""
    try:
        return float(str(text).replace("%", "").strip())
    except ValueError:
        return None


def sample(engine, prefix):
    """Obtiene una muestra de los contenedores cuyo nombre inicia con el prefijo."""
    result = subprocess.run(
        [engine, "stats", "--no-stream", "--format", "{{json .}}"],
        capture_output=True,
        text=True,
        check=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        name = item.get("Name") or item.get("name", "")
        if not name.startswith(prefix):
            continue
        usage, _, limit = (item.get("MemUsage") or item.get("mem_usage", "")).partition("/")
        rows.append(
            {
                "container": name,
                "cpu_percent": to_float(item.get("CPUPerc") or item.get("cpu_percent")),
                "mem_usage_mib": to_mib(usage),
                "mem_limit_mib": to_mib(limit),
                "mem_percent": to_float(item.get("MemPerc") or item.get("mem_percent")),
                "pids": to_float(item.get("PIDs") or item.get("pids")),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--interval", type=float, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine", choices=["docker", "podman"], default="docker")
    parser.add_argument("--prefix", default="mibici-equipo15")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    count = 0

    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()

        while time.monotonic() - start < args.duration:
            tick = time.monotonic()
            timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
            elapsed = round(tick - start, 1)

            for row in sample(args.engine, args.prefix):
                row = {k: ("" if v is None else v) for k, v in row.items()}
                writer.writerow({"timestamp": timestamp, "elapsed_s": elapsed, **row})
                count += 1
            file.flush()

            time.sleep(max(0.0, args.interval - (time.monotonic() - tick)))

    print(f"{count} registros guardados en {args.output}")


if __name__ == "__main__":
    main()
