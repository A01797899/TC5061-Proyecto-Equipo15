#!/usr/bin/env bash
# Ejecuta el escenario S5 por escalones con registro simultáneo de recursos.
# Uso: bash scripts/week10/run_capacity_test.sh [docker|podman]

set -euo pipefail

ENGINE="${1:-docker}"
HOST="http://127.0.0.1:8000"
PY=".venv-locust/bin/python"
LOCUST=".venv-locust/bin/locust"
SHAPE_SECONDS=360
LEAD=10
TAIL=15

OUT="reports/week10/locust/S5_capacity"
METRICS_DIR="reports/week10/metrics"
LOG="$METRICS_DIR/run_log.txt"
mkdir -p "$OUT" "$METRICS_DIR"

restarts() {
  for c in mibici-equipo15-api-1 mibici-equipo15-mlflow-1; do
    printf '%s=%s ' "$c" "$("$ENGINE" inspect -f '{{.RestartCount}}' "$c")"
  done
}

curl -fsS "$HOST/ready" > /dev/null || { echo "La API no está lista"; exit 1; }
echo "S5_capacity inicio: $(date -u +%Y-%m-%dT%H:%M:%SZ) escalones=50-300 cada 60s commit=$(git rev-parse --short HEAD)" >> "$LOG"

"$PY" scripts/week10/collect_container_stats.py --engine "$ENGINE" \
  --duration $((LEAD + SHAPE_SECONDS + TAIL)) --interval 2 \
  --output "$METRICS_DIR/S5_resources.csv" &
collector=$!

sleep "$LEAD"
"$LOCUST" -f load_tests/locustfile_capacity.py --host "$HOST" --headless \
  --csv "$OUT/locust" --csv-full-history \
  --html "$OUT/report.html" --only-summary \
  > "$OUT/console.log" 2>&1 || true

wait "$collector"
echo "S5_capacity fin: $(date -u +%Y-%m-%dT%H:%M:%SZ) reinicios: $(restarts)" >> "$LOG"
echo "Listo. Resultados en $OUT"
