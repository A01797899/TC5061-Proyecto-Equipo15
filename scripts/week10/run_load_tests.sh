#!/usr/bin/env bash
# Ejecuta la línea base y los escenarios S1-S4 con registro simultáneo de recursos.
# Uso: bash scripts/week10/run_load_tests.sh [docker|podman]

set -euo pipefail

ENGINE="${1:-docker}"
HOST="http://127.0.0.1:8000"
PY=".venv-locust/bin/python"
LOCUST=".venv-locust/bin/locust"
RUN_TIME=120
LEAD=10
TAIL=15
PAUSE=30
BASELINE=30

LOCUST_DIR="reports/week10/locust"
METRICS_DIR="reports/week10/metrics"
mkdir -p "$LOCUST_DIR" "$METRICS_DIR"

SCENARIOS=(
  "S1_low 5 1"
  "S2_medium 20 2"
  "S3_high 50 5"
  "S4_stress 100 10"
)

check_ready() {
  curl -fsS "$HOST/ready" > /dev/null || { echo "La API no está lista"; exit 1; }
}

restarts() {
  for c in mibici-equipo15-api-1 mibici-equipo15-mlflow-1; do
    printf '%s=%s ' "$c" "$("$ENGINE" inspect -f '{{.RestartCount}}' "$c")"
  done
}

LOG="$METRICS_DIR/run_log.txt"
{
  echo "inicio: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "commit: $(git rev-parse --short HEAD)"
  echo "motor: $ENGINE"
  echo "reinicios: $(restarts)"
} > "$LOG"

check_ready
echo "Línea base sin carga (${BASELINE}s)"
"$PY" scripts/week10/collect_container_stats.py --engine "$ENGINE" \
  --duration "$BASELINE" --interval 2 \
  --output "$METRICS_DIR/baseline_resources.csv"

for entry in "${SCENARIOS[@]}"; do
  read -r name users rate <<< "$entry"
  out="$LOCUST_DIR/$name"
  mkdir -p "$out"

  sleep "$PAUSE"
  check_ready
  echo "$name: $users usuarios, spawn $rate/s"
  echo "$name inicio: $(date -u +%Y-%m-%dT%H:%M:%SZ) usuarios=$users spawn=$rate" >> "$LOG"

  "$PY" scripts/week10/collect_container_stats.py --engine "$ENGINE" \
    --duration $((LEAD + RUN_TIME + TAIL)) --interval 2 \
    --output "$METRICS_DIR/${name%%_*}_resources.csv" &
  collector=$!

  sleep "$LEAD"
  "$LOCUST" -f load_tests/locustfile.py --host "$HOST" --headless \
    --users "$users" --spawn-rate "$rate" --run-time "${RUN_TIME}s" \
    --csv "$out/locust" --csv-full-history \
    --html "$out/report.html" --only-summary \
    > "$out/console.log" 2>&1 || true

  wait "$collector"
  echo "$name fin: $(date -u +%Y-%m-%dT%H:%M:%SZ) reinicios: $(restarts)" >> "$LOG"
done

check_ready
echo "fin: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
echo "Listo. Registro en $LOG"
