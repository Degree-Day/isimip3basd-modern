#!/usr/bin/env bash
set -euo pipefail

# Serial, restartable production queue restricted to SSP2-4.5 and SSP3-7.0.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}
PYTHON=${DOWNSCALE_PYTHON:-/home/dmr/isimip3basd-v3.0.2/modern-env/bin/python}
CANONICAL_ROOT=${CANONICAL_ROOT:-/nas/dat1/cmip6_fwi/inputs/standardized_1deg}
PUBLISHED_ROOT=${PUBLISHED_ROOT:-/nas/dat1/cmip6_fwi/published}
OPERATIONS_ROOT=${OPERATIONS_ROOT:-/nas/dat1/cmip6_fwi/operations/model_queue}
WORKERS=${WORKERS:-18}
THREADS_PER_WORKER=${THREADS_PER_WORKER:-1}
PUBLISH_WORKERS=${PUBLISH_WORKERS:-8}

mkdir -p "$OPERATIONS_ROOT"
exec 9>"$OPERATIONS_ROOT/queue.lock"
if ! flock -n 9; then
  printf 'Another global model queue is already running.\n' >&2
  exit 1
fi

timestamp=$(date +%Y%m%dT%H%M%S)
plan="$OPERATIONS_ROOT/plan-$timestamp.json"
log="$OPERATIONS_ROOT/queue-$timestamp.log"
latest="$OPERATIONS_ROOT/latest-plan.json"

cd "$REPO"
export PYTHONPATH=src
"$PYTHON" scripts/plan_global_model_queue.py \
  --canonical-root "$CANONICAL_ROOT" \
  --published-root "$PUBLISHED_ROOT" \
  --output "$plan" "$@" | tee "$log"
ln -sfn "$(basename "$plan")" "$latest"

mapfile -t jobs < <(
  "$PYTHON" - "$plan" <<'PY'
import json
import sys

for item in json.load(open(sys.argv[1])):
    if item["status"] == "ready":
        print(item["model"], item["scenario"])
PY
)

if (( ${#jobs[@]} == 0 )); then
  printf 'No ready model-scenario pathways remain.\n' | tee -a "$log"
  exit 0
fi

for job in "${jobs[@]}"; do
  read -r model scenario <<<"$job"
  start_stage=historical_downscale
  historical_weather="$PUBLISHED_ROOT/$model/historical/hist/weather/publication-manifest.json"
  historical_fwi="$PUBLISHED_ROOT/$model/historical/hist/fwi/global/daily_fire_weather_indices_1989-2014.zarr.qc.json"
  if [[ -f "$historical_weather" ]] &&
     [[ -f "$historical_fwi" ]] &&
     grep -q '"valid": true' "$historical_fwi"; then
    start_stage=projection_downscale
  fi
  printf '\nQUEUE START %s %s from %s at %s\n' \
    "$model" "$scenario" "$start_stage" "$(date -Is)" | tee -a "$log"
  env SCENARIO="$scenario" \
    WORKERS="$WORKERS" \
    THREADS_PER_WORKER="$THREADS_PER_WORKER" \
    PUBLISH_WORKERS="$PUBLISH_WORKERS" \
    "$SCRIPT_DIR/run_global_model_pipeline.sh" "$model" "$start_stage" \
    2>&1 | tee -a "$log"
  printf 'QUEUE DONE %s %s at %s\n' \
    "$model" "$scenario" "$(date -Is)" | tee -a "$log"
done

printf 'QUEUE COMPLETE at %s\n' "$(date -Is)" | tee -a "$log"
