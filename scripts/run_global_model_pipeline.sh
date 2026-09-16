#!/usr/bin/env bash
set -euo pipefail

# Restartable end-to-end global workflow for one standardized CMIP6 model.
MODEL=${1:?Usage: run_global_model_pipeline.sh MODEL [START_STAGE]}
START_STAGE=${2:-preprocess}
SCENARIO=${SCENARIO:-ssp245}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}
DOWNSCALE_PYTHON=${DOWNSCALE_PYTHON:-/home/dmr/isimip3basd-v3.0.2/modern-env/bin/python}
FWI_PYTHON=${FWI_PYTHON:-/home/dmr/miniconda3/envs/xr-zarr3/bin/python}
RAW_ROOT=${RAW_ROOT:-/nas/dat1/cmip6_fwi_inputs}
CANONICAL_ROOT=${CANONICAL_ROOT:-/nas/dat1/cmip6_fwi_1deg}
REFERENCE_ROOT=${REFERENCE_ROOT:-/nas/dat1/era5ref-global-localnoon}
REFERENCE_SOURCE=${REFERENCE_SOURCE:-/nas/dat1/era5land-fwi/noon_daily.zarr}
ERA5_DAILY_ROOT=${ERA5_DAILY_ROOT:-/nas/dat1/ERA5/daily}
LULC_LAND_AREA=${LULC_LAND_AREA:-/nas/dat1/LULC/global_landarea_30as_km2.tif}
REFERENCE_COASTAL_PLAN=${REFERENCE_COASTAL_PLAN:-/nas/dat1/cmip6_downscaled_global/reference_qc/era5land_coastal_fill_plan.zarr}
ADJUSTED_ROOT=${ADJUSTED_ROOT:-/nas/dat1/cmip6_bias_adjusted_1deg}
DOWNSCALED_ROOT=${DOWNSCALED_ROOT:-/nas/dat1/cmip6_downscaled_global}
FWI_ROOT=${FWI_ROOT:-/nas/dat1/cmip6_fwi_global}
PUBLISHED_ROOT=${PUBLISHED_ROOT:-/nas/dat1/cmip6_published}
WORKERS=${WORKERS:-16}
THREADS_PER_WORKER=${THREADS_PER_WORKER:-1}
PUBLISH_WORKERS=${PUBLISH_WORKERS:-8}
FIT_CACHE_ROOT=${FIT_CACHE_ROOT:-/nas/dat1/cmip6_bias_fit_cache}
PIPELINE_REVISION=${PIPELINE_REVISION:-hurs-isimip3b-v2}

FIT_CACHE_ARGS=()
if [[ -n "$FIT_CACHE_ROOT" ]]; then
  FIT_CACHE_ARGS=(--fit-cache-root "$FIT_CACHE_ROOT")
fi

MODEL_ROOT="$DOWNSCALED_ROOT/$MODEL"
HIST_ROOT="$MODEL_ROOT/historical/hist"
PROJ_ROOT="$MODEL_ROOT/$SCENARIO/projection"
FWI_MODEL_ROOT="$FWI_ROOT/$MODEL"
LOG_ROOT="$MODEL_ROOT/logs/$SCENARIO"
STATE_ROOT="$MODEL_ROOT/pipeline_state/$SCENARIO"

STAGES=(
  reference_preparation
  preprocess
  historical_downscale
  historical_support_qc
  projection_downscale
  projection_support_qc
  climate_indicators
  daily_fwi
  fwi_indicators
  final_qc
  publication
)

stage_index() {
  local wanted=$1
  local index
  for index in "${!STAGES[@]}"; do
    if [[ ${STAGES[$index]} == "$wanted" ]]; then
      printf '%s\n' "$index"
      return 0
    fi
  done
  printf 'Unknown stage: %s\n' "$wanted" >&2
  return 2
}

START_INDEX=$(stage_index "$START_STAGE")
mkdir -p "$LOG_ROOT" "$STATE_ROOT"
cd "$REPO"
export PYTHONPATH=src
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

run_stage() {
  local stage=$1
  shift
  local index
  local marker="$STATE_ROOT/$stage.success"
  index=$(stage_index "$stage")
  if (( index < START_INDEX )); then
    printf 'SKIP stage %s (starting at %s)\n' "$stage" "$START_STAGE"
    return 0
  fi
  case "$stage" in
    historical_downscale|historical_support_qc|projection_downscale|projection_support_qc)
      marker="$STATE_ROOT/$stage.$PIPELINE_REVISION.success"
      ;;
    daily_fwi|fwi_indicators|final_qc)
      marker="$STATE_ROOT/$stage.continuous-history-v1.$PIPELINE_REVISION.success"
      ;;
    publication)
      marker="$STATE_ROOT/$stage.read-optimized-v1.$PIPELINE_REVISION.success"
      ;;
  esac
  if [[ -f "$marker" ]]; then
    printf 'SKIP completed stage %s\n' "$stage"
    return 0
  fi
  printf 'START stage %s at %s\n' "$stage" "$(date -Is)"
  "$@" 2>&1 | tee "$LOG_ROOT/$stage.log"
  touch "$marker"
  printf 'DONE stage %s at %s\n' "$stage" "$(date -Is)"
}

downscale() {
  local scenario=$1
  local simulation_stage=$2
  local start=$3
  local end=$4
  local output=$5
  "$DOWNSCALE_PYTHON" scripts/run_global_downscale_tiles.py \
    --model "$MODEL" \
    --scenario "$scenario" \
    --simulation-stage "$simulation_stage" \
    --simulation-start "$start" \
    --simulation-end "$end" \
    --reference-root "$REFERENCE_ROOT" \
    --canonical-root "$CANONICAL_ROOT" \
    --adjusted-root "$ADJUSTED_ROOT" \
    --output-root "$output" \
    --tile-lat-degrees 5 \
    --tile-lon-degrees 10 \
    --tile-workers "$WORKERS" \
    --threads-per-worker "$THREADS_PER_WORKER" \
    "${FIT_CACHE_ARGS[@]}"
}

verify_land_support() {
  local root=$1
  "$DOWNSCALE_PYTHON" scripts/verify_downscaled_land_support.py \
    "$root/global" "$REFERENCE_ROOT/lulc_land_mask.zarr"
}

prepare_reference() {
  local variable
  local complete=1
  for variable in tas hurs pr sfcWind; do
    [[ -d "$REFERENCE_ROOT/fine/$variable.zarr" ]] || complete=0
    [[ -d "$REFERENCE_ROOT/coarse/$variable.zarr" ]] || complete=0
  done
  [[ -d "$REFERENCE_ROOT/lulc_land_mask.zarr" ]] || complete=0
  [[ -f "$REFERENCE_ROOT/reference-final-qc.json" ]] || complete=0

  if (( complete )); then
    printf 'REUSE verified prepared reference at %s\n' "$REFERENCE_ROOT"
    return 0
  fi
  if [[ ! -d "$REFERENCE_SOURCE" ]]; then
    printf 'Prepared reference is incomplete and source is unavailable: %s\n' \
      "$REFERENCE_SOURCE" >&2
    return 1
  fi

  "$DOWNSCALE_PYTHON" scripts/prepare_era5land_reference.py \
    "$REFERENCE_SOURCE" "$REFERENCE_ROOT" \
    --workers "$WORKERS" \
    --era5-daily-root "$ERA5_DAILY_ROOT" \
    --lulc-land-area "$LULC_LAND_AREA" \
    --coastal-fill-plan "$REFERENCE_COASTAL_PLAN" \
    --variables tas hurs pr sfcWind
}

climate_indicators() {
  local slug
  slug=$(printf '%s' "$MODEL" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '_')
  mkdir -p "$HIST_ROOT/qc" "$PROJ_ROOT/qc"
  "$FWI_PYTHON" scripts/plot_global_etccdi_qc.py \
    --root "$HIST_ROOT" \
    --output "$HIST_ROOT/qc/${slug}_historical_etccdi_qc_1989_2014.png" \
    --annual-output "$HIST_ROOT/qc/annual_climate_indicators_1989_2014.zarr" \
    --start-year 1989 \
    --end-year 2014 \
    --title-prefix "$MODEL historical" \
    --workers "$WORKERS"
  "$FWI_PYTHON" scripts/plot_global_etccdi_qc.py \
    --root "$PROJ_ROOT" \
    --output "$PROJ_ROOT/qc/${slug}_${SCENARIO}_etccdi_qc_2015_2100.png" \
    --annual-output "$PROJ_ROOT/qc/annual_climate_indicators_2015_2100.zarr" \
    --start-year 2015 \
    --end-year 2100 \
    --title-prefix "$MODEL $SCENARIO" \
    --workers "$WORKERS"
}

daily_fwi() {
  "$FWI_PYTHON" scripts/calc_global_fwi.py \
    "$HIST_ROOT" "$FWI_MODEL_ROOT/historical/hist" \
    --compute-start 1989-01-01 \
    --compute-end 2014-12-31 \
    --output-start 1989-01-01 \
    --output-end 2014-12-31 \
    --period-label 1989-2014 \
    --tile-size 40 \
    --workers "$WORKERS" \
    --threads-per-worker 1 \
    --support-mask-store "$SUPPORT_MASK" \
    "${COASTAL_ARGS[@]}"
  "$FWI_PYTHON" scripts/calc_global_fwi.py \
    "$PROJ_ROOT" "$FWI_MODEL_ROOT/$SCENARIO/projection" \
    --history-input-root "$HIST_ROOT" \
    --history-start 1989-01-01 \
    --history-end 2014-12-31 \
    --compute-start 2015-01-01 \
    --compute-end 2100-12-31 \
    --output-start 2015-01-01 \
    --output-end 2100-12-31 \
    --period-label 2015-2100 \
    --tile-size 40 \
    --workers "$WORKERS" \
    --threads-per-worker 1 \
    --support-mask-store "$SUPPORT_MASK" \
    "${COASTAL_ARGS[@]}"
}

pack_store() {
  local source=$1
  local output=$2
  local chunks=$3
  shift 3
  local qc="${output}.qc.json"
  local variable_args=()
  if (( $# )); then
    variable_args=(--variables "$@")
  fi
  if [[ -f "$qc" ]] && grep -q '"valid": true' "$qc"; then
    printf 'SKIP verified publication store %s\n' "$output"
    return 0
  fi
  mkdir -p "$(dirname "$output")"
  "$FWI_PYTHON" -m isimip3basd_modern.cli pack \
    "$source" "$output" \
    --chunks "$chunks" \
    --workers "$PUBLISH_WORKERS" \
    --threads-per-worker 1 \
    --memory-limit 12GB \
    --overwrite \
    "${variable_args[@]}"
}

publish_products() {
  local published_model="$PUBLISHED_ROOT/$MODEL"
  local published_hist="$published_model/historical/hist"
  local published_future="$published_model/$SCENARIO/projection"

  "$DOWNSCALE_PYTHON" scripts/publish_global_outputs.py \
    "$HIST_ROOT" "$published_hist/weather" \
    --chunks time=365,lat=100,lon=100 \
    --workers "$PUBLISH_WORKERS" --threads-per-worker 1
  "$DOWNSCALE_PYTHON" scripts/publish_global_outputs.py \
    "$PROJ_ROOT" "$published_future/weather" \
    --chunks time=365,lat=100,lon=100 \
    --workers "$PUBLISH_WORKERS" --threads-per-worker 1

  pack_store \
    "$HIST_DAILY" \
    "$published_hist/fwi/global/daily_fire_weather_indices_1989-2014.zarr" \
    time=365,lat=80,lon=80 \
    ffmc dmc dc isi bui fwi
  pack_store \
    "$FUTURE_DAILY" \
    "$published_future/fwi/global/daily_fire_weather_indices_2015-2100.zarr" \
    time=365,lat=80,lon=80 \
    ffmc dmc dc isi bui fwi
  pack_store \
    "$ANNUAL_ROOT/annual_fwi_indicators_1989_2100.zarr" \
    "$published_future/fwi/annual/annual_fwi_indicators_1989_2100.zarr" \
    time=-1,lat=160,lon=160 \
    fwixx fwixd fwils fwisa
  pack_store \
    "$ANNUAL_ROOT/fwi_reference_thresholds_1995_2014.zarr" \
    "$published_future/fwi/annual/fwi_reference_thresholds_1995_2014.zarr" \
    lat=160,lon=160 \
    fwi_q95_reference fwi_midrange_reference
}

run_stage reference_preparation prepare_reference

run_stage preprocess \
  "$DOWNSCALE_PYTHON" scripts/preprocess_collection.py \
  "$RAW_ROOT" "$CANONICAL_ROOT" \
  --model "$MODEL" \
  --workers "$WORKERS" \
  --threads-per-worker 1 \
  --memory-limit 12GB \
  --spatial-chunk 20 \
  --require-variable tas \
  --require-variable hurs \
  --require-variable pr \
  --require-variable sfcWind

run_stage historical_downscale downscale historical hist 1989 2014 "$HIST_ROOT"
run_stage historical_support_qc verify_land_support "$HIST_ROOT"
run_stage projection_downscale downscale "$SCENARIO" projection 2015 2100 "$PROJ_ROOT"
run_stage projection_support_qc verify_land_support "$PROJ_ROOT"

run_stage climate_indicators climate_indicators

SUPPORT_MASK="$HIST_ROOT/global/spatial_valid_mask.zarr"
COASTAL_FILL="$PROJ_ROOT/global/coastal_fill_plan.zarr"
COASTAL_ARGS=()
if [[ -d "$COASTAL_FILL" ]]; then
  COASTAL_ARGS=(--coastal-fill-plan "$COASTAL_FILL")
fi
HIST_DAILY="$FWI_MODEL_ROOT/historical/hist/global/daily_fire_weather_indices_1989-2014.zarr"
FUTURE_DAILY="$FWI_MODEL_ROOT/$SCENARIO/projection/global/daily_fire_weather_indices_2015-2100.zarr"
ANNUAL_ROOT="$FWI_MODEL_ROOT/$SCENARIO/projection/annual"

run_stage daily_fwi daily_fwi

run_stage fwi_indicators \
  "$FWI_PYTHON" scripts/calc_global_fwi_indicators.py \
  "$HIST_DAILY" "$FUTURE_DAILY" "$ANNUAL_ROOT" \
  --reference-start-year 1995 \
  --reference-end-year 2014 \
  --tile-size 40 \
  --workers "$WORKERS" \
  --support-mask-store "$SUPPORT_MASK" \
  "${COASTAL_ARGS[@]}"

run_stage final_qc \
  "$FWI_PYTHON" scripts/qc_global_fwi_products.py \
  "$HIST_DAILY" \
  "$FUTURE_DAILY" \
  "$ANNUAL_ROOT/annual_fwi_indicators_1989_2100.zarr" \
  "$ANNUAL_ROOT/fwi_reference_thresholds_1995_2014.zarr" \
  "$SUPPORT_MASK" \
  "$ANNUAL_ROOT/${MODEL}_${SCENARIO}_global_fwi_support_qc.json" \
  "${COASTAL_ARGS[@]}"

run_stage publication publish_products

printf 'Global pipeline complete for %s %s at %s\n' "$MODEL" "$SCENARIO" "$(date -Is)"
