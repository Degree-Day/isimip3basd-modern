#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/dmr/miniconda3/envs/xr-zarr3/bin/python
REPO=/home/dmr/isimip3basd-modern
INPUT=/data0/cmip6_downscaled_global/ACCESS-CM2
OUTPUT=/data0/cmip6_fwi_global/ACCESS-CM2
LAND_MASK=/data0/cmip6_downscaled_global/ACCESS-CM2/historical/hist/qc/access_cm2_historical_annual_indicators_1989_2020.zarr

cd "$REPO"
export PYTHONPATH=src
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

"$PYTHON" scripts/calc_global_fwi.py \
  "$INPUT/historical/hist" \
  "$OUTPUT/historical/hist" \
  --compute-start 1989-01-01 \
  --compute-end 2020-12-31 \
  --output-start 1989-01-01 \
  --output-end 2020-12-31 \
  --period-label 1989-2020 \
  --tile-size 40 \
  --workers 12 \
  --threads-per-worker 1 \
  --land-mask-store "$LAND_MASK" \
  --land-mask-variable tg_mean

"$PYTHON" scripts/calc_global_fwi.py \
  "$INPUT/ssp245/proj" \
  "$OUTPUT/ssp245/proj" \
  --compute-start 2034-01-01 \
  --compute-end 2095-12-31 \
  --output-start 2034-01-01 \
  --output-end 2095-12-31 \
  --period-label 2034-2095 \
  --tile-size 40 \
  --workers 12 \
  --threads-per-worker 1 \
  --land-mask-store "$LAND_MASK" \
  --land-mask-variable tg_mean

"$PYTHON" scripts/calc_global_fwi_indicators.py \
  "$OUTPUT/historical/hist/global/daily_fire_weather_indices_1989-2020.zarr" \
  "$OUTPUT/ssp245/proj/global/daily_fire_weather_indices_2034-2095.zarr" \
  "$OUTPUT/annual" \
  --reference-start-year 1995 \
  --reference-end-year 2014 \
  --tile-size 40 \
  --workers 12
