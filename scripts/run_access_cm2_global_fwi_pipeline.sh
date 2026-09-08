#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/dmr/miniconda3/envs/xr-zarr3/bin/python
REPO=/home/dmr/isimip3basd-modern
INPUT=/data0/cmip6_downscaled_global/ACCESS-CM2
OUTPUT=/data0/cmip6_fwi_global/ACCESS-CM2
SUPPORT_MASK=/data0/cmip6_downscaled_global/ACCESS-CM2/historical/hist/global/spatial_valid_mask.zarr
COASTAL_FILL=/data0/cmip6_downscaled_global/ACCESS-CM2/ssp245/projection/global/coastal_fill_plan.zarr
QC_REPORT="$OUTPUT/annual/access_cm2_global_fwi_support_qc.json"

cd "$REPO"
export PYTHONPATH=src
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

"$PYTHON" scripts/calc_global_fwi.py \
  "$INPUT/historical/hist" \
  "$OUTPUT/historical/hist" \
  --compute-start 1989-01-01 \
  --compute-end 2014-12-31 \
  --output-start 1989-01-01 \
  --output-end 2014-12-31 \
  --period-label 1989-2014 \
  --tile-size 40 \
  --workers 12 \
  --threads-per-worker 1 \
  --support-mask-store "$SUPPORT_MASK" \
  --coastal-fill-plan "$COASTAL_FILL"

"$PYTHON" scripts/calc_global_fwi.py \
  "$INPUT/ssp245/projection" \
  "$OUTPUT/ssp245/projection" \
  --compute-start 2015-01-01 \
  --compute-end 2100-12-31 \
  --output-start 2015-01-01 \
  --output-end 2100-12-31 \
  --period-label 2015-2100 \
  --tile-size 40 \
  --workers 12 \
  --threads-per-worker 1 \
  --support-mask-store "$SUPPORT_MASK" \
  --coastal-fill-plan "$COASTAL_FILL"

"$PYTHON" scripts/calc_global_fwi_indicators.py \
  "$OUTPUT/historical/hist/global/daily_fire_weather_indices_1989-2014.zarr" \
  "$OUTPUT/ssp245/projection/global/daily_fire_weather_indices_2015-2100.zarr" \
  "$OUTPUT/annual" \
  --reference-start-year 1995 \
  --reference-end-year 2014 \
  --tile-size 40 \
  --workers 12 \
  --support-mask-store "$SUPPORT_MASK" \
  --coastal-fill-plan "$COASTAL_FILL"

"$PYTHON" scripts/qc_global_fwi_products.py \
  "$OUTPUT/historical/hist/global/daily_fire_weather_indices_1989-2014.zarr" \
  "$OUTPUT/ssp245/projection/global/daily_fire_weather_indices_2015-2100.zarr" \
  "$OUTPUT/annual/annual_fwi_indicators_1989_2100.zarr" \
  "$OUTPUT/annual/fwi_reference_thresholds_1995_2014.zarr" \
  "$SUPPORT_MASK" \
  "$QC_REPORT" \
  --coastal-fill-plan "$COASTAL_FILL"
