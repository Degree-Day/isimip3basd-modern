#!/usr/bin/env bash
set -euo pipefail

# One-time verified migration of the production climate pipeline to /nas/dat1.
# Run independent groups in parallel; each source is removed only after rsync
# reports that no source file differs from its destination copy.

LOG_ROOT=${LOG_ROOT:-/nas/dat1/cmip6_migration_logs}
mkdir -p "$LOG_ROOT"

migrate() {
  local source=$1
  local destination=$2
  local label=$3

  if [[ ! -e "$source" ]]; then
    printf 'SKIP %s: source absent (%s)\n' "$label" "$source"
    return
  fi

  mkdir -p "$destination"
  printf 'START %s: %s -> %s at %s\n' \
    "$label" "$source" "$destination" "$(date -Is)"
  rsync -aO --no-owner --no-group --human-readable --info=progress2,stats2 \
    "$source/" "$destination/"

  if [[ -n "$(rsync -aniO --no-owner --no-group "$source/" "$destination/")" ]]; then
    printf 'VERIFY FAILED %s; source retained at %s\n' "$label" "$source" >&2
    return 1
  fi

  find "$source" -depth -delete
  printf 'DONE %s at %s\n' "$label" "$(date -Is)"
}

case "${1:-all}" in
  inputs)
    migrate /data1/cmip6_fwi_inputs /nas/dat1/cmip6_fwi_inputs inputs
    migrate /data1/cmip6_fwi_1deg /nas/dat1/cmip6_fwi_1deg standardized_1deg
    migrate /data1/era5land-fwi /nas/dat1/era5land-fwi era5land_products
    ;;
  products)
    migrate /data0/cmip6_bias_adjusted_1deg_localnoon \
      /nas/dat1/cmip6_bias_adjusted_1deg bias_adjusted_1deg
    migrate /data0/cmip6_bias_fit_cache_localnoon \
      /nas/dat1/cmip6_bias_fit_cache bias_fit_cache
    migrate /data0/cmip6_downscaled_global \
      /nas/dat1/cmip6_downscaled_global downscaled_0p1deg
    migrate /data0/cmip6_fwi_global /nas/dat1/cmip6_fwi_global fwi
    ;;
  reference)
    migrate /data0/era5ref-global-localnoon \
      /nas/dat1/era5ref-global-localnoon prepared_reference
    migrate /data0/data1_archive/era5land-fwi/noon_daily.zarr \
      /nas/dat1/era5land-fwi/noon_daily.zarr raw_local_noon_reference
    ;;
  all)
    "$0" inputs
    "$0" products
    "$0" reference
    ;;
  *)
    printf 'Usage: %s {inputs|products|reference|all}\n' "$0" >&2
    exit 2
    ;;
esac
