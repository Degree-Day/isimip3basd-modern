#!/usr/bin/env bash
set -euo pipefail

# One-time verified migration of the production climate pipeline to /nas/dat1.
# Run independent groups in parallel; each source is removed only after rsync
# reports that no source file differs from its destination copy.

LOG_ROOT=${LOG_ROOT:-/nas/dat1/cmip6_fwi/operations/migration_logs}
mkdir -p "$LOG_ROOT"

migrate() {
  local source=$1
  local destination=$2
  local label=$3
  local copy_destination=$destination
  local replace_symlink=0

  if [[ ! -e "$source" ]]; then
    printf 'SKIP %s: source absent (%s)\n' "$label" "$source"
    return
  fi

  if [[ -L "$destination" ]]; then
    copy_destination="${destination}.materializing"
    replace_symlink=1
    if [[ -e "$copy_destination" || -L "$copy_destination" ]]; then
      printf 'REFUSE %s: temporary destination exists (%s)\n' \
        "$label" "$copy_destination" >&2
      return 1
    fi
  elif [[ -e "$destination" && ! -d "$destination" ]]; then
    printf 'REFUSE %s: destination is not a directory (%s)\n' \
      "$label" "$destination" >&2
    return 1
  fi
  mkdir -p "$copy_destination"
  printf 'START %s: %s -> %s at %s\n' \
    "$label" "$source" "$destination" "$(date -Is)"
  rsync -aLO --no-owner --no-group --human-readable --info=progress2,stats2 \
    "$source/" "$copy_destination/"

  if [[ -n "$(rsync -aniLO --no-owner --no-group "$source/" "$copy_destination/")" ]]; then
    printf 'VERIFY FAILED %s; source retained at %s\n' "$label" "$source" >&2
    return 1
  fi

  if (( replace_symlink )); then
    rm "$destination"
    mv "$copy_destination" "$destination"
  fi

  find "$source" -depth -delete
  printf 'DONE %s at %s\n' "$label" "$(date -Is)"
}

case "${1:-all}" in
  inputs)
    migrate /data1/cmip6_fwi_inputs /nas/dat1/cmip6_fwi/inputs/raw inputs
    migrate /data1/cmip6_fwi_1deg /nas/dat1/cmip6_fwi/inputs/standardized_1deg standardized_1deg
    migrate /data1/era5land-fwi /nas/dat1/cmip6_fwi/reference/era5land era5land_products
    ;;
  products)
    migrate /data0/cmip6_bias_adjusted_1deg_localnoon \
      /nas/dat1/cmip6_fwi/processing/bias_adjusted_1deg bias_adjusted_1deg
    migrate /data0/cmip6_bias_fit_cache_localnoon \
      /nas/dat1/cmip6_fwi/processing/bias_fit_cache bias_fit_cache
    migrate /data0/cmip6_downscaled_global \
      /nas/dat1/cmip6_fwi/processing/downscaled_0p1deg downscaled_0p1deg
    migrate /data0/cmip6_fwi_global /nas/dat1/cmip6_fwi/processing/fwi fwi
    ;;
  reference)
    migrate /data0/era5ref-global-localnoon \
      /nas/dat1/cmip6_fwi/reference/prepared_local_noon prepared_reference
    migrate /data0/data1_archive/era5land-fwi/noon_daily.zarr \
      /nas/dat1/cmip6_fwi/reference/era5land/noon_daily.zarr raw_local_noon_reference
    ;;
  repair-symlinks)
    migrate /data0/data1_archive/cmip6_fwi_inputs/MRI-ESM2-0 \
      /nas/dat1/cmip6_fwi/inputs/raw/MRI-ESM2-0 mri_raw_inputs
    migrate /data0/data1_archive/cmip6_fwi_1deg/MRI-ESM2-0 \
      /nas/dat1/cmip6_fwi/inputs/standardized_1deg/MRI-ESM2-0 mri_standardized_1deg
    ;;
  all)
    "$0" inputs
    "$0" products
    "$0" reference
    ;;
  *)
    printf 'Usage: %s {inputs|products|reference|repair-symlinks|all}\n' "$0" >&2
    exit 2
    ;;
esac
