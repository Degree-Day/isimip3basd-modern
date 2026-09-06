#!/usr/bin/env python3
"""Verify that a downscaled product covers every mapped reference land cell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr


def coverage_report(support: xr.DataArray, land: xr.DataArray) -> dict[str, object]:
    support, land = xr.align(support.astype(bool), land.astype(bool), join="exact")
    support_values = np.asarray(support.values, dtype=bool)
    land_values = np.asarray(land.values, dtype=bool)
    missing = land_values & ~support_values
    return {
        "valid": not bool(missing.any()),
        "support_cells": int(support_values.sum()),
        "mapped_land_cells": int(land_values.sum()),
        "mapped_land_cells_missing_support": int(missing.sum()),
        "support_cells_outside_mapped_land": int((support_values & ~land_values).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("region_root", type=Path)
    parser.add_argument("land_mask", type=Path)
    args = parser.parse_args()

    support = xr.open_zarr(
        args.region_root / "spatial_valid_mask.zarr", consolidated=False, chunks=None
    )["spatial_valid_mask"]
    land = xr.open_zarr(args.land_mask, consolidated=False, chunks=None)["lulc_land"]
    report = coverage_report(support, land)
    report.update(
        region_root=str(args.region_root),
        land_mask=str(args.land_mask),
        action="verification only; no post-downscaling nearest-neighbor fill",
    )
    output = args.region_root / "land-support-qc.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
