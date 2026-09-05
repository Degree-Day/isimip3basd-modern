#!/usr/bin/env python3
"""Validate global daily and annual FWI publication products."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import numpy as np
import xarray as xr
import zarr


DAILY_VARIABLES = ("ffmc", "dmc", "dc", "isi", "bui", "fwi")
ANNUAL_VARIABLES = ("fwixx", "fwixd", "fwils", "fwisa")
THRESHOLD_VARIABLES = ("fwi_q95_reference", "fwi_midrange_reference")


def _coverage_counts(
    raw: np.ndarray, support: np.ndarray, fill: np.int16
) -> tuple[int, int]:
    valid = raw != fill
    return int((support & ~valid).sum()), int((~support & valid).sum())


def _inspect_daily(path: Path) -> dict[str, object]:
    dataset = xr.open_zarr(path, consolidated=False, chunks=None)
    group = zarr.open_group(str(path), mode="r")
    variables = {}
    valid = set(dataset.data_vars) == set(DAILY_VARIABLES)
    for name in DAILY_VARIABLES:
        array = group[name]
        item = {
            "raw_dtype": str(array.dtype),
            "shape": list(array.shape),
            "scale_factor": float(array.attrs["scale_factor"]),
            "add_offset": float(array.attrs["add_offset"]),
        }
        variables[name] = item
        valid &= array.dtype == np.dtype("int16")
    time = dataset.time
    years = time.dt.year.values
    months = time.dt.month.values
    days = time.dt.day.values
    noleap = not bool(((months == 2) & (days == 29)).any())
    deltas = np.diff(time.values)
    regular = bool(
        time.size > 1
        and len(np.unique(time.values)) == time.size
        and all(delta == timedelta(days=1) for delta in deltas)
    )
    resolution = float(np.median(np.diff(dataset.lon.values)))
    valid &= noleap and regular and np.isclose(resolution, 0.1)
    return {
        "path": str(path),
        "shape": dict(dataset.sizes),
        "start": str(time.values[0]),
        "end": str(time.values[-1]),
        "calendar": "noleap" if noleap else "contains_february_29",
        "longitude_resolution_degrees": resolution,
        "variables": variables,
        "valid": bool(valid),
    }


def _inspect_packed_layers(
    path: Path,
    names: tuple[str, ...],
    support: np.ndarray,
) -> tuple[dict[str, object], bool]:
    group = zarr.open_group(str(path), mode="r")
    reports = {}
    all_valid = True
    for name in names:
        array = group[name]
        fill = np.int16(array.attrs["_FillValue"])
        scale = float(array.attrs["scale_factor"])
        offset = float(array.attrs["add_offset"])
        missing_supported = 0
        valid_outside_support = 0
        minimum = np.inf
        maximum = -np.inf
        layers = range(array.shape[0]) if array.ndim == 3 else (None,)
        for layer in layers:
            raw = np.asarray(array[layer] if layer is not None else array[:])
            missing, outside = _coverage_counts(raw, support, fill)
            missing_supported += missing
            valid_outside_support += outside
            selected = raw[raw != fill]
            if selected.size:
                decoded = selected.astype("float64") * scale + offset
                minimum = min(minimum, float(decoded.min()))
                maximum = max(maximum, float(decoded.max()))
        item_valid = bool(
            array.dtype == np.dtype("int16")
            and missing_supported == 0
            and valid_outside_support == 0
            and np.isfinite(minimum)
            and np.isfinite(maximum)
        )
        reports[name] = {
            "raw_dtype": str(array.dtype),
            "shape": list(array.shape),
            "scale_factor": scale,
            "add_offset": offset,
            "missing_supported": missing_supported,
            "valid_outside_support": valid_outside_support,
            "decoded_minimum": minimum,
            "decoded_maximum": maximum,
            "valid": item_valid,
        }
        all_valid &= item_valid
    return reports, all_valid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("historical_daily", type=Path)
    parser.add_argument("future_daily", type=Path)
    parser.add_argument("annual", type=Path)
    parser.add_argument("thresholds", type=Path)
    parser.add_argument("support_mask", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--coastal-fill-plan", type=Path)
    args = parser.parse_args()

    support_dataset = xr.open_zarr(
        args.support_mask, consolidated=False, chunks=None
    )
    support = np.asarray(support_dataset.spatial_valid_mask.values, dtype=bool)
    if args.coastal_fill_plan:
        coastal = xr.open_zarr(
            args.coastal_fill_plan, consolidated=False, chunks=None
        ).coastal_fill
        support |= np.asarray(coastal.values, dtype=bool)

    historical = _inspect_daily(args.historical_daily)
    future = _inspect_daily(args.future_daily)
    annual, annual_valid = _inspect_packed_layers(
        args.annual, ANNUAL_VARIABLES, support
    )
    thresholds, threshold_valid = _inspect_packed_layers(
        args.thresholds, THRESHOLD_VARIABLES, support
    )
    valid = bool(
        historical["valid"] and future["valid"] and annual_valid and threshold_valid
    )
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "support_cells": int(support.sum()),
        "outside_support_cells": int((~support).sum()),
        "historical_daily": historical,
        "future_daily": future,
        "annual": annual,
        "thresholds": thresholds,
        "valid": valid,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output)
    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
