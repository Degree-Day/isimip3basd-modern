#!/usr/bin/env python3
"""Run an independent, reproducible QC audit of the global reference dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
import zarr


VARIABLES = ("tas", "hurs", "pr", "sfcWind")
EXPECTED = {
    "tas": ("K", "air_temperature", 180.0, 340.0),
    "hurs": ("%", "relative_humidity", 0.0, 100.0),
    "pr": ("kg m-2 s-1", "precipitation_flux", 0.0, 0.035),
    "sfcWind": ("m s-1", "wind_speed", 0.0, 75.0),
}
SOURCE_NAMES = {
    0: "outside_or_unavailable",
    1: "era5_land",
    2: "era5_land_coastal_repair",
    3: "era5",
}


def json_value(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def coordinate_report(data: xr.DataArray, resolution: float) -> dict[str, Any]:
    lat = np.asarray(data.lat)
    lon = np.asarray(data.lon)
    time = data.indexes["time"]
    inferred = xr.infer_freq(time)
    years = np.asarray(data.time.dt.year)
    month = np.asarray(data.time.dt.month)
    day = np.asarray(data.time.dt.day)
    counts = np.bincount(years - years.min())
    return {
        "shape": list(data.shape),
        "dimension_order": list(data.dims),
        "lat_range": [float(lat[0]), float(lat[-1])],
        "lon_range": [float(lon[0]), float(lon[-1])],
        "lat_resolution": float(np.median(np.diff(lat))),
        "lon_resolution": float(np.median(np.diff(lon))),
        "grid_regular": bool(
            np.allclose(np.diff(lat), resolution)
            and np.allclose(np.diff(lon), resolution)
        ),
        "time_start": str(time[0]),
        "time_end": str(time[-1]),
        "time_count": len(time),
        "time_frequency": inferred,
        "time_unique": bool(time.is_unique),
        "time_monotonic": bool(time.is_monotonic_increasing),
        "calendar": str(data.time.dt.calendar),
        "year_day_counts": sorted(set(int(value) for value in counts)),
        "contains_february_29": bool(((month == 2) & (day == 29)).any()),
    }


def distribution(values: np.ndarray) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": 0}
    quantiles = np.quantile(finite, [0, 0.001, 0.01, 0.5, 0.99, 0.999, 1])
    return {
        "count": int(finite.size),
        "mean": float(finite.mean(dtype="float64")),
        "standard_deviation": float(finite.std(dtype="float64")),
        "quantiles": dict(
            zip(("min", "p0.1", "p1", "p50", "p99", "p99.9", "max"), quantiles)
        ),
    }


def boundary_differences(field: np.ndarray, source: np.ndarray) -> dict[str, Any]:
    records: dict[str, list[np.ndarray]] = {"boundary": [], "same_source": []}
    source_pairs: dict[tuple[int, int], list[np.ndarray]] = {}
    for axis in (0, 1):
        shifted_field = np.roll(field, -1, axis=axis)
        shifted_source = np.roll(source, -1, axis=axis)
        active = (source > 0) & (shifted_source > 0)
        if axis == 0:
            active[-1] = False
        difference = np.abs(field - shifted_field)
        records["boundary"].append(
            difference[active & (source != shifted_source)]
        )
        records["same_source"].append(
            difference[active & (source == shifted_source)]
        )
        for first in (1, 2, 3):
            for second in range(first + 1, 4):
                pair = active & (
                    ((source == first) & (shifted_source == second))
                    | ((source == second) & (shifted_source == first))
                )
                source_pairs.setdefault((first, second), []).append(difference[pair])
    result = {}
    for name, parts in records.items():
        values = np.concatenate(parts)
        result[name] = {
            "edge_count": int(values.size),
            "median": float(np.nanmedian(values)) if values.size else None,
            "p95": float(np.nanquantile(values, 0.95)) if values.size else None,
            "p99": float(np.nanquantile(values, 0.99)) if values.size else None,
        }
    denominator = result["same_source"]["p95"]
    numerator = result["boundary"]["p95"]
    result["boundary_to_same_source_p95_ratio"] = (
        float(numerator / denominator)
        if numerator is not None and denominator not in (None, 0)
        else None
    )
    result["by_source_pair"] = {}
    for pair, parts in source_pairs.items():
        values = np.concatenate(parts)
        result["by_source_pair"][
            f"{SOURCE_NAMES[pair[0]]}--{SOURCE_NAMES[pair[1]]}"
        ] = {
            "edge_count": int(values.size),
            "median": float(np.nanmedian(values)) if values.size else None,
            "p95": float(np.nanquantile(values, 0.95)) if values.size else None,
            "p99": float(np.nanquantile(values, 0.99)) if values.size else None,
        }
    return result


def coarse_consistency(
    fine: xr.DataArray, coarse: xr.DataArray, indices: np.ndarray
) -> dict[str, Any]:
    selected = fine.isel(time=indices).load()
    weights = np.cos(np.deg2rad(selected.lat)).broadcast_like(selected)
    expected = (
        (selected * weights).coarsen(lat=10, lon=10, boundary="exact").sum()
        / weights.where(selected.notnull())
        .coarsen(lat=10, lon=10, boundary="exact")
        .sum()
    ).assign_coords(lat=coarse.lat, lon=coarse.lon)
    actual = coarse.isel(time=indices).load()
    difference = np.abs(np.asarray(expected) - np.asarray(actual))
    finite = difference[np.isfinite(difference)]
    return {
        "sample_dates": [str(value) for value in actual.time.values],
        "compared_values": int(finite.size),
        "mean_absolute_error": float(finite.mean()) if finite.size else None,
        "p99_absolute_error": float(np.quantile(finite, 0.99)) if finite.size else None,
        "maximum_absolute_error": float(finite.max()) if finite.size else None,
    }


def temporal_report(data: xr.DataArray, source: np.ndarray) -> dict[str, Any]:
    series = data.isel(lat=slice(None, None, 100), lon=slice(None, None, 100)).load()
    values = np.asarray(series)
    sampled_source = source[::100, ::100]
    supported = sampled_source > 0
    supported_values = values[:, supported]
    finite_count = np.isfinite(supported_values).sum(axis=0)
    partial = finite_count != data.sizes["time"]
    standard_deviation = np.nanstd(supported_values, axis=0)
    jumps = np.abs(np.diff(supported_values, axis=0))
    finite_jumps = jumps[np.isfinite(jumps)]
    return {
        "sampled_supported_cells": int(supported.sum()),
        "partial_time_series": int(partial.sum()),
        "constant_time_series": int((standard_deviation == 0).sum()),
        "absolute_daily_change": distribution(finite_jumps),
    }


def variable_report(
    root: Path, variable: str, source: np.ndarray, time_stride: int
) -> tuple[dict[str, Any], list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    fine_path = root / "fine" / f"{variable}.zarr"
    coarse_path = root / "coarse" / f"{variable}.zarr"
    fine_ds = xr.open_zarr(fine_path, consolidated=False, chunks=None)
    coarse_ds = xr.open_zarr(coarse_path, consolidated=False, chunks=None)
    fine = fine_ds[variable]
    coarse = coarse_ds[variable]
    units, standard_name, lower, upper = EXPECTED[variable]

    fine_coordinates = coordinate_report(fine, 0.1)
    coarse_coordinates = coordinate_report(coarse, 1.0)
    if not fine_coordinates["grid_regular"] or not coarse_coordinates["grid_regular"]:
        errors.append(f"{variable}: irregular grid")
    if fine_coordinates["time_count"] != 8030:
        errors.append(f"{variable}: expected 8030 reference days")
    if fine_coordinates["calendar"] != "noleap":
        errors.append(f"{variable}: calendar is not noleap")
    if fine.attrs.get("units") != units:
        errors.append(f"{variable}: unexpected units {fine.attrs.get('units')!r}")
    if fine.attrs.get("standard_name") != standard_name:
        errors.append(f"{variable}: unexpected standard_name")

    array = zarr.open_group(fine_path, mode="r")[variable]
    coarse_array = zarr.open_group(coarse_path, mode="r")[variable]
    time_indices = np.arange(0, fine.sizes["time"], time_stride)
    sampled = fine.isel(
        time=time_indices,
        lat=slice(None, None, 10),
        lon=slice(None, None, 10),
    ).load()
    sampled_values = np.asarray(sampled)
    sampled_source = source[::10, ::10]
    expected_finite = sampled_source > 0
    partial = expected_finite & ~np.isfinite(sampled_values).all(axis=0)
    unexpected = (~expected_finite) & np.isfinite(sampled_values).any(axis=0)
    if partial.any():
        errors.append(f"{variable}: sampled supported cells contain missing values")
    if unexpected.any():
        errors.append(f"{variable}: sampled values occur outside support")
    finite = sampled_values[np.isfinite(sampled_values)]
    if finite.size and (finite.min() < lower or finite.max() > upper):
        errors.append(f"{variable}: sampled values violate audit bounds")

    by_source = {}
    for code, name in SOURCE_NAMES.items():
        if code == 0:
            continue
        mask = sampled_source == code
        by_source[name] = distribution(sampled_values[:, mask])

    climatology_indices = np.linspace(0, fine.sizes["time"] - 1, 12).astype(int)
    climatology = np.asarray(fine.isel(time=climatology_indices).mean("time").load())
    boundary = boundary_differences(climatology, source)
    ratio = boundary["boundary_to_same_source_p95_ratio"]
    if ratio is not None and ratio > 5:
        warnings.append(f"{variable}: source-boundary p95 difference ratio exceeds 5")

    consistency_indices = np.linspace(0, fine.sizes["time"] - 1, 6).astype(int)
    consistency = coarse_consistency(fine, coarse, consistency_indices)
    consistency_tolerance = 2e-5
    consistency["absolute_tolerance"] = consistency_tolerance
    if (consistency["maximum_absolute_error"] or 0) > consistency_tolerance:
        errors.append(f"{variable}: coarse field does not reproduce fine aggregation")

    build_qc_path = root / f"{variable}.qc.json"
    build_qc = json.loads(build_qc_path.read_text()) if build_qc_path.exists() else None
    if not build_qc or not build_qc.get("valid"):
        errors.append(f"{variable}: missing or failed build-time full-array QC")

    report = {
        "fine_coordinates": fine_coordinates,
        "coarse_coordinates": coarse_coordinates,
        "metadata": dict(fine.attrs),
        "storage": {
            "fine_dtype": str(array.dtype),
            "fine_chunks": list(array.chunks),
            "coarse_dtype": str(coarse_array.dtype),
            "coarse_chunks": list(coarse_array.chunks),
        },
        "build_time_full_array_qc": build_qc,
        "sample": {
            "time_stride_days": time_stride,
            "spatial_stride_cells": 10,
            "distribution": distribution(sampled_values),
            "by_source": by_source,
            "partial_supported_cells": int(partial.sum()),
            "values_outside_support": int(unexpected.sum()),
        },
        "temporal": temporal_report(fine, source),
        "source_boundary_climatology": boundary,
        "fine_to_coarse_consistency": consistency,
    }
    return report, errors, warnings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--time-stride", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root
    errors: list[str] = []
    warnings: list[str] = []

    sources = []
    for variable in VARIABLES:
        dataset = xr.open_zarr(
            root / "source" / f"{variable}.zarr",
            consolidated=False,
            chunks=None,
        )
        sources.append(np.asarray(dataset.reference_source))
    source_masks_identical = all(np.array_equal(sources[0], item) for item in sources[1:])
    if not source_masks_identical:
        errors.append("reference source masks differ between variables")
    source = sources[0]

    land = np.asarray(
        xr.open_zarr(
            root / "lulc_land_mask.zarr", consolidated=False, chunks=None
        ).lulc_land,
        dtype=bool,
    )
    missing_land = land & (source == 0)
    values_outside_land = (~land) & (source > 0)
    if missing_land.any():
        errors.append("mapped LULC land remains unsupported")
    if values_outside_land.any():
        warnings.append("reference support includes cells outside the LULC land mask")

    reports = {}
    for variable in VARIABLES:
        report, variable_errors, variable_warnings = variable_report(
            root, variable, source, args.time_stride
        )
        reports[variable] = report
        errors.extend(variable_errors)
        warnings.extend(variable_warnings)

    result = {
        "valid": not errors,
        "root": str(root),
        "errors": errors,
        "warnings": warnings,
        "source": {
            "masks_identical": source_masks_identical,
            "counts": {
                SOURCE_NAMES[code]: int((source == code).sum())
                for code in SOURCE_NAMES
            },
            "mapped_land_cells": int(land.sum()),
            "mapped_land_cells_without_reference": int(missing_land.sum()),
            "supported_cells_outside_land_mask": int(values_outside_land.sum()),
        },
        "variables": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=json_value) + "\n")
    print(json.dumps({
        "valid": result["valid"],
        "errors": errors,
        "warnings": warnings,
        "output": str(args.output),
    }, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
