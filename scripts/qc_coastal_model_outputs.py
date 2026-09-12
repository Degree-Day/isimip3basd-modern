#!/usr/bin/env python3
"""Audit coastal source transitions in downscaled weather and daily FWI output."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import xarray as xr
import zarr


WEATHER_VARIABLES = ("tas", "hurs", "pr", "sfcWind")
FWI_VARIABLES = ("ffmc", "dmc", "dc", "isi", "bui", "fwi")
SOURCE_NAMES = {1: "era5_land", 2: "era5_land_coastal_repair", 3: "era5_extension"}


def common_source(reference_root: Path) -> xr.DataArray:
    sources = [
        xr.open_zarr(
            reference_root / "source" / f"{variable}.zarr", consolidated=False
        )["reference_source"].load()
        for variable in WEATHER_VARIABLES
    ]
    for source in sources[1:]:
        if not np.array_equal(source.values, sources[0].values):
            raise ValueError("reference-source footprints differ among variables")
    return sources[0]


def _all_edges(source: np.ndarray) -> tuple[np.ndarray, ...]:
    lat_size, lon_size = source.shape
    rows, columns = np.indices(source.shape)
    first_lat = np.concatenate((rows[:-1].ravel(), rows.ravel()))
    first_lon = np.concatenate((columns[:-1].ravel(), columns.ravel()))
    second_lat = np.concatenate((rows[1:].ravel(), rows.ravel()))
    second_lon = np.concatenate(
        (columns[1:].ravel(), np.roll(columns, -1, axis=1).ravel())
    )
    # The second block includes the periodic longitude edge at 360/0 degrees.
    assert first_lat.size == (lat_size - 1) * lon_size + lat_size * lon_size
    return first_lat, first_lon, second_lat, second_lon


def sampled_edges(
    source: np.ndarray, maximum_per_group: int, seed: int
) -> dict[str, dict[str, list[int]]]:
    first_lat, first_lon, second_lat, second_lon = _all_edges(source)
    first_source = source[first_lat, first_lon]
    second_source = source[second_lat, second_lon]
    active = (first_source > 0) & (second_source > 0)
    groups: dict[str, np.ndarray] = {}
    for first in SOURCE_NAMES:
        same = active & (first_source == first) & (second_source == first)
        groups[f"same--{SOURCE_NAMES[first]}"] = np.flatnonzero(same)
        for second in range(first + 1, 4):
            transition = active & (
                ((first_source == first) & (second_source == second))
                | ((first_source == second) & (second_source == first))
            )
            groups[f"{SOURCE_NAMES[first]}--{SOURCE_NAMES[second]}"] = np.flatnonzero(
                transition
            )

    rng = np.random.default_rng(seed)
    result = {}
    for name, available in groups.items():
        if available.size > maximum_per_group:
            selected = np.sort(
                rng.choice(available, size=maximum_per_group, replace=False)
            )
        else:
            selected = available
        result[name] = {
            "available_edges": int(available.size),
            "first_lat": first_lat[selected].tolist(),
            "first_lon": first_lon[selected].tolist(),
            "second_lat": second_lat[selected].tolist(),
            "second_lon": second_lon[selected].tolist(),
        }
    return result


def _statistics(values: np.ndarray) -> dict[str, float | int | None]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": 0, "median": None, "p95": None, "p99": None, "maximum": None}
    quantiles = np.quantile(finite, (0.5, 0.95, 0.99))
    return {
        "count": int(finite.size),
        "median": float(quantiles[0]),
        "p95": float(quantiles[1]),
        "p99": float(quantiles[2]),
        "maximum": float(finite.max()),
    }


def audit_array(
    data: xr.DataArray, groups: dict[str, dict[str, list[int]]], multiplier: float
) -> dict[str, object]:
    report = {}
    for name, group in groups.items():
        if not group["first_lat"]:
            report[name] = {"sampled_edges": 0}
            continue
        edge = xr.DataArray(np.arange(len(group["first_lat"])), dims="edge")
        first_data = data.isel(
            lat=xr.DataArray(group["first_lat"], dims="edge", coords={"edge": edge}),
            lon=xr.DataArray(group["first_lon"], dims="edge", coords={"edge": edge}),
        )
        second_data = data.isel(
            lat=xr.DataArray(group["second_lat"], dims="edge", coords={"edge": edge}),
            lon=xr.DataArray(group["second_lon"], dims="edge", coords={"edge": edge}),
        )
        first = first_data.values
        second = second_data.values
        paired = np.isfinite(first) & np.isfinite(second)
        edge_axis = first_data.get_axis_num("edge")
        time_axes = tuple(axis for axis in range(paired.ndim) if axis != edge_axis)
        paired_by_edge = np.any(paired, axis=time_axes)
        first_valid_by_edge = np.any(np.isfinite(first), axis=time_axes)
        second_valid_by_edge = np.any(np.isfinite(second), axis=time_axes)
        difference = np.where(paired, np.abs(first - second) * multiplier, np.nan)
        endpoint = np.concatenate((first[np.isfinite(first)], second[np.isfinite(second)]))
        item = {
            "available_edges": group["available_edges"],
            "sampled_edges": len(group["first_lat"]),
            "paired_observations": int(paired.sum()),
            "missing_endpoint_observations": int((~np.isfinite(first)).sum() + (~np.isfinite(second)).sum()),
            "missing_endpoint_fraction": float(
                ((~np.isfinite(first)).sum() + (~np.isfinite(second)).sum())
                / (first.size + second.size)
            ),
            "edges_without_any_paired_observation": int((~paired_by_edge).sum()),
            "edges_with_either_endpoint_never_valid": int(
                (~(first_valid_by_edge & second_valid_by_edge)).sum()
            ),
            "absolute_neighbor_difference": _statistics(difference),
            "endpoint_minimum": float(endpoint.min() * multiplier) if endpoint.size else None,
            "endpoint_maximum": float(endpoint.max() * multiplier) if endpoint.size else None,
        }
        report[name] = item
    for first, second in ((1, 2), (1, 3), (2, 3)):
        transition = f"{SOURCE_NAMES[first]}--{SOURCE_NAMES[second]}"
        if transition not in report:
            continue
        boundary = report[transition].get("absolute_neighbor_difference", {}).get("p95")
        controls = [
            report[f"same--{SOURCE_NAMES[source]}"].get(
                "absolute_neighbor_difference", {}
            ).get("p95")
            for source in (first, second)
            if f"same--{SOURCE_NAMES[source]}" in report
        ]
        controls = [value for value in controls if value not in (None, 0)]
        report[transition]["boundary_to_same_source_p95_ratio"] = (
            float(boundary / np.mean(controls)) if boundary is not None and controls else None
        )
    return report


def _open_weather(root: Path, variable: str) -> tuple[xr.DataArray, str, float]:
    path = root / f"{variable}_downscaled.zarr"
    data = xr.open_zarr(path, consolidated=False)[variable]
    unit = str(data.attrs.get("units", ""))
    if variable == "tas":
        return data - 273.15, "degC", 1.0
    if variable == "pr":
        return data, "mm/day", 86_400.0
    return data, unit, 1.0


def audit_weather(
    root: Path, groups: dict[str, dict[str, list[int]]]
) -> dict[str, object]:
    result = {}
    for variable in WEATHER_VARIABLES:
        data, unit, multiplier = _open_weather(root, variable)
        path = root / f"{variable}_downscaled.zarr"
        raw = zarr.open_group(str(path), mode="r")[variable]
        result[variable] = {
            "units": unit,
            "raw_dtype": str(raw.dtype),
            "time_start": str(data.time.values[0]),
            "time_end": str(data.time.values[-1]),
            "groups": audit_array(data, groups, multiplier),
        }
    return result


def audit_fwi(path: Path, groups: dict[str, dict[str, list[int]]]) -> dict[str, object]:
    dataset = xr.open_zarr(path, consolidated=False)
    group = zarr.open_group(str(path), mode="r")
    return {
        variable: {
            "units": "1",
            "raw_dtype": str(group[variable].dtype),
            "groups": audit_array(dataset[variable], groups, 1.0),
        }
        for variable in FWI_VARIABLES
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_root", type=Path)
    parser.add_argument("historical_weather_root", type=Path)
    parser.add_argument("projection_weather_root", type=Path)
    parser.add_argument("historical_fwi", type=Path)
    parser.add_argument("projection_fwi", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--maximum-edges", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260911)
    args = parser.parse_args()

    source = common_source(args.reference_root)
    groups = sampled_edges(np.asarray(source), args.maximum_edges, args.seed)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reference_root": str(args.reference_root),
        "method": (
            "Deterministic global samples of adjacent source-transition and "
            "same-source edges evaluated over every daily time step."
        ),
        "edge_groups": {
            name: {
                "available_edges": group["available_edges"],
                "sampled_edges": len(group["first_lat"]),
            }
            for name, group in groups.items()
        },
        "historical_weather": audit_weather(args.historical_weather_root, groups),
        "projection_weather": audit_weather(args.projection_weather_root, groups),
        "historical_fwi": audit_fwi(args.historical_fwi, groups),
        "projection_fwi": audit_fwi(args.projection_fwi, groups),
    }
    missing = []
    warnings = []
    for section in ("historical_weather", "projection_weather"):
        for variable, variable_report in report[section].items():
            for name, item in variable_report["groups"].items():
                if not name.startswith("same--") and item.get("missing_endpoint_observations", 0):
                    missing.append(f"{section}/{variable}/{name}")
    for section in ("historical_fwi", "projection_fwi"):
        for variable, variable_report in report[section].items():
            for name, item in variable_report["groups"].items():
                if not name.startswith("same--") and item.get(
                    "edges_with_either_endpoint_never_valid", 0
                ):
                    missing.append(f"{section}/{variable}/{name}")
    for section in (
        "historical_weather",
        "projection_weather",
        "historical_fwi",
        "projection_fwi",
    ):
        for variable, variable_report in report[section].items():
            for name, item in variable_report["groups"].items():
                ratio = item.get("boundary_to_same_source_p95_ratio")
                if not name.startswith("same--") and ratio is not None and ratio > 5:
                    warnings.append(
                        f"{section}/{variable}/{name}: p95 boundary ratio {ratio:.2f}"
                    )
    report["missing_transition_groups"] = missing
    report["warnings"] = warnings
    report["valid"] = not missing
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output)
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
