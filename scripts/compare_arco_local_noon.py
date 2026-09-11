#!/usr/bin/env python3
"""Compare daily ERA5 reference joins with Google ARCO ERA5 local noon."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


ARCO_URI = (
    "gs://gcp-public-data-arco-era5/ar/"
    "full_37-1h-0p25deg-chunk-1.zarr-v3"
)
ARCO_VARIABLES = (
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
)
VARIABLES = ("tas", "hurs", "sfcWind")
SOURCE_NAMES = {1: "era5_land", 2: "era5_land_coastal_repair"}
DEFAULT_DATES = (
    "1995-01-15",
    "1995-04-15",
    "1995-07-15",
    "1995-10-15",
    "2013-01-15",
    "2013-04-15",
    "2013-07-15",
    "2013-10-15",
)


@dataclass(frozen=True)
class BoundaryEdges:
    target_row: np.ndarray
    target_column: np.ndarray
    neighbor_row: np.ndarray
    neighbor_column: np.ndarray
    neighbor_source: np.ndarray

    def take(self, indices: np.ndarray) -> "BoundaryEdges":
        return BoundaryEdges(
            *(getattr(self, field)[indices] for field in self.__dataclass_fields__)
        )

    def __len__(self) -> int:
        return self.target_row.size


def json_value(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def find_boundary_edges(source: np.ndarray) -> BoundaryEdges:
    """Find cardinal edges with regular ERA5 on one side and ERA5-Land on the other."""
    parts: list[tuple[np.ndarray, ...]] = []
    for row_offset, column_offset in ((1, 0), (0, 1)):
        first = source[
            : source.shape[0] - row_offset or None,
            : source.shape[1] - column_offset or None,
        ]
        second = source[
            row_offset:,
            column_offset:,
        ]
        active = ((first == 3) & np.isin(second, (1, 2))) | (
            (second == 3) & np.isin(first, (1, 2))
        )
        row, column = np.where(active)
        first_is_target = first[row, column] == 3
        target_row = row + np.where(first_is_target, 0, row_offset)
        target_column = column + np.where(first_is_target, 0, column_offset)
        neighbor_row = row + np.where(first_is_target, row_offset, 0)
        neighbor_column = column + np.where(first_is_target, column_offset, 0)
        parts.append(
            (
                target_row,
                target_column,
                neighbor_row,
                neighbor_column,
                source[neighbor_row, neighbor_column],
            )
        )
    return BoundaryEdges(*(np.concatenate(values) for values in zip(*parts)))


def signed_longitude(longitude: np.ndarray) -> np.ndarray:
    return (longitude + 180.0) % 360.0 - 180.0


def local_noon_utc_hour(longitude: np.ndarray) -> np.ndarray:
    return np.rint(12.0 - signed_longitude(longitude) / 15.0).astype(int) % 24


def sample_edges(
    edges: BoundaryEdges,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    per_hour_source: int,
) -> BoundaryEdges:
    """Sample evenly in latitude within each UTC-hour and neighbor-source group."""
    target_lat = latitudes[edges.target_row]
    target_lon = longitudes[edges.target_column]
    hours = local_noon_utc_hour(target_lon)
    selected: list[np.ndarray] = []
    for source in (1, 2):
        for hour in range(24):
            group = np.where((edges.neighbor_source == source) & (hours == hour))[0]
            if not group.size:
                continue
            order = group[np.argsort(target_lat[group])]
            count = min(per_hour_source, order.size)
            positions = np.linspace(0, order.size - 1, count).round().astype(int)
            selected.append(order[positions])
    return edges.take(np.unique(np.concatenate(selected)))


def utc_timestamps(local_dates: list[str], longitudes: np.ndarray) -> np.ndarray:
    timestamps = []
    offsets = signed_longitude(longitudes) / 15.0
    for date in local_dates:
        local_noon = pd.Timestamp(date) + pd.Timedelta(hours=12)
        values = [
            (local_noon - pd.Timedelta(hours=float(offset))).round("h")
            for offset in offsets
        ]
        timestamps.append(np.asarray(values, dtype="datetime64[ns]"))
    return np.stack(timestamps)


def bilinear(field: np.ndarray, latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    """Bilinearly sample the 0.25-degree global ARCO grid."""
    y = (90.0 - latitude) / 0.25
    x = (longitude % 360.0) / 0.25
    row0 = np.floor(y).astype(int)
    column0 = np.floor(x).astype(int) % field.shape[1]
    row1 = np.minimum(row0 + 1, field.shape[0] - 1)
    column1 = (column0 + 1) % field.shape[1]
    fy = y - row0
    fx = x - np.floor(x)
    return (
        field[row0, column0] * (1.0 - fy) * (1.0 - fx)
        + field[row1, column0] * fy * (1.0 - fx)
        + field[row0, column1] * (1.0 - fy) * fx
        + field[row1, column1] * fy * fx
    )


def relative_humidity(temperature: np.ndarray, dewpoint: np.ndarray) -> np.ndarray:
    """Derive percent RH using the Magnus saturation-vapour-pressure relation."""
    temperature_c = temperature - 273.15
    dewpoint_c = dewpoint - 273.15
    numerator = np.exp(17.625 * dewpoint_c / (243.04 + dewpoint_c))
    denominator = np.exp(17.625 * temperature_c / (243.04 + temperature_c))
    return np.clip(100.0 * numerator / denominator, 0.0, 100.0)


def date_indices(time: xr.DataArray, dates: list[str]) -> np.ndarray:
    year = np.asarray(time.dt.year)
    month = np.asarray(time.dt.month)
    day = np.asarray(time.dt.day)
    result = []
    for date in dates:
        stamp = pd.Timestamp(date)
        matches = np.where(
            (year == stamp.year) & (month == stamp.month) & (day == stamp.day)
        )[0]
        if matches.size != 1:
            raise ValueError(f"Reference date {date} matched {matches.size} entries")
        result.append(matches[0])
    return np.asarray(result)


def read_reference_points(
    root: Path,
    variable: str,
    dates: list[str],
    edges: BoundaryEdges,
) -> tuple[np.ndarray, np.ndarray]:
    data = xr.open_zarr(
        root / "fine" / f"{variable}.zarr", consolidated=False, chunks=None
    )[variable]
    times = date_indices(data.time, dates)
    target = data.isel(
        time=xr.DataArray(times, dims="date"),
        lat=xr.DataArray(edges.target_row, dims="edge"),
        lon=xr.DataArray(edges.target_column, dims="edge"),
    ).load()
    neighbor = data.isel(
        time=xr.DataArray(times, dims="date"),
        lat=xr.DataArray(edges.neighbor_row, dims="edge"),
        lon=xr.DataArray(edges.neighbor_column, dims="edge"),
    ).load()
    return np.asarray(target), np.asarray(neighbor)


def read_arco_local_noon(
    dataset: xr.Dataset,
    timestamps: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> dict[str, np.ndarray]:
    shape = timestamps.shape
    output = {variable: np.full(shape, np.nan, dtype="float32") for variable in VARIABLES}
    flat_timestamps = timestamps.ravel()
    flat_latitudes = np.broadcast_to(latitudes, shape).ravel()
    flat_longitudes = np.broadcast_to(longitudes, shape).ravel()

    def load_field(variable: str, timestamp: np.datetime64) -> np.ndarray:
        return np.asarray(dataset[variable].sel(time=timestamp).load())

    for timestamp in np.unique(flat_timestamps):
        active = np.where(flat_timestamps == timestamp)[0]
        with ThreadPoolExecutor(max_workers=4) as executor:
            fields = list(
                executor.map(
                    lambda variable: load_field(variable, timestamp), ARCO_VARIABLES
                )
            )
        temperature, dewpoint, u_wind, v_wind = (
            bilinear(field, flat_latitudes[active], flat_longitudes[active])
            for field in fields
        )
        output["tas"].ravel()[active] = temperature
        output["hurs"].ravel()[active] = relative_humidity(temperature, dewpoint)
        output["sfcWind"].ravel()[active] = np.hypot(u_wind, v_wind)
    return output


def statistics(values: np.ndarray) -> dict[str, float | int | None]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return {
            "count": 0,
            "mean": None,
            "p05": None,
            "p50": None,
            "p95": None,
            "p99": None,
        }
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "p05": float(np.quantile(finite, 0.05)),
        "p50": float(np.quantile(finite, 0.5)),
        "p95": float(np.quantile(finite, 0.95)),
        "p99": float(np.quantile(finite, 0.99)),
    }


def comparison(
    current: np.ndarray,
    local_noon: np.ndarray,
    neighbor: np.ndarray,
) -> dict[str, Any]:
    current_join = np.abs(current - neighbor)
    local_noon_join = np.abs(local_noon - neighbor)
    current_stats = statistics(current_join)
    noon_stats = statistics(local_noon_join)
    valid = np.isfinite(current_join) & np.isfinite(local_noon_join)
    return {
        "arco_local_noon_minus_reference": statistics(local_noon - current),
        "absolute_join_difference_reference": current_stats,
        "absolute_join_difference_arco_local_noon": noon_stats,
        "paired_join_difference_change": statistics(local_noon_join - current_join),
        "fraction_of_join_observations_improved": (
            float(np.mean(local_noon_join[valid] < current_join[valid]))
            if valid.any()
            else None
        ),
        "p95_change": (
            noon_stats["p95"] - current_stats["p95"]
            if noon_stats["p95"] is not None and current_stats["p95"] is not None
            else None
        ),
        "p95_percent_change": (
            100.0 * (noon_stats["p95"] / current_stats["p95"] - 1.0)
            if noon_stats["p95"] is not None and current_stats["p95"]
            else None
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dates", nargs="+", default=list(DEFAULT_DATES))
    parser.add_argument("--samples-per-hour-source", type=int, default=6)
    parser.add_argument("--arco-uri", default=ARCO_URI)
    parser.add_argument(
        "--reference-semantics",
        default="existing reference values",
        help="description of the values being compared with ARCO local noon",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_data = xr.open_zarr(
        args.reference_root / "source" / "tas.zarr",
        consolidated=False,
        chunks=None,
    )
    source = np.asarray(source_data.reference_source)
    latitudes = np.asarray(source_data.lat)
    longitudes = np.asarray(source_data.lon)
    all_edges = find_boundary_edges(source)
    edges = sample_edges(
        all_edges, latitudes, longitudes, args.samples_per_hour_source
    )
    target_latitudes = latitudes[edges.target_row]
    target_longitudes = longitudes[edges.target_column]
    timestamps = utc_timestamps(args.dates, target_longitudes)

    arco = xr.open_zarr(
        args.arco_uri,
        consolidated=True,
        storage_options={"token": "anon"},
        chunks=None,
    )
    local_noon = read_arco_local_noon(
        arco, timestamps, target_latitudes, target_longitudes
    )

    reports: dict[str, Any] = {}
    for variable in VARIABLES:
        current, neighbor = read_reference_points(
            args.reference_root, variable, args.dates, edges
        )
        reports[variable] = {"all_edges": comparison(current, local_noon[variable], neighbor)}
        for source_code, source_name in SOURCE_NAMES.items():
            active = edges.neighbor_source == source_code
            reports[variable][source_name] = comparison(
                current[:, active], local_noon[variable][:, active], neighbor[:, active]
            )

    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reference_root": str(args.reference_root),
        "arco_uri": args.arco_uri,
        "method": {
            "comparison": (
                f"{args.reference_semantics} and Google ARCO ERA5 hourly values at "
                "nearest-hour local solar noon are each compared with the same "
                "adjacent ERA5-Land or repaired ERA5-Land reference cell."
            ),
            "arco_spatial_interpolation": "bilinear 0.25 degree to 0.1 degree",
            "relative_humidity": (
                "derived from 2 m temperature and dewpoint using the Magnus relation"
            ),
            "wind_speed": "sqrt(u10**2 + v10**2)",
            "precipitation": "not compared; the reference requires a daily total",
        },
        "sample": {
            "dates": args.dates,
            "available_boundary_edges": len(all_edges),
            "sampled_boundary_edges": len(edges),
            "sampled_edges_by_neighbor_source": {
                SOURCE_NAMES[code]: int((edges.neighbor_source == code).sum())
                for code in SOURCE_NAMES
            },
            "unique_arco_timestamps": int(np.unique(timestamps).size),
            "latitude_range": [float(target_latitudes.min()), float(target_latitudes.max())],
            "longitude_range_0_360": [
                float(target_longitudes.min()),
                float(target_longitudes.max()),
            ],
        },
        "variables": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=json_value) + "\n")
    print(json.dumps({"output": str(args.output), "sample": result["sample"]}, indent=2))


if __name__ == "__main__":
    main()
