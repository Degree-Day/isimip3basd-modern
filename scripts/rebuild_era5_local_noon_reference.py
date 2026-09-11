#!/usr/bin/env python3
"""Rebuild regular ERA5 reference cells from hourly Google ARCO local noon."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

import dask.array as da
import numpy as np
import xarray as xr
import zarr

from isimip3basd_modern.io import open_dataset
from isimip3basd_modern.validation import validate_variable


ARCO_URI = (
    "gs://gcp-public-data-arco-era5/ar/"
    "full_37-1h-0p25deg-chunk-1.zarr-v3"
)
ARCO_VARIABLES = {
    "temperature": "2m_temperature",
    "dewpoint": "2m_dewpoint_temperature",
    "u_wind": "10m_u_component_of_wind",
    "v_wind": "10m_v_component_of_wind",
}
OUTPUT_VARIABLES = ("tas", "hurs", "sfcWind")
ALL_VARIABLES = ("tas", "hurs", "pr", "sfcWind")
BOUNDS = {
    "tas": (180.0, 340.0),
    "hurs": (0.0, 100.0),
    "sfcWind": (0.0, 75.0),
}


def signed_longitude(longitude: np.ndarray) -> np.ndarray:
    return (longitude + 180.0) % 360.0 - 180.0


def local_noon_utc_offset(longitude: np.ndarray) -> np.ndarray:
    """Return the integer UTC-hour offset from a local calendar date's midnight."""
    return np.rint(12.0 - signed_longitude(longitude) / 15.0).astype(int)


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
    temperature_c = temperature - 273.15
    dewpoint_c = dewpoint - 273.15
    numerator = np.exp(17.625 * dewpoint_c / (243.04 + dewpoint_c))
    denominator = np.exp(17.625 * temperature_c / (243.04 + temperature_c))
    return np.clip(100.0 * numerator / denominator, 0.0, 100.0)


def initialize_sparse_store(
    path: Path,
    time_coordinate: xr.DataArray,
    rows: np.ndarray,
    columns: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    shape = (time_coordinate.size, rows.size)
    chunks = (365, min(2048, rows.size))
    variables = {
        variable: xr.DataArray(
            da.full(shape, np.nan, chunks=chunks, dtype="float32"),
            dims=("time", "cell"),
            attrs={"source": "Google ARCO ERA5 nearest-hour local solar noon"},
        )
        for variable in OUTPUT_VARIABLES
    }
    dataset = xr.Dataset(
        variables,
        coords={
            "time": time_coordinate,
            "cell": np.arange(rows.size, dtype="int32"),
            "row": ("cell", rows.astype("int32")),
            "column": ("cell", columns.astype("int32")),
            "lat": ("cell", latitudes.astype("float32")),
            "lon": ("cell", longitudes.astype("float32")),
        },
        attrs={
            "arco_uri": ARCO_URI,
            "temporal_sampling": "nearest UTC hour to local solar noon",
            "spatial_sampling": "bilinear 0.25 degree to 0.1 degree",
        },
    )
    dataset.to_zarr(
        path,
        mode="w",
        compute=False,
        consolidated=False,
        zarr_format=3,
        encoding={variable: {"_FillValue": np.nan} for variable in OUTPUT_VARIABLES},
    )


def local_dates(time_coordinate: xr.DataArray, year: int) -> tuple[np.ndarray, list[str]]:
    years = np.asarray(time_coordinate.dt.year)
    indices = np.where(years == year)[0]
    dates = [
        f"{int(time_coordinate.dt.year[index]):04d}-"
        f"{int(time_coordinate.dt.month[index]):02d}-"
        f"{int(time_coordinate.dt.day[index]):02d}"
        for index in indices
    ]
    return indices, dates


def load_field(
    dataset: xr.Dataset,
    variable: str,
    timestamp: np.datetime64,
    latitude: np.ndarray,
    longitude: np.ndarray,
    retries: int,
) -> np.ndarray:
    for attempt in range(retries):
        try:
            field = np.asarray(dataset[variable].sel(time=timestamp).load())
            sampled = bilinear(field, latitude, longitude).astype("float32")
            if not np.isfinite(sampled).all():
                raise RuntimeError(f"non-finite {variable} at {timestamp}")
            return sampled
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def extract_timestamp(
    dataset: xr.Dataset,
    timestamp: np.datetime64,
    assignments: list[tuple[int, int]],
    groups: dict[int, np.ndarray],
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    retries: int,
) -> tuple[list[tuple[int, int]], dict[str, np.ndarray]]:
    positions = np.concatenate([groups[offset] for _, offset in assignments])
    latitude = latitudes[positions]
    longitude = longitudes[positions]
    temperature = load_field(
        dataset, ARCO_VARIABLES["temperature"], timestamp, latitude, longitude, retries
    )
    dewpoint = load_field(
        dataset, ARCO_VARIABLES["dewpoint"], timestamp, latitude, longitude, retries
    )
    u_wind = load_field(
        dataset, ARCO_VARIABLES["u_wind"], timestamp, latitude, longitude, retries
    )
    v_wind = load_field(
        dataset, ARCO_VARIABLES["v_wind"], timestamp, latitude, longitude, retries
    )
    return assignments, {
        "tas": temperature,
        "hurs": relative_humidity(temperature, dewpoint).astype("float32"),
        "sfcWind": np.hypot(u_wind, v_wind).astype("float32"),
    }


def extract_year(
    dataset: xr.Dataset,
    year: int,
    dates: list[str],
    groups: dict[int, np.ndarray],
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    workers: int,
    retries: int,
) -> dict[str, np.ndarray]:
    output = {
        variable: np.full((len(dates), latitudes.size), np.nan, dtype="float32")
        for variable in OUTPUT_VARIABLES
    }
    jobs: dict[np.datetime64, list[tuple[int, int]]] = defaultdict(list)
    for day_index, date in enumerate(dates):
        midnight = np.datetime64(date, "h")
        for offset in groups:
            jobs[midnight + np.timedelta64(offset, "h")].append((day_index, offset))

    started = time.monotonic()
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                extract_timestamp,
                dataset,
                timestamp,
                assignments,
                groups,
                latitudes,
                longitudes,
                retries,
            ): timestamp
            for timestamp, assignments in jobs.items()
        }
        for future in as_completed(futures):
            assignments, values = future.result()
            start = 0
            for day_index, offset in assignments:
                positions = groups[offset]
                stop = start + positions.size
                for variable in OUTPUT_VARIABLES:
                    output[variable][day_index, positions] = values[variable][start:stop]
                start = stop
            completed += 1
            if completed % 240 == 0 or completed == len(futures):
                elapsed = time.monotonic() - started
                rate = completed / elapsed
                remaining = (len(futures) - completed) / rate if rate else 0
                print(
                    f"{year}: {completed}/{len(futures)} hourly fields "
                    f"({100 * completed / len(futures):.1f}%), ETA {remaining / 60:.1f} min",
                    flush=True,
                )

    for variable, values in output.items():
        lower, upper = BOUNDS[variable]
        if not np.isfinite(values).all():
            raise RuntimeError(f"{year} {variable} contains missing extracted values")
        if values.min() < lower or values.max() > upper:
            raise RuntimeError(
                f"{year} {variable} outside bounds: {values.min()} to {values.max()}"
            )
    return output


def write_sparse_year(
    path: Path,
    time_indices: np.ndarray,
    values: dict[str, np.ndarray],
) -> None:
    group = zarr.open_group(path, mode="r+")
    region = slice(int(time_indices[0]), int(time_indices[-1]) + 1)
    for variable in OUTPUT_VARIABLES:
        group[variable][region, :] = values[variable]


def group_spatial_chunks(
    rows: np.ndarray, columns: np.ndarray, chunk_size: int
) -> dict[tuple[int, int], np.ndarray]:
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, (row, column) in enumerate(zip(rows, columns)):
        groups[(int(row) // chunk_size, int(column) // chunk_size)].append(index)
    return {key: np.asarray(value, dtype=int) for key, value in groups.items()}


def aggregate_block(block: np.ndarray, latitude: np.ndarray) -> np.ndarray:
    time_size, rows, columns = block.shape
    if rows % 10 or columns % 10:
        raise ValueError("fine reference blocks must align with the nested 1 degree grid")
    reshaped = block.reshape(time_size, rows // 10, 10, columns // 10, 10)
    weights = np.cos(np.deg2rad(latitude)).reshape(1, rows // 10, 10, 1, 1)
    valid = np.isfinite(reshaped)
    numerator = np.nansum(reshaped * weights, axis=(2, 4))
    denominator = np.sum(valid * weights, axis=(2, 4))
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype="float32"),
        where=denominator > 0,
    ).astype("float32")


def clone_reference(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["cp", "-a", "--reflink=auto", str(source), str(destination)],
        check=True,
    )


def merge_sparse_reference(
    source_root: Path,
    destination_root: Path,
    sparse_path: Path,
    rows: np.ndarray,
    columns: np.ndarray,
    years: list[int],
    time_coordinate: xr.DataArray,
) -> None:
    clone_reference(source_root, destination_root)
    shutil.rmtree(destination_root / "qc_extensive", ignore_errors=True)
    sparse = zarr.open_group(sparse_path, mode="r")
    chunk_groups = group_spatial_chunks(rows, columns, 50)
    source_mask = np.asarray(
        xr.open_zarr(
            source_root / "source" / "tas.zarr", consolidated=False, chunks=None
        ).reference_source
    )
    fine_latitude = np.asarray(
        xr.open_zarr(
            source_root / "fine" / "tas.zarr", consolidated=False, chunks=None
        ).lat
    )

    for variable in OUTPUT_VARIABLES:
        source_fine = zarr.open_group(source_root / "fine" / f"{variable}.zarr", mode="r")[
            variable
        ]
        destination_fine = zarr.open_group(
            destination_root / "fine" / f"{variable}.zarr", mode="r+"
        )[variable]
        destination_coarse = zarr.open_group(
            destination_root / "coarse" / f"{variable}.zarr", mode="r+"
        )[variable]
        for year in years:
            time_indices, _ = local_dates(time_coordinate, year)
            time_slice = slice(int(time_indices[0]), int(time_indices[-1]) + 1)
            sparse_year = np.asarray(sparse[variable][time_slice, :])
            for (row_chunk, column_chunk), cell_indices in chunk_groups.items():
                row_start = row_chunk * 50
                column_start = column_chunk * 50
                row_stop = min(row_start + 50, source_mask.shape[0])
                column_stop = min(column_start + 50, source_mask.shape[1])
                row_slice = slice(row_start, row_stop)
                column_slice = slice(column_start, column_stop)
                original = np.asarray(
                    source_fine[time_slice, row_slice, column_slice]
                )
                rebuilt = original.copy()
                local_rows = rows[cell_indices] - row_start
                local_columns = columns[cell_indices] - column_start
                rebuilt[:, local_rows, local_columns] = sparse_year[:, cell_indices]
                unchanged = source_mask[row_slice, column_slice] != 3
                if not np.array_equal(
                    rebuilt[:, unchanged], original[:, unchanged], equal_nan=True
                ):
                    raise RuntimeError(f"{variable} merge changed non-ERA5 cells")
                destination_fine[time_slice, row_slice, column_slice] = rebuilt
                coarse = aggregate_block(rebuilt, fine_latitude[row_slice])
                destination_coarse[
                    time_slice,
                    slice(row_start // 10, row_stop // 10),
                    slice(column_start // 10, column_stop // 10),
                ] = coarse
            print(f"merged {variable} {year}", flush=True)

        provenance = {
            "reference_dataset": "ERA5-Land local noon with ERA5 local-noon extension",
            "reference_era5_source": ARCO_URI,
            "reference_era5_temporal_sampling": "nearest UTC hour to local solar noon",
            "reference_era5_spatial_sampling": "bilinear 0.25 degree to 0.1 degree",
            "reference_era5_rebuilt_variables": "tas hurs sfcWind",
            "reference_precipitation_semantics": "existing ERA5 daily accumulation retained",
        }
        destination_fine.attrs.update(provenance)
        destination_coarse.attrs.update(provenance)

    for variable in ALL_VARIABLES:
        source_store = zarr.open_group(
            destination_root / "source" / f"{variable}.zarr", mode="r+"
        )["reference_source"]
        source_store.attrs.update(
            flag_values=[0, 1, 2, 3],
            flag_meanings=(
                "outside_or_unavailable era5_land era5_land_coastal_repair "
                "era5_bilinear_extension"
            ),
        )


def write_fresh_qc(
    destination_root: Path,
    published_root: Path,
    rebuilt_variables: tuple[str, ...] = OUTPUT_VARIABLES,
) -> None:
    """Run full-array validation and replace copied build-time QC records."""
    source = np.asarray(
        xr.open_zarr(
            destination_root / "source" / "tas.zarr",
            consolidated=False,
            chunks=None,
        ).reference_source
    )
    source_names = {
        0: "outside LULC land or unavailable",
        1: "ERA5-Land",
        2: "ERA5-Land coastal repair",
        3: "ERA5 local-noon bilinear extension",
    }
    codes, counts = np.unique(source, return_counts=True)
    source_counts = {
        source_names[int(code)]: int(count)
        for code, count in zip(codes, counts, strict=True)
    }
    records = []
    for variable in ALL_VARIABLES:
        qc_path = destination_root / f"{variable}.qc.json"
        if variable in rebuilt_variables:
            reports = {}
            for label in ("fine", "coarse"):
                path = destination_root / label / f"{variable}.zarr"
                with open_dataset(path, {"time": 365}) as dataset:
                    report = validate_variable(
                        dataset[variable], variable, statistical=False
                    )
                if not report.valid:
                    raise RuntimeError(
                        f"rebuilt {label} {variable} QC failed: {report.errors}"
                    )
                reports[label] = report.to_dict()
            record = {
                "variable": variable,
                "fine": str(published_root / "fine" / f"{variable}.zarr"),
                "coarse": str(published_root / "coarse" / f"{variable}.zarr"),
                "valid": True,
                "qc": reports,
                "reference_source_cells": source_counts,
                "qc_created_utc": datetime.now(timezone.utc).isoformat(),
            }
        else:
            record = json.loads(qc_path.read_text())
            record["fine"] = str(published_root / "fine" / f"{variable}.zarr")
            record["coarse"] = str(published_root / "coarse" / f"{variable}.zarr")
            record["reference_source_cells"] = source_counts
        qc_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        records.append(record)
        print(f"validated {variable}", flush=True)

    manifest = {
        "source": "ERA5-Land local noon with Google ARCO ERA5 local-noon extension",
        "output": str(published_root),
        "valid_records": len(records),
        "records": records,
    }
    (destination_root / "reference-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--arco-uri", default=ARCO_URI)
    parser.add_argument("--start-year", type=int, default=1993)
    parser.add_argument("--end-year", type=int, default=2014)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--extract-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root == args.source_root:
        raise ValueError("output_root must differ from source_root")
    years = list(range(args.start_year, args.end_year + 1))
    source_dataset = xr.open_zarr(
        args.source_root / "source" / "tas.zarr", consolidated=False, chunks=None
    )
    source = np.asarray(source_dataset.reference_source)
    rows, columns = np.where(source == 3)
    latitudes = np.asarray(source_dataset.lat)[rows]
    longitudes = np.asarray(source_dataset.lon)[columns]
    offsets = local_noon_utc_offset(longitudes)
    groups = {offset: np.where(offsets == offset)[0] for offset in np.unique(offsets)}

    reference_data = xr.open_zarr(
        args.source_root / "fine" / "tas.zarr", consolidated=False, chunks=None
    )["tas"]
    sparse_path = args.checkpoint_root / "era5_local_noon_sparse.zarr"
    marker_root = args.checkpoint_root / "years"
    marker_root.mkdir(parents=True, exist_ok=True)
    initialize_sparse_store(
        sparse_path,
        reference_data.time,
        rows,
        columns,
        latitudes,
        longitudes,
    )

    arco = xr.open_zarr(
        args.arco_uri,
        consolidated=True,
        storage_options={"token": "anon"},
        chunks=None,
    )
    for year in years:
        marker = marker_root / f"{year}.json"
        if marker.exists():
            print(f"{year}: checkpoint complete", flush=True)
            continue
        time_indices, dates = local_dates(reference_data.time, year)
        values = extract_year(
            arco,
            year,
            dates,
            groups,
            latitudes,
            longitudes,
            args.workers,
            args.retries,
        )
        write_sparse_year(sparse_path, time_indices, values)
        marker.write_text(
            json.dumps(
                {
                    "year": year,
                    "days": len(dates),
                    "cells": rows.size,
                    "completed_utc": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            )
            + "\n"
        )
        print(f"{year}: checkpoint written", flush=True)

    if args.extract_only:
        return
    partial = args.output_root.with_name(f"{args.output_root.name}.partial")
    merge_sparse_reference(
        args.source_root,
        partial,
        sparse_path,
        rows,
        columns,
        years,
        reference_data.time,
    )
    write_fresh_qc(partial, args.output_root)
    provenance = {
        "source_reference": str(args.source_root),
        "arco_uri": args.arco_uri,
        "regular_era5_cells": int(rows.size),
        "years": years,
        "variables_rebuilt": list(OUTPUT_VARIABLES),
        "precipitation": "existing daily accumulation retained",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    (partial / "era5_local_noon_rebuild.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    partial.rename(args.output_root)
    print(f"completed reference: {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
