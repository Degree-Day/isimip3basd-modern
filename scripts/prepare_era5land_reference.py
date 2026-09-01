#!/usr/bin/env python3
"""Prepare nested no-leap ERA5-Land references for 1 to 0.1 degree MBCnSD."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import dask.array as da
import numpy as np
import xarray as xr
from xclim.core.units import convert_units_to

from isimip3basd_modern.io import open_dataset
from isimip3basd_modern.preprocessing import _normalize_calendar
from isimip3basd_modern.validation import validate_variable


METADATA = {
    "hurs": ("%", "relative_humidity"),
    "pr": ("mm d-1", "precipitation_flux"),
    "sfcWind": ("m s-1", "wind_speed"),
    "tas": ("K", "air_temperature"),
}
TARGET_UNITS = {
    "hurs": "%",
    "pr": "kg m-2 s-1",
    "sfcWind": "m s-1",
    "tas": "K",
}


def nested_grids() -> tuple[xr.Dataset, xr.Dataset]:
    """Return the ERA5-Land domain on exactly nested 0.1 and 1 degree grids."""
    coarse_lat = np.arange(-56.5, 90.0, 1.0)
    coarse_lon = np.arange(0.5, 360.0, 1.0)
    fine_lat = np.arange(-56.95, 90.0, 0.1)
    fine_lon = np.arange(0.05, 360.0, 0.1)

    def grid(lat: np.ndarray, lon: np.ndarray) -> xr.Dataset:
        return xr.Dataset(
            coords={
                "lat": xr.DataArray(
                    lat,
                    dims="lat",
                    attrs={"standard_name": "latitude", "units": "degrees_north"},
                ),
                "lon": xr.DataArray(
                    lon,
                    dims="lon",
                    attrs={"standard_name": "longitude", "units": "degrees_east"},
                ),
            }
        )

    return grid(fine_lat, fine_lon), grid(coarse_lat, coarse_lon)


def prepare_fine(
    source: xr.DataArray,
    variable: str,
    fine_grid: xr.Dataset | None = None,
) -> xr.DataArray:
    units, standard_name = METADATA[variable]
    renames = {
        name: replacement
        for name, replacement in (("latitude", "lat"), ("longitude", "lon"))
        if name in source.dims
    }
    source = source.rename(renames).sortby("lat")
    source.attrs.update(units=units, standard_name=standard_name)
    source = source.sel(time=slice("1993-01-01", "2014-12-31"))
    source = convert_units_to(
        source,
        TARGET_UNITS[variable],
        context="hydro" if variable == "pr" else None,
    )
    source, source_calendar, day_delta = _normalize_calendar(source, variable)
    if fine_grid is None:
        fine_grid = nested_grids()[0]
    prepared = source.interp(
        lat=fine_grid.lat,
        lon=fine_grid.lon,
        method="linear",
        assume_sorted=True,
    )
    prepared = prepared.astype("float32").chunk(
        {"time": 365, "lat": 20, "lon": 20}
    )
    prepared.name = variable
    prepared.attrs.update(
        units=TARGET_UNITS[variable],
        standard_name=standard_name,
        reference_dataset="ERA5-Land local-noon daily weather",
        reference_period="1993-2014",
        preprocessing_calendar="noleap",
        preprocessing_source_calendar=source_calendar,
        preprocessing_calendar_day_delta=day_delta,
        preprocessing_grid="nested_0.1_degree_cell_centers",
        preprocessing_created_utc=datetime.now(timezone.utc).isoformat(),
    )
    return prepared


def aggregate_coarse(fine: xr.DataArray) -> xr.DataArray:
    """Area-average a nested fine reference to its 1 degree training grid."""
    weights = np.cos(np.deg2rad(fine.lat)).broadcast_like(fine)
    numerator = (fine * weights).coarsen(lat=10, lon=10, boundary="exact").sum()
    denominator = weights.where(fine.notnull()).coarsen(
        lat=10, lon=10, boundary="exact"
    ).sum()
    coarse = numerator / denominator
    _, coarse_grid = nested_grids()
    coarse = coarse.assign_coords(lat=coarse_grid.lat, lon=coarse_grid.lon)
    coarse.name = fine.name
    coarse.attrs.update(fine.attrs)
    coarse.attrs["preprocessing_grid"] = "nested_1_degree_area_mean"
    return coarse.astype("float32").chunk({"time": 365, "lat": 20, "lon": 20})


def initialize_store(path: Path, variable: str, grid: xr.Dataset) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    time_coord = xr.date_range(
        "1993-01-01", periods=22 * 365, freq="D", calendar="noleap", use_cftime=True
    )
    spatial_chunks = 50 if grid.sizes["lon"] == 3600 else 5
    values = da.empty(
        (time_coord.size, grid.sizes["lat"], grid.sizes["lon"]),
        chunks=(365, spatial_chunks, spatial_chunks),
        dtype="float32",
    )
    units, standard_name = METADATA[variable]
    skeleton = xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={"time": time_coord, "lat": grid.lat, "lon": grid.lon},
        name=variable,
        attrs={
            "units": TARGET_UNITS[variable],
            "standard_name": standard_name,
            "reference_dataset": "ERA5-Land local-noon daily weather",
            "reference_period": "1993-2014",
            "preprocessing_calendar": "noleap",
            "preprocessing_grid": (
                "nested_0.1_degree_cell_centers"
                if spatial_chunks == 50
                else "nested_1_degree_area_mean"
            ),
        },
    )
    skeleton.to_dataset().to_zarr(
        path,
        mode="w",
        compute=False,
        consolidated=False,
        zarr_format=3,
        encoding={variable: {"_FillValue": np.nan}},
    )


def process_tile(
    source_path: str,
    output_root: str,
    variable: str,
    lat_start: int,
    lat_stop: int,
    lon_start: int,
    lon_stop: int,
) -> str:
    fine_grid, coarse_grid = nested_grids()
    target = fine_grid.isel(
        lat=slice(lat_start, lat_stop), lon=slice(lon_start, lon_stop)
    )
    with xr.open_zarr(source_path, consolidated=False, chunks=None) as dataset:
        source = dataset[variable].sel(time=slice("1993-01-01", "2014-12-31"))
        source = source.rename(latitude="lat", longitude="lon").sortby("lat")
        source = source.sel(
            lat=slice(float(target.lat[0]) - 0.11, float(target.lat[-1]) + 0.11)
        )
        lon_lower = float(target.lon[0]) - 0.11
        lon_upper = float(target.lon[-1]) + 0.11
        source_lon = source.sel(lon=slice(max(0.0, lon_lower), min(359.9, lon_upper)))
        if lon_upper > 359.9:
            wrapped = source.isel(lon=[0]).assign_coords(lon=[360.0])
            source_lon = xr.concat((source_lon, wrapped), dim="lon")
        source = source_lon
        fine = prepare_fine(source, variable, target).compute()

    weights = np.cos(np.deg2rad(fine.lat)).broadcast_like(fine)
    numerator = (fine * weights).coarsen(lat=10, lon=10, boundary="exact").sum()
    denominator = weights.where(fine.notnull()).coarsen(
        lat=10, lon=10, boundary="exact"
    ).sum()
    coarse = (numerator / denominator).astype("float32")
    coarse.name = variable
    coarse_lat = slice(lat_start // 10, lat_stop // 10)
    coarse_lon = slice(lon_start // 10, lon_stop // 10)
    coarse = coarse.assign_coords(
        lat=coarse_grid.lat.isel(lat=coarse_lat),
        lon=coarse_grid.lon.isel(lon=coarse_lon),
    )

    root = Path(output_root)
    region_fine = {
        "time": slice(0, fine.sizes["time"]),
        "lat": slice(lat_start, lat_stop),
        "lon": slice(lon_start, lon_stop),
    }
    region_coarse = {
        "time": slice(0, coarse.sizes["time"]),
        "lat": coarse_lat,
        "lon": coarse_lon,
    }
    fine.to_dataset().drop_vars(["time", "lat", "lon"]).to_zarr(
        root / "fine" / f"{variable}.zarr",
        mode="r+",
        region=region_fine,
        consolidated=False,
    )
    coarse.to_dataset().drop_vars(["time", "lat", "lon"]).to_zarr(
        root / "coarse" / f"{variable}.zarr",
        mode="r+",
        region=region_coarse,
        consolidated=False,
    )
    marker = f"{lat_start:04d}-{lon_start:04d}"
    state = root / "state" / variable
    state.mkdir(parents=True, exist_ok=True)
    (state / marker).write_text("done\n")
    return marker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--memory-limit", default="16GB")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []

    fine_grid, coarse_grid = nested_grids()
    tiles = [
        (lat, min(lat + 50, fine_grid.sizes["lat"]), lon, lon + 50)
        for lat in range(0, fine_grid.sizes["lat"], 50)
        for lon in range(0, fine_grid.sizes["lon"], 50)
    ]
    for variable in METADATA:
        started = time.perf_counter()
        fine_path = args.output / "fine" / f"{variable}.zarr"
        coarse_path = args.output / "coarse" / f"{variable}.zarr"
        qc_path = args.output / f"{variable}.qc.json"
        if fine_path.exists() and coarse_path.exists() and qc_path.exists():
            existing = json.loads(qc_path.read_text())
            if not existing.get("valid"):
                raise RuntimeError(f"existing QC is invalid: {qc_path}")
            records.append(existing)
            print(f"SKIP {variable}", flush=True)
            continue
        initialize_store(fine_path, variable, fine_grid)
        initialize_store(coarse_path, variable, coarse_grid)
        state = args.output / "state" / variable
        pending = [
            tile
            for tile in tiles
            if not (state / f"{tile[0]:04d}-{tile[2]:04d}").exists()
        ]
        print(f"START {variable}: {len(pending)}/{len(tiles)} tiles", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_tile,
                    str(args.source),
                    str(args.output),
                    variable,
                    *tile,
                ): tile
                for tile in pending
            }
            for index, future in enumerate(as_completed(futures), start=1):
                marker = future.result()
                if index % 25 == 0 or index == len(pending):
                    print(
                        f"{variable} {index}/{len(pending)} tiles; last {marker}",
                        flush=True,
                    )

        reports = {}
        for label, path in (("fine", fine_path), ("coarse", coarse_path)):
            with open_dataset(path, {"time": 365}) as written:
                report = validate_variable(
                    written[variable], variable, statistical=False
                )
            if not report.valid:
                raise RuntimeError(f"{label} {variable} QC failed: {report.errors}")
            reports[label] = report.to_dict()
        record = {
            "variable": variable,
            "fine": str(fine_path),
            "coarse": str(coarse_path),
            "valid": True,
            "elapsed_seconds": time.perf_counter() - started,
            "qc": reports,
        }
        records.append(record)
        qc_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        print(f"DONE {variable} {record['elapsed_seconds']:.1f}s", flush=True)

    manifest = {
        "source": str(args.source),
        "output": str(args.output),
        "valid_records": len(records),
        "records": records,
    }
    (args.output / "reference-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
