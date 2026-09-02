#!/usr/bin/env python3
"""Fill mapped coastal land gaps in completed global downscaled stores."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import shutil
import time

from cartopy.io import shapereader
import numpy as np
import shapely
import xarray as xr
import zarr

from isimip3basd_modern.coastal import build_coastal_fill_plan


def open_variable(path: Path, variable: str) -> xr.DataArray:
    return xr.open_zarr(path, consolidated=False)[variable]


def write_zarr_atomic(dataset: xr.Dataset, path: Path) -> None:
    partial = path.with_name(f"{path.name}.partial")
    shutil.rmtree(partial, ignore_errors=True)
    dataset.to_zarr(partial, mode="w", consolidated=False, zarr_format=3)
    partial.rename(path)


def natural_earth_land(resolution: str) -> object:
    path = shapereader.natural_earth(resolution, "physical", "land")
    return shapely.union_all(list(shapereader.Reader(path).geometries()))


def ensure_plan(region_root: Path, resolution: str) -> Path:
    path = region_root / "coastal_fill_plan.zarr"
    if path.exists():
        return path
    valid = open_variable(
        region_root / "spatial_valid_mask.zarr", "spatial_valid_mask"
    ).load()
    plan = build_coastal_fill_plan(valid, natural_earth_land(resolution)).chunk(
        {"lat": 100, "lon": 100}
    )
    plan.attrs.update(
        land_geometry=f"Natural Earth physical land {resolution}",
        grid_cell_rule="cell footprint intersects land polygon",
    )
    write_zarr_atomic(plan, path)
    return path


def fill_tile(
    store: str,
    variable: str,
    plan_path: str,
    lat_start: int,
    lat_stop: int,
    lon_start: int,
    lon_stop: int,
    marker: str,
) -> dict[str, object]:
    marker_path = Path(marker)
    if marker_path.exists():
        return {"skipped": True, "cells": 0}
    plan = xr.open_zarr(plan_path, consolidated=False, chunks=None)
    core = plan.isel(
        lat=slice(lat_start, lat_stop), lon=slice(lon_start, lon_stop)
    )
    target_lat, target_lon = np.where(np.asarray(core.coastal_fill.values))
    if not target_lat.size:
        return {"skipped": True, "cells": 0}
    source_lat = np.asarray(core.source_lat_index.values)[target_lat, target_lon]
    source_lon = np.asarray(core.source_lon_index.values)[target_lat, target_lon]

    with xr.open_zarr(store, consolidated=False, chunks=None) as dataset:
        data = dataset[variable]
        written = data.isel(
            lat=slice(lat_start, lat_stop), lon=slice(lon_start, lon_stop)
        ).load()
        donors = data.isel(
            lat=xr.DataArray(source_lat, dims="coastal_cell"),
            lon=xr.DataArray(source_lon, dims="coastal_cell"),
        ).load()
    values = np.asarray(written.values)
    values[:, target_lat, target_lon] = np.asarray(
        donors.transpose("time", "coastal_cell").values
    )
    if not np.isfinite(values[:, target_lat, target_lon]).all():
        raise RuntimeError("coastal donor produced non-finite values")
    filled = xr.DataArray(
        values,
        dims=written.dims,
        coords=written.coords,
        name=variable,
        attrs=written.attrs,
    )
    filled.to_dataset().drop_vars(["time", "lat", "lon"]).to_zarr(
        store,
        mode="r+",
        region={
            "time": slice(0, filled.sizes["time"]),
            "lat": slice(lat_start, lat_stop),
            "lon": slice(lon_start, lon_stop),
        },
        consolidated=False,
    )
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.touch()
    return {"skipped": False, "cells": int(target_lat.size)}


def completed_spatial_tiles(region_root: Path, variable: str) -> int:
    state = region_root / "state_spatial_global_context" / variable
    return sum(1 for _ in state.glob("*.success"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("region_root", type=Path)
    parser.add_argument("--variables", nargs="+", default=["tas", "hurs", "pr", "sfcWind"])
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--expected-spatial-tiles", type=int, default=536)
    parser.add_argument("--land-resolution", choices=("110m", "50m", "10m"), default="10m")
    args = parser.parse_args()

    plan_path = ensure_plan(args.region_root, args.land_resolution)
    plan = xr.open_zarr(plan_path, consolidated=False, chunks=None)
    fill = np.asarray(plan.coastal_fill.values)
    tile_size = 10
    tiles = [
        (lat, min(lat + tile_size, fill.shape[0]), lon, min(lon + tile_size, fill.shape[1]))
        for lat in range(0, fill.shape[0], tile_size)
        for lon in range(0, fill.shape[1], tile_size)
        if fill[lat : lat + tile_size, lon : lon + tile_size].any()
    ]
    for variable in args.variables:
        completed = completed_spatial_tiles(args.region_root, variable)
        if completed != args.expected_spatial_tiles:
            print(
                f"SKIP {variable}: {completed}/{args.expected_spatial_tiles} spatial tiles complete",
                flush=True,
            )
            continue
        store = args.region_root / f"{variable}_downscaled.zarr"
        state = args.region_root / "state_coastal_fill" / variable
        started = time.perf_counter()
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            futures = []
            for lat_start, lat_stop, lon_start, lon_stop in tiles:
                marker = state / f"{lat_start:04d}-{lon_start:04d}.success"
                futures.append(
                    executor.submit(
                        fill_tile,
                        str(store),
                        variable,
                        str(plan_path),
                        lat_start,
                        lat_stop,
                        lon_start,
                        lon_stop,
                        str(marker),
                    )
                )
            filled_cells = 0
            for index, future in enumerate(as_completed(futures), start=1):
                record = future.result()
                filled_cells += int(record["cells"])
                if index % 100 == 0 or index == len(futures):
                    print(f"{variable}: {index}/{len(futures)} coastal chunks", flush=True)
        zarr.open_group(store, mode="r+")[variable].attrs.update(
            coastal_fill_method=plan.attrs["method"],
            coastal_fill_plan=str(plan_path),
            coastal_fill_cells=int(plan.attrs["coastal_fill_cell_count"]),
        )
        report = {
            "variable": variable,
            "store": str(store),
            "plan": str(plan_path),
            "coastal_cells": int(plan.attrs["coastal_fill_cell_count"]),
            "cells_written_this_run": filled_cells,
            "chunks": len(tiles),
            "elapsed_seconds": time.perf_counter() - started,
        }
        state.mkdir(parents=True, exist_ok=True)
        (state / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
