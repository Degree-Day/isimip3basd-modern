#!/usr/bin/env python3
"""Run the western-Europe ACCESS-CM2 adjustment and 0.1 degree MBCnSD pilot."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import time

import dask
from distributed import Client, LocalCluster
import xarray as xr

from isimip3basd_modern.downscaling import (
    coarse_scale_conservation,
    downscale_variable,
)
from isimip3basd_modern.pipeline import adjust_variable
from isimip3basd_modern.validation import validate_variable


VARIABLES = ("hurs", "pr", "sfcWind", "tas")


def open_variable(path: Path, variable: str) -> xr.DataArray:
    return xr.open_zarr(path, consolidated=False)[variable]


def write_zarr_atomic(data: xr.DataArray, path: Path) -> None:
    partial = path.with_name(f"{path.name}.partial")
    shutil.rmtree(partial, ignore_errors=True)
    data.to_dataset().to_zarr(
        partial,
        mode="w",
        consolidated=False,
        zarr_format=3,
    )
    shutil.rmtree(path, ignore_errors=True)
    partial.rename(path)
    success_path(path).touch()


def is_complete(path: Path) -> bool:
    return success_path(path).exists()


def success_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.success")


def main() -> None:
    output = Path("/data1/access_europe_downscale_pilot")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    with dask.config.set({"array.rechunk.method": "tasks"}), LocalCluster(
        n_workers=12, threads_per_worker=1, memory_limit="16GB", processes=True
    ) as cluster, Client(cluster):
        for variable in VARIABLES:
            started = time.perf_counter()
            adjusted_path = output / f"{variable}_adjusted.zarr"
            downscaled_path = output / f"{variable}_downscaled.zarr"
            obs_fine = open_variable(
                Path(f"/data1/era5ref-europe-pilot/fine/{variable}.zarr"), variable
            ).isel(lat=slice(1050, 1100), lon=slice(100, 150))
            obs_coarse = open_variable(
                Path(f"/data1/era5ref-europe-pilot/coarse/{variable}.zarr"), variable
            ).isel(lat=slice(105, 110), lon=slice(10, 15))
            historical = open_variable(
                Path(
                    f"/data1/cmip6_fwi_1deg/ACCESS-CM2/historical/hist/{variable}.zarr"
                ),
                variable,
            ).sel(
                time=slice("1993", "2014"),
                lat=slice(48.5, 52.5),
                lon=slice(10.5, 14.5),
            )
            simulation = open_variable(
                Path(
                    f"/data1/cmip6_fwi_1deg/ACCESS-CM2/ssp245/proj/{variable}.zarr"
                ),
                variable,
            ).sel(lat=slice(48.5, 52.5), lon=slice(10.5, 14.5))

            if adjusted_path.exists() and not is_complete(adjusted_path):
                # Products written before atomic completion markers were introduced
                # are validated by opening them before being marked complete.
                open_variable(adjusted_path, variable).isel(time=0).load()
                success_path(adjusted_path).touch()
            if not is_complete(adjusted_path):
                print(f"START {variable} adjustment", flush=True)
                adjusted = adjust_variable(
                    obs_coarse,
                    historical,
                    simulation,
                    variable=variable,
                    chunks={"lat": 1, "lon": 1},
                )
                write_zarr_atomic(adjusted, adjusted_path)
            adjusted = open_variable(adjusted_path, variable)

            if downscaled_path.exists() and not is_complete(downscaled_path):
                shutil.rmtree(downscaled_path)
            if not is_complete(downscaled_path):
                print(f"START {variable} MBCnSD", flush=True)
                downscaled = downscale_variable(
                    obs_fine,
                    adjusted,
                    variable=variable,
                    iterations=20,
                    quantiles=50,
                    chunks={"lat": 10, "lon": 10},
                )
                write_zarr_atomic(downscaled, downscaled_path)
            downscaled = open_variable(downscaled_path, variable)
            report = validate_variable(downscaled, variable, statistical=False)
            conservation = coarse_scale_conservation(downscaled, adjusted)
            record = {
                "variable": variable,
                "valid": report.valid,
                "minimum": report.minimum,
                "maximum": report.maximum,
                "conservation": conservation,
                "elapsed_seconds": time.perf_counter() - started,
            }
            records.append(record)
            print(f"DONE {variable}: {record}", flush=True)
    (output / "pilot-report.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
