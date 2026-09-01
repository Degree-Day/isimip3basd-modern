#!/usr/bin/env python3
"""Run Europe MBCnSD as independent longitude stripes for better CPU use."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import time
from importlib.metadata import version

for name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ.setdefault(name, "1")

import dask
import dask.array as da
import numpy as np
import xarray as xr

from isimip3basd_modern.downscaling import (
    coarse_scale_conservation,
    downscale_variable,
)
from isimip3basd_modern import __version__
from isimip3basd_modern.pipeline import adjust_variable
from isimip3basd_modern.validation import validate_variable


VARIABLES = ("hurs", "pr", "sfcWind", "tas")
REGIONS = {
    "west": {
        "fine_lat": slice(920, 1290),
        "fine_lon": slice(3490, 3600),
        "coarse_lat": slice(92, 129),
        "coarse_lon": slice(349, 360),
        "description": "35.05-71.95N, 10.95W-0.05W",
    },
    "east": {
        "fine_lat": slice(920, 1290),
        "fine_lon": slice(0, 320),
        "coarse_lat": slice(92, 129),
        "coarse_lon": slice(0, 32),
        "description": "35.05-71.95N, 0.05-31.95E",
    },
}


def open_variable(path: Path, variable: str) -> xr.DataArray:
    return xr.open_zarr(path, consolidated=False)[variable]


def success_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.success")


def is_complete(path: Path) -> bool:
    return success_path(path).exists()


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


def variable_only_dataset(data: xr.DataArray) -> xr.Dataset:
    """Return only the data variable for a Zarr region write."""
    return xr.Dataset(
        {data.name: (data.dims, data.data, data.attrs)},
        coords={dim: (dim, data[dim].data, data[dim].attrs) for dim in data.dims},
    )


def initialize_output_store(
    adjusted: xr.DataArray,
    fine_reference: xr.DataArray,
    path: Path,
    *,
    iterations: int,
    quantiles: int,
) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    dims = ("time", "lat", "lon")
    shape = (
        adjusted.sizes["time"],
        fine_reference.sizes["lat"],
        fine_reference.sizes["lon"],
    )
    chunks = (adjusted.sizes["time"], 10, 10)
    template = xr.DataArray(
        da.empty(shape, chunks=chunks, dtype=adjusted.dtype),
        dims=dims,
        coords={
            "time": adjusted.time,
            "lat": fine_reference.lat,
            "lon": fine_reference.lon,
        },
        name=adjusted.name,
        attrs={
            **adjusted.attrs,
            "statistical_downscaling_method": "MBCnSD",
            "statistical_downscaling_iterations": iterations,
            "statistical_downscaling_quantiles": quantiles,
            "statistical_downscaling_software": (
                f"isimip3basd-modern/{__version__}; xarray/{version('xarray')}; "
                f"scipy/{version('scipy')}"
            ),
            "statistical_downscaling_source": (
                "ISIMIP3BASD/3.0.2; https://doi.org/10.5281/zenodo.7151476"
            ),
        },
    )
    template.to_dataset().to_zarr(
        path,
        mode="w",
        compute=False,
        consolidated=False,
        zarr_format=3,
        encoding={adjusted.name: {"_FillValue": float("nan")}},
    )


def initialize_adjusted_store(
    simulation: xr.DataArray,
    path: Path,
) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    chunks = (simulation.sizes["time"], 1, 1)
    template = xr.DataArray(
        da.empty(simulation.shape, chunks=chunks, dtype=simulation.dtype),
        dims=simulation.dims,
        coords={dim: simulation[dim] for dim in simulation.dims},
        name=simulation.name,
        attrs=simulation.attrs,
    )
    template.to_dataset().to_zarr(
        path,
        mode="w",
        compute=False,
        consolidated=False,
        zarr_format=3,
        encoding={simulation.name: {"_FillValue": float("nan")}},
    )


def _region_coarse(data: xr.DataArray, region: str) -> xr.DataArray:
    spec = REGIONS[region]
    return data.isel(lat=spec["coarse_lat"], lon=spec["coarse_lon"])


def _region_fine(data: xr.DataArray, region: str) -> xr.DataArray:
    spec = REGIONS[region]
    return data.isel(lat=spec["fine_lat"], lon=spec["fine_lon"])


def ensure_adjusted(
    *,
    model: str,
    scenario: str,
    region: str,
    variable: str,
    reference_root: Path,
    canonical_root: Path,
    output_root: Path,
) -> Path:
    output = output_root / region
    output.mkdir(parents=True, exist_ok=True)
    adjusted_path = output / f"{variable}_adjusted.zarr"
    if adjusted_path.exists() and not is_complete(adjusted_path):
        shutil.rmtree(adjusted_path)
    if is_complete(adjusted_path):
        return adjusted_path

    obs_coarse = _region_coarse(
        open_variable(reference_root / "coarse" / f"{variable}.zarr", variable),
        region,
    )
    historical = (
        open_variable(
            canonical_root / model / "historical" / "hist" / f"{variable}.zarr",
            variable,
        )
        .sel(time=slice("1993", "2014"))
        .sel(lat=obs_coarse.lat, lon=obs_coarse.lon)
    )
    simulation = (
        open_variable(
            canonical_root / model / scenario / "proj" / f"{variable}.zarr",
            variable,
        )
        .sel(lat=obs_coarse.lat, lon=obs_coarse.lon)
    )
    print(f"START {variable} {region} adjustment", flush=True)
    adjusted = adjust_variable(
        obs_coarse,
        historical,
        simulation,
        variable=variable,
        chunks={"lat": 1, "lon": 1},
    )
    write_zarr_atomic(adjusted, adjusted_path)
    print(f"DONE {variable} {region} adjustment", flush=True)
    return adjusted_path


def run_adjustment_tile(
    *,
    model: str,
    scenario: str,
    region: str,
    variable: str,
    tile: dict[str, int],
    reference_root: str,
    canonical_root: str,
    output_root: str,
) -> dict[str, object]:
    configure_worker_runtime()
    started = time.perf_counter()
    reference = Path(reference_root)
    canonical = Path(canonical_root)
    output = Path(output_root)
    adjusted_path = output / region / f"{variable}_adjusted.zarr"
    tile_marker = output / region / "state_adjusted" / variable / f"{_tile_name(tile)}.success"
    if tile_marker.exists():
        return {
            "variable": variable,
            "region": region,
            "tile": _tile_name(tile),
            "stage": "adjustment",
            "skipped": True,
        }

    obs_region = _region_coarse(
        open_variable(reference / "coarse" / f"{variable}.zarr", variable),
        region,
    )
    local_lon_start = tile["coarse_lon_start"] - REGIONS[region]["coarse_lon"].start
    local_lon_stop = tile["coarse_lon_stop"] - REGIONS[region]["coarse_lon"].start
    obs_coarse = obs_region.isel(lon=slice(local_lon_start, local_lon_stop))
    historical = (
        open_variable(
            canonical / model / "historical" / "hist" / f"{variable}.zarr",
            variable,
        )
        .sel(time=slice("1993", "2014"))
        .sel(lat=obs_coarse.lat, lon=obs_coarse.lon)
    )
    simulation = (
        open_variable(
            canonical / model / scenario / "proj" / f"{variable}.zarr",
            variable,
        )
        .sel(lat=obs_coarse.lat, lon=obs_coarse.lon)
    )
    adjusted = adjust_variable(
        obs_coarse,
        historical,
        simulation,
        variable=variable,
        chunks={"lat": 1, "lon": 1},
    )
    variable_only_dataset(adjusted).to_zarr(
        adjusted_path,
        mode="r+",
        region={
            "time": slice(0, adjusted.sizes["time"]),
            "lat": slice(0, adjusted.sizes["lat"]),
            "lon": slice(local_lon_start, local_lon_stop),
        },
        consolidated=False,
    )
    tile_marker.parent.mkdir(parents=True, exist_ok=True)
    tile_marker.touch()
    written_tile = open_variable(adjusted_path, variable).isel(
        lon=slice(local_lon_start, local_lon_stop)
    )
    report = validate_variable(
        written_tile,
        variable,
        min_valid_fraction=0.95,
        statistical=False,
        allow_out_of_bounds_hurs=variable == "hurs",
    )
    if not report.valid:
        raise RuntimeError(
            f"{variable} {region} {_tile_name(tile)} adjustment QC failed: {report.errors}"
        )
    record = {
        "model": model,
        "scenario": scenario,
        "region": region,
        "description": REGIONS[region]["description"],
        "variable": variable,
        "tile": _tile_name(tile),
        "stage": "adjustment",
        "path": str(adjusted_path),
        "valid": report.valid,
        "minimum": report.minimum,
        "maximum": report.maximum,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (tile_marker.with_suffix(".report.json")).write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    return record


def tile_specs(region: str, tile_lon_degrees: int) -> list[dict[str, int]]:
    if tile_lon_degrees < 2:
        raise ValueError("tile_lon_degrees must be at least 2 for MBCnSD")
    spec = REGIONS[region]
    lon_start = spec["coarse_lon"].start
    lon_stop = spec["coarse_lon"].stop
    tiles = []
    for coarse_lon in range(lon_start, lon_stop, tile_lon_degrees):
        coarse_lon_stop = min(coarse_lon + tile_lon_degrees, lon_stop)
        fine_lon = coarse_lon * 10
        fine_lon_stop = coarse_lon_stop * 10
        tiles.append(
            {
                "coarse_lon_start": coarse_lon,
                "coarse_lon_stop": coarse_lon_stop,
                "fine_lon_start": fine_lon,
                "fine_lon_stop": fine_lon_stop,
            }
        )
    if len(tiles) > 1 and (
        tiles[-1]["coarse_lon_stop"] - tiles[-1]["coarse_lon_start"]
    ) < 2:
        last = tiles.pop()
        tiles[-1]["coarse_lon_stop"] = last["coarse_lon_stop"]
        tiles[-1]["fine_lon_stop"] = last["fine_lon_stop"]
    return tiles


def _tile_name(tile: dict[str, int]) -> str:
    return f"lon{tile['coarse_lon_start']:03d}-{tile['coarse_lon_stop']:03d}"


def marker_path(
    output_root: Path, region: str, variable: str, tile: dict[str, int]
) -> Path:
    return output_root / region / "state" / variable / f"{_tile_name(tile)}.success"


def report_path(marker: Path) -> Path:
    return marker.with_suffix(".report.json")


def tile_complete(marker: Path) -> bool:
    return marker.exists() and report_path(marker).exists()


def configure_worker_runtime() -> None:
    dask.config.set(scheduler="synchronous", num_workers=1)


def run_tile(
    *,
    model: str,
    scenario: str,
    region: str,
    variable: str,
    tile: dict[str, int],
    reference_root: str,
    output_root: str,
    iterations: int,
    quantiles: int,
) -> dict[str, object]:
    configure_worker_runtime()
    started = time.perf_counter()
    reference = Path(reference_root)
    output = Path(output_root)
    tile_marker = marker_path(output, region, variable, tile)
    downscaled_path = output / region / f"{variable}_downscaled.zarr"
    if tile_complete(tile_marker):
        return {
            "variable": variable,
            "region": region,
            "tile": _tile_name(tile),
            "skipped": True,
        }

    adjusted = open_variable(output / region / f"{variable}_adjusted.zarr", variable)
    sim = adjusted.isel(
        lon=slice(
            tile["coarse_lon_start"] - REGIONS[region]["coarse_lon"].start,
            tile["coarse_lon_stop"] - REGIONS[region]["coarse_lon"].start,
        )
    )
    local_lon_start = tile["fine_lon_start"] - REGIONS[region]["fine_lon"].start
    local_lon_stop = tile["fine_lon_stop"] - REGIONS[region]["fine_lon"].start
    if not tile_marker.exists():
        obs_fine = _region_fine(
            open_variable(reference / "fine" / f"{variable}.zarr", variable),
            region,
        ).isel(lon=slice(local_lon_start, local_lon_stop))
        with dask.config.set({"array.rechunk.method": "tasks"}):
            downscaled = downscale_variable(
                obs_fine,
                sim,
                variable=variable,
                iterations=iterations,
                quantiles=quantiles,
                chunks={"lat": 10, "lon": 10},
            )
            variable_only_dataset(downscaled).to_zarr(
                downscaled_path,
                mode="r+",
                region={
                    "time": slice(0, downscaled.sizes["time"]),
                    "lat": slice(0, downscaled.sizes["lat"]),
                    "lon": slice(local_lon_start, local_lon_stop),
                },
                consolidated=False,
            )
        tile_marker.parent.mkdir(parents=True, exist_ok=True)
        tile_marker.touch()

    written_tile = open_variable(downscaled_path, variable).isel(
        lon=slice(
            tile["fine_lon_start"] - REGIONS[region]["fine_lon"].start,
            tile["fine_lon_stop"] - REGIONS[region]["fine_lon"].start,
        )
    )
    active_cells = int(((np.isfinite(written_tile).sum("time")) > 0).sum().compute())
    if active_cells:
        report = validate_variable(
            written_tile,
            variable,
            min_valid_fraction=0.95,
            statistical=False,
        )
        conservation = coarse_scale_conservation(written_tile, sim)
        valid = report.valid
        minimum = report.minimum
        maximum = report.maximum
        errors = report.errors
    else:
        conservation = {
            "valid": True,
            "not_applicable": True,
            "reason": "tile has no active fine-grid cells",
            "units": sim.attrs.get("units", ""),
        }
        valid = True
        minimum = None
        maximum = None
        errors = ()
    if not valid:
        raise RuntimeError(f"{variable} {region} {_tile_name(tile)} QC failed: {errors}")
    if not conservation["valid"]:
        raise RuntimeError(
            f"{variable} {region} {_tile_name(tile)} conservation failed: {conservation}"
        )
    record = {
        "model": model,
        "scenario": scenario,
        "region": region,
        "description": REGIONS[region]["description"],
        "variable": variable,
        "tile": _tile_name(tile),
        "path": str(downscaled_path),
        "valid": valid,
        "active_cells": active_cells,
        "minimum": minimum,
        "maximum": maximum,
        "conservation": conservation,
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path(tile_marker).write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="ACCESS-CM2")
    parser.add_argument("--scenario", default="ssp245")
    parser.add_argument("--regions", nargs="+", choices=REGIONS, default=list(REGIONS))
    parser.add_argument("--variables", nargs="+", choices=VARIABLES, default=list(VARIABLES))
    parser.add_argument("--reference-root", type=Path, default=Path("/data1/era5ref-europe-full"))
    parser.add_argument("--canonical-root", type=Path, default=Path("/data1/cmip6_fwi_1deg"))
    parser.add_argument("--output-root", type=Path, default=Path("/data1/access_europe_downscale_full"))
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--quantiles", type=int, default=50)
    parser.add_argument("--tile-lon-degrees", type=int, default=2)
    parser.add_argument("--tile-workers", type=int, default=8)
    args = parser.parse_args()

    manifest_records = []
    for region in args.regions:
        for variable in args.variables:
            tiles = tile_specs(region, args.tile_lon_degrees)
            adjusted_path = args.output_root / region / f"{variable}_adjusted.zarr"
            if not is_complete(adjusted_path):
                obs_coarse = _region_coarse(
                    open_variable(
                        args.reference_root / "coarse" / f"{variable}.zarr",
                        variable,
                    ),
                    region,
                )
                simulation = (
                    open_variable(
                        args.canonical_root
                        / args.model
                        / args.scenario
                        / "proj"
                        / f"{variable}.zarr",
                        variable,
                    )
                    .sel(lat=obs_coarse.lat, lon=obs_coarse.lon)
                )
                initialize_adjusted_store(simulation, adjusted_path)
                pending_adjustment = [
                    tile
                    for tile in tiles
                    if not (
                        args.output_root
                        / region
                        / "state_adjusted"
                        / variable
                        / f"{_tile_name(tile)}.success"
                    ).exists()
                ]
                print(
                    f"START {variable} {region} adjustment tiles: "
                    f"{len(pending_adjustment)}/{len(tiles)}",
                    flush=True,
                )
                with ProcessPoolExecutor(max_workers=args.tile_workers) as executor:
                    futures = [
                        executor.submit(
                            run_adjustment_tile,
                            model=args.model,
                            scenario=args.scenario,
                            region=region,
                            variable=variable,
                            tile=tile,
                            reference_root=str(args.reference_root),
                            canonical_root=str(args.canonical_root),
                            output_root=str(args.output_root),
                        )
                        for tile in pending_adjustment
                    ]
                    for index, future in enumerate(as_completed(futures), start=1):
                        record = future.result()
                        manifest_records.append(record)
                        print(
                            f"DONE {variable} {region} adjustment tile "
                            f"{index}/{len(pending_adjustment)}: {record.get('tile')}",
                            flush=True,
                        )
                success_path(adjusted_path).touch()

            adjusted = open_variable(
                adjusted_path, variable
            )
            fine_reference = _region_fine(
                open_variable(args.reference_root / "fine" / f"{variable}.zarr", variable),
                region,
            )
            initialize_output_store(
                adjusted,
                fine_reference,
                args.output_root / region / f"{variable}_downscaled.zarr",
                iterations=args.iterations,
                quantiles=args.quantiles,
            )
            pending = [
                tile
                for tile in tiles
                if not tile_complete(marker_path(args.output_root, region, variable, tile))
            ]
            print(
                f"START {variable} {region} MBCnSD tiles: {len(pending)}/{len(tiles)}",
                flush=True,
            )
            with ProcessPoolExecutor(max_workers=args.tile_workers) as executor:
                futures = [
                    executor.submit(
                        run_tile,
                        model=args.model,
                        scenario=args.scenario,
                        region=region,
                        variable=variable,
                        tile=tile,
                        reference_root=str(args.reference_root),
                        output_root=str(args.output_root),
                        iterations=args.iterations,
                        quantiles=args.quantiles,
                    )
                    for tile in pending
                ]
                for index, future in enumerate(as_completed(futures), start=1):
                    record = future.result()
                    manifest_records.append(record)
                    print(
                        f"DONE {variable} {region} tile {index}/{len(pending)}: "
                        f"{record.get('tile')}",
                        flush=True,
                    )

    manifest = {
        "model": args.model,
        "scenario": args.scenario,
        "regions": {key: REGIONS[key]["description"] for key in args.regions},
        "tile_lon_degrees": args.tile_lon_degrees,
        "records": manifest_records,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "downscale-tiles-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
