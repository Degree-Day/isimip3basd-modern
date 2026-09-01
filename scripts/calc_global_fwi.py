#!/usr/bin/env python3
"""Calculate daily CFFWIS indices over a restartable tiled global domain."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time

import dask
import dask.array as da
import numpy as np
import xarray as xr
from xclim.indices.fire import cffwis_indices


INPUT_VARIABLES = ("tas", "hurs", "pr", "sfcWind")
INDEX_METADATA = {
    "ffmc": "Fine Fuel Moisture Code",
    "dmc": "Duff Moisture Code",
    "dc": "Drought Code",
    "isi": "Initial Spread Index",
    "bui": "Build Up Index",
    "fwi": "Fire Weather Index",
}


def tile_specs(lat_size: int, lon_size: int, tile_size: int) -> list[dict[str, int]]:
    return [
        {
            "lat_start": lat,
            "lat_stop": min(lat + tile_size, lat_size),
            "lon_start": lon,
            "lon_stop": min(lon + tile_size, lon_size),
        }
        for lat in range(0, lat_size, tile_size)
        for lon in range(0, lon_size, tile_size)
    ]


def tile_name(tile: dict[str, int]) -> str:
    return f"lat{tile['lat_start']:04d}_lon{tile['lon_start']:04d}"


def _input_path(root: Path, region: str, variable: str) -> Path:
    return root / region / f"{variable}_downscaled.zarr"


def _open_variable(path: Path, variable: str) -> xr.DataArray:
    return xr.open_zarr(path, consolidated=False)[variable].transpose("time", "lat", "lon")


def _prepare_inputs(arrays: dict[str, xr.DataArray]) -> dict[str, xr.DataArray]:
    prepared = {name: data.chunk({"time": -1, "lat": -1, "lon": -1}) for name, data in arrays.items()}
    for data in prepared.values():
        data.lat.attrs.update(
            standard_name="latitude", units="degrees_north"
        )
    units = str(prepared["pr"].attrs.get("units", "")).replace(" ", "")
    if units in {"kgm-2s-1", "kg/m2/s"}:
        prepared["pr"] = prepared["pr"] * 86400.0
        prepared["pr"].attrs["units"] = "mm/day"
    for name, unit in {"tas": "K", "hurs": "%", "sfcWind": "m s-1"}.items():
        prepared[name].attrs["units"] = unit
    return prepared


def compute_indices(
    arrays: dict[str, xr.DataArray], output_start: str, output_end: str
) -> xr.Dataset:
    inputs = _prepare_inputs(arrays)
    dc, dmc, ffmc, isi, bui, fwi = cffwis_indices(
        tas=inputs["tas"],
        pr=inputs["pr"],
        sfcWind=inputs["sfcWind"],
        hurs=inputs["hurs"],
        lat=inputs["tas"].lat,
        season_method="WF93",
        overwintering=True,
        dry_start="GFWED",
        initial_start_up=True,
    )
    outputs = {"ffmc": ffmc, "dmc": dmc, "dc": dc, "isi": isi, "bui": bui, "fwi": fwi}
    cleaned = {}
    for name, data in outputs.items():
        data = data.sel(time=slice(output_start, output_end)).transpose("time", "lat", "lon")
        data = data.astype("float32").rename(name)
        data.attrs = {
            "long_name": INDEX_METADATA[name],
            "units": "1",
            "xclim_function": "xclim.indices.fire.cffwis_indices",
            "fwi_season_method": "WF93",
            "fwi_overwintering": "true",
            "fwi_dry_start": "GFWED",
            "initial_start_up": "true",
        }
        cleaned[name] = data
    return xr.Dataset(cleaned)


def initialize_output(
    template: xr.DataArray,
    output: Path,
    output_start: str,
    output_end: str,
    tile_size: int,
) -> None:
    if output.exists():
        existing = xr.open_zarr(output, consolidated=False)
        expected_time = template.sel(time=slice(output_start, output_end)).time
        if set(existing.data_vars) != set(INDEX_METADATA) or dict(existing.sizes) != {
            "time": expected_time.size,
            "lat": template.sizes["lat"],
            "lon": template.sizes["lon"],
        }:
            raise ValueError(f"existing FWI store shape is incompatible: {output}")
        if not (
            np.array_equal(existing.time.values, expected_time.values)
            and np.array_equal(existing.lat.values, template.lat.values)
            and np.array_equal(existing.lon.values, template.lon.values)
        ):
            raise ValueError(f"existing FWI store coordinates are incompatible: {output}")
        return
    selected_time = template.sel(time=slice(output_start, output_end)).time
    chunks = (min(365, selected_time.size), tile_size, tile_size)
    variables = {}
    for name, long_name in INDEX_METADATA.items():
        values = da.empty(
            (selected_time.size, template.sizes["lat"], template.sizes["lon"]),
            chunks=chunks,
            dtype="float32",
        )
        variables[name] = xr.DataArray(
            values,
            dims=("time", "lat", "lon"),
            coords={"time": selected_time, "lat": template.lat, "lon": template.lon},
            attrs={"long_name": long_name, "units": "1"},
        )
    dataset = xr.Dataset(
        variables,
        attrs={
            "title": "Daily Canadian Forest Fire Weather Index System outputs",
            "xclim_function": "xclim.indices.fire.cffwis_indices",
            "fwi_season_method": "WF93",
            "fwi_overwintering": "true",
            "fwi_dry_start": "GFWED",
        },
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_zarr(
        output,
        mode="w",
        compute=False,
        consolidated=False,
        zarr_format=3,
        encoding={name: {"_FillValue": np.nan} for name in INDEX_METADATA},
    )


def run_tile(
    input_root: str,
    region: str,
    output: str,
    state_root: str,
    tile: dict[str, int],
    compute_start: str,
    compute_end: str,
    output_start: str,
    output_end: str,
    threads: int,
) -> dict[str, object]:
    os.environ.update(
        OMP_NUM_THREADS=str(threads),
        MKL_NUM_THREADS=str(threads),
        OPENBLAS_NUM_THREADS=str(threads),
        NUMBA_NUM_THREADS=str(threads),
    )
    dask.config.set(scheduler="threads", num_workers=threads)
    marker = Path(state_root) / f"{tile_name(tile)}.success"
    if marker.exists():
        return {"tile": tile_name(tile), "skipped": True}
    started = time.perf_counter()
    spatial = {
        "lat": slice(tile["lat_start"], tile["lat_stop"]),
        "lon": slice(tile["lon_start"], tile["lon_stop"]),
    }
    arrays = {
        variable: _open_variable(
            _input_path(Path(input_root), region, variable), variable
        ).sel(time=slice(compute_start, compute_end)).isel(**spatial)
        for variable in INPUT_VARIABLES
    }
    dataset = compute_indices(arrays, output_start, output_end)
    variable_only = xr.Dataset(
        {
            name: (data.dims, data.data, data.attrs)
            for name, data in dataset.data_vars.items()
        }
    )
    variable_only.to_zarr(
        output,
        mode="r+",
        region={
            "time": slice(0, dataset.sizes["time"]),
            "lat": spatial["lat"],
            "lon": spatial["lon"],
        },
        consolidated=False,
    )
    sample = xr.open_zarr(output, consolidated=False).isel(**spatial)
    qc_values = dask.compute(
        *(sample[name].min(skipna=True) for name in INDEX_METADATA),
        *(np.isinf(sample[name]).any() for name in INDEX_METADATA),
    )
    minimum = min(float(value) for value in qc_values[: len(INDEX_METADATA)])
    has_inf = any(bool(value) for value in qc_values[len(INDEX_METADATA) :])
    if minimum < 0 or has_inf:
        raise RuntimeError(f"tile QC failed: minimum={minimum}, has_inf={has_inf}")
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    return {
        "tile": tile_name(tile),
        "skipped": False,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--region", default="global")
    parser.add_argument("--compute-start", required=True)
    parser.add_argument("--compute-end", required=True)
    parser.add_argument("--output-start", required=True)
    parser.add_argument("--output-end", required=True)
    parser.add_argument("--period-label", required=True)
    parser.add_argument("--tile-size", type=int, default=40)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.tile_size < 1 or args.workers < 1 or args.threads_per_worker < 1:
        parser.error("tile size, workers, and threads must be positive")
    for variable in INPUT_VARIABLES:
        path = _input_path(args.input_root, args.region, variable)
        if not path.exists():
            parser.error(f"missing input store: {path}")
    template = _open_variable(
        _input_path(args.input_root, args.region, "tas"), "tas"
    ).sel(time=slice(args.compute_start, args.compute_end))
    output = args.output_root / args.region / f"daily_fire_weather_indices_{args.period_label}.zarr"
    state = args.output_root / "state" / args.region / args.period_label
    if args.overwrite:
        shutil.rmtree(output, ignore_errors=True)
        shutil.rmtree(state, ignore_errors=True)
    initialize_output(template, output, args.output_start, args.output_end, args.tile_size)
    tiles = tile_specs(template.sizes["lat"], template.sizes["lon"], args.tile_size)
    pending = [tile for tile in tiles if not (state / f"{tile_name(tile)}.success").exists()]
    print(f"START global CFFWIS: {len(pending)}/{len(tiles)} tiles", flush=True)
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                run_tile,
                str(args.input_root),
                args.region,
                str(output),
                str(state),
                tile,
                args.compute_start,
                args.compute_end,
                args.output_start,
                args.output_end,
                args.threads_per_worker,
            )
            for tile in pending
        ]
        for index, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            records.append(record)
            if index % 25 == 0 or index == len(pending):
                print(f"DONE {index}/{len(pending)}; {record['tile']}", flush=True)

    written = xr.open_zarr(output, consolidated=False)
    valid = (
        set(written.data_vars) == set(INDEX_METADATA)
        and written.sizes["time"] == template.sel(
            time=slice(args.output_start, args.output_end)
        ).sizes["time"]
        and all((state / f"{tile_name(tile)}.success").exists() for tile in tiles)
    )
    manifest = {
        "input_root": str(args.input_root),
        "output": str(output),
        "region": args.region,
        "compute_period": [args.compute_start, args.compute_end],
        "output_period": [args.output_start, args.output_end],
        "variables": list(INDEX_METADATA),
        "tile_size": args.tile_size,
        "workers": args.workers,
        "threads_per_worker": args.threads_per_worker,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "valid": valid,
        "completed_tiles": len(tiles),
        "records": records,
    }
    manifest_path = output.with_suffix(".zarr.manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if not valid:
        raise SystemExit(1)
    print(f"Wrote {output}; manifest: {manifest_path}")


if __name__ == "__main__":
    main()
