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
import warnings

import dask
import dask.array as da
import numpy as np
import xarray as xr
import zarr
from xclim.indices.fire import cffwis_indices

from isimip3basd_modern.publication import (
    PACKED_FILL_VALUE,
    PACKED_MAX_CODE,
    PACKED_MIN_CODE,
    PACKING_SPECS,
    packing_encoding,
)


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
        group = zarr.open_group(str(output), mode="r")
        expected_time = template.sel(time=slice(output_start, output_end)).time
        if set(existing.data_vars) != set(INDEX_METADATA) or dict(existing.sizes) != {
            "time": expected_time.size,
            "lat": template.sizes["lat"],
            "lon": template.sizes["lon"],
        }:
            raise ValueError(f"existing FWI store shape is incompatible: {output}")
        if any(group[name].dtype != np.dtype("int16") for name in INDEX_METADATA):
            raise ValueError(f"existing FWI store is not physically int16: {output}")
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
            "publication_format": "scaled int16 Zarr v3",
        },
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    encoding = {
        name: {**packing_encoding(name), "chunks": chunks}
        for name in INDEX_METADATA
    }
    dataset.to_zarr(
        output,
        mode="w",
        compute=False,
        consolidated=False,
        zarr_format=3,
        encoding=encoding,
    )


def pack_indices(dataset: xr.Dataset) -> dict[str, np.ndarray]:
    """Compute and pack one tile while rejecting out-of-range values."""
    loaded = dataset.load()
    packed = {}
    for name in INDEX_METADATA:
        values = np.asarray(loaded[name].values)
        finite = np.isfinite(values)
        spec = PACKING_SPECS[name]
        if finite.any():
            minimum = float(values[finite].min())
            maximum = float(values[finite].max())
            if minimum < spec.minimum or maximum > spec.maximum:
                raise ValueError(
                    f"{name} range {minimum}..{maximum} exceeds int16 packing "
                    f"range {spec.minimum}..{spec.maximum}"
                )
        result = np.full(values.shape, PACKED_FILL_VALUE, dtype="int16")
        if finite.any():
            codes = np.rint((values[finite] - spec.add_offset) / spec.scale_factor)
            if (codes < PACKED_MIN_CODE).any() or (codes > PACKED_MAX_CODE).any():
                raise ValueError(f"{name} generated an invalid packed code")
            result[finite] = codes.astype("int16")
        packed[name] = result
    return packed


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
    land_mask_store: str | None = None,
    land_mask_variable: str = "tg_mean",
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
    if land_mask_store:
        mask = xr.open_zarr(land_mask_store, consolidated=False, chunks=None)[
            land_mask_variable
        ]
        if "time" in mask.dims:
            mask = mask.isel(time=0)
        if not bool(mask.isel(**spatial).notnull().any().item()):
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch()
            return {"tile": tile_name(tile), "skipped_ocean": True}
    arrays = {
        variable: _open_variable(
            _input_path(Path(input_root), region, variable), variable
        ).sel(time=slice(compute_start, compute_end)).isel(**spatial)
        for variable in INPUT_VARIABLES
    }
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="numba")
        dataset = compute_indices(arrays, output_start, output_end)
        packed = pack_indices(dataset)
    group = zarr.open_group(output, mode="r+")
    output_region = (
        slice(0, dataset.sizes["time"]),
        spatial["lat"],
        spatial["lon"],
    )
    for name, values in packed.items():
        group[name][output_region] = values
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
    parser.add_argument("--land-mask-store", type=Path)
    parser.add_argument("--land-mask-variable", default="tg_mean")
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
                str(args.land_mask_store) if args.land_mask_store else None,
                args.land_mask_variable,
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
        "land_mask_store": str(args.land_mask_store) if args.land_mask_store else None,
        "land_mask_variable": args.land_mask_variable,
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
