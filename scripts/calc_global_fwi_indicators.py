#!/usr/bin/env python3
"""Calculate restartable annual FWI indicators from packed daily global FWI."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

import dask.array as da
import numpy as np
import xarray as xr
import zarr
from zarr.codecs import BloscCodec


FILL = np.int16(-32768)
INDICATORS = {
    "fwixx": {
        "long_name": "Annual extreme value of the Fire Weather Index",
        "units": "1",
        "scale": 0.04,
        "offset": 1000.0,
    },
    "fwixd": {
        "long_name": "Annual number of days with extreme fire weather",
        "units": "d",
        "scale": 1.0,
        "offset": 0.0,
    },
    "fwils": {
        "long_name": "Annual length of the fire season",
        "units": "d",
        "scale": 1.0,
        "offset": 0.0,
    },
    "fwisa": {
        "long_name": "Seasonal average of the Fire Weather Index",
        "units": "1",
        "scale": 0.04,
        "offset": 1000.0,
    },
}
THRESHOLDS = {
    "fwi_q95_reference": "Local 95th percentile of daily FWI in the reference period",
    "fwi_midrange_reference": "Local midpoint of minimum and maximum daily FWI in the reference period",
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


def _annual_values(
    values: np.ndarray,
    years: np.ndarray,
    q95: np.ndarray,
    midrange: np.ndarray,
) -> dict[str, np.ndarray]:
    unique_years = np.unique(years)
    shape = (unique_years.size, *values.shape[1:])
    outputs = {
        name: np.full(shape, np.nan, dtype="float32") for name in INDICATORS
    }
    for index, year in enumerate(unique_years):
        annual = values[years == year]
        valid = np.isfinite(annual)
        any_valid = valid.any(axis=0)
        with np.errstate(all="ignore"):
            maximum = np.nanmax(annual, axis=0)
        outputs["fwixx"][index] = np.where(any_valid, maximum, np.nan)
        outputs["fwixd"][index] = np.where(
            any_valid, np.sum(valid & (annual > q95), axis=0), np.nan
        )
        outputs["fwils"][index] = np.where(
            any_valid, np.sum(valid & (annual > midrange), axis=0), np.nan
        )

        filled = np.where(valid, annual, 0.0).astype("float64")
        counts = valid.astype("int16")
        sums = np.concatenate(
            [np.zeros((1, *annual.shape[1:])), np.cumsum(filled, axis=0)], axis=0
        )
        count_sums = np.concatenate(
            [np.zeros((1, *annual.shape[1:]), dtype="int32"), np.cumsum(counts, axis=0)],
            axis=0,
        )
        if annual.shape[0] >= 90:
            window_sum = sums[90:] - sums[:-90]
            window_count = count_sums[90:] - count_sums[:-90]
            window_mean = np.where(window_count == 90, window_sum / 90.0, np.nan)
            with np.errstate(all="ignore"):
                seasonal = np.nanmax(window_mean, axis=0)
            outputs["fwisa"][index] = np.where(
                np.isfinite(seasonal), seasonal, np.nan
            )
    return outputs


def _pack(values: np.ndarray, scale: float, offset: float) -> np.ndarray:
    finite = np.isfinite(values)
    result = np.full(values.shape, FILL, dtype="int16")
    if finite.any():
        codes = np.rint((values[finite] - offset) / scale)
        if (codes < -32767).any() or (codes > 32767).any():
            raise ValueError(
                f"values {values[finite].min()}..{values[finite].max()} exceed packing range"
            )
        result[finite] = codes.astype("int16")
    return result


def _encoding(scale: float, offset: float, chunks: tuple[int, ...]) -> dict:
    return {
        "dtype": "int16",
        "_FillValue": FILL,
        "fill_value": FILL,
        "scale_factor": scale,
        "add_offset": offset,
        "chunks": chunks,
        "compressors": [BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")],
    }


def initialize_outputs(
    historical: xr.Dataset,
    future: xr.Dataset,
    annual_output: Path,
    threshold_output: Path,
    tile_size: int,
    reference_period: str,
) -> None:
    annual_times = []
    for dataset in (historical, future):
        dataset_years = dataset.time.dt.year.values
        first = np.r_[0, np.flatnonzero(np.diff(dataset_years)) + 1]
        annual_times.append(dataset.time.values[first])
    years = xr.DataArray(
        np.concatenate(annual_times),
        dims="time",
        name="time",
        attrs=historical.time.attrs,
    )
    annual_chunks = (1, tile_size, tile_size)
    if not annual_output.exists():
        data_vars = {}
        encoding = {}
        for name, metadata in INDICATORS.items():
            data_vars[name] = xr.DataArray(
                da.empty(
                    (years.size, historical.sizes["lat"], historical.sizes["lon"]),
                    chunks=annual_chunks,
                    dtype="float32",
                ),
                dims=("time", "lat", "lon"),
                coords={"time": years, "lat": historical.lat, "lon": historical.lon},
                attrs={"long_name": metadata["long_name"], "units": metadata["units"]},
            )
            encoding[name] = _encoding(
                metadata["scale"], metadata["offset"], annual_chunks
            )
        xr.Dataset(
            data_vars,
            attrs={
                "title": "Annual Canadian Fire Weather Index indicators",
                "reference_period": reference_period,
                "publication_format": "scaled int16 Zarr v3",
                "fwixx_definition": "local annual maximum of daily FWI",
                "fwixd_definition": "annual count of daily FWI above the local reference-period 95th percentile",
                "fwils_definition": "annual count of daily FWI above the local reference-period midrange",
                "fwisa_definition": "local annual maximum of the 90-day running mean of daily FWI",
            },
        ).to_zarr(
            annual_output,
            mode="w",
            compute=False,
            consolidated=False,
            zarr_format=3,
            encoding=encoding,
        )

    threshold_chunks = (tile_size, tile_size)
    if not threshold_output.exists():
        variables = {
            name: xr.DataArray(
                da.empty(
                    (historical.sizes["lat"], historical.sizes["lon"]),
                    chunks=threshold_chunks,
                    dtype="float32",
                ),
                dims=("lat", "lon"),
                coords={"lat": historical.lat, "lon": historical.lon},
                attrs={"long_name": long_name, "units": "1"},
            )
            for name, long_name in THRESHOLDS.items()
        }
        xr.Dataset(
            variables,
            attrs={
                "title": "Local FWI thresholds for annual indicators",
                "reference_period": reference_period,
                "publication_format": "scaled int16 Zarr v3",
            },
        ).to_zarr(
            threshold_output,
            mode="w",
            compute=False,
            consolidated=False,
            zarr_format=3,
            encoding={
                name: _encoding(0.04, 1000.0, threshold_chunks)
                for name in THRESHOLDS
            },
        )


def run_tile(
    historical_path: str,
    future_path: str,
    annual_output: str,
    threshold_output: str,
    state_root: str,
    tile: dict[str, int],
    reference_start_year: int,
    reference_end_year: int,
) -> dict[str, object]:
    os.environ.update(
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    marker = Path(state_root) / f"{tile_name(tile)}.success"
    if marker.exists():
        return {"tile": tile_name(tile), "skipped": True}
    spatial = {
        "lat": slice(tile["lat_start"], tile["lat_stop"]),
        "lon": slice(tile["lon_start"], tile["lon_stop"]),
    }
    historical = xr.open_zarr(historical_path, consolidated=False, chunks=None)[
        "fwi"
    ].isel(**spatial)
    future = xr.open_zarr(future_path, consolidated=False, chunks=None)["fwi"].isel(
        **spatial
    )
    historical_values = historical.values.astype("float32")
    future_values = future.values.astype("float32")
    historical_years = historical.time.dt.year.values
    reference_mask = (historical_years >= reference_start_year) & (
        historical_years <= reference_end_year
    )
    reference_values = historical_values[reference_mask]
    with np.errstate(all="ignore"):
        q95 = np.nanquantile(reference_values, 0.95, axis=0).astype("float32")
        minimum = np.nanmin(reference_values, axis=0)
        maximum = np.nanmax(reference_values, axis=0)
    midrange = ((minimum + maximum) / 2).astype("float32")

    historical_annual = _annual_values(
        historical_values,
        historical_years,
        q95,
        midrange,
    )
    future_annual = _annual_values(
        future_values,
        future.time.dt.year.values,
        q95,
        midrange,
    )
    annual_group = zarr.open_group(annual_output, mode="r+")
    region = (slice(None), spatial["lat"], spatial["lon"])
    for name, metadata in INDICATORS.items():
        values = np.concatenate([historical_annual[name], future_annual[name]], axis=0)
        annual_group[name][region] = _pack(
            values, metadata["scale"], metadata["offset"]
        )
    threshold_group = zarr.open_group(threshold_output, mode="r+")
    threshold_region = (spatial["lat"], spatial["lon"])
    threshold_group["fwi_q95_reference"][threshold_region] = _pack(q95, 0.04, 1000.0)
    threshold_group["fwi_midrange_reference"][threshold_region] = _pack(
        midrange, 0.04, 1000.0
    )
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    return {"tile": tile_name(tile), "skipped": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("historical_daily", type=Path)
    parser.add_argument("future_daily", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--reference-start-year", type=int, default=1995)
    parser.add_argument("--reference-end-year", type=int, default=2014)
    parser.add_argument("--tile-size", type=int, default=40)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    historical = xr.open_zarr(args.historical_daily, consolidated=False, chunks=None)
    future = xr.open_zarr(args.future_daily, consolidated=False, chunks=None)
    for label, dataset in (("historical", historical), ("future", future)):
        if "fwi" not in dataset or dataset.fwi.dims != ("time", "lat", "lon"):
            parser.error(f"{label} daily store has no compatible fwi variable")
    if not np.array_equal(historical.lat.values, future.lat.values) or not np.array_equal(
        historical.lon.values, future.lon.values
    ):
        parser.error("historical and future grids do not match")

    if args.reference_start_year > args.reference_end_year:
        parser.error("reference start year must not exceed reference end year")
    historical_years = historical.time.dt.year.values
    if (
        args.reference_start_year < historical_years.min()
        or args.reference_end_year > historical_years.max()
    ):
        parser.error("reference period must lie within the historical daily store")
    reference_period = f"{args.reference_start_year}-{args.reference_end_year}"
    annual_output = args.output_root / "annual_fwi_indicators_1989_2095.zarr"
    threshold_output = (
        args.output_root
        / f"fwi_reference_thresholds_{args.reference_start_year}_{args.reference_end_year}.zarr"
    )
    state = args.output_root / "state" / "annual_fwi_indicators_1989_2095"
    if args.overwrite:
        shutil.rmtree(annual_output, ignore_errors=True)
        shutil.rmtree(threshold_output, ignore_errors=True)
        shutil.rmtree(state, ignore_errors=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    initialize_outputs(
        historical,
        future,
        annual_output,
        threshold_output,
        args.tile_size,
        reference_period,
    )
    tiles = tile_specs(historical.sizes["lat"], historical.sizes["lon"], args.tile_size)
    pending = [tile for tile in tiles if not (state / f"{tile_name(tile)}.success").exists()]
    print(f"START annual FWI indicators: {len(pending)}/{len(tiles)} tiles", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                run_tile,
                str(args.historical_daily),
                str(args.future_daily),
                str(annual_output),
                str(threshold_output),
                str(state),
                tile,
                args.reference_start_year,
                args.reference_end_year,
            )
            for tile in pending
        ]
        for index, future_result in enumerate(as_completed(futures), 1):
            record = future_result.result()
            if index % 25 == 0 or index == len(pending):
                print(f"DONE {index}/{len(pending)}; {record['tile']}", flush=True)

    complete = all((state / f"{tile_name(tile)}.success").exists() for tile in tiles)
    manifest = {
        "historical_daily": str(args.historical_daily),
        "future_daily": str(args.future_daily),
        "annual_output": str(annual_output),
        "threshold_output": str(threshold_output),
        "reference_period": reference_period,
        "variables": list(INDICATORS),
        "tile_size": args.tile_size,
        "workers": args.workers,
        "completed_tiles": len(tiles) if complete else None,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "valid": complete,
    }
    manifest_path = args.output_root / "annual_fwi_indicators_1989_2095.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if not complete:
        raise SystemExit(1)
    print(annual_output)
    print(threshold_output)
    print(manifest_path)


if __name__ == "__main__":
    main()
