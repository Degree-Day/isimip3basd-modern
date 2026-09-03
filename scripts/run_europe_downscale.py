#!/usr/bin/env python3
"""Run ACCESS-CM2 MBCnSD over the full Europe 0.1 degree domain."""

from __future__ import annotations

import argparse
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
from isimip3basd_modern.publication import packing_encoding
from isimip3basd_modern.validation import validate_variable


VARIABLES = ("hurs", "pr", "sfcWind", "tas")
REGIONS = {
    "west": {
        "fine": {"lat": slice(920, 1290), "lon": slice(3490, 3600)},
        "coarse": {"lat": slice(92, 129), "lon": slice(349, 360)},
        "description": "35.05-71.95N, 10.95W-0.05W",
    },
    "east": {
        "fine": {"lat": slice(920, 1290), "lon": slice(0, 320)},
        "coarse": {"lat": slice(92, 129), "lon": slice(0, 32)},
        "description": "35.05-71.95N, 0.05-31.95E",
    },
}


def open_variable(path: Path, variable: str) -> xr.DataArray:
    return xr.open_zarr(path, consolidated=False)[variable]


def success_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.success")


def is_complete(path: Path) -> bool:
    return success_path(path).exists()


def write_zarr_atomic(
    data: xr.DataArray, path: Path, *, packed: bool = False
) -> None:
    partial = path.with_name(f"{path.name}.partial")
    shutil.rmtree(partial, ignore_errors=True)
    data.to_dataset().to_zarr(
        partial,
        mode="w",
        consolidated=False,
        zarr_format=3,
        encoding={data.name: packing_encoding(data.name)} if packed else None,
    )
    shutil.rmtree(path, ignore_errors=True)
    partial.rename(path)
    success_path(path).touch()


def _select_region(data: xr.DataArray, region: str, resolution: str) -> xr.DataArray:
    return data.isel(REGIONS[region][resolution])


def run_region(
    *,
    model: str,
    scenario: str,
    region: str,
    reference_root: Path,
    canonical_root: Path,
    output_root: Path,
    iterations: int,
    quantiles: int,
) -> list[dict[str, object]]:
    output = output_root / region
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for variable in VARIABLES:
        started = time.perf_counter()
        adjusted_path = output / f"{variable}_adjusted.zarr"
        downscaled_path = output / f"{variable}_downscaled.zarr"

        obs_fine = _select_region(
            open_variable(reference_root / "fine" / f"{variable}.zarr", variable),
            region,
            "fine",
        )
        obs_coarse = _select_region(
            open_variable(reference_root / "coarse" / f"{variable}.zarr", variable),
            region,
            "coarse",
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

        if adjusted_path.exists() and not is_complete(adjusted_path):
            shutil.rmtree(adjusted_path)
        if not is_complete(adjusted_path):
            print(f"START {variable} {region} adjustment", flush=True)
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
            print(f"START {variable} {region} MBCnSD", flush=True)
            downscaled = downscale_variable(
                obs_fine,
                adjusted,
                variable=variable,
                iterations=iterations,
                quantiles=quantiles,
                chunks={"lat": 10, "lon": 10},
            )
            write_zarr_atomic(downscaled, downscaled_path, packed=True)
        downscaled = open_variable(downscaled_path, variable)

        report = validate_variable(downscaled, variable, statistical=False)
        conservation = coarse_scale_conservation(downscaled, adjusted)
        record = {
            "model": model,
            "scenario": scenario,
            "region": region,
            "description": REGIONS[region]["description"],
            "variable": variable,
            "valid": report.valid,
            "minimum": report.minimum,
            "maximum": report.maximum,
            "conservation": conservation,
            "elapsed_seconds": time.perf_counter() - started,
        }
        records.append(record)
        print(f"DONE {variable} {region}: {record}", flush=True)
        (output / f"{variable}.report.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n"
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="ACCESS-CM2")
    parser.add_argument("--scenario", default="ssp245")
    parser.add_argument("--regions", nargs="+", choices=REGIONS, default=list(REGIONS))
    parser.add_argument("--reference-root", type=Path, default=Path("/data1/era5ref-europe-full"))
    parser.add_argument("--canonical-root", type=Path, default=Path("/data1/cmip6_fwi_1deg"))
    parser.add_argument("--output-root", type=Path, default=Path("/data1/access_europe_downscale_full"))
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--quantiles", type=int, default=50)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--memory-limit", default="16GB")
    args = parser.parse_args()

    all_records = []
    with dask.config.set({"array.rechunk.method": "tasks"}), LocalCluster(
        n_workers=args.workers,
        threads_per_worker=1,
        memory_limit=args.memory_limit,
        processes=True,
    ) as cluster, Client(cluster):
        for region in args.regions:
            all_records.extend(
                run_region(
                    model=args.model,
                    scenario=args.scenario,
                    region=region,
                    reference_root=args.reference_root,
                    canonical_root=args.canonical_root,
                    output_root=args.output_root,
                    iterations=args.iterations,
                    quantiles=args.quantiles,
                )
            )

    manifest = {
        "model": args.model,
        "scenario": args.scenario,
        "regions": {key: REGIONS[key]["description"] for key in args.regions},
        "records": all_records,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "downscale-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
