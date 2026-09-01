#!/usr/bin/env python3
"""Calculate and save daily fire-weather indices for downscaled Europe."""

from __future__ import annotations

import argparse
import json
import logging
import os
import warnings
from pathlib import Path

import dask
import numpy as np
import xarray as xr
from xclim.indices.fire import cffwis_indices


VARIABLES = ("tas", "hurs", "pr", "sfcWind")
FWI_VARIABLES = ("ffmc", "dmc", "dc", "isi", "bui", "fwi")
HALVES = ("west", "east")


def _open_var(path: Path, var: str, lat_slice: slice | None = None, lon_slice: slice | None = None) -> xr.DataArray:
    ds = xr.open_zarr(path, consolidated=False)
    da = ds[var]
    if lat_slice is not None:
        da = da.sel(lat=lat_slice)
    if lon_slice is not None:
        da = da.sel(lon=lon_slice)
    return da


def _prep_inputs(arrays: dict[str, xr.DataArray], lat_chunk: int, lon_chunk: int) -> dict[str, xr.DataArray]:
    out: dict[str, xr.DataArray] = {}
    for name, da in arrays.items():
        da = da.transpose("time", "lat", "lon")
        da = da.chunk({"time": -1, "lat": lat_chunk, "lon": lon_chunk})
        out[name] = da

    pr = out["pr"]
    units = str(pr.attrs.get("units", "")).replace(" ", "")
    if units in {"kgm-2s-1", "kg/m2/s"}:
        pr = (pr * 86400.0).astype("float32")
        pr.attrs["units"] = "mm/day"
        pr.attrs["long_name"] = "Daily precipitation amount"
    out["pr"] = pr

    for name, units in {
        "tas": "K",
        "hurs": "%",
        "sfcWind": "m s-1",
    }.items():
        out[name].attrs["units"] = units

    return out


def _compute_daily_indices(
    arrays: dict[str, xr.DataArray],
    selection_start: str,
    selection_end: str,
    lat_chunk: int,
    lon_chunk: int,
) -> xr.Dataset:
    inputs = _prep_inputs(arrays, lat_chunk=lat_chunk, lon_chunk=lon_chunk)
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
    daily = xr.Dataset(
        {
            "ffmc": ffmc,
            "dmc": dmc,
            "dc": dc,
            "isi": isi,
            "bui": bui,
            "fwi": fwi,
        }
    ).sel(time=slice(selection_start, selection_end))
    daily.attrs.update(
        {
            "title": "Daily Canadian Forest Fire Weather Index System outputs",
            "xclim_function": "xclim.indices.fire.cffwis_indices",
            "fwi_season_method": "WF93",
            "fwi_overwintering": "true",
            "fwi_dry_start": "GFWED",
            "initial_start_up": "true",
        }
    )
    return daily


def _mean_annual_max(daily_fwi: xr.DataArray) -> xr.Dataset:
    annual_max = daily_fwi.resample(time="YS").max("time")
    mean_annual_max = annual_max.mean("time").rename("mean_annual_max_fwi")
    mean_annual_max.attrs.update(
        {
            "long_name": "Mean annual maximum Fire Weather Index",
            "units": "1",
            "xclim_function": "xclim.indices.fire.cffwis_indices",
            "fwi_season_method": "WF93",
            "fwi_overwintering": "true",
            "fwi_dry_start": "GFWED",
        }
    )
    return xr.Dataset({"mean_annual_max_fwi": mean_annual_max})


def _summarize(da: xr.DataArray) -> dict[str, float | int]:
    finite = np.isfinite(da)
    return {
        "finite_cells": int(finite.sum().compute()),
        "min": float(da.min(skipna=True).compute()),
        "mean": float(da.mean(skipna=True).compute()),
        "p95": float(da.quantile(0.95, skipna=True).compute()),
        "max": float(da.max(skipna=True).compute()),
    }


def _write_dataset(ds: xr.Dataset, path: Path) -> None:
    if path.exists():
        import shutil

        shutil.rmtree(path)
    chunked = ds.chunk(
        {
            "lat": min(50, ds.sizes["lat"]),
            "lon": min(50, ds.sizes["lon"]),
        }
    )
    encoding = {
        "mean_annual_max_fwi": {
            "chunks": (
                min(50, ds.sizes["lat"]),
                min(50, ds.sizes["lon"]),
            )
        }
    }
    chunked.to_zarr(path, mode="w", consolidated=True, encoding=encoding)


def _write_daily_dataset(
    ds: xr.Dataset,
    path: Path,
    lat_chunk: int,
    lon_chunk: int,
) -> None:
    if path.exists():
        import shutil

        shutil.rmtree(path)
    chunks = {
        "time": min(365, ds.sizes["time"]),
        "lat": min(lat_chunk, ds.sizes["lat"]),
        "lon": min(lon_chunk, ds.sizes["lon"]),
    }
    ds.chunk(chunks).to_zarr(path, mode="w", consolidated=True)


def _daily_store_summary(path: Path) -> dict[str, int | str | list[str]]:
    ds = xr.open_zarr(path, consolidated=True)
    return {
        "path": str(path),
        "variables": list(ds.data_vars),
        "time_steps": ds.sizes["time"],
        "lat_cells": ds.sizes["lat"],
        "lon_cells": ds.sizes["lon"],
        "start": str(ds.time.values[0]),
        "end": str(ds.time.values[-1]),
    }


def _combined_period_output(out_root: Path, label: str) -> dict[str, float | int | str]:
    parts: list[xr.DataArray] = []
    for half in HALVES:
        da = xr.open_zarr(out_root / f"mean_annual_max_fwi_{label}_{half}.zarr", consolidated=True)[
            "mean_annual_max_fwi"
        ]
        if half == "west":
            da = da.assign_coords(lon=((da.lon + 180) % 360) - 180).sortby("lon")
        parts.append(da)

    da = xr.concat(parts, dim="lon").sortby("lon").rename("mean_annual_max_fwi")
    ds = da.to_dataset()
    ds.attrs.update({"period": label, "source": "combined west/east Europe", "lon_convention": "-180..180"})
    out_path = out_root / f"mean_annual_max_fwi_{label}_europe.zarr"
    _write_dataset(ds, out_path)
    opened = xr.open_zarr(out_path, consolidated=True)["mean_annual_max_fwi"]
    return {"path": str(out_path), **_summarize(opened)}


def _period_inputs(
    half: str,
    data_root: Path,
    compute_start: str,
    compute_end: str,
) -> dict[str, xr.DataArray]:
    arrays: dict[str, xr.DataArray] = {}
    base = data_root / half
    for var in VARIABLES:
        arrays[var] = _open_var(base / f"{var}_downscaled.zarr", var).sel(
            time=slice(compute_start, compute_end)
        )
    return arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--future-root",
        type=Path,
        default=Path("/data1/access_europe_downscale_global_context"),
    )
    parser.add_argument(
        "--historical-root",
        type=Path,
        default=Path("/data1/access_europe_downscale_historical_global_context"),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("/data1/access_europe_downscale_global_context/fwi"),
    )
    parser.add_argument("--lat-chunk", type=int, default=10)
    parser.add_argument("--lon-chunk", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads-per-worker", type=int, default=3)
    parser.add_argument("--scheduler", choices=("threads", "distributed"), default="distributed")
    args = parser.parse_args()

    os.environ["PYTHONWARNINGS"] = "ignore"
    logging.getLogger("distributed").setLevel(logging.WARNING)
    logging.getLogger("tornado").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="numba")
    warnings.filterwarnings("ignore", category=UserWarning, module="zarr")
    warnings.filterwarnings("ignore", message=".*Compilation requested for previously compiled argument types.*")

    client = None
    if args.scheduler == "distributed":
        try:
            from dask.distributed import Client, LocalCluster

            cluster = LocalCluster(
                n_workers=args.workers,
                threads_per_worker=args.threads_per_worker,
                processes=True,
                dashboard_address=None,
            )
            client = Client(cluster)
            print(
                "Dask distributed scheduler: "
                f"{args.workers} worker processes x {args.threads_per_worker} threads",
                flush=True,
            )
        except Exception as exc:
            print(f"Falling back to threaded Dask scheduler: {exc}", flush=True)
            dask.config.set(scheduler="threads", num_workers=args.workers)
    else:
        dask.config.set(scheduler="threads", num_workers=args.workers)
        print(f"Dask threaded scheduler: {args.workers} workers", flush=True)

    args.out_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, dict[str, float | int | str | list[str]]]] = {}
    periods = {
        "1995-2014": (args.historical_root, "1993-01-01", "2014-12-31", "1995-01-01", "2014-12-31"),
        "2070-2090": (args.future_root, "2068-01-01", "2090-12-31", "2070-01-01", "2090-12-31"),
    }

    for label, (data_root, compute_start, compute_end, selection_start, selection_end) in periods.items():
        summary[label] = {}
        for half in HALVES:
            print(f"Calculating {label} {half}", flush=True)
            arrays = _period_inputs(half, data_root, compute_start, compute_end)
            daily = _compute_daily_indices(
                arrays,
                selection_start,
                selection_end,
                args.lat_chunk,
                args.lon_chunk,
            )
            daily.attrs.update(
                {
                    "period": label,
                    "compute_start": compute_start,
                    "compute_end": compute_end,
                    "selection_start": selection_start,
                    "selection_end": selection_end,
                    "source": str(data_root),
                    "domain_half": half,
                }
            )
            daily_path = args.out_root / f"daily_fire_weather_indices_{label}_{half}.zarr"
            print(f"Writing daily indices to {daily_path}", flush=True)
            _write_daily_dataset(daily, daily_path, args.lat_chunk, args.lon_chunk)

            opened_daily = xr.open_zarr(daily_path, consolidated=True)
            ds = _mean_annual_max(opened_daily["fwi"])
            ds.attrs.update(daily.attrs)
            out_path = args.out_root / f"mean_annual_max_fwi_{label}_{half}.zarr"
            _write_dataset(ds, out_path)
            opened = xr.open_zarr(out_path, consolidated=True)["mean_annual_max_fwi"]
            summary[label][half] = {
                **_daily_store_summary(daily_path),
                "mean_annual_max_path": str(out_path),
                **_summarize(opened),
            }

    combined_summary = {label: _combined_period_output(args.out_root, label) for label in periods}
    combined_summary_path = args.out_root / "mean_annual_max_fwi_combined_summary.json"
    combined_summary_path.write_text(json.dumps(combined_summary, indent=2) + "\n")

    summary_path = args.out_root / "mean_annual_max_fwi_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(json.dumps(combined_summary, indent=2), flush=True)
    if client is not None:
        client.close()


if __name__ == "__main__":
    main()
