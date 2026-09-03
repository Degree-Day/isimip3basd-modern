#!/usr/bin/env python3
"""Calculate and plot global xclim/ETCCDI-style QC climatologies."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy.util import add_cyclic_point
from xclim import indices as xci
from xclim import set_options


_TAS: xr.DataArray | None = None
_PR: xr.DataArray | None = None


def _initialize_worker(root: str, start_year: int, end_year: int) -> None:
    global _TAS, _PR
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    period = slice(str(start_year), str(end_year))
    _TAS = xr.open_zarr(
        f"{root}/tas.zarr", consolidated=False, chunks=None
    )["tas"].sel(time=period)
    _PR = xr.open_zarr(
        f"{root}/pr.zarr", consolidated=False, chunks=None
    )["pr"].sel(time=period)


def _calculate_tile(tile: tuple[int, int, int, int]) -> tuple:
    if _TAS is None or _PR is None:
        raise RuntimeError("worker was not initialized")
    lat0, lat1, lon0, lon1 = tile
    selection = {"lat": slice(lat0, lat1), "lon": slice(lon0, lon1)}
    tas = _TAS.isel(selection).load()
    pr = _PR.isel(selection).load()
    valid = tas.notnull().any("time") & pr.notnull().any("time")

    with set_options(check_missing="skip"):
        tg = xci.tg_mean(tas, freq="YS").mean("time") - 273.15
        prcptot = xci.prcptot(pr, thresh="1 mm/day", freq="YS").mean("time")
        rx1day = xci.max_1day_precipitation_amount(pr, freq="YS").mean("time")
        cdd = xci.maximum_consecutive_dry_days(
            pr, thresh="1 mm/day", freq="YS"
        ).mean("time")

    values = (
        tg.where(valid).values.astype("float32"),
        prcptot.where(valid).values.astype("float32"),
        (rx1day * 86400).where(valid).values.astype("float32"),
        cdd.where(valid).values.astype("float32"),
    )
    return tile, values


def _open_result_arrays(work_dir: Path, shape: tuple[int, int]) -> dict[str, np.memmap]:
    work_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for name in ("tg_mean", "prcptot", "rx1day", "cdd"):
        path = work_dir / f"{name}.npy"
        mode = "r+" if path.exists() else "w+"
        arrays[name] = np.lib.format.open_memmap(
            path, mode=mode, dtype="float32", shape=shape
        )
        if mode == "w+":
            arrays[name][:] = np.nan
    return arrays


def _boundary_ratio(values: np.ndarray, axis: int, stride: int = 10) -> float:
    gradients = np.abs(np.diff(values, axis=axis))
    finite = np.isfinite(gradients)
    boundary = np.zeros(gradients.shape[axis], dtype=bool)
    boundary[stride - 1 :: stride] = True
    on = np.compress(boundary, gradients, axis=axis)
    off = np.compress(~boundary, gradients, axis=axis)
    on = on[np.isfinite(on)]
    off = off[np.isfinite(off)]
    if not on.size or not off.size:
        return float("nan")
    denominator = float(np.median(off))
    return float(np.median(on) / denominator) if denominator else float("nan")


def _summarize(arrays: dict[str, np.ndarray]) -> dict:
    report = {}
    for name, values in arrays.items():
        finite = values[np.isfinite(values)]
        report[name] = {
            "valid_cells": int(finite.size),
            "minimum": float(np.min(finite)),
            "p01": float(np.percentile(finite, 1)),
            "median": float(np.median(finite)),
            "p99": float(np.percentile(finite, 99)),
            "maximum": float(np.max(finite)),
            "one_degree_longitude_boundary_gradient_ratio": _boundary_ratio(values, axis=1),
            "one_degree_latitude_boundary_gradient_ratio": _boundary_ratio(values, axis=0),
        }
    return report


def _plot(
    arrays: dict[str, np.ndarray],
    lat: np.ndarray,
    lon: np.ndarray,
    output: Path,
    title: str,
) -> None:
    panels = (
        ("tg_mean", "Annual mean temperature", "°C", "RdYlBu_r"),
        ("prcptot", "Wet-day precipitation total (PRCPTOT)", "mm/year", "YlGnBu"),
        ("rx1day", "Maximum 1-day precipitation (Rx1day)", "mm/day", "PuBu"),
        ("cdd", "Maximum consecutive dry days (CDD)", "days", "YlOrBr"),
    )
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16, 8.8),
        subplot_kw={"projection": ccrs.Robinson()},
        constrained_layout=True,
    )
    for axis, (name, label, units, cmap) in zip(axes.flat, panels, strict=True):
        values, cyclic_lon = add_cyclic_point(np.asarray(arrays[name]), coord=lon)
        finite = np.asarray(values[np.isfinite(values)])
        vmin, vmax = np.percentile(finite, [1, 99])
        mesh = axis.pcolormesh(
            cyclic_lon,
            lat,
            values,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="auto",
            rasterized=True,
        )
        axis.add_feature(cfeature.LAND, facecolor="none", edgecolor="0.25", linewidth=0.25)
        axis.coastlines(linewidth=0.35)
        axis.set_global()
        axis.set_title(label, fontsize=11)
        colorbar = fig.colorbar(mesh, ax=axis, orientation="horizontal", pad=0.025, shrink=0.88)
        colorbar.set_label(units)
    fig.suptitle(title, fontsize=15)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_zoom(
    arrays: dict[str, np.ndarray],
    lat: np.ndarray,
    lon: np.ndarray,
    output: Path,
    extent: tuple[float, float, float, float],
    title: str,
) -> None:
    panels = (
        ("tg_mean", "Annual mean temperature", "°C", "RdYlBu_r"),
        ("prcptot", "Wet-day total (PRCPTOT)", "mm/year", "YlGnBu"),
        ("rx1day", "Maximum 1-day precipitation (Rx1day)", "mm/day", "PuBu"),
        ("cdd", "Maximum consecutive dry days (CDD)", "days", "YlOrBr"),
    )
    west, east, south, north = extent
    lon_for_selection = np.mod(lon + 180, 360) - 180
    lat_selection = (lat >= south) & (lat <= north)
    lon_selection = (lon_for_selection >= west) & (lon_for_selection <= east)
    selected_lon = lon_for_selection[lon_selection]
    order = np.argsort(selected_lon)
    selected_lon = selected_lon[order]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 10),
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True,
    )
    for axis, (name, label, units, cmap) in zip(axes.flat, panels, strict=True):
        values = np.asarray(arrays[name])[np.ix_(lat_selection, lon_selection)][:, order]
        finite = values[np.isfinite(values)]
        vmin, vmax = np.percentile(finite, [1, 99])
        mesh = axis.pcolormesh(
            selected_lon,
            lat[lat_selection],
            values,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="auto",
            rasterized=True,
        )
        axis.coastlines(resolution="50m", linewidth=0.6)
        axis.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.35)
        gridlines = axis.gridlines(
            xlocs=np.arange(np.floor(west), np.ceil(east) + 1),
            ylocs=np.arange(np.floor(south), np.ceil(north) + 1),
            linewidth=0.18,
            color="0.15",
            alpha=0.25,
        )
        gridlines.xlines = True
        gridlines.ylines = True
        axis.set_extent(extent, crs=ccrs.PlateCarree())
        axis.set_title(label, fontsize=11)
        colorbar = fig.colorbar(mesh, ax=axis, orientation="horizontal", pad=0.035, shrink=0.88)
        colorbar.set_label(units)
    fig.suptitle(title, fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-year", type=int, default=2070)
    parser.add_argument("--end-year", type=int, default=2090)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--tile-lon", type=int, default=200)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--zoom-output", type=Path)
    parser.add_argument(
        "--zoom-extent",
        type=float,
        nargs=4,
        metavar=("WEST", "EAST", "SOUTH", "NORTH"),
        default=(-12, 35, 34, 72),
    )
    args = parser.parse_args()

    tas = xr.open_zarr(args.root / "tas.zarr", consolidated=False, chunks=None)["tas"]
    lat = tas.lat.values
    lon = tas.lon.values
    shape = (lat.size, lon.size)
    work_dir = args.work_dir or args.output.with_suffix(".work")
    arrays = _open_result_arrays(work_dir, shape)
    done_path = work_dir / "completed.npy"
    lat_step = 10
    lon_step = args.tile_lon
    tile_shape = (
        (shape[0] + lat_step - 1) // lat_step,
        (shape[1] + lon_step - 1) // lon_step,
    )
    if done_path.exists():
        completed = np.load(done_path)
    else:
        completed = np.zeros(tile_shape, dtype=bool)

    tiles = []
    for ilat, lat0 in enumerate(range(0, shape[0], lat_step)):
        for ilon, lon0 in enumerate(range(0, shape[1], lon_step)):
            if not completed[ilat, ilon]:
                tiles.append((lat0, min(lat0 + lat_step, shape[0]), lon0, min(lon0 + lon_step, shape[1])))

    root = str(args.root)
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_initialize_worker,
        initargs=(root, args.start_year, args.end_year),
    ) as pool:
        futures = {pool.submit(_calculate_tile, tile): tile for tile in tiles}
        for count, future in enumerate(as_completed(futures), 1):
            tile, values = future.result()
            lat0, lat1, lon0, lon1 = tile
            for name, value in zip(arrays, values, strict=True):
                arrays[name][lat0:lat1, lon0:lon1] = value
            completed[lat0 // lat_step, lon0 // lon_step] = True
            if count % 25 == 0 or count == len(tiles):
                for array in arrays.values():
                    array.flush()
                np.save(done_path, completed)
                print(f"completed {completed.sum()}/{completed.size} tiles", flush=True)

    report = {
        "source": str(args.root),
        "period": f"{args.start_year}-{args.end_year}",
        "resolution_degrees": 0.1,
        "methods": {
            "tg_mean": "xclim.indices.tg_mean; annual values averaged across years",
            "prcptot": "xclim.indices.prcptot at 1 mm/day; annual values averaged across years",
            "rx1day": "xclim.indices.max_1day_precipitation_amount; annual values averaged across years",
            "cdd": "xclim.indices.maximum_consecutive_dry_days at 1 mm/day; annual values averaged across years",
        },
        "statistics": _summarize(arrays),
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    _plot(
        arrays,
        lat,
        lon,
        args.output,
        f"ACCESS-CM2 SSP2-4.5 downscaled climate indicators, {args.start_year}–{args.end_year}\n"
        "xclim ETCCDI-style annual climatologies • 0.1° MBCnSD land grid",
    )
    if args.zoom_output:
        _plot_zoom(
            arrays,
            lat,
            lon,
            args.zoom_output,
            tuple(args.zoom_extent),
            f"ACCESS-CM2 SSP2-4.5 climate-indicator QC, {args.start_year}–{args.end_year}\n"
            "Europe at 0.1° • thin lines show inherited 1° grid boundaries",
        )
    print(args.output)
    if args.zoom_output:
        print(args.zoom_output)
    print(report_path)


if __name__ == "__main__":
    main()
