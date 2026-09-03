#!/usr/bin/env python3
"""Calculate annual xclim climate indicators and plot QC climatologies."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy.util import add_cyclic_point
from xclim import indices as xci
from xclim import set_options
from zarr.codecs import BloscCodec


_TAS: xr.DataArray | None = None
_PR: xr.DataArray | None = None

INDICATORS = ("tg_mean", "prcptot", "rx1day", "cdd", "cdd65", "hdd65")
PANELS = (
    ("tg_mean", "Annual mean temperature", "°C", "RdYlBu_r"),
    ("prcptot", "Wet-day precipitation total (PRCPTOT)", "mm/year", "YlGnBu"),
    ("rx1day", "Maximum 1-day precipitation (Rx1day)", "mm/day", "PuBu"),
    ("cdd", "Maximum consecutive dry days (CDD)", "days", "YlOrBr"),
    ("cdd65", "Cooling degree-days, base 65°F (CDD65)", "°C days", "YlOrRd"),
    ("hdd65", "Heating degree-days, base 65°F (HDD65)", "°C days", "PuBuGn"),
)

# Each scale preserves useful precision while reserving int16 code -32768 for missing data.
PACKING = {
    "tg_mean": (0.01, 0.0),
    "prcptot": (1.0, 32767.0),
    "rx1day": (0.1, 3276.7),
    "cdd": (1.0, 0.0),
    "cdd65": (0.25, 8191.75),
    "hdd65": (0.25, 8191.75),
}


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
        tg = xci.tg_mean(tas, freq="YS") - 273.15
        prcptot = xci.prcptot(pr, thresh="1 mm/day", freq="YS")
        rx1day = xci.max_1day_precipitation_amount(pr, freq="YS")
        cdd = xci.maximum_consecutive_dry_days(
            pr, thresh="1 mm/day", freq="YS"
        )
        cdd65 = xci.cooling_degree_days(tas, thresh="65 degF", freq="YS")
        hdd65 = xci.heating_degree_days(tas, thresh="65 degF", freq="YS")

    values = (
        tg.where(valid).values.astype("float32"),
        prcptot.where(valid).values.astype("float32"),
        (rx1day * 86400).where(valid).values.astype("float32"),
        cdd.where(valid).values.astype("float32"),
        cdd65.where(valid).values.astype("float32"),
        hdd65.where(valid).values.astype("float32"),
    )
    return tile, values


def _open_result_arrays(
    work_dir: Path, shape: tuple[int, int, int]
) -> dict[str, np.memmap]:
    work_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for name in INDICATORS:
        path = work_dir / f"{name}.npy"
        mode = "r+" if path.exists() else "w+"
        arrays[name] = np.lib.format.open_memmap(
            path, mode=mode, dtype="float32", shape=shape
        )
        if mode == "w+":
            arrays[name][:] = np.nan
    return arrays


def _climatologies(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return {
            name: np.nanmean(values, axis=0, dtype="float64").astype("float32")
            for name, values in arrays.items()
        }


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


def _write_annual_zarr(
    arrays: dict[str, np.ndarray],
    years: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    output: Path,
    source: Path,
) -> None:
    partial = output.with_name(f"{output.name}.partial")
    shutil.rmtree(partial, ignore_errors=True)
    data_vars = {}
    encoding = {}
    units = {
        "tg_mean": "degC",
        "prcptot": "mm",
        "rx1day": "mm/day",
        "cdd": "d",
        "cdd65": "K d",
        "hdd65": "K d",
    }
    long_names = {name: label for name, label, _, _ in PANELS}
    for name, values in arrays.items():
        scale, offset = PACKING[name]
        finite = values[np.isfinite(values)]
        packed_min = offset - 32767 * scale
        packed_max = offset + 32767 * scale
        if finite.size and (finite.min() < packed_min or finite.max() > packed_max):
            raise ValueError(
                f"{name} range {finite.min()}..{finite.max()} exceeds int16 packing "
                f"range {packed_min}..{packed_max}"
            )
        data_vars[name] = xr.DataArray(
            da.from_array(values, chunks=(1, 100, 200), asarray=False),
            dims=("time", "lat", "lon"),
            attrs={"units": units[name], "long_name": long_names[name]},
        )
        encoding[name] = {
            "dtype": "int16",
            "_FillValue": np.int16(-32768),
            "fill_value": np.int16(-32768),
            "scale_factor": scale,
            "add_offset": offset,
            "compressors": [
                BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")
            ],
        }
    dataset = xr.Dataset(
        data_vars,
        coords={"time": years, "lat": lat, "lon": lon},
        attrs={
            "source": str(source),
            "frequency": "annual",
            "calendar": "noleap",
            "publication_format": "scaled int16 Zarr v3",
            "indicator_library": "xclim",
            "etccdi_note": (
                "PRCPTOT, Rx1day, and consecutive-dry-days CDD are ETCCDI indices; "
                "CDD65 and HDD65 are energy degree-day indicators included alongside them"
            ),
            "degree_day_base": "65 degF (18.333333 degC)",
        },
    )
    dataset.to_zarr(
        partial,
        mode="w",
        consolidated=False,
        zarr_format=3,
        encoding=encoding,
    )
    if output.exists():
        shutil.rmtree(output)
    partial.rename(output)


def _plot(
    arrays: dict[str, np.ndarray],
    lat: np.ndarray,
    lon: np.ndarray,
    output: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(16, 13),
        subplot_kw={"projection": ccrs.Robinson()},
        constrained_layout=True,
    )
    for axis, (name, label, units, cmap) in zip(axes.flat, PANELS, strict=True):
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
    west, east, south, north = extent
    lon_for_selection = np.mod(lon + 180, 360) - 180
    lat_selection = (lat >= south) & (lat <= north)
    lon_selection = (lon_for_selection >= west) & (lon_for_selection <= east)
    selected_lon = lon_for_selection[lon_selection]
    order = np.argsort(selected_lon)
    selected_lon = selected_lon[order]

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(13, 14),
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True,
    )
    for axis, (name, label, units, cmap) in zip(axes.flat, PANELS, strict=True):
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
    parser.add_argument("--annual-output", type=Path)
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
    spatial_shape = (lat.size, lon.size)
    years = xr.date_range(
        f"{args.start_year}-01-01",
        periods=args.end_year - args.start_year + 1,
        freq="YS",
        calendar="noleap",
        use_cftime=True,
    ).values
    work_dir = args.work_dir or args.output.with_name(
        f"{args.output.stem}.annual.work"
    )
    annual_output = args.annual_output or args.output.with_name(
        f"{args.output.stem}.annual.zarr"
    )
    arrays = _open_result_arrays(work_dir, (years.size, *spatial_shape))
    done_path = work_dir / "completed.npy"
    lat_step = 10
    lon_step = args.tile_lon
    tile_shape = (
        (spatial_shape[0] + lat_step - 1) // lat_step,
        (spatial_shape[1] + lon_step - 1) // lon_step,
    )
    if done_path.exists():
        completed = np.load(done_path)
    else:
        completed = np.zeros(tile_shape, dtype=bool)

    tiles = []
    for ilat, lat0 in enumerate(range(0, spatial_shape[0], lat_step)):
        for ilon, lon0 in enumerate(range(0, spatial_shape[1], lon_step)):
            if not completed[ilat, ilon]:
                tiles.append(
                    (
                        lat0,
                        min(lat0 + lat_step, spatial_shape[0]),
                        lon0,
                        min(lon0 + lon_step, spatial_shape[1]),
                    )
                )

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
                arrays[name][:, lat0:lat1, lon0:lon1] = value
            completed[lat0 // lat_step, lon0 // lon_step] = True
            if count % 25 == 0 or count == len(tiles):
                for array in arrays.values():
                    array.flush()
                np.save(done_path, completed)
                print(f"completed {completed.sum()}/{completed.size} tiles", flush=True)

    climatologies = _climatologies(arrays)
    _write_annual_zarr(arrays, years, lat, lon, annual_output, args.root)
    report = {
        "source": str(args.root),
        "annual_output": str(annual_output),
        "period": f"{args.start_year}-{args.end_year}",
        "resolution_degrees": 0.1,
        "methods": {
            "tg_mean": "xclim.indices.tg_mean",
            "prcptot": "xclim.indices.prcptot at 1 mm/day",
            "rx1day": "xclim.indices.max_1day_precipitation_amount",
            "cdd": "xclim.indices.maximum_consecutive_dry_days at 1 mm/day",
            "cdd65": "xclim.indices.cooling_degree_days at 65 degF",
            "hdd65": "xclim.indices.heating_degree_days at 65 degF",
        },
        "statistics": _summarize(climatologies),
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    _plot(
        climatologies,
        lat,
        lon,
        args.output,
        f"ACCESS-CM2 SSP2-4.5 downscaled climate indicators, {args.start_year}–{args.end_year}\n"
        "xclim ETCCDI-style annual climatologies • 0.1° MBCnSD land grid",
    )
    if args.zoom_output:
        _plot_zoom(
            climatologies,
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
    print(annual_output)
    print(report_path)


if __name__ == "__main__":
    main()
