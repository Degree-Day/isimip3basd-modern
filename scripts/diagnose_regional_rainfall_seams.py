#!/usr/bin/env python3
"""Diagnose daily-rainfall discontinuities between regional output stores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import dask
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


SECONDS_PER_DAY = 86_400.0


def open_rainfall(data_root: Path) -> tuple[xr.DataArray, xr.DataArray]:
    west = xr.open_zarr(
        data_root / "west" / "pr_downscaled.zarr", consolidated=False
    )["pr"]
    east = xr.open_zarr(
        data_root / "east" / "pr_downscaled.zarr", consolidated=False
    )["pr"]
    west = west.assign_coords(lon=xr.where(west.lon > 180, west.lon - 360, west.lon))
    return west * SECONDS_PER_DAY, east * SECONDS_PER_DAY


def seam_diagnostics(
    west: xr.DataArray, east: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    seam = abs(west.isel(lon=-1) - east.isel(lon=0))
    left = abs(west.isel(lon=-2) - west.isel(lon=-1))
    right = abs(east.isel(lon=0) - east.isel(lon=1))
    valid = np.isfinite(seam) & np.isfinite(left) & np.isfinite(right)
    diagnostic_lat = slice(40, 65)
    seam_mean = seam.where(valid).sel(lat=diagnostic_lat).mean("lat")
    neighbor_mean = 0.5 * (
        left.where(valid).sel(lat=diagnostic_lat).mean("lat")
        + right.where(valid).sel(lat=diagnostic_lat).mean("lat")
    )
    seam_rain = 0.5 * (
        west.isel(lon=-1).where(valid).sel(lat=diagnostic_lat).mean("lat")
        + east.isel(lon=0).where(valid).sel(lat=diagnostic_lat).mean("lat")
    )
    ratio = seam_mean / (neighbor_mean + 0.01)
    return seam, seam_mean, neighbor_mean, ratio.where(seam_rain >= 1)


def add_map_features(axes) -> None:
    axes.add_feature(cfeature.OCEAN, facecolor="#edf3f5", zorder=0)
    axes.add_feature(cfeature.LAND, facecolor="#f5f4ef", zorder=0)
    axes.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="#555555")
    axes.coastlines(resolution="50m", linewidth=0.6, color="#303030")


def plot_diagnostic(
    daily: xr.DataArray,
    seam: xr.DataArray,
    neighbor: xr.DataArray,
    *,
    date: str,
    output: Path,
) -> None:
    levels = [0.1, 1, 2.5, 5, 10, 20, 40, 80]
    cmap = plt.get_cmap("Blues", len(levels) - 1).copy()
    cmap.set_under("white")
    norm = colors.BoundaryNorm(levels, cmap.N)
    projection = ccrs.PlateCarree()
    figure = plt.figure(figsize=(14, 8.5), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=(1.35, 1), height_ratios=(1, 1))

    full = figure.add_subplot(grid[:, 0], projection=projection)
    add_map_features(full)
    full.set_extent((-11, 32, 35, 72), crs=projection)
    image = full.pcolormesh(
        daily.lon,
        daily.lat,
        daily,
        cmap=cmap,
        norm=norm,
        shading="auto",
        transform=projection,
    )
    full.axvline(0, color="#d62728", linewidth=1.2, linestyle="--")
    full.set_title(f"Daily rainfall on {date}")

    zoom = figure.add_subplot(grid[0, 1], projection=projection)
    add_map_features(zoom)
    zoom.set_extent((-3, 3, 42, 61), crs=projection)
    zoom.pcolormesh(
        daily.lon,
        daily.lat,
        daily,
        cmap=cmap,
        norm=norm,
        shading="auto",
        transform=projection,
    )
    zoom.axvline(0, color="#d62728", linewidth=1.5, linestyle="--")
    zoom.set_title("0° regional seam")

    profile = figure.add_subplot(grid[1, 1])
    profile.plot(seam, seam.lat, color="#d62728", label="Across regional seam")
    profile.plot(
        neighbor,
        neighbor.lat,
        color="#2c7fb8",
        label="Mean adjacent 0.1° gradient",
    )
    profile.set_xlabel("Absolute rainfall difference (mm day$^{-1}$)")
    profile.set_ylabel("Latitude")
    profile.set_ylim(35, 72)
    profile.grid(color="#dddddd", linewidth=0.6)
    profile.legend(frameon=False, loc="upper right")

    colorbar = figure.colorbar(
        image,
        ax=(full, zoom),
        orientation="horizontal",
        pad=0.04,
        shrink=0.78,
        extend="max",
        ticks=levels,
    )
    colorbar.set_label("Rainfall (mm day$^{-1}$)")
    figure.suptitle(
        "Regional processing diagnostic: ACCESS-CM2 SSP2-4.5, 0.1°",
        fontsize=15,
    )
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/data1/access_europe_downscale_full"),
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--date", default=None)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    output_root = args.output_root or args.data_root / "plots"
    output_root.mkdir(parents=True, exist_ok=True)
    west, east = open_rainfall(args.data_root)
    seam, seam_mean, neighbor_mean, wet_ratio = seam_diagnostics(west, east)
    with dask.config.set(scheduler="threads", num_workers=args.workers):
        if args.date:
            day_index = int(west.get_index("time").get_loc(args.date))
        else:
            day_index = int(wet_ratio.argmax("time").compute())
        seam_day, neighbor_day = dask.compute(
            seam.isel(time=day_index),
            0.5
            * (
                abs(west.isel(time=day_index, lon=-2) - west.isel(time=day_index, lon=-1))
                + abs(east.isel(time=day_index, lon=0) - east.isel(time=day_index, lon=1))
            ),
        )
        daily = xr.concat(
            [west.isel(time=day_index), east.isel(time=day_index)], dim="lon"
        ).sortby("lon").compute()
        selected_seam_mean, selected_neighbor_mean = dask.compute(
            seam_mean.isel(time=day_index), neighbor_mean.isel(time=day_index)
        )

    date = str(west.time.values[day_index])[:10]
    ratio = float(selected_seam_mean / (selected_neighbor_mean + 0.01))
    output = output_root / f"regional_rainfall_seam_{date}_cartopy.png"
    plot_diagnostic(daily, seam_day, neighbor_day, date=date, output=output)
    summary = {
        "date": date,
        "map": str(output),
        "mean_cross_seam_difference_mm_day": float(selected_seam_mean),
        "mean_adjacent_difference_mm_day": float(selected_neighbor_mean),
        "cross_seam_to_adjacent_ratio": ratio,
        "selection_latitude_band": "40-65N",
        "interpretation": (
            "A large ratio indicates a daily discontinuity at this 1-degree "
            "parent-cell boundary. Compare all parent boundaries before "
            "attributing it to regional processing."
        ),
    }
    summary_path = output_root / "regional_rainfall_seam_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
