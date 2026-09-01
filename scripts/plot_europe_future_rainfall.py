#!/usr/bin/env python3
"""Plot future European annual rainfall and a reproducible random rainy day."""

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


def open_europe(data_root: Path) -> xr.DataArray:
    pieces = []
    for region in ("west", "east"):
        data = xr.open_zarr(
            data_root / region / "pr_downscaled.zarr", consolidated=False
        )["pr"]
        longitude = xr.where(data.lon > 180, data.lon - 360, data.lon)
        pieces.append(data.assign_coords(lon=longitude))
    result = xr.concat(pieces, dim="lon").sortby("lon")
    result.name = "pr"
    return result


def map_axes(title: str):
    projection = ccrs.LambertConformal(
        central_longitude=10,
        central_latitude=52,
        standard_parallels=(35, 65),
    )
    figure = plt.figure(figsize=(10.8, 8.2), constrained_layout=True)
    axes = figure.add_subplot(1, 1, 1, projection=projection)
    axes.set_extent((-11, 32, 35, 72), crs=ccrs.PlateCarree())
    axes.add_feature(cfeature.LAND, facecolor="#f1f1ed", zorder=0)
    axes.add_feature(cfeature.OCEAN, facecolor="#e8f1f5", zorder=0)
    axes.add_feature(cfeature.BORDERS, linewidth=0.45, edgecolor="#555555")
    axes.coastlines(resolution="50m", linewidth=0.65, color="#303030")
    axes.set_title(title, fontsize=15, pad=12)
    return figure, axes


def plot_annual(data: xr.DataArray, output: Path, start: int, end: int) -> None:
    finite = np.asarray(data.values)[np.isfinite(data.values)]
    upper = float(np.nanpercentile(finite, 99))
    upper = max(500.0, 100.0 * np.ceil(upper / 100.0))
    figure, axes = map_axes(
        f"Mean annual total rainfall, {start}-{end}\nACCESS-CM2 SSP2-4.5, 0.1°"
    )
    image = axes.pcolormesh(
        data.lon,
        data.lat,
        data,
        cmap="YlGnBu",
        vmin=0,
        vmax=upper,
        shading="auto",
        transform=ccrs.PlateCarree(),
    )
    colorbar = figure.colorbar(
        image,
        ax=axes,
        orientation="horizontal",
        pad=0.045,
        shrink=0.78,
        extend="max",
    )
    colorbar.set_label("Rainfall (mm year$^{-1}$)")
    figure.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(figure)


def plot_day(data: xr.DataArray, output: Path, date: str) -> None:
    levels = [0.1, 1, 2.5, 5, 10, 20, 40, 80]
    cmap = plt.get_cmap("Blues", len(levels) - 1).copy()
    cmap.set_under("white", alpha=0)
    norm = colors.BoundaryNorm(levels, cmap.N)
    figure, axes = map_axes(
        f"Daily rainfall on {date}\nRandom rainy day, ACCESS-CM2 SSP2-4.5"
    )
    image = axes.pcolormesh(
        data.lon,
        data.lat,
        data,
        cmap=cmap,
        norm=norm,
        shading="auto",
        transform=ccrs.PlateCarree(),
    )
    colorbar = figure.colorbar(
        image,
        ax=axes,
        orientation="horizontal",
        pad=0.045,
        shrink=0.82,
        extend="max",
        ticks=levels,
    )
    colorbar.set_label("Rainfall (mm day$^{-1}$)")
    figure.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/data1/access_europe_downscale_full"),
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--start-year", type=int, default=2070)
    parser.add_argument("--end-year", type=int, default=2090)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    output_root = args.output_root or args.data_root / "plots"
    output_root.mkdir(parents=True, exist_ok=True)
    period = open_europe(args.data_root).sel(
        time=slice(str(args.start_year), str(args.end_year))
    )
    expected_days = (args.end_year - args.start_year + 1) * 365
    if period.sizes["time"] != expected_days:
        raise ValueError(
            f"expected {expected_days} noleap days, found {period.sizes['time']}"
        )

    rainfall = period * SECONDS_PER_DAY
    rainfall.attrs["units"] = "mm d-1"
    annual_mean = rainfall.sum("time", skipna=True, min_count=1) / (
        args.end_year - args.start_year + 1
    )
    valid_land = period.isel(time=0).notnull()
    wet_fraction = ((rainfall >= 1) & valid_land).sum(("lat", "lon")) / valid_land.sum()
    with dask.config.set(scheduler="threads", num_workers=args.workers):
        annual_mean, wet_fraction = dask.compute(annual_mean, wet_fraction)

    eligible = np.flatnonzero(np.asarray(wet_fraction.values) >= 0.05)
    if not eligible.size:
        eligible = np.flatnonzero(np.asarray(wet_fraction.values) > 0)
    if not eligible.size:
        raise RuntimeError("the selected future period contains no rainy days")
    rng = np.random.default_rng(args.random_seed)
    day_index = int(rng.choice(eligible))
    with dask.config.set(scheduler="threads", num_workers=args.workers):
        daily = rainfall.isel(time=day_index).compute()
    date = str(period.time.values[day_index])[:10]

    annual_path = output_root / (
        f"mean_annual_total_rainfall_{args.start_year}-{args.end_year}_cartopy.png"
    )
    daily_path = output_root / f"random_rainy_day_{date}_cartopy.png"
    plot_annual(annual_mean, annual_path, args.start_year, args.end_year)
    plot_day(daily, daily_path, date)

    finite_annual = annual_mean.values[np.isfinite(annual_mean.values)]
    finite_daily = daily.values[np.isfinite(daily.values)]
    summary = {
        "period": f"{args.start_year}-{args.end_year}",
        "annual_map": str(annual_path),
        "annual_rainfall_mm": {
            "minimum": float(np.min(finite_annual)),
            "mean": float(np.mean(finite_annual)),
            "maximum": float(np.max(finite_annual)),
        },
        "random_seed": args.random_seed,
        "random_day": date,
        "random_day_wet_land_fraction_ge_1mm": float(wet_fraction.values[day_index]),
        "random_day_map": str(daily_path),
        "random_day_rainfall_mm": {
            "mean": float(np.mean(finite_daily)),
            "maximum": float(np.max(finite_daily)),
        },
    }
    summary_path = output_root / "future_rainfall_plot_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
