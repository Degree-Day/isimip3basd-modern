#!/usr/bin/env python3
"""Compare regional and global-context daily rainfall downscaling."""

from __future__ import annotations

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def open_day(root: Path, date: str) -> xr.DataArray:
    pieces = []
    for region in ("west", "east"):
        data = xr.open_zarr(
            root / region / "pr_downscaled.zarr", consolidated=False
        )["pr"].sel(time=date).squeeze("time", drop=True)
        pieces.append(
            data.assign_coords(lon=xr.where(data.lon > 180, data.lon - 360, data.lon))
        )
    return (xr.concat(pieces, "lon").sortby("lon") * 86_400).compute()


def decorate(axes, title: str) -> None:
    axes.set_extent((-11, 32, 35, 72), crs=ccrs.PlateCarree())
    axes.add_feature(cfeature.OCEAN, facecolor="#edf3f5")
    axes.add_feature(cfeature.LAND, facecolor="#f5f4ef")
    axes.add_feature(cfeature.BORDERS, linewidth=0.35, edgecolor="#555555")
    axes.coastlines(resolution="50m", linewidth=0.55, color="#303030")
    axes.axvline(0, color="#d62728", linewidth=1.0, linestyle="--")
    axes.set_title(title)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--date", default="2046-04-03")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    old = open_day(args.old_root, args.date)
    new = open_day(args.new_root, args.date)
    difference = new - old
    levels = [0.1, 1, 2.5, 5, 10, 20, 40, 80]
    rain_cmap = plt.get_cmap("Blues", len(levels) - 1).copy()
    rain_cmap.set_under("white")
    rain_norm = colors.BoundaryNorm(levels, rain_cmap.N)
    difference_limit = max(1.0, float(abs(difference).quantile(0.995)))

    projection = ccrs.PlateCarree()
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(17, 6.3),
        constrained_layout=True,
        subplot_kw={"projection": projection},
    )
    for axis, data, title in zip(
        axes[:2],
        (old, new),
        ("Independent regional context", "Shared global context"),
        strict=True,
    ):
        decorate(axis, title)
        rain_image = axis.pcolormesh(
            data.lon,
            data.lat,
            data,
            cmap=rain_cmap,
            norm=rain_norm,
            shading="auto",
            transform=projection,
        )
    decorate(axes[2], "Global context minus regional")
    difference_image = axes[2].pcolormesh(
        difference.lon,
        difference.lat,
        difference,
        cmap="RdBu",
        vmin=-difference_limit,
        vmax=difference_limit,
        shading="auto",
        transform=projection,
    )
    rain_bar = figure.colorbar(
        rain_image, ax=axes[:2], orientation="horizontal", shrink=0.75, pad=0.03
    )
    rain_bar.set_label("Rainfall (mm day$^{-1}$)")
    difference_bar = figure.colorbar(
        difference_image,
        ax=axes[2],
        orientation="horizontal",
        shrink=0.75,
        pad=0.03,
        extend="both",
    )
    difference_bar.set_label("Change (mm day$^{-1}$)")
    figure.suptitle(f"Daily rainfall context test: {args.date}", fontsize=15)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(args.output)
    print(f"maximum absolute change: {float(abs(difference).max()):.6f} mm/day")


if __name__ == "__main__":
    main()
