#!/usr/bin/env python3
"""Plot native ERA5-Land and regular ERA5 fallback coverage."""

from __future__ import annotations

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np
import xarray as xr


VARIABLES = ("tas", "hurs", "pr", "sfcWind")


def common_source_classes(root: Path) -> xr.DataArray:
    sources = [
        xr.open_zarr(root / "source" / f"{variable}.zarr", consolidated=False)[
            "reference_source"
        ].load()
        for variable in VARIABLES
    ]
    template = sources[0]
    for source in sources[1:]:
        xr.align(template, source, join="exact")
    values = np.stack([np.asarray(source) for source in sources])
    land = xr.open_zarr(root / "lulc_land_mask.zarr", consolidated=False)[
        "lulc_land"
    ].load()
    classes = np.zeros(template.shape, dtype="uint8")
    classes[np.all(values == 1, axis=0)] = 1
    complete = np.all(values > 0, axis=0)
    classes[complete & np.any(values == 2, axis=0)] = 2
    classes[np.asarray(land, dtype=bool) & ~complete] = 3
    return xr.DataArray(
        classes,
        dims=template.dims,
        coords=template.coords,
        name="reference_source_class",
    )


def plot_coverage(classes: xr.DataArray, output: Path) -> None:
    shifted = classes.assign_coords(lon=((classes.lon + 180) % 360) - 180).sortby(
        "lon"
    )
    colors = ["#d9edf3", "#238b8d", "#efb366", "#d73027"]
    labels = [
        "Ocean / outside reference support",
        "ERA5-Land (all variables)",
        "ERA5 fallback (one or more variables)",
        "Remaining mapped-land gap",
    ]
    counts = np.bincount(np.asarray(classes).ravel(), minlength=4)
    figure = plt.figure(figsize=(15, 7.2), constrained_layout=True)
    axis = figure.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    axis.set_global()
    axis.set_facecolor(colors[0])
    axis.pcolormesh(
        shifted.lon,
        shifted.lat,
        shifted,
        transform=ccrs.PlateCarree(),
        cmap=ListedColormap(colors),
        norm=BoundaryNorm(np.arange(-0.5, 4.5), 4),
        shading="nearest",
        rasterized=True,
    )
    axis.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.45)
    axis.add_feature(cfeature.BORDERS.with_scale("110m"), linewidth=0.25)
    axis.set_title(
        "Global 0.1 degree training-reference coverage\n"
        "ERA5-Land primary; ERA5 bilinear fallback on LULC-confirmed land",
        fontsize=15,
    )
    legend_labels = [
        f"{label}: {count:,} cells" for label, count in zip(labels, counts, strict=True)
    ]
    axis.legend(
        [Patch(facecolor=color, edgecolor="0.35") for color in colors],
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.11),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    figure.text(
        0.01,
        0.01,
        "Common support across tas, hurs, pr, and sfcWind; training period 1993-2014.",
        fontsize=9,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot_coverage(common_source_classes(args.reference_root), args.output)
    print(args.output)


if __name__ == "__main__":
    main()
