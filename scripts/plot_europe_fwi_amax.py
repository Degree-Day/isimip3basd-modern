#!/usr/bin/env python3
"""Plot historical, future, and change in mean annual maximum Europe FWI."""

from __future__ import annotations

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import TwoSlopeNorm


def _open(root: Path, period: str) -> xr.DataArray:
    path = root / f"mean_annual_max_fwi_{period}_europe.zarr"
    return xr.open_zarr(path, consolidated=True)["mean_annual_max_fwi"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fwi-root",
        type=Path,
        default=Path("/data1/access_europe_downscale_global_context/fwi"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    historical = _open(args.fwi_root, "1995-2014")
    future = _open(args.fwi_root, "2070-2090")
    historical, future = xr.align(historical, future, join="exact")
    change = future - historical

    projection = ccrs.LambertConformal(central_longitude=10, central_latitude=52)
    data_crs = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(20, 8.5),
        subplot_kw={"projection": projection},
    )
    fig.subplots_adjust(left=0.02, right=0.99, bottom=0.12, top=0.90, wspace=0.08)
    panels = (
        (historical, "1995-2014", "YlOrRd", 0, 120, None),
        (future, "2070-2090", "YlOrRd", 0, 120, None),
        (change, "Change", "RdBu_r", None, None, TwoSlopeNorm(vmin=-30, vcenter=0, vmax=30)),
    )

    for ax, (data, title, cmap, vmin, vmax, norm) in zip(axes, panels, strict=True):
        ax.set_extent((-11, 32, 34, 72), crs=data_crs)
        ax.add_feature(cfeature.OCEAN, facecolor="#e8f0f3", zorder=0)
        ax.add_feature(cfeature.LAND, facecolor="#f7f7f2", zorder=0)
        mesh = ax.pcolormesh(
            data.lon,
            data.lat,
            data,
            transform=data_crs,
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            norm=norm,
        )
        ax.add_feature(cfeature.BORDERS, linewidth=0.45, edgecolor="#555555")
        ax.coastlines(linewidth=0.65, color="#444444")
        ax.gridlines(linewidth=0.35, color="#888888", alpha=0.35)
        ax.set_title(title, fontsize=17, pad=10)
        label = "FWI change" if title == "Change" else "Mean annual max FWI"
        fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.04, shrink=0.98, label=label)

    fig.suptitle("Europe Mean Annual Maximum Fire Weather Index", fontsize=21, y=0.975)
    fig.text(
        0.5,
        0.025,
        "xclim CFFWIS, WF93 season method; ACCESS-CM2 downscaled historical and SSP2-4.5",
        ha="center",
        fontsize=11,
    )
    output = args.output or args.fwi_root / "mean_annual_max_fwi_1995-2014_2070-2090_cartopy.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    finite_change = change.values[np.isfinite(change.values)]
    print(f"Wrote {output}")
    print(
        "Change summary: "
        f"mean={finite_change.mean():.3f}, "
        f"p05={np.quantile(finite_change, 0.05):.3f}, "
        f"p95={np.quantile(finite_change, 0.95):.3f}"
    )


if __name__ == "__main__":
    main()
