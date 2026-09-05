#!/usr/bin/env python3
"""Plot global historical, future, and change maps for annual FWI indicators."""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


INDICATORS = (
    ("fwixx", "Annual maximum FWI (fwixx)", "FWI", "inferno"),
    ("fwixd", "Extreme-fire-weather days (fwixd)", "days", "YlOrRd"),
    ("fwils", "Fire-season length (fwils)", "days", "YlGnBu"),
    ("fwisa", "Maximum 90-day mean FWI (fwisa)", "FWI", "magma"),
)

ROW_LABELS = {
    "fwixx": "FWIXX\nAnnual maximum",
    "fwixd": "FWIXD\nExtreme days",
    "fwils": "FWILS\nSeason length",
    "fwisa": "FWISA\n90-day mean",
}


def period_mean(data: xr.DataArray, start: int, end: int) -> np.ndarray:
    years = data.time.dt.year
    selected = data.where((years >= start) & (years <= end), drop=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return selected.mean("time", skipna=True).values.astype("float32")


def robust_limits(reference: np.ndarray, future: np.ndarray) -> tuple[float, float]:
    finite = np.concatenate(
        [reference[np.isfinite(reference)], future[np.isfinite(future)]]
    )
    low, high = np.percentile(finite, [1, 99])
    return max(0.0, float(low)), float(high)


def missing_summary(reference: np.ndarray, future: np.ndarray) -> dict[str, int]:
    ref = np.isfinite(reference)
    fut = np.isfinite(future)
    return {
        "reference_valid": int(ref.sum()),
        "future_valid": int(fut.sum()),
        "lost_in_future": int((ref & ~fut).sum()),
        "gained_in_future": int((~ref & fut).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-start", type=int, default=1995)
    parser.add_argument("--reference-end", type=int, default=2014)
    parser.add_argument("--future-start", type=int, default=2076)
    parser.add_argument("--future-end", type=int, default=2095)
    args = parser.parse_args()

    dataset = xr.open_zarr(args.input, consolidated=False, chunks=None)
    missing = {item[0] for item in INDICATORS} - set(dataset.data_vars)
    if missing:
        raise ValueError(f"input is missing variables: {sorted(missing)}")
    lon = dataset.lon.values
    lat = dataset.lat.values
    extent = [
        float(lon.min() - 0.05),
        float(lon.max() + 0.05),
        float(lat.min() - 0.05),
        float(lat.max() + 0.05),
    ]

    projection = ccrs.Robinson()
    source_crs = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        4,
        3,
        figsize=(18, 10.5),
        subplot_kw={"projection": projection},
    )
    summaries = {}
    for row, (variable, title, units, cmap_name) in enumerate(INDICATORS):
        reference = period_mean(
            dataset[variable], args.reference_start, args.reference_end
        )
        future = period_mean(dataset[variable], args.future_start, args.future_end)
        change = future - reference
        summaries[variable] = missing_summary(reference, future)
        vmin, vmax = robust_limits(reference, future)
        finite_change = np.abs(change[np.isfinite(change)])
        change_limit = float(np.percentile(finite_change, 98))
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("#b8bdc2")
        change_cmap = plt.get_cmap("RdBu_r").copy()
        change_cmap.set_bad("#b8bdc2")

        images = []
        for column, values in enumerate((reference, future, change)):
            axis = axes[row, column]
            image = axis.imshow(
                values,
                origin="lower",
                extent=extent,
                transform=source_crs,
                interpolation="nearest",
                cmap=change_cmap if column == 2 else cmap,
                vmin=-change_limit if column == 2 else vmin,
                vmax=change_limit if column == 2 else vmax,
                rasterized=True,
            )
            images.append(image)
            axis.add_feature(cfeature.LAND, facecolor="none", edgecolor="none")
            axis.coastlines(linewidth=0.35, color="#303030")
            axis.set_global()
            if column == 0:
                axis.text(
                    -0.035,
                    0.5,
                    ROW_LABELS[variable],
                    transform=axis.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=10,
                    weight="bold",
                )
        value_cax = axes[row, 1].inset_axes([0.28, 0.035, 0.44, 0.035])
        change_cax = axes[row, 2].inset_axes([0.28, 0.035, 0.44, 0.035])
        fig.colorbar(
            images[0],
            cax=value_cax,
            orientation="horizontal",
        )
        value_cax.tick_params(labelsize=7, pad=1)
        value_cax.set_xlabel(units, fontsize=7, labelpad=1)
        fig.colorbar(
            images[2],
            cax=change_cax,
            orientation="horizontal",
        )
        change_cax.tick_params(labelsize=7, pad=1)
        change_cax.set_xlabel(f"change ({units})", fontsize=7, labelpad=1)
        note = summaries[variable]
        axes[row, 2].text(
            0.5,
            0.082,
            f"lost: {note['lost_in_future']:,} | gained: {note['gained_in_future']:,}",
            transform=axes[row, 2].transAxes,
            ha="center",
            va="bottom",
            fontsize=7,
            color="#444444",
        )

    headings = (
        f"Reference mean\n{args.reference_start}-{args.reference_end}",
        f"SSP2-4.5 mean\n{args.future_start}-{args.future_end}",
        "Late-century minus reference",
    )
    for axis, heading in zip(axes[0], headings, strict=True):
        axis.set_title(heading, fontsize=12, weight="bold")
    fig.suptitle(
        "ACCESS-CM2 global annual Fire Weather Index indicators",
        fontsize=16,
        y=0.992,
    )
    fig.text(
        0.5,
        0.955,
        "0.1 deg MBCnSD product | gray is outside supported land/coastal domain | robust global color limits",
        ha="center",
        fontsize=10,
        color="#444444",
    )
    fig.subplots_adjust(
        left=0.045,
        right=0.995,
        bottom=0.015,
        top=0.895,
        wspace=0.01,
        hspace=0.055,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(args.output)
    for variable, summary in summaries.items():
        print(variable, summary)


if __name__ == "__main__":
    main()
