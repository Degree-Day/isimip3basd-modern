#!/usr/bin/env python3
"""Plot annual CDD65 and HDD65 trends for climate-diverse global cities."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


@dataclass(frozen=True)
class City:
    name: str
    climate: str
    lat: float
    lon: float


CITIES = (
    City("Singapore", "tropical rainforest", 1.35, 103.82),
    City("Lagos", "tropical savanna", 6.52, 3.38),
    City("Dubai", "hot desert", 25.20, 55.27),
    City("Delhi", "monsoon subtropical", 28.61, 77.21),
    City("Mexico City", "subtropical highland", 19.43, -99.13),
    City("Sao Paulo", "humid subtropical", -23.55, -46.63),
    City("Sydney", "humid subtropical", -33.87, 151.21),
    City("Tokyo", "humid subtropical", 35.68, 139.69),
    City("New York", "humid continental", 40.71, -74.01),
    City("London", "oceanic", 51.51, -0.13),
    City("Moscow", "cold continental", 55.76, 37.62),
    City("Reykjavik", "subpolar oceanic", 64.15, -21.94),
)


def _nearest_valid_cell(
    valid: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    city: City,
) -> tuple[int, int]:
    target_lon = city.lon % 360.0
    ilat = int(np.abs(lat - city.lat).argmin())
    ilon = int(np.abs(((lon - target_lon + 180.0) % 360.0) - 180.0).argmin())
    if valid[ilat, ilon]:
        return ilat, ilon

    radius = 8
    lat_indices = np.arange(max(0, ilat - radius), min(lat.size, ilat + radius + 1))
    lon_indices = np.arange(ilon - radius, ilon + radius + 1) % lon.size
    yy, xx = np.meshgrid(lat_indices, lon_indices, indexing="ij")
    candidates = valid[yy, xx]
    if not candidates.any():
        raise ValueError(f"no valid land cell found near {city.name}")
    dlat = lat[yy] - city.lat
    dlon = ((lon[xx] - target_lon + 180.0) % 360.0) - 180.0
    distance2 = dlat**2 + (dlon * np.cos(np.deg2rad(city.lat))) ** 2
    distance2 = np.where(candidates, distance2, np.inf)
    index = np.unravel_index(np.argmin(distance2), distance2.shape)
    return int(yy[index]), int(xx[index])


def _linear_trend(years: np.ndarray, values: np.ndarray) -> float:
    good = np.isfinite(values)
    if good.sum() < 2:
        return np.nan
    return float(np.polyfit(years[good] - years[good].mean(), values[good], 1)[0] * 10)


def _rolling_mean(years: np.ndarray, values: np.ndarray, width: int = 5) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=float)
    breaks = np.flatnonzero(np.diff(years) > 1) + 1
    for indices in np.split(np.arange(years.size), breaks):
        result[indices] = (
            xr.DataArray(values[indices], dims="time")
            .rolling(time=width, center=True, min_periods=3)
            .mean()
            .values
        )
    return result


def _set_focused_limits(axis: plt.Axes, values: np.ndarray) -> None:
    good = values[np.isfinite(values)]
    low = float(good.min())
    high = float(good.max())
    span = high - low
    if span == 0:
        span = max(abs(high) * 0.1, 1.0)
    padding = 0.08 * span
    axis.set_ylim(max(0.0, low - padding), high + padding)


def _with_year_gaps(
    years: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    complete_years = np.arange(years[0], years[-1] + 1)
    complete_values = np.full(complete_years.shape, np.nan, dtype=float)
    complete_values[years - years[0]] = values
    return complete_years, complete_values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--future-input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path)
    args = parser.parse_args()

    ds = xr.open_zarr(args.input, consolidated=False, chunks=None)
    required = {"cdd65", "hdd65"}
    missing = required - set(ds.data_vars)
    if missing:
        raise ValueError(f"missing required variables: {sorted(missing)}")

    datasets = [("historical", ds)]
    if args.future_input:
        future = xr.open_zarr(args.future_input, consolidated=False, chunks=None)
        if not np.array_equal(ds.lat.values, future.lat.values) or not np.array_equal(
            ds.lon.values, future.lon.values
        ):
            raise ValueError("historical and future grids do not match")
        if required - set(future.data_vars):
            raise ValueError("future input is missing cdd65 or hdd65")
        datasets.append(("ssp245", future))

    years = np.concatenate(
        [part.time.dt.year.values.astype(int) for _, part in datasets]
    )
    lat = ds.lat.values
    lon = ds.lon.values
    valid = np.isfinite(ds.cdd65.isel(time=0).values) & np.isfinite(
        ds.hdd65.isel(time=0).values
    )

    records = []
    series = {}
    for city in CITIES:
        ilat, ilon = _nearest_valid_cell(valid, lat, lon, city)
        cdd = np.concatenate(
            [part.cdd65.isel(lat=ilat, lon=ilon).values for _, part in datasets]
        ).astype(float)
        hdd = np.concatenate(
            [part.hdd65.isel(lat=ilat, lon=ilon).values for _, part in datasets]
        ).astype(float)
        series[city.name] = (city, ilat, ilon, cdd, hdd)
        experiment = np.concatenate(
            [np.repeat(name, part.sizes["time"]) for name, part in datasets]
        )
        for year, cdd_value, hdd_value, segment in zip(
            years, cdd, hdd, experiment, strict=True
        ):
            records.append(
                {
                    "city": city.name,
                    "climate": city.climate,
                    "year": int(year),
                    "experiment": segment,
                    "cdd65_degC_days": float(cdd_value),
                    "hdd65_degC_days": float(hdd_value),
                    "grid_lat": float(lat[ilat]),
                    "grid_lon": float(((lon[ilon] + 180.0) % 360.0) - 180.0),
                }
            )

    csv_output = args.csv_output or args.output.with_suffix(".csv")
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    with csv_output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(4, 3, figsize=(16, 15), sharex=True)
    cdd_color = "#c43c2f"
    hdd_color = "#2676a6"
    for panel_index, (axis, city) in enumerate(zip(axes.flat, CITIES, strict=True)):
        _, ilat, ilon, cdd, hdd = series[city.name]
        hdd_axis = axis.twinx()
        if args.future_input:
            axis.axvspan(2034, years[-1], color="#eeeeee", alpha=0.7, zorder=0)
        plot_years, plot_cdd = _with_year_gaps(years, cdd)
        _, plot_hdd = _with_year_gaps(years, hdd)
        _, smooth_cdd = _with_year_gaps(years, _rolling_mean(years, cdd))
        _, smooth_hdd = _with_year_gaps(years, _rolling_mean(years, hdd))
        axis.plot(plot_years, plot_cdd, color=cdd_color, alpha=0.28, linewidth=1)
        hdd_axis.plot(
            plot_years, plot_hdd, color=hdd_color, alpha=0.28, linewidth=1
        )
        axis.plot(
            plot_years,
            smooth_cdd,
            color=cdd_color,
            linewidth=2.2,
            label="CDD65",
        )
        hdd_axis.plot(
            plot_years,
            smooth_hdd,
            color=hdd_color,
            linewidth=2.2,
            label="HDD65",
        )
        trend_mask = years >= 2034 if args.future_input else np.ones(years.shape, bool)
        cdd_trend = _linear_trend(years[trend_mask], cdd[trend_mask])
        hdd_trend = _linear_trend(years[trend_mask], hdd[trend_mask])
        axis.set_title(f"{city.name} | {city.climate}", fontsize=10, weight="bold")
        axis.text(
            0.02,
            0.96,
            f"Future trend: CDD {cdd_trend:+.0f} / HDD {hdd_trend:+.0f} per decade",
            transform=axis.transAxes,
            va="top",
            fontsize=8.5,
        )
        axis.text(
            0.02,
            0.04,
            f"cell {lat[ilat]:.2f}, {((lon[ilon] + 180.0) % 360.0) - 180.0:.2f}",
            transform=axis.transAxes,
            fontsize=7.5,
            color="#555555",
        )
        axis.set_xlim(years[0], years[-1])
        _set_focused_limits(axis, cdd)
        _set_focused_limits(hdd_axis, hdd)
        axis.tick_params(axis="y", colors=cdd_color, labelsize=8)
        axis.tick_params(axis="x", labelsize=8)
        hdd_axis.tick_params(axis="y", colors=hdd_color, labelsize=8)
        axis.spines["left"].set_color(cdd_color)
        hdd_axis.spines["right"].set_color(hdd_color)
        if panel_index % 3 == 2:
            hdd_axis.set_ylabel("HDD65, degC-days/year", fontsize=9, color=hdd_color)

    for axis in axes[:, 0]:
        axis.set_ylabel("CDD65, degC-days/year", fontsize=9, color=cdd_color)
    for axis in axes[-1, :]:
        axis.set_xlabel("Year", fontsize=9)
    handles = [
        plt.Line2D([], [], color=cdd_color, linewidth=2.2, label="CDD65 (left axis)"),
        plt.Line2D([], [], color=hdd_color, linewidth=2.2, label="HDD65 (right axis)"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.965),
    )
    fig.suptitle(
        "Annual cooling and heating degree-day trends at major global cities\n"
        f"ACCESS-CM2 0.1 deg MBCnSD | base 65 F | {years[0]}-{years[-1]} | shaded area is SSP2-4.5",
        fontsize=16,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(args.output)
    print(csv_output)


if __name__ == "__main__":
    main()
