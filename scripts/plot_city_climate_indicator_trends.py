#!/usr/bin/env python3
"""Plot annual climate-indicator trends for climate-diverse global cities."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from plot_city_degree_day_trends import (
    CITIES,
    _linear_trend,
    _nearest_valid_cell,
    _rolling_mean,
    _set_focused_limits,
    _with_year_gaps,
)


@dataclass(frozen=True)
class Indicator:
    variable: str
    title: str
    units: str
    color: str
    trend_decimals: int


INDICATORS = (
    Indicator("tg_mean", "Annual mean temperature", "degC", "#c43c2f", 2),
    Indicator("prcptot", "Wet-day precipitation total", "mm/year", "#16817a", 0),
    Indicator("rx1day", "Maximum one-day precipitation", "mm/day", "#2676a6", 1),
    Indicator("cdd", "Maximum consecutive dry days", "days", "#9a5b13", 1),
)


def _plot_indicator(
    indicator: Indicator,
    years: np.ndarray,
    city_series: dict[str, tuple[int, int, np.ndarray]],
    lat: np.ndarray,
    lon: np.ndarray,
    output: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(4, 3, figsize=(16, 14), sharex=True)
    future_mask = years >= 2034
    for axis, city in zip(axes.flat, CITIES, strict=True):
        ilat, ilon, values = city_series[city.name]
        plot_years, plot_values = _with_year_gaps(years, values)
        _, smooth = _with_year_gaps(years, _rolling_mean(years, values))
        axis.axvspan(2034, years[-1], color="#eeeeee", alpha=0.7, zorder=0)
        axis.plot(
            plot_years,
            plot_values,
            color=indicator.color,
            alpha=0.28,
            linewidth=1,
        )
        axis.plot(plot_years, smooth, color=indicator.color, linewidth=2.2)
        trend = _linear_trend(years[future_mask], values[future_mask])
        axis.set_title(f"{city.name} | {city.climate}", fontsize=10, weight="bold")
        axis.text(
            0.02,
            0.96,
            f"Future trend: {trend:+.{indicator.trend_decimals}f} {indicator.units}/decade",
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
        _set_focused_limits(axis, values)
        axis.tick_params(labelsize=8, colors=indicator.color)
        axis.spines["left"].set_color(indicator.color)

    for axis in axes[:, 0]:
        axis.set_ylabel(indicator.units, fontsize=9, color=indicator.color)
    for axis in axes[-1, :]:
        axis.set_xlabel("Year", fontsize=9)
    fig.suptitle(
        f"{indicator.title} at major global cities\n"
        "ACCESS-CM2 0.1 deg MBCnSD | 1989-2095 | shaded area is SSP2-4.5 | bold line is 5-year mean",
        fontsize=16,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--future-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path)
    args = parser.parse_args()

    historical = xr.open_zarr(args.input, consolidated=False, chunks=None)
    future = xr.open_zarr(args.future_input, consolidated=False, chunks=None)
    required = {item.variable for item in INDICATORS}
    for name, dataset in (("historical", historical), ("future", future)):
        missing = required - set(dataset.data_vars)
        if missing:
            raise ValueError(f"{name} input is missing variables: {sorted(missing)}")
    if not np.array_equal(historical.lat.values, future.lat.values) or not np.array_equal(
        historical.lon.values, future.lon.values
    ):
        raise ValueError("historical and future grids do not match")

    datasets = (("historical", historical), ("ssp245", future))
    years = np.concatenate(
        [part.time.dt.year.values.astype(int) for _, part in datasets]
    )
    experiment = np.concatenate(
        [np.repeat(name, part.sizes["time"]) for name, part in datasets]
    )
    lat = historical.lat.values
    lon = historical.lon.values
    valid = np.ones((lat.size, lon.size), dtype=bool)
    for indicator in INDICATORS:
        valid &= np.isfinite(historical[indicator.variable].isel(time=0).values)

    records = []
    all_series: dict[str, dict[str, tuple[int, int, np.ndarray]]] = {
        item.variable: {} for item in INDICATORS
    }
    for city in CITIES:
        ilat, ilon = _nearest_valid_cell(valid, lat, lon, city)
        values_by_variable = {}
        for indicator in INDICATORS:
            values = np.concatenate(
                [
                    part[indicator.variable].isel(lat=ilat, lon=ilon).values
                    for _, part in datasets
                ]
            ).astype(float)
            values_by_variable[indicator.variable] = values
            all_series[indicator.variable][city.name] = (ilat, ilon, values)
        for index, year in enumerate(years):
            record = {
                "city": city.name,
                "climate": city.climate,
                "year": int(year),
                "experiment": experiment[index],
                "grid_lat": float(lat[ilat]),
                "grid_lon": float(((lon[ilon] + 180.0) % 360.0) - 180.0),
            }
            for indicator in INDICATORS:
                record[indicator.variable] = float(
                    values_by_variable[indicator.variable][index]
                )
            records.append(record)

    csv_output = args.csv_output or args.output_dir / "city_climate_indicators.csv"
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    with csv_output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    for indicator in INDICATORS:
        output = args.output_dir / f"access_cm2_city_{indicator.variable}_trends_1989_2095.png"
        _plot_indicator(
            indicator,
            years,
            all_series[indicator.variable],
            lat,
            lon,
            output,
        )
        print(output)
    print(csv_output)


if __name__ == "__main__":
    main()
