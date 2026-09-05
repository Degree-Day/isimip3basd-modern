#!/usr/bin/env python3
"""Plot annual FWI indicators in peri-urban rings around wildfire-prone cities."""

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
    region: str
    lat: float
    lon: float


@dataclass(frozen=True)
class Indicator:
    variable: str
    title: str
    units: str
    color: str
    decimals: int


CITIES = (
    City("Boulder", "Colorado Front Range", 40.0150, -105.2705),
    City("Los Angeles", "Southern California", 34.0522, -118.2437),
    City("Spokane", "Inland Northwest", 47.6588, -117.4260),
    City("Santa Rosa", "Northern California", 38.4405, -122.7144),
    City("Vancouver", "Pacific Northwest", 49.2827, -123.1207),
    City("Sydney", "New South Wales", -33.8688, 151.2093),
    City("Canberra", "Australian Capital Territory", -35.2809, 149.1300),
    City("Kelowna", "British Columbia", 49.8880, -119.4960),
    City("Cape Town", "Western Cape", -33.9249, 18.4241),
    City("Athens", "Attica", 37.9838, 23.7275),
    City("Madrid", "Central Spain", 40.4168, -3.7038),
    City("Santiago", "Central Chile", -33.4489, -70.6693),
)

INDICATORS = (
    Indicator("fwixx", "Annual maximum FWI (fwixx)", "FWI", "#b83a2d", 1),
    Indicator("fwixd", "Days above local reference-period FWI 95th percentile (fwixd)", "days", "#d07a21", 0),
    Indicator("fwils", "Fire-season length above local reference midrange (fwils)", "days", "#56805b", 0),
    Indicator("fwisa", "Maximum 90-day mean FWI (fwisa)", "FWI", "#76507b", 1),
)


def haversine_km(lat: np.ndarray, lon: np.ndarray, city: City) -> np.ndarray:
    """Return great-circle distances from grid-cell centers to a city."""
    radius_km = 6371.0088
    lat1 = np.deg2rad(lat)
    lat2 = np.deg2rad(city.lat)
    dlat = lat1 - lat2
    dlon = np.deg2rad(((lon - city.lon + 180.0) % 360.0) - 180.0)
    value = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(
        dlon / 2.0
    ) ** 2
    return 2.0 * radius_km * np.arcsin(np.sqrt(value))


def city_ring(
    dataset: xr.Dataset, city: City, inner_km: float, outer_km: float
) -> tuple[xr.Dataset, np.ndarray]:
    """Subset a city neighborhood and return its annular area weights."""
    lat_pad = outer_km / 110.0 + 0.2
    lon_pad = outer_km / (110.0 * max(np.cos(np.deg2rad(city.lat)), 0.2)) + 0.2
    target_lon = city.lon % 360.0
    lat_values = dataset.lat.values
    lon_values = dataset.lon.values
    ilat = np.flatnonzero(np.abs(lat_values - city.lat) <= lat_pad)
    dlon = np.abs(((lon_values - target_lon + 180.0) % 360.0) - 180.0)
    ilon = np.flatnonzero(dlon <= lon_pad)
    subset = dataset.isel(lat=ilat, lon=ilon)
    yy, xx = np.meshgrid(subset.lat.values, subset.lon.values, indexing="ij")
    distance = haversine_km(yy, xx, city)
    annulus = (distance >= inner_km) & (distance <= outer_km)
    weights = np.where(annulus, np.cos(np.deg2rad(yy)), 0.0)
    return subset, weights


def weighted_series(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, int]:
    """Calculate an annual area-weighted mean over finite annulus cells."""
    finite = np.isfinite(values)
    annual_weights = finite * weights[None, :, :]
    denominator = annual_weights.sum(axis=(1, 2))
    numerator = np.nansum(values * annual_weights, axis=(1, 2))
    result = np.divide(
        numerator,
        denominator,
        out=np.full(denominator.shape, np.nan),
        where=denominator > 0,
    )
    cells = int(np.any(finite, axis=0)[weights > 0].sum())
    return result, cells


def rolling_mean(years: np.ndarray, values: np.ndarray, width: int = 5) -> np.ndarray:
    result = np.full(values.shape, np.nan)
    breaks = np.flatnonzero(np.diff(years) > 1) + 1
    for indices in np.split(np.arange(years.size), breaks):
        result[indices] = (
            xr.DataArray(values[indices], dims="time")
            .rolling(time=width, center=True, min_periods=3)
            .mean()
            .values
        )
    return result


def with_year_gap(years: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    complete = np.arange(years.min(), years.max() + 1)
    result = np.full(complete.shape, np.nan)
    result[years - complete[0]] = values
    return complete, result


def plot_indicator(
    indicator: Indicator,
    years: np.ndarray,
    series: dict[str, np.ndarray],
    cells: dict[str, int],
    inner_km: float,
    outer_km: float,
    output: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(4, 3, figsize=(16, 13.5), sharex=True)
    for axis, city in zip(axes.flat, CITIES, strict=True):
        values = series[city.name]
        plot_years, annual = with_year_gap(years, values)
        _, smooth = with_year_gap(years, rolling_mean(years, values))
        baseline = float(np.nanmean(values[(years >= 1995) & (years <= 2014)]))
        late = float(np.nanmean(values[(years >= 2076) & (years <= 2095)]))
        change = late - baseline

        axis.axvspan(2034, 2095, color="#eef0f2", zorder=0)
        axis.axhline(baseline, color="#777777", linestyle="--", linewidth=1)
        axis.plot(plot_years, annual, color=indicator.color, alpha=0.25, linewidth=0.9)
        axis.plot(plot_years, smooth, color=indicator.color, linewidth=2.1)
        axis.set_title(f"{city.name} | {city.region}", fontsize=9.5, weight="bold")
        axis.text(
            0.02,
            0.95,
            f"Late-century change: {change:+.{indicator.decimals}f} {indicator.units}",
            transform=axis.transAxes,
            va="top",
            fontsize=8,
        )
        axis.text(
            0.02,
            0.05,
            f"{cells[city.name]} land cells",
            transform=axis.transAxes,
            fontsize=7.5,
            color="#555555",
        )
        finite = values[np.isfinite(values)]
        span = float(finite.max() - finite.min())
        pad = max(span * 0.08, 0.5)
        axis.set_ylim(max(0.0, float(finite.min()) - pad), float(finite.max()) + pad)
        axis.set_xlim(1989, 2095)
        axis.tick_params(labelsize=8)

    for axis in axes[:, 0]:
        axis.set_ylabel(indicator.units, fontsize=9)
    for axis in axes[-1, :]:
        axis.set_xlabel("Year", fontsize=9)
    fig.suptitle(
        f"{indicator.title} in major-city peri-urban fire zones\n"
        f"ACCESS-CM2 0.1 deg | area-weighted {inner_km:g}-{outer_km:g} km land ring | "
        "SSP2-4.5 shaded | 5-year mean in bold",
        fontsize=15,
        y=0.995,
    )
    fig.text(
        0.5,
        0.006,
        "WUI proxy only: a distance-based peri-urban ring, not a mapped settlement-vegetation WUI mask. "
        "Change compares 2076-2095 with 1995-2014.",
        ha="center",
        fontsize=8.5,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inner-km", type=float, default=10.0)
    parser.add_argument("--outer-km", type=float, default=50.0)
    args = parser.parse_args()
    if not 0 <= args.inner_km < args.outer_km:
        parser.error("require 0 <= inner-km < outer-km")

    dataset = xr.open_zarr(args.input, consolidated=False, chunks=None)
    missing = {item.variable for item in INDICATORS} - set(dataset.data_vars)
    if missing:
        raise ValueError(f"input is missing variables: {sorted(missing)}")
    years = dataset.time.dt.year.values.astype(int)
    experiment = np.where(years <= 2020, "historical", "ssp245")
    all_series = {item.variable: {} for item in INDICATORS}
    cell_counts: dict[str, int] = {}
    records = []

    for city in CITIES:
        subset, weights = city_ring(dataset, city, args.inner_km, args.outer_km)
        values_by_variable = {}
        for indicator in INDICATORS:
            values, cells = weighted_series(subset[indicator.variable].values, weights)
            finite = np.isfinite(values)
            baseline = finite & (years >= 1995) & (years <= 2014)
            late = finite & (years >= 2076) & (years <= 2095)
            if finite.sum() < years.size - 2 or baseline.sum() < 18 or late.sum() < 18:
                raise ValueError(
                    f"insufficient {indicator.variable} coverage for {city.name}"
                )
            values_by_variable[indicator.variable] = values
            all_series[indicator.variable][city.name] = values
            cell_counts[city.name] = max(cell_counts.get(city.name, 0), cells)
        for index, year in enumerate(years):
            record = {
                "city": city.name,
                "region": city.region,
                "year": int(year),
                "experiment": experiment[index],
                "center_lat": city.lat,
                "center_lon": city.lon,
                "inner_km": args.inner_km,
                "outer_km": args.outer_km,
                "land_cells": cell_counts[city.name],
            }
            record.update(
                {
                    item.variable: float(values_by_variable[item.variable][index])
                    for item in INDICATORS
                }
            )
            records.append(record)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_output = args.output_dir / "access_cm2_city_wui_fwi_indicators_1989_2095.csv"
    with csv_output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    for indicator in INDICATORS:
        output = args.output_dir / f"access_cm2_city_wui_{indicator.variable}_1989_2095.png"
        plot_indicator(
            indicator,
            years,
            all_series[indicator.variable],
            cell_counts,
            args.inner_km,
            args.outer_km,
            output,
        )
        print(output)
    print(csv_output)


if __name__ == "__main__":
    main()
