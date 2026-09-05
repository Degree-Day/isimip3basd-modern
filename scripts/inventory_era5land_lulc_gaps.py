#!/usr/bin/env python3
"""Inventory LULC land missing from an ERA5-Land 0.1-degree support mask."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy import ndimage
import xarray as xr


def aggregate_raster(path: Path, factor: int = 12) -> tuple[np.ndarray, object]:
    """Sum a nested fine area raster to 0.1-degree cells without resampling."""
    with rasterio.open(path) as source:
        if source.width % factor or source.height % factor:
            raise ValueError(f"raster shape is not divisible by {factor}: {path}")
        output = np.empty(
            (source.height // factor, source.width // factor), dtype=np.float64
        )
        for row in range(output.shape[0]):
            values = source.read(
                1,
                window=Window(0, row * factor, source.width, factor),
                masked=True,
            ).filled(0)
            output[row] = values.reshape(factor, output.shape[1], factor).sum(
                axis=(0, 2), dtype=np.float64
            )
        return output, source.transform


def align_lulc_to_support(
    values_north_to_south: np.ndarray,
    transform: object,
    support: xr.DataArray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align a global -180/180 LULC grid to ascending-latitude 0/360 support."""
    resolution = abs(float(transform.e))
    if not np.isclose(resolution * 12, 0.1):
        raise ValueError(f"expected a 30 arc-second source, found {resolution} degrees")
    lat = float(transform.f) - (np.arange(values_north_to_south.shape[0]) + 0.5) * 0.1
    lon = float(transform.c) + (np.arange(values_north_to_south.shape[1]) + 0.5) * 0.1
    values = values_north_to_south[::-1]
    lat = lat[::-1]

    support_lat = np.asarray(support.lat.values)
    support_lon = ((np.asarray(support.lon.values) + 180) % 360) - 180
    lon_order = np.argsort(support_lon)
    support_lon = support_lon[lon_order]
    if not np.allclose(lon, support_lon, atol=1e-5):
        raise ValueError("LULC and support longitudes do not align")

    selected = (lat >= support_lat[0] - 1e-5) & (lat <= support_lat[-1] + 1e-5)
    aligned_lat = lat[selected]
    support_selection = (support_lat >= lat[0] - 1e-5) & (
        support_lat <= lat[-1] + 1e-5
    )
    if not np.allclose(aligned_lat, support_lat[support_selection], atol=1e-5):
        raise ValueError("LULC and support latitudes do not align")
    return values[selected], support_selection, lon_order


def periodic_labels(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Label eight-connected components while joining across the dateline."""
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    parent = np.arange(count + 1)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        if left and right:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[right_root] = left_root

    for row in range(mask.shape[0]):
        for neighbor_row in range(max(0, row - 1), min(mask.shape[0], row + 2)):
            union(int(labels[row, 0]), int(labels[neighbor_row, -1]))
    roots = np.array([find(value) for value in range(count + 1)])
    unique = np.unique(roots[1:])
    remap = np.zeros(count + 1, dtype=np.int32)
    remap[unique] = np.arange(1, unique.size + 1)
    return remap[roots[labels]], int(unique.size)


def cluster_records(
    missing: np.ndarray,
    land_area: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> list[dict[str, object]]:
    labels, count = periodic_labels(missing)
    records = []
    for label in range(1, count + 1):
        rows, columns = np.where(labels == label)
        area = land_area[rows, columns]
        total_area = float(area.sum())
        weights = area if total_area > 0 else np.ones_like(area)
        records.append(
            {
                "cluster": label,
                "cells": int(rows.size),
                "land_area_km2": total_area,
                "centroid_lat": float(np.average(lat[rows], weights=weights)),
                "centroid_lon": float(np.average(lon[columns], weights=weights)),
                "south": float(lat[rows].min() - 0.05),
                "north": float(lat[rows].max() + 0.05),
                "west": float(lon[columns].min() - 0.05),
                "east": float(lon[columns].max() + 0.05),
            }
        )
    return sorted(records, key=lambda record: record["land_area_km2"], reverse=True)


def summary_for_mask(
    name: str,
    land_area: np.ndarray,
    cell_area: np.ndarray,
    coverage: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    land_fraction = np.divide(
        land_area,
        cell_area,
        out=np.zeros_like(land_area),
        where=cell_area > 0,
    )
    thresholds = {
        "any_land": 0.0,
        "land_fraction_1pct": 0.01,
        "land_fraction_10pct": 0.1,
        "land_fraction_50pct": 0.5,
    }
    result: dict[str, object] = {"name": name}
    for label, threshold in thresholds.items():
        land = land_area > 0 if threshold == 0 else land_fraction >= threshold
        missing = land & ~coverage
        result[label] = {
            "land_cells": int(land.sum()),
            "missing_cells": int(missing.sum()),
            "missing_land_area_km2": float(land_area[missing].sum()),
            "missing_fraction_of_land_area": float(
                land_area[missing].sum() / land_area[land].sum()
            ),
        }
    any_missing = (land_area > 0) & ~coverage
    clusters = cluster_records(any_missing, land_area, lat, lon)
    result["any_land"]["missing_clusters"] = len(clusters)
    return result, clusters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--land-area", type=Path, required=True)
    parser.add_argument("--cell-area", type=Path, required=True)
    parser.add_argument("--support-mask", type=Path, required=True)
    parser.add_argument("--coastal-plan", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    land_area, land_transform = aggregate_raster(args.land_area)
    cell_area, cell_transform = aggregate_raster(args.cell_area)
    if land_transform != cell_transform or land_area.shape != cell_area.shape:
        raise ValueError("land-area and cell-area rasters do not share a grid")

    support_data = xr.open_zarr(
        args.support_mask, consolidated=False, chunks=None
    )["spatial_valid_mask"]
    aligned_land, lat_selection, lon_order = align_lulc_to_support(
        land_area, land_transform, support_data
    )
    aligned_cell, _, _ = align_lulc_to_support(
        cell_area, cell_transform, support_data
    )
    support = np.asarray(support_data.values, dtype=bool)[lat_selection][:, lon_order]
    lat = np.asarray(support_data.lat.values)[lat_selection]
    lon = ((np.asarray(support_data.lon.values) + 180) % 360) - 180
    lon = lon[lon_order]

    summaries = []
    raw_summary, raw_clusters = summary_for_mask(
        "native_era5land_support", aligned_land, aligned_cell, support, lat, lon
    )
    summaries.append(raw_summary)
    cluster_sets = {"native": raw_clusters}
    if args.coastal_plan:
        plan = xr.open_zarr(args.coastal_plan, consolidated=False, chunks=None)
        coastal = np.asarray(plan.coastal_fill.values, dtype=bool)[lat_selection][
            :, lon_order
        ]
        repaired_summary, repaired_clusters = summary_for_mask(
            "after_coastal_repair",
            aligned_land,
            aligned_cell,
            support | coastal,
            lat,
            lon,
        )
        summaries.append(repaired_summary)
        cluster_sets["after_coastal_repair"] = repaired_clusters

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "land_area_source": str(args.land_area),
        "cell_area_source": str(args.cell_area),
        "support_mask": str(args.support_mask),
        "coastal_plan": str(args.coastal_plan) if args.coastal_plan else None,
        "comparison_domain": {
            "south": float(lat.min() - 0.05),
            "north": float(lat.max() + 0.05),
            "west": -180.0,
            "east": 180.0,
            "resolution_degrees": 0.1,
        },
        "summaries": summaries,
    }
    json_path = args.output_prefix.with_suffix(".json")
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for label, records in cluster_sets.items():
        csv_path = args.output_prefix.with_name(
            f"{args.output_prefix.name}.{label}.clusters.csv"
        )
        with csv_path.open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(records[0]) if records else ["cluster"]
            )
            writer.writeheader()
            writer.writerows(records)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
