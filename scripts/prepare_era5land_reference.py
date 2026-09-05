#!/usr/bin/env python3
"""Prepare nested no-leap ERA5-Land references for 1 to 0.1 degree MBCnSD."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import multiprocessing
from pathlib import Path
import time

import dask.array as da
import numpy as np
import rasterio
from rasterio.windows import Window
import xarray as xr
from xclim.core.units import convert_units_to

from isimip3basd_modern.io import open_dataset
from isimip3basd_modern.preprocessing import CLIP_BOUNDS, TARGET_UNITS, _normalize_calendar
from isimip3basd_modern.validation import validate_variable


METADATA = {
    "hurs": ("%", "relative_humidity"),
    "pr": ("mm d-1", "precipitation_flux"),
    "prsnratio": ("1", "snowfall_precipitation_ratio"),
    "ps": ("Pa", "surface_air_pressure"),
    "rlds": ("W m-2", "surface_downwelling_longwave_flux_in_air"),
    "rsds": ("W m-2", "surface_downwelling_shortwave_flux_in_air"),
    "sfcWind": ("m s-1", "wind_speed"),
    "tas": ("K", "air_temperature"),
    "tasrange": ("K", "air_temperature_range"),
    "tasskew": ("1", "air_temperature_skewness"),
}
DEFAULT_VARIABLES = ("tas", "hurs", "pr", "sfcWind")
REFERENCE_SOURCE_CODES = {
    0: "outside LULC land or unavailable",
    1: "ERA5-Land",
    2: "ERA5 bilinear fallback",
}


def nested_grids() -> tuple[xr.Dataset, xr.Dataset]:
    """Return the ERA5-Land domain on exactly nested 0.1 and 1 degree grids."""
    coarse_lat = np.arange(-56.5, 90.0, 1.0)
    coarse_lon = np.arange(0.5, 360.0, 1.0)
    fine_lat = np.arange(-56.95, 90.0, 0.1)
    fine_lon = np.arange(0.05, 360.0, 0.1)

    def grid(lat: np.ndarray, lon: np.ndarray) -> xr.Dataset:
        return xr.Dataset(
            coords={
                "lat": xr.DataArray(
                    lat,
                    dims="lat",
                    attrs={"standard_name": "latitude", "units": "degrees_north"},
                ),
                "lon": xr.DataArray(
                    lon,
                    dims="lon",
                    attrs={"standard_name": "longitude", "units": "degrees_east"},
                ),
            }
        )

    return grid(fine_lat, fine_lon), grid(coarse_lat, coarse_lon)


def prepare_fine(
    source: xr.DataArray,
    variable: str,
    fine_grid: xr.Dataset | None = None,
) -> xr.DataArray:
    units, standard_name = METADATA[variable]
    renames = {
        name: replacement
        for name, replacement in (("latitude", "lat"), ("longitude", "lon"))
        if name in source.dims
    }
    source = source.rename(renames).sortby("lat")
    source.attrs.update(units=units, standard_name=standard_name)
    source = source.sel(time=slice("1993-01-01", "2014-12-31"))
    source = convert_units_to(
        source,
        TARGET_UNITS[variable],
        context="hydro" if variable == "pr" else None,
    )
    bounds = CLIP_BOUNDS.get(variable)
    if bounds is not None:
        source = source.clip(min=bounds[0], max=bounds[1])
    source, source_calendar, day_delta = _normalize_calendar(source, variable)
    if fine_grid is None:
        fine_grid = nested_grids()[0]
    prepared = source.interp(
        lat=fine_grid.lat,
        lon=fine_grid.lon,
        method="linear",
        assume_sorted=True,
    )
    prepared = prepared.astype("float32").chunk(
        {"time": 365, "lat": 20, "lon": 20}
    )
    prepared.name = variable
    prepared.attrs.update(
        units=TARGET_UNITS[variable],
        standard_name=standard_name,
        reference_dataset="ERA5-Land local-noon daily weather",
        reference_period="1993-2014",
        preprocessing_calendar="noleap",
        preprocessing_source_calendar=source_calendar,
        preprocessing_calendar_day_delta=day_delta,
        preprocessing_grid="nested_0.1_degree_cell_centers",
        preprocessing_created_utc=datetime.now(timezone.utc).isoformat(),
    )
    return prepared


def aggregate_coarse(fine: xr.DataArray) -> xr.DataArray:
    """Area-average a nested fine reference to its 1 degree training grid."""
    weights = np.cos(np.deg2rad(fine.lat)).broadcast_like(fine)
    numerator = (fine * weights).coarsen(lat=10, lon=10, boundary="exact").sum()
    denominator = weights.where(fine.notnull()).coarsen(
        lat=10, lon=10, boundary="exact"
    ).sum()
    coarse = numerator / denominator
    _, coarse_grid = nested_grids()
    coarse = coarse.assign_coords(lat=coarse_grid.lat, lon=coarse_grid.lon)
    coarse.name = fine.name
    coarse.attrs.update(fine.attrs)
    coarse.attrs["preprocessing_grid"] = "nested_1_degree_area_mean"
    return coarse.astype("float32").chunk({"time": 365, "lat": 20, "lon": 20})


def build_lulc_land_mask(path: Path, fine_grid: xr.Dataset) -> xr.DataArray:
    """Aggregate 30 arc-second land area to the nested 0.1-degree grid."""
    factor = 12
    with rasterio.open(path) as source:
        if source.crs is None or source.crs.to_epsg() != 4326:
            raise ValueError(f"LULC land-area raster must use EPSG:4326: {path}")
        if source.width % factor or source.height % factor:
            raise ValueError(f"LULC raster shape is not divisible by {factor}: {path}")
        if not np.isclose(abs(source.transform.e) * factor, 0.1):
            raise ValueError("LULC raster must have 30 arc-second cells")
        rows = source.height // factor
        columns = source.width // factor
        land = np.empty((rows, columns), dtype=bool)
        for row in range(rows):
            values = source.read(
                1,
                window=Window(0, row * factor, source.width, factor),
                masked=True,
            ).filled(0)
            land[row] = values.reshape(factor, columns, factor).sum(axis=(0, 2)) > 0
        lat = source.transform.f - (np.arange(rows) + 0.5) * 0.1
        lon = source.transform.c + (np.arange(columns) + 0.5) * 0.1

    source_mask = xr.DataArray(
        land[::-1],
        dims=("lat", "lon"),
        coords={"lat": lat[::-1], "lon": lon},
    )
    source_mask = source_mask.assign_coords(lon=(source_mask.lon % 360)).sortby("lon")
    lat_keys = np.round(np.asarray(source_mask.lat), 5)
    lon_keys = np.round(np.asarray(source_mask.lon), 5)
    target_lat = np.round(np.asarray(fine_grid.lat), 5)
    target_lon = np.round(np.asarray(fine_grid.lon), 5)
    lat_lookup = {value: index for index, value in enumerate(lat_keys)}
    lon_lookup = {value: index for index, value in enumerate(lon_keys)}
    output = np.zeros((target_lat.size, target_lon.size), dtype=bool)
    matched_lat = np.array([value in lat_lookup for value in target_lat])
    matched_lon = np.array([value in lon_lookup for value in target_lon])
    source_rows = [lat_lookup[value] for value in target_lat[matched_lat]]
    source_columns = [lon_lookup[value] for value in target_lon[matched_lon]]
    output[np.ix_(matched_lat, matched_lon)] = np.asarray(source_mask)[
        np.ix_(source_rows, source_columns)
    ]
    return xr.DataArray(
        output,
        dims=("lat", "lon"),
        coords=fine_grid.coords,
        name="lulc_land",
        attrs={
            "source": str(path),
            "definition": "positive 30 arc-second land area within 0.1-degree cell",
        },
    )


def normalize_era5_daily(source: xr.DataArray, variable: str) -> xr.DataArray:
    """Apply explicit units and physical bounds to the regular ERA5 daily archive."""
    source = source.rename(
        {
            name: replacement
            for name, replacement in (("latitude", "lat"), ("longitude", "lon"))
            if name in source.dims
        }
    )
    source = source.assign_coords(lon=(source.lon % 360)).sortby("lon").sortby("lat")
    if variable == "hurs":
        source.attrs["units"] = "1"
    elif variable == "pr":
        source.attrs["units"] = "mm d-1"
    else:
        source.attrs["units"] = METADATA[variable][0]
    source = convert_units_to(
        source,
        TARGET_UNITS[variable],
        context="hydro" if variable == "pr" else None,
    )
    bounds = CLIP_BOUNDS.get(variable)
    if bounds is not None:
        source = source.clip(min=bounds[0], max=bounds[1])
    return source


def interpolate_era5_year(
    path: Path,
    variable: str,
    target: xr.Dataset,
    expected_time: xr.DataArray,
) -> xr.DataArray:
    """Bilinearly interpolate one regular ERA5 year, including the dateline."""
    with xr.open_zarr(path, consolidated=True, chunks=None) as dataset:
        source = normalize_era5_daily(dataset[variable], variable)
        source = xr.concat(
            (
                source.isel(lon=[-1]).assign_coords(lon=[float(source.lon[-1]) - 360]),
                source,
                source.isel(lon=[0]).assign_coords(lon=[float(source.lon[0]) + 360]),
            ),
            dim="lon",
        )
        source = source.sel(
            lat=slice(float(target.lat[0]) - 0.3, float(target.lat[-1]) + 0.3),
            lon=slice(float(target.lon[0]) - 0.3, float(target.lon[-1]) + 0.3),
        )
        source, _, _ = _normalize_calendar(source, variable)
        interpolated = source.interp(
            lat=target.lat,
            lon=target.lon,
            method="linear",
            assume_sorted=True,
        )
        if interpolated.sizes["time"] != expected_time.size:
            raise ValueError(f"ERA5 time axis does not match target for {path}")
        return interpolated.assign_coords(time=expected_time).astype("float32").load()


def initialize_store(path: Path, variable: str, grid: xr.Dataset) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    time_coord = xr.date_range(
        "1993-01-01", periods=22 * 365, freq="D", calendar="noleap", use_cftime=True
    )
    spatial_chunks = 50 if grid.sizes["lon"] == 3600 else 5
    values = da.empty(
        (time_coord.size, grid.sizes["lat"], grid.sizes["lon"]),
        chunks=(365, spatial_chunks, spatial_chunks),
        dtype="float32",
    )
    units, standard_name = METADATA[variable]
    skeleton = xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={"time": time_coord, "lat": grid.lat, "lon": grid.lon},
        name=variable,
        attrs={
            "units": TARGET_UNITS[variable],
            "standard_name": standard_name,
            "reference_dataset": "ERA5-Land local-noon daily weather",
            "reference_period": "1993-2014",
            "preprocessing_calendar": "noleap",
            "preprocessing_grid": (
                "nested_0.1_degree_cell_centers"
                if spatial_chunks == 50
                else "nested_1_degree_area_mean"
            ),
        },
    )
    skeleton.to_dataset().to_zarr(
        path,
        mode="w",
        compute=False,
        consolidated=False,
        zarr_format=3,
        encoding={variable: {"_FillValue": np.nan}},
    )


def initialize_source_mask_store(path: Path, grid: xr.Dataset) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = xr.DataArray(
        da.zeros(
            (grid.sizes["lat"], grid.sizes["lon"]),
            chunks=(50, 50),
            dtype="uint8",
        ),
        dims=("lat", "lon"),
        coords=grid.coords,
        name="reference_source",
        attrs={
            "flag_values": list(REFERENCE_SOURCE_CODES),
            "flag_meanings": "outside_or_unavailable era5_land era5_bilinear_fallback",
        },
    )
    mask.to_dataset().to_zarr(
        path,
        mode="w",
        compute=False,
        consolidated=False,
        zarr_format=3,
    )


def process_tile(
    source_path: str,
    output_root: str,
    variable: str,
    lat_start: int,
    lat_stop: int,
    lon_start: int,
    lon_stop: int,
    era5_daily_root: str | None = None,
    lulc_mask_path: str | None = None,
) -> str:
    fine_grid, coarse_grid = nested_grids()
    target = fine_grid.isel(
        lat=slice(lat_start, lat_stop), lon=slice(lon_start, lon_stop)
    )
    with xr.open_zarr(source_path, consolidated=False, chunks=None) as dataset:
        source = dataset[variable].sel(time=slice("1993-01-01", "2014-12-31"))
        source = source.rename(latitude="lat", longitude="lon").sortby("lat")
        source = source.sel(
            lat=slice(float(target.lat[0]) - 0.11, float(target.lat[-1]) + 0.11)
        )
        lon_lower = float(target.lon[0]) - 0.11
        lon_upper = float(target.lon[-1]) + 0.11
        source_lon = source.sel(lon=slice(max(0.0, lon_lower), min(359.9, lon_upper)))
        if lon_upper > 359.9:
            wrapped = source.isel(lon=[0]).assign_coords(lon=[360.0])
            source_lon = xr.concat((source_lon, wrapped), dim="lon")
        source = source_lon
        fine = prepare_fine(source, variable, target).compute()

    reference_source = xr.where(fine.notnull().all("time"), 1, 0).astype("uint8")
    if era5_daily_root and lulc_mask_path:
        lulc_land = xr.open_zarr(
            lulc_mask_path, consolidated=False, chunks=None
        )["lulc_land"].isel(
            lat=slice(lat_start, lat_stop), lon=slice(lon_start, lon_stop)
        )
        fallback_cells = lulc_land & fine.isnull().any("time")
        if bool(fallback_cells.any()):
            annual = []
            for year in range(1993, 2015):
                path = Path(era5_daily_root) / str(year) / (
                    f"{variable}.era5.day.global.30km.{year}.zarr"
                )
                if not path.exists():
                    raise FileNotFoundError(path)
                expected = fine.time.where(fine.time.dt.year == year, drop=True)
                annual.append(interpolate_era5_year(path, variable, target, expected))
            fallback = xr.concat(annual, dim="time").assign_coords(time=fine.time)
            fine = fine.fillna(fallback.where(fallback_cells))
            reference_source = xr.where(
                fallback_cells & fine.notnull().all("time"), 2, reference_source
            ).astype("uint8")
            fine.attrs.update(
                reference_dataset="ERA5-Land with ERA5 fallback",
                reference_fallback="ERA5 daily data bilinearly interpolated to 0.1 degree",
                reference_fallback_temporal_semantics=(
                    "daily means for tas, hurs, and sfcWind; daily total for pr"
                ),
            )
    reference_source.name = "reference_source"
    reference_source.attrs.update(
        flag_values=list(REFERENCE_SOURCE_CODES),
        flag_meanings="outside_or_unavailable era5_land era5_bilinear_fallback",
    )

    weights = np.cos(np.deg2rad(fine.lat)).broadcast_like(fine)
    numerator = (fine * weights).coarsen(lat=10, lon=10, boundary="exact").sum()
    denominator = weights.where(fine.notnull()).coarsen(
        lat=10, lon=10, boundary="exact"
    ).sum()
    coarse = (numerator / denominator).astype("float32")
    coarse.name = variable
    coarse_lat = slice(lat_start // 10, lat_stop // 10)
    coarse_lon = slice(lon_start // 10, lon_stop // 10)
    coarse = coarse.assign_coords(
        lat=coarse_grid.lat.isel(lat=coarse_lat),
        lon=coarse_grid.lon.isel(lon=coarse_lon),
    )

    root = Path(output_root)
    region_fine = {
        "time": slice(0, fine.sizes["time"]),
        "lat": slice(lat_start, lat_stop),
        "lon": slice(lon_start, lon_stop),
    }
    region_coarse = {
        "time": slice(0, coarse.sizes["time"]),
        "lat": coarse_lat,
        "lon": coarse_lon,
    }
    fine.to_dataset().drop_vars(["time", "lat", "lon"]).to_zarr(
        root / "fine" / f"{variable}.zarr",
        mode="r+",
        region=region_fine,
        consolidated=False,
    )
    reference_source.to_dataset().drop_vars(["lat", "lon"]).to_zarr(
        root / "source" / f"{variable}.zarr",
        mode="r+",
        region={
            "lat": slice(lat_start, lat_stop),
            "lon": slice(lon_start, lon_stop),
        },
        consolidated=False,
    )
    coarse.to_dataset().drop_vars(["time", "lat", "lon"]).to_zarr(
        root / "coarse" / f"{variable}.zarr",
        mode="r+",
        region=region_coarse,
        consolidated=False,
    )
    marker = f"{lat_start:04d}-{lon_start:04d}"
    state = root / "state" / variable
    state.mkdir(parents=True, exist_ok=True)
    (state / marker).write_text("done\n")
    return marker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--memory-limit", default="16GB")
    parser.add_argument(
        "--era5-daily-root",
        type=Path,
        help="optional regular ERA5 year-store root used only for missing LULC land",
    )
    parser.add_argument(
        "--lulc-land-area",
        type=Path,
        help="30 arc-second land-area raster defining ERA5 fallback cells",
    )
    parser.add_argument(
        "--variables", nargs="+", choices=tuple(METADATA), default=list(DEFAULT_VARIABLES)
    )
    args = parser.parse_args()
    if bool(args.era5_daily_root) != bool(args.lulc_land_area):
        parser.error("--era5-daily-root and --lulc-land-area must be used together")
    args.output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []

    fine_grid, coarse_grid = nested_grids()
    lulc_mask_path = None
    if args.lulc_land_area:
        lulc_mask_path = args.output / "lulc_land_mask.zarr"
        if not lulc_mask_path.exists():
            land_mask = build_lulc_land_mask(args.lulc_land_area, fine_grid)
            land_mask.to_dataset().to_zarr(
                lulc_mask_path, mode="w", consolidated=False, zarr_format=3
            )
    tiles = [
        (lat, min(lat + 50, fine_grid.sizes["lat"]), lon, lon + 50)
        for lat in range(0, fine_grid.sizes["lat"], 50)
        for lon in range(0, fine_grid.sizes["lon"], 50)
    ]
    for variable in args.variables:
        started = time.perf_counter()
        fine_path = args.output / "fine" / f"{variable}.zarr"
        coarse_path = args.output / "coarse" / f"{variable}.zarr"
        qc_path = args.output / f"{variable}.qc.json"
        source_mask_path = args.output / "source" / f"{variable}.zarr"
        if (
            fine_path.exists()
            and coarse_path.exists()
            and qc_path.exists()
            and (not args.era5_daily_root or source_mask_path.exists())
        ):
            existing = json.loads(qc_path.read_text())
            if not existing.get("valid"):
                raise RuntimeError(f"existing QC is invalid: {qc_path}")
            records.append(existing)
            print(f"SKIP {variable}", flush=True)
            continue
        initialize_store(fine_path, variable, fine_grid)
        initialize_store(coarse_path, variable, coarse_grid)
        initialize_source_mask_store(source_mask_path, fine_grid)
        state = args.output / "state" / variable
        pending = [
            tile
            for tile in tiles
            if not (state / f"{tile[0]:04d}-{tile[2]:04d}").exists()
        ]
        print(f"START {variable}: {len(pending)}/{len(tiles)} tiles", flush=True)
        # Full-variable Dask QC leaves worker threads alive. Spawning avoids the
        # fork-after-threads deadlock when the next variable starts its pool.
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            futures = {
                executor.submit(
                    process_tile,
                    str(args.source),
                    str(args.output),
                    variable,
                    *tile,
                    str(args.era5_daily_root) if args.era5_daily_root else None,
                    str(lulc_mask_path) if lulc_mask_path else None,
                ): tile
                for tile in pending
            }
            for index, future in enumerate(as_completed(futures), start=1):
                marker = future.result()
                if index % 25 == 0 or index == len(pending):
                    print(
                        f"{variable} {index}/{len(pending)} tiles; last {marker}",
                        flush=True,
                    )

        reports = {}
        for label, path in (("fine", fine_path), ("coarse", coarse_path)):
            with open_dataset(path, {"time": 365}) as written:
                report = validate_variable(
                    written[variable], variable, statistical=False
                )
            if not report.valid:
                raise RuntimeError(f"{label} {variable} QC failed: {report.errors}")
            reports[label] = report.to_dict()
        record = {
            "variable": variable,
            "fine": str(fine_path),
            "coarse": str(coarse_path),
            "valid": True,
            "elapsed_seconds": time.perf_counter() - started,
            "qc": reports,
        }
        with xr.open_zarr(source_mask_path, consolidated=False, chunks=None) as source_ds:
            codes, counts = np.unique(source_ds.reference_source.values, return_counts=True)
        record["reference_source_cells"] = {
            REFERENCE_SOURCE_CODES[int(code)]: int(count)
            for code, count in zip(codes, counts, strict=True)
        }
        records.append(record)
        qc_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        print(f"DONE {variable} {record['elapsed_seconds']:.1f}s", flush=True)

    manifest = {
        "source": str(args.source),
        "output": str(args.output),
        "valid_records": len(records),
        "records": records,
    }
    (args.output / "reference-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
