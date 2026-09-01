#!/usr/bin/env python3
"""Run regional or global MBCnSD as restartable two-dimensional tiles."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import time
import warnings
from importlib.metadata import version

for name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ.setdefault(name, "1")

import dask
import dask.array as da
import numpy as np
import xarray as xr
from xclim.core.units import convert_units_to

warnings.filterwarnings("ignore", message="invalid value encountered in divide")
warnings.filterwarnings(
    "ignore", message="All-nan slice encountered in interp_on_quantiles"
)
warnings.filterwarnings("ignore", message="Increasing number of chunks by factor")

from isimip3basd_modern.downscaling import (
    CIL_PRECIPITATION_CEILING,
    CIL_TEMPERATURE_VALID_RANGE,
    apply_downscaled_value_controls,
    downscale_variable,
)
from isimip3basd_modern import __version__
from isimip3basd_modern.pipeline import adjust_variable
from isimip3basd_modern.validation import validate_variable


DEFAULT_VARIABLES = ("hurs", "pr", "sfcWind", "tas")
SUPPORTED_VARIABLES = (
    "hurs",
    "pr",
    "prsnratio",
    "ps",
    "rlds",
    "rsds",
    "sfcWind",
    "tas",
    "tasrange",
    "tasskew",
)
REGIONS = {
    "west": {
        "fine_lat": slice(920, 1290),
        "fine_lon": slice(3490, 3600),
        "coarse_lat": slice(92, 129),
        "coarse_lon": slice(349, 360),
        "description": "35.05-71.95N, 10.95W-0.05W",
    },
    "east": {
        "fine_lat": slice(920, 1290),
        "fine_lon": slice(0, 320),
        "coarse_lat": slice(92, 129),
        "coarse_lon": slice(0, 32),
        "description": "35.05-71.95N, 0.05-31.95E",
    },
}


def resolve_regions(
    reference_root: Path, requested: list[str], variable: str
) -> dict[str, dict[str, object]]:
    """Resolve named domains against the actual nested reference grids."""
    regions = {name: dict(REGIONS[name]) for name in requested if name in REGIONS}
    if "global" not in requested:
        return regions

    coarse = open_variable(reference_root / "coarse" / f"{variable}.zarr", variable)
    fine = open_variable(reference_root / "fine" / f"{variable}.zarr", variable)
    if (
        fine.sizes["lat"] % coarse.sizes["lat"]
        or fine.sizes["lon"] % coarse.sizes["lon"]
    ):
        raise ValueError("global fine reference grid is not nested in its coarse grid")
    lat_factor = fine.sizes["lat"] // coarse.sizes["lat"]
    lon_factor = fine.sizes["lon"] // coarse.sizes["lon"]
    if lat_factor <= 1 or lon_factor <= 1:
        raise ValueError("global reference must be finer than the coarse grid")
    regions["global"] = {
        "fine_lat": slice(0, fine.sizes["lat"]),
        "fine_lon": slice(0, fine.sizes["lon"]),
        "coarse_lat": slice(0, coarse.sizes["lat"]),
        "coarse_lon": slice(0, coarse.sizes["lon"]),
        "lat_factor": lat_factor,
        "lon_factor": lon_factor,
        "periodic_lon": bool(
            np.isclose(
                float(coarse.lon[-1] - coarse.lon[0])
                + 360 / coarse.sizes["lon"],
                360,
            )
        ),
        "description": (
            f"full reference domain: {float(fine.lat[0]):.2f}-"
            f"{float(fine.lat[-1]):.2f}N, 0-360 longitude"
        ),
    }
    return regions


def open_variable(path: Path, variable: str) -> xr.DataArray:
    return xr.open_zarr(path, consolidated=False)[variable]


def simulation_path(
    canonical_root: Path,
    model: str,
    experiment: str,
    stage: str,
    variable: str,
) -> Path:
    return canonical_root / model / experiment / stage / f"{variable}.zarr"


def adjusted_store_path(
    adjusted_root: Path,
    model: str,
    experiment: str,
    stage: str,
    variable: str,
) -> Path:
    """Return the shared coarse-grid product used by spatial downscaling."""
    return adjusted_root / model / experiment / stage / f"{variable}.zarr"


def adjustment_marker_path(
    adjusted_root: Path,
    model: str,
    experiment: str,
    stage: str,
    variable: str,
    tile: dict[str, int],
) -> Path:
    return (
        adjusted_root
        / model
        / experiment
        / stage
        / "state"
        / variable
        / f"{_tile_name(tile)}.success"
    )


def select_simulation_period(
    data: xr.DataArray, start: str | None, end: str | None
) -> xr.DataArray:
    if start is None and end is None:
        return data
    return data.sel(time=slice(start, end))


def default_output_root(
    model: str,
    scenario: str,
    simulation_stage: str,
    regions: list[str],
) -> Path:
    """Return a collision-proof default for regional or global products."""
    if regions == ["global"]:
        return Path("/data1/cmip6_downscaled_global") / model / scenario / simulation_stage
    return Path("/data1/access_europe_downscale_full")


def success_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.success")


def is_complete(path: Path) -> bool:
    return success_path(path).exists()


def write_zarr_atomic(data: xr.DataArray, path: Path) -> None:
    partial = path.with_name(f"{path.name}.partial")
    shutil.rmtree(partial, ignore_errors=True)
    data.to_dataset().to_zarr(
        partial,
        mode="w",
        consolidated=False,
        zarr_format=3,
    )
    shutil.rmtree(path, ignore_errors=True)
    partial.rename(path)
    success_path(path).touch()


def variable_only_dataset(data: xr.DataArray) -> xr.Dataset:
    """Return only the data variable for a Zarr region write."""
    return xr.Dataset(
        {data.name: (data.dims, data.data, data.attrs)},
        coords={dim: (dim, data[dim].data, data[dim].attrs) for dim in data.dims},
    )


def initialize_output_store(
    adjusted: xr.DataArray,
    fine_reference: xr.DataArray,
    path: Path,
    *,
    iterations: int,
    quantiles: int,
) -> None:
    if path.exists():
        existing = open_variable(path, adjusted.name)
        expected_sizes = {
            "time": adjusted.sizes["time"],
            "lat": fine_reference.sizes["lat"],
            "lon": fine_reference.sizes["lon"],
        }
        if dict(existing.sizes) != expected_sizes:
            raise ValueError(
                f"existing output shape does not match requested run: {path}"
            )
        if not (
            np.array_equal(existing.time.values, adjusted.time.values)
            and np.array_equal(existing.lat.values, fine_reference.lat.values)
            and np.array_equal(existing.lon.values, fine_reference.lon.values)
        ):
            raise ValueError(
                f"existing output coordinates do not match requested run: {path}"
            )
        if (
            existing.attrs.get("statistical_downscaling_iterations") != iterations
            or existing.attrs.get("statistical_downscaling_quantiles") != quantiles
        ):
            raise ValueError(
                f"existing output algorithm settings do not match requested run: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    dims = ("time", "lat", "lon")
    shape = (
        adjusted.sizes["time"],
        fine_reference.sizes["lat"],
        fine_reference.sizes["lon"],
    )
    chunks = (adjusted.sizes["time"], 10, 10)
    template = xr.DataArray(
        da.empty(shape, chunks=chunks, dtype=adjusted.dtype),
        dims=dims,
        coords={
            "time": adjusted.time,
            "lat": fine_reference.lat,
            "lon": fine_reference.lon,
        },
        name=adjusted.name,
        attrs={
            **adjusted.attrs,
            "statistical_downscaling_method": "MBCnSD",
            "statistical_downscaling_iterations": iterations,
            "statistical_downscaling_quantiles": quantiles,
            "statistical_downscaling_software": (
                f"isimip3basd-modern/{__version__}; xarray/{version('xarray')}; "
                f"scipy/{version('scipy')}"
            ),
            "statistical_downscaling_source": (
                "ISIMIP3BASD/3.0.2; https://doi.org/10.5281/zenodo.7151476"
            ),
            "spatial_processing_context": (
                "global coarse and fine grids with a one-coarse-cell halo; "
                "only the disjoint tile core is written"
            ),
        },
    )
    template.to_dataset().to_zarr(
        path,
        mode="w",
        compute=False,
        consolidated=False,
        zarr_format=3,
        encoding={adjusted.name: {"_FillValue": float("nan")}},
    )


def initialize_adjusted_store(
    simulation: xr.DataArray,
    path: Path,
) -> None:
    if path.exists():
        existing = open_variable(path, simulation.name)
        if dict(existing.sizes) != dict(simulation.sizes) or any(
            not np.array_equal(existing[dim].values, simulation[dim].values)
            for dim in simulation.dims
        ):
            raise ValueError(
                f"existing adjusted store does not match requested simulation: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    chunks = (simulation.sizes["time"], 1, 1)
    template = xr.DataArray(
        da.empty(simulation.shape, chunks=chunks, dtype=simulation.dtype),
        dims=simulation.dims,
        coords={dim: simulation[dim] for dim in simulation.dims},
        name=simulation.name,
        attrs=simulation.attrs,
    )
    template.to_dataset().to_zarr(
        path,
        mode="w",
        compute=False,
        consolidated=False,
        zarr_format=3,
        encoding={simulation.name: {"_FillValue": float("nan")}},
    )


def required_adjustment_mask(
    global_spec: dict[str, object],
    target_specs: list[dict[str, object]],
    *,
    halo: int = 1,
) -> np.ndarray:
    """Return global coarse cells needed by targets and their spatial halos."""
    coarse_lat = global_spec["coarse_lat"]
    coarse_lon = global_spec["coarse_lon"]
    required = np.zeros(
        (coarse_lat.stop - coarse_lat.start, coarse_lon.stop - coarse_lon.start),
        dtype=bool,
    )
    for spec in target_specs:
        lat = spec["coarse_lat"]
        lon = spec["coarse_lon"]
        lat_indices = np.arange(
            max(lat.start - halo, 0), min(lat.stop + halo, required.shape[0])
        )
        lon_indices = np.arange(lon.start - halo, lon.stop + halo) % required.shape[1]
        required[np.ix_(lat_indices, lon_indices)] = True
    return required


def required_adjustment_tiles(
    global_spec: dict[str, object],
    target_specs: list[dict[str, object]],
    tile_lat_degrees: int,
    tile_lon_degrees: int,
    *,
    halo: int = 1,
) -> list[dict[str, int]]:
    """Return disjoint global tiles covering targets and their spatial halos."""
    required = required_adjustment_mask(global_spec, target_specs, halo=halo)

    return [
        tile
        for tile in tile_specs(global_spec, tile_lat_degrees, tile_lon_degrees)
        if required[
            tile["coarse_lat_start"] : tile["coarse_lat_stop"],
            tile["coarse_lon_start"] : tile["coarse_lon_stop"],
        ].any()
    ]


def missing_cell_tiles(missing: np.ndarray, lon_width: int) -> list[dict[str, int]]:
    """Group uncovered cells into disjoint one-row adjustment tiles."""
    tiles: list[dict[str, int]] = []
    for lat in range(missing.shape[0]):
        indices = np.flatnonzero(missing[lat])
        start = 0
        while start < indices.size:
            lon_start = int(indices[start])
            stop = start + 1
            while (
                stop < indices.size
                and indices[stop] == indices[stop - 1] + 1
                and indices[stop] < lon_start + lon_width
            ):
                stop += 1
            lon_stop = int(indices[stop - 1]) + 1
            tiles.append(
                {
                    "coarse_lat_start": lat,
                    "coarse_lat_stop": lat + 1,
                    "coarse_lon_start": lon_start,
                    "coarse_lon_stop": lon_stop,
                    "fine_lat_start": lat,
                    "fine_lat_stop": lat + 1,
                    "fine_lon_start": lon_start,
                    "fine_lon_stop": lon_stop,
                }
            )
            start = stop
    return tiles


def initialize_coverage_store(simulation: xr.DataArray, path: Path) -> None:
    if path.exists():
        existing = open_variable(path, "coverage")
        if dict(existing.sizes) != {
            "lat": simulation.sizes["lat"],
            "lon": simulation.sizes["lon"],
        } or not (
            np.array_equal(existing.lat.values, simulation.lat.values)
            and np.array_equal(existing.lon.values, simulation.lon.values)
        ):
            raise ValueError(
                f"existing coverage store does not match requested simulation: {path}"
            )
        return
    coverage = xr.DataArray(
        da.zeros(
            (simulation.sizes["lat"], simulation.sizes["lon"]),
            chunks=(1, 1),
            dtype=bool,
        ),
        dims=("lat", "lon"),
        coords={"lat": simulation.lat, "lon": simulation.lon},
        name="coverage",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    coverage.to_dataset().to_zarr(
        path, mode="w", compute=False, consolidated=False, zarr_format=3
    )


def seed_adjusted_store(
    adjusted_path: Path,
    coverage_path: Path,
    source_root: Path,
    region_specs: dict[str, dict[str, object]],
    regions: list[str],
    variable: str,
) -> int:
    """Copy reusable pointwise regional adjustments into the shared store."""
    target = open_variable(adjusted_path, variable)
    existing_coverage = open_variable(coverage_path, "coverage")
    seeded_cells = 0
    for region in regions:
        source_path = source_root / region / f"{variable}_adjusted.zarr"
        if not source_path.exists():
            continue
        source = open_variable(source_path, variable)
        spec = region_specs[region]
        if bool(_region_coarse(existing_coverage, spec).all().compute()):
            continue
        target_region = _region_coarse(target, spec)
        if not (
            source.sizes == target_region.sizes
            and np.array_equal(source.time.values, target_region.time.values)
            and np.array_equal(source.lat.values, target_region.lat.values)
            and np.array_equal(source.lon.values, target_region.lon.values)
        ):
            raise ValueError(f"seed store is incompatible: {source_path}")
        region_indexers = {
            "time": slice(0, source.sizes["time"]),
            "lat": spec["coarse_lat"],
            "lon": spec["coarse_lon"],
        }
        variable_only_dataset(source).to_zarr(
            adjusted_path,
            mode="r+",
            region=region_indexers,
            consolidated=False,
        )
        coverage = xr.DataArray(
            np.ones((source.sizes["lat"], source.sizes["lon"]), dtype=bool),
            dims=("lat", "lon"),
            coords={"lat": source.lat, "lon": source.lon},
            name="coverage",
        )
        variable_only_dataset(coverage).to_zarr(
            coverage_path,
            mode="r+",
            region={"lat": spec["coarse_lat"], "lon": spec["coarse_lon"]},
            consolidated=False,
        )
        seeded_cells += source.sizes["lat"] * source.sizes["lon"]
    return seeded_cells


def ensure_spatial_valid_mask(
    *,
    model: str,
    scenario: str,
    simulation_stage: str,
    simulation_start: str | None,
    simulation_end: str | None,
    region: str,
    region_spec: dict[str, object],
    reference_root: Path,
    canonical_root: Path,
    output_root: Path,
    variables: tuple[str, ...] = DEFAULT_VARIABLES,
) -> Path:
    """Create the common fine-grid support mask used by every variable."""
    path = output_root / region / "spatial_valid_mask.zarr"
    if is_complete(path):
        existing = open_variable(path, "spatial_valid_mask")
        if existing.attrs.get("variables") == ",".join(variables):
            return path
        shutil.rmtree(path)
        success_path(path).unlink(missing_ok=True)
    if not variables:
        raise ValueError("at least one variable is required for the support mask")
    reference_support = None
    model_support = None
    fine_template = None
    for variable in variables:
        fine = _region_fine(
            open_variable(reference_root / "fine" / f"{variable}.zarr", variable),
            region_spec,
        )
        reference_coarse = _region_coarse(
            open_variable(
                reference_root / "coarse" / f"{variable}.zarr", variable
            ),
            region_spec,
        )
        model_data = open_variable(
            simulation_path(
                canonical_root, model, scenario, simulation_stage, variable
            ),
            variable,
        ).sel(lat=reference_coarse.lat, lon=reference_coarse.lon)
        model_data = select_simulation_period(
            model_data, simulation_start, simulation_end
        )
        valid_model = np.isfinite(model_data)
        if variable == "tas":
            model_temperature = convert_units_to(model_data, "K")
            valid_model = (
                valid_model
                & (model_temperature > 130)
                & (model_temperature < 377)
            )
        fine_valid = fine.notnull().all("time").compute()
        model_valid = valid_model.all("time").compute()
        reference_support = (
            fine_valid
            if reference_support is None
            else reference_support & fine_valid
        )
        model_support = (
            model_valid if model_support is None else model_support & model_valid
        )
        fine_template = fine

    assert fine_template is not None
    assert reference_support is not None
    assert model_support is not None
    lat_factor = int(region_spec.get("lat_factor", 10))
    lon_factor = int(region_spec.get("lon_factor", 10))
    expanded = np.repeat(
        np.repeat(np.asarray(model_support.values), lat_factor, axis=0),
        lon_factor,
        axis=1,
    )
    if expanded.shape != (
        fine_template.sizes["lat"],
        fine_template.sizes["lon"],
    ):
        raise ValueError("expanded model support mask does not match the fine grid")
    mask = xr.DataArray(
        expanded & np.asarray(reference_support.values),
        dims=("lat", "lon"),
        coords={"lat": fine_template.lat, "lon": fine_template.lon},
        name="spatial_valid_mask",
        attrs={
            "long_name": "common downscaling support mask",
            "definition": (
                "all requested fine reference variables and parent model variables "
                "are complete; model tas also remains within 130-377 K"
            ),
            "variables": ",".join(variables),
            "model": model,
            "scenario": scenario,
            "simulation_stage": simulation_stage,
            "simulation_period": (
                f"{simulation_start or 'start'}-{simulation_end or 'end'}"
            ),
        },
    ).chunk({"lat": 100, "lon": 100})
    path.parent.mkdir(parents=True, exist_ok=True)
    write_zarr_atomic(mask, path)
    return path


def _region_coarse(data: xr.DataArray, spec: dict[str, object]) -> xr.DataArray:
    return data.isel(lat=spec["coarse_lat"], lon=spec["coarse_lon"])


def _region_fine(data: xr.DataArray, spec: dict[str, object]) -> xr.DataArray:
    return data.isel(lat=spec["fine_lat"], lon=spec["fine_lon"])


def _context_subset(
    data: xr.DataArray,
    *,
    lat_start: int,
    lat_stop: int,
    lon_start: int,
    lon_stop: int,
    lat_halo: int,
    lon_halo: int,
    periodic_lon: bool,
) -> tuple[xr.DataArray, slice, slice]:
    """Select a haloed tile and return slices locating its unhaloed center."""
    context_lat_start = max(lat_start - lat_halo, 0)
    context_lat_stop = min(lat_stop + lat_halo, data.sizes["lat"])
    lat_center = slice(
        lat_start - context_lat_start,
        lat_stop - context_lat_start,
    )

    raw_lon = np.arange(lon_start - lon_halo, lon_stop + lon_halo)
    if periodic_lon:
        indices = raw_lon % data.sizes["lon"]
        subset = data.isel(
            lat=slice(context_lat_start, context_lat_stop), lon=indices
        )
        periods = np.floor_divide(raw_lon, data.sizes["lon"])
        subset = subset.assign_coords(
            lon=np.asarray(subset.lon.values, dtype=np.float64) + 360 * periods
        )
        lon_center = slice(lon_halo, lon_halo + lon_stop - lon_start)
    else:
        context_lon_start = max(lon_start - lon_halo, 0)
        context_lon_stop = min(lon_stop + lon_halo, data.sizes["lon"])
        subset = data.isel(
            lat=slice(context_lat_start, context_lat_stop),
            lon=slice(context_lon_start, context_lon_stop),
        )
        lon_center = slice(
            lon_start - context_lon_start,
            lon_stop - context_lon_start,
        )
    return subset, lat_center, lon_center


def global_tile_contexts(
    adjusted: xr.DataArray,
    fine_reference: xr.DataArray,
    region_spec: dict[str, object],
    global_spec: dict[str, object],
    tile: dict[str, int],
) -> tuple[xr.DataArray, xr.DataArray, slice, slice]:
    """Read a target tile plus halos from global-coordinate input stores."""
    lat_factor = int(region_spec.get("lat_factor", 10))
    lon_factor = int(region_spec.get("lon_factor", 10))
    coarse_lat_start = region_spec["coarse_lat"].start + tile["coarse_lat_start"]
    coarse_lat_stop = region_spec["coarse_lat"].start + tile["coarse_lat_stop"]
    coarse_lon_start = region_spec["coarse_lon"].start + tile["coarse_lon_start"]
    coarse_lon_stop = region_spec["coarse_lon"].start + tile["coarse_lon_stop"]
    periodic_lon = bool(global_spec.get("periodic_lon", False))
    simulation, _, _ = _context_subset(
        adjusted,
        lat_start=coarse_lat_start,
        lat_stop=coarse_lat_stop,
        lon_start=coarse_lon_start,
        lon_stop=coarse_lon_stop,
        lat_halo=1,
        lon_halo=1,
        periodic_lon=periodic_lon,
    )

    fine_lat_start = region_spec["fine_lat"].start + tile["fine_lat_start"]
    fine_lat_stop = region_spec["fine_lat"].start + tile["fine_lat_stop"]
    fine_lon_start = region_spec["fine_lon"].start + tile["fine_lon_start"]
    fine_lon_stop = region_spec["fine_lon"].start + tile["fine_lon_stop"]
    observations, lat_center, lon_center = _context_subset(
        fine_reference,
        lat_start=fine_lat_start,
        lat_stop=fine_lat_stop,
        lon_start=fine_lon_start,
        lon_stop=fine_lon_stop,
        lat_halo=lat_factor,
        lon_halo=lon_factor,
        periodic_lon=periodic_lon,
    )
    return simulation, observations, lat_center, lon_center


def run_adjustment_tile(
    *,
    model: str,
    scenario: str,
    simulation_stage: str,
    simulation_start: str | None,
    simulation_end: str | None,
    global_spec: dict[str, object],
    variable: str,
    tile: dict[str, int],
    reference_root: str,
    canonical_root: str,
    adjusted_root: str,
    coverage_path: str,
    threads_per_worker: int,
) -> dict[str, object]:
    configure_worker_runtime(threads_per_worker)
    started = time.perf_counter()
    reference = Path(reference_root)
    canonical = Path(canonical_root)
    adjusted_root_path = Path(adjusted_root)
    adjusted_path = adjusted_store_path(
        adjusted_root_path, model, scenario, simulation_stage, variable
    )
    tile_marker = adjustment_marker_path(
        adjusted_root_path,
        model,
        scenario,
        simulation_stage,
        variable,
        tile,
    )
    if tile_complete(tile_marker):
        return {
            "variable": variable,
            "tile": _tile_name(tile),
            "stage": "adjustment",
            "skipped": True,
        }

    obs_region = _region_coarse(
        open_variable(reference / "coarse" / f"{variable}.zarr", variable),
        global_spec,
    )
    local_lat_start = tile["coarse_lat_start"]
    local_lat_stop = tile["coarse_lat_stop"]
    local_lon_start = tile["coarse_lon_start"]
    local_lon_stop = tile["coarse_lon_stop"]
    obs_coarse = obs_region.isel(
        lat=slice(local_lat_start, local_lat_stop),
        lon=slice(local_lon_start, local_lon_stop),
    )
    historical = (
        open_variable(
            canonical / model / "historical" / "hist" / f"{variable}.zarr",
            variable,
        )
        .sel(time=slice("1993", "2014"))
        .sel(lat=obs_coarse.lat, lon=obs_coarse.lon)
    )
    simulation = open_variable(
        simulation_path(
            canonical, model, scenario, simulation_stage, variable
        ),
        variable,
    ).sel(lat=obs_coarse.lat, lon=obs_coarse.lon)
    simulation = select_simulation_period(
        simulation, simulation_start, simulation_end
    )
    adjusted = adjust_variable(
        obs_coarse,
        historical,
        simulation,
        variable=variable,
        chunks={"lat": 1, "lon": 1},
    )
    variable_only_dataset(adjusted).to_zarr(
        adjusted_path,
        mode="r+",
        region={
            "time": slice(0, adjusted.sizes["time"]),
            "lat": slice(local_lat_start, local_lat_stop),
            "lon": slice(local_lon_start, local_lon_stop),
        },
        consolidated=False,
    )
    tile_marker.parent.mkdir(parents=True, exist_ok=True)
    tile_marker.touch()
    written_tile = open_variable(adjusted_path, variable).isel(
        lat=slice(local_lat_start, local_lat_stop),
        lon=slice(local_lon_start, local_lon_stop),
    )
    report = validate_variable(
        written_tile,
        variable,
        min_valid_fraction=0.95,
        statistical=False,
        allow_out_of_bounds_hurs=variable == "hurs",
    )
    if not report.valid:
        raise RuntimeError(
            f"{variable} global {_tile_name(tile)} adjustment QC failed: "
            f"{report.errors}"
        )
    record = {
        "model": model,
        "scenario": scenario,
        "simulation_stage": simulation_stage,
        "simulation_start": simulation_start,
        "simulation_end": simulation_end,
        "region": "global-context",
        "description": global_spec["description"],
        "variable": variable,
        "tile": _tile_name(tile),
        "stage": "adjustment",
        "path": str(adjusted_path),
        "valid": report.valid,
        "minimum": report.minimum,
        "maximum": report.maximum,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (tile_marker.with_suffix(".report.json")).write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    coverage = xr.DataArray(
        np.ones(
            (local_lat_stop - local_lat_start, local_lon_stop - local_lon_start),
            dtype=bool,
        ),
        dims=("lat", "lon"),
        coords={"lat": adjusted.lat, "lon": adjusted.lon},
        name="coverage",
    )
    variable_only_dataset(coverage).to_zarr(
        Path(coverage_path),
        mode="r+",
        region={
            "lat": slice(local_lat_start, local_lat_stop),
            "lon": slice(local_lon_start, local_lon_stop),
        },
        consolidated=False,
    )
    return record


def _tile_edges(size: int, width: int) -> list[tuple[int, int]]:
    if width < 1:
        raise ValueError("tile dimensions must be at least one coarse cell")
    edges = [(start, min(start + width, size)) for start in range(0, size, width)]
    if width > 1 and len(edges) > 1 and edges[-1][1] - edges[-1][0] < 2:
        previous = edges[-2]
        edges[-2:] = [(previous[0], edges[-1][1])]
    return edges


def tile_specs(
    spec: dict[str, object], tile_lat_degrees: int, tile_lon_degrees: int
) -> list[dict[str, int]]:
    """Return disjoint local coarse/fine regions for a nested-grid domain."""
    coarse_lat = spec["coarse_lat"]
    coarse_lon = spec["coarse_lon"]
    coarse_lat_size = coarse_lat.stop - coarse_lat.start
    coarse_lon_size = coarse_lon.stop - coarse_lon.start
    lat_factor = int(spec.get("lat_factor", 10))
    lon_factor = int(spec.get("lon_factor", 10))
    return [
        {
            "coarse_lat_start": lat_start,
            "coarse_lat_stop": lat_stop,
            "coarse_lon_start": lon_start,
            "coarse_lon_stop": lon_stop,
            "fine_lat_start": lat_start * lat_factor,
            "fine_lat_stop": lat_stop * lat_factor,
            "fine_lon_start": lon_start * lon_factor,
            "fine_lon_stop": lon_stop * lon_factor,
        }
        for lat_start, lat_stop in _tile_edges(coarse_lat_size, tile_lat_degrees)
        for lon_start, lon_stop in _tile_edges(coarse_lon_size, tile_lon_degrees)
    ]


def _tile_name(tile: dict[str, int]) -> str:
    return (
        f"lat{tile['coarse_lat_start']:03d}-{tile['coarse_lat_stop']:03d}_"
        f"lon{tile['coarse_lon_start']:03d}-{tile['coarse_lon_stop']:03d}"
    )


def _legacy_tile_marker(
    output_root: Path,
    region: str,
    variable: str,
    tile: dict[str, int],
    region_spec: dict[str, object],
    *,
    adjustment: bool = False,
) -> Path | None:
    """Locate markers written by the original Europe longitude-only runner."""
    coarse_lat = region_spec["coarse_lat"]
    full_latitude = (
        tile["coarse_lat_start"] == 0
        and tile["coarse_lat_stop"] == coarse_lat.stop - coarse_lat.start
    )
    if region == "global" or not full_latitude:
        return None
    coarse_lon = region_spec["coarse_lon"]
    start = coarse_lon.start + tile["coarse_lon_start"]
    stop = coarse_lon.start + tile["coarse_lon_stop"]
    state = "state_adjusted" if adjustment else "state"
    return (
        output_root
        / region
        / state
        / variable
        / f"lon{start:03d}-{stop:03d}.success"
    )


def _tile_already_written(
    output_root: Path,
    region: str,
    variable: str,
    tile: dict[str, int],
    region_spec: dict[str, object],
    *,
    adjustment: bool = False,
) -> bool:
    if not adjustment:
        return tile_complete(marker_path(output_root, region, variable, tile))
    if adjustment:
        current = (
            output_root
            / region
            / "state_adjusted"
            / variable
            / f"{_tile_name(tile)}.success"
        )
    legacy = _legacy_tile_marker(
        output_root,
        region,
        variable,
        tile,
        region_spec,
        adjustment=adjustment,
    )
    current_complete = tile_complete(current)
    legacy_complete = bool(legacy and tile_complete(legacy))
    return current_complete or legacy_complete


def marker_path(
    output_root: Path, region: str, variable: str, tile: dict[str, int]
) -> Path:
    return (
        output_root
        / region
        / "state_spatial_global_context"
        / variable
        / f"{_tile_name(tile)}.success"
    )


def report_path(marker: Path) -> Path:
    return marker.with_suffix(".report.json")


def tile_complete(marker: Path) -> bool:
    return marker.exists() and report_path(marker).exists()


def tile_written(marker: Path) -> bool:
    return marker.exists()


def configure_worker_runtime(threads_per_worker: int) -> None:
    if threads_per_worker < 1:
        raise ValueError("threads_per_worker must be at least one")
    scheduler = "threads" if threads_per_worker > 1 else "synchronous"
    dask.config.set(scheduler=scheduler, num_workers=threads_per_worker)


def quick_tile_qc(
    written_tile: xr.DataArray,
    reference_tile: xr.DataArray,
    variable: str,
    *,
    min_valid_fraction: float = 0.95,
) -> dict[str, object]:
    qc_steps = min(written_tile.sizes["time"], 365)
    written_sample = written_tile.isel(time=slice(0, qc_steps))
    finite = np.isfinite(written_sample)
    valid_count = finite.sum("time")
    active = valid_count > 0
    partial = active & (valid_count / qc_steps < min_valid_fraction)
    reference_sample = reference_tile.isel(
        time=slice(0, min(reference_tile.sizes["time"], 366))
    )
    reference_active = reference_sample.notnull().any("time")
    written_active = written_sample.notnull().any("time")
    (
        active_cells,
        partial_cells,
        missing_reference_cells,
        extra_cells,
        has_inf,
        minimum,
        maximum,
    ) = dask.compute(
        active.sum(),
        partial.sum(),
        (reference_active & ~written_active).sum(),
        (written_active & ~reference_active).sum(),
        np.isinf(written_sample).any(),
        written_sample.min(skipna=True),
        written_sample.max(skipna=True),
    )
    minimum_value = float(minimum)
    maximum_value = float(maximum)
    errors: list[str] = []
    if bool(has_inf):
        errors.append("variable contains infinite values")
    if int(partial_cells):
        errors.append(
            f"{int(partial_cells)} active spatial cells are below the required "
            f"{min_valid_fraction:.3f} valid fraction"
        )
    if int(missing_reference_cells):
        errors.append(
            f"{int(missing_reference_cells)} fine reference-land cells are missing "
            "from the downscaled tile"
        )
    if int(extra_cells):
        errors.append(
            f"{int(extra_cells)} downscaled cells are active where reference is missing"
        )
    if variable in {"pr", "sfcWind"} and minimum_value < 0:
        errors.append(f"{variable} minimum is below zero ({minimum_value})")
    if variable == "pr":
        ceiling = float(
            convert_units_to(
                CIL_PRECIPITATION_CEILING, written_sample, context="hydro"
            )
        )
        if maximum_value > ceiling:
            errors.append(
                f"pr exceeds CIL precipitation ceiling {CIL_PRECIPITATION_CEILING} "
                f"({maximum_value})"
            )
    if variable == "tas":
        lower = float(convert_units_to(CIL_TEMPERATURE_VALID_RANGE[0], written_sample))
        upper = float(convert_units_to(CIL_TEMPERATURE_VALID_RANGE[1], written_sample))
        static_floor_cells = int(((written_sample == 150).all("time")).sum().compute())
        if minimum_value < lower or maximum_value > upper:
            errors.append(
                f"tas violates CIL validation range {CIL_TEMPERATURE_VALID_RANGE} "
                f"({minimum_value}, {maximum_value})"
            )
        if static_floor_cells:
            errors.append(
                f"{static_floor_cells} cells are static at the MBCnSD tas floor"
            )
    if variable == "hurs" and (minimum_value < 0 or maximum_value > 100):
        errors.append(
            f"hurs is outside [0, 100] ({minimum_value}, {maximum_value})"
        )
    return {
        "valid": not errors,
        "active_cells": int(active_cells),
        "partial_missing_cells": int(partial_cells),
        "missing_reference_cells": int(missing_reference_cells),
        "extra_cells": int(extra_cells),
        "minimum": minimum_value,
        "maximum": maximum_value,
        "errors": tuple(errors),
        "qc_time_steps": qc_steps,
    }


def apply_static_sentinel_mask_to_region(
    output_root: Path, region: str
) -> dict[str, object]:
    """Mask all downscaled variables where tas is static at the MBCnSD floor."""
    common_mask = output_root / region / "spatial_valid_mask.zarr"
    if is_complete(common_mask):
        return {
            "region": region,
            "skipped": True,
            "reason": "common spatial validity mask was applied during tile writes",
        }
    tas_path = output_root / region / "tas_downscaled.zarr"
    if not tas_path.exists():
        return {"region": region, "skipped": True, "reason": "tas store is missing"}

    tas = open_variable(tas_path, "tas")
    static_floor = (tas == 150).all("time").compute()
    cells = int(static_floor.sum())
    report: dict[str, object] = {"region": region, "static_tas_floor_cells": cells}
    if cells == 0:
        return report

    for variable in SUPPORTED_VARIABLES:
        path = output_root / region / f"{variable}_downscaled.zarr"
        if not path.exists():
            continue
        data = open_variable(path, variable)
        cleaned = data.where(~static_floor)
        cleaned.attrs.update(
            data.attrs,
            static_temperature_floor_mask_source="tas == 150 K for all time steps",
            static_temperature_floor_cells_masked=cells,
        )
        temporary = path.with_name(f"{path.name}.static-mask")
        write_zarr_atomic(cleaned, temporary)
        success_path(temporary).unlink(missing_ok=True)
        shutil.rmtree(path)
        temporary.rename(path)
        success_path(path).touch()
        report[variable] = str(path)
    return report


def run_tile(
    *,
    model: str,
    scenario: str,
    region: str,
    region_spec: dict[str, object],
    variable: str,
    tile: dict[str, int],
    reference_root: str,
    adjusted_path: str,
    global_spec: dict[str, object],
    output_root: str,
    iterations: int,
    quantiles: int,
    threads_per_worker: int,
) -> dict[str, object]:
    configure_worker_runtime(threads_per_worker)
    started = time.perf_counter()
    reference = Path(reference_root)
    output = Path(output_root)
    tile_marker = marker_path(output, region, variable, tile)
    downscaled_path = output / region / f"{variable}_downscaled.zarr"
    if tile_complete(tile_marker):
        return {
            "variable": variable,
            "region": region,
            "tile": _tile_name(tile),
            "skipped": True,
        }

    adjusted = open_variable(Path(adjusted_path), variable)
    lat_factor = int(region_spec.get("lat_factor", 10))
    lon_factor = int(region_spec.get("lon_factor", 10))
    local_lat_start = tile["fine_lat_start"]
    local_lat_stop = tile["fine_lat_stop"]
    local_lon_start = tile["fine_lon_start"]
    local_lon_stop = tile["fine_lon_stop"]
    obs_fine_global = open_variable(
        reference / "fine" / f"{variable}.zarr", variable
    )
    sim, obs_fine_context, fine_lat_center, fine_lon_center = global_tile_contexts(
        adjusted,
        obs_fine_global,
        region_spec,
        global_spec,
        tile,
    )
    obs_fine = obs_fine_context.isel(
        lat=fine_lat_center,
        lon=fine_lon_center,
    )
    spatial_mask = open_variable(
        output / region / "spatial_valid_mask.zarr", "spatial_valid_mask"
    ).isel(
        lat=slice(local_lat_start, local_lat_stop),
        lon=slice(local_lon_start, local_lon_stop),
    )
    obs_fine = obs_fine.where(spatial_mask)
    if not tile_complete(tile_marker):
        with (
            dask.config.set({"array.rechunk.method": "tasks"}),
            warnings.catch_warnings(),
        ):
            warnings.filterwarnings(
                "ignore", message="invalid value encountered in divide"
            )
            downscaled = downscale_variable(
                obs_fine_context,
                sim,
                variable=variable,
                iterations=iterations,
                quantiles=quantiles,
                chunks={"lat": lat_factor, "lon": lon_factor},
            )
            downscaled = downscaled.isel(
                lat=fine_lat_center,
                lon=fine_lon_center,
            )
            downscaled = apply_downscaled_value_controls(downscaled, variable)
            downscaled = downscaled.where(spatial_mask)
            variable_only_dataset(downscaled).to_zarr(
                downscaled_path,
                mode="r+",
                region={
                    "time": slice(0, downscaled.sizes["time"]),
                    "lat": slice(local_lat_start, local_lat_stop),
                    "lon": slice(local_lon_start, local_lon_stop),
                },
                consolidated=False,
            )
        tile_marker.parent.mkdir(parents=True, exist_ok=True)
        tile_marker.touch()

    written_tile = open_variable(downscaled_path, variable).isel(
        lat=slice(local_lat_start, local_lat_stop),
        lon=slice(local_lon_start, local_lon_stop),
    )
    qc = quick_tile_qc(written_tile, obs_fine, variable)
    active_cells = qc["active_cells"]
    if active_cells:
        conservation = {
            "valid": True,
            "not_applicable": True,
            "reason": (
                "per-tile conservation skipped for independently haloed tiles"
            ),
            "units": sim.attrs.get("units", ""),
        }
        valid = bool(qc["valid"])
        minimum = qc["minimum"]
        maximum = qc["maximum"]
        errors = qc["errors"]
    else:
        conservation = {
            "valid": True,
            "not_applicable": True,
            "reason": "tile has no active fine-grid cells",
            "units": sim.attrs.get("units", ""),
        }
        valid = True
        minimum = None
        maximum = None
        errors = ()
    if not valid:
        raise RuntimeError(
            f"{variable} {region} {_tile_name(tile)} QC failed: {errors}"
        )
    record = {
        "model": model,
        "scenario": scenario,
        "region": region,
        "description": region_spec["description"],
        "variable": variable,
        "tile": _tile_name(tile),
        "path": str(downscaled_path),
        "valid": valid,
        "active_cells": active_cells,
        "missing_reference_cells": qc["missing_reference_cells"],
        "extra_cells": qc["extra_cells"],
        "partial_missing_cells": qc["partial_missing_cells"],
        "qc_time_steps": qc["qc_time_steps"],
        "minimum": minimum,
        "maximum": maximum,
        "conservation": conservation,
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path(tile_marker).write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="ACCESS-CM2")
    parser.add_argument("--scenario", default="ssp245")
    parser.add_argument(
        "--simulation-stage",
        choices=("hist", "proj"),
        default=None,
        help="canonical input stage; defaults to hist for historical and proj otherwise",
    )
    parser.add_argument("--simulation-start", default=None)
    parser.add_argument("--simulation-end", default=None)
    parser.add_argument(
        "--regions",
        nargs="+",
        choices=(*REGIONS, "global"),
        default=list(REGIONS),
        help="use 'global' for the complete domain available in the reference stores",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        choices=SUPPORTED_VARIABLES,
        default=list(DEFAULT_VARIABLES),
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path("/data1/era5ref-europe-full"),
    )
    parser.add_argument(
        "--canonical-root", type=Path, default=Path("/data1/cmip6_fwi_1deg")
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--adjusted-root",
        type=Path,
        default=Path("/data1/cmip6_bias_adjusted_1deg"),
        help="shared global-coordinate 1-degree bias-adjusted Zarr products",
    )
    parser.add_argument(
        "--seed-adjusted-from",
        type=Path,
        default=None,
        help="reuse compatible regional *_adjusted.zarr stores before filling halos",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("adjust", "spatial"),
        default=("adjust", "spatial"),
        help="run coarse bias adjustment, spatial downscaling, or both",
    )
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--quantiles", type=int, default=50)
    parser.add_argument(
        "--tile-lat-degrees",
        type=int,
        default=None,
        help=(
            "coarse latitude cells per tile; defaults to 5 globally or the full "
            "regional height"
        ),
    )
    parser.add_argument("--tile-lon-degrees", type=int, default=2)
    parser.add_argument("--tile-workers", type=int, default=16)
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        default=1,
        help="Dask threads inside each tile process; total slots are workers times threads",
    )
    args = parser.parse_args()

    simulation_stage = args.simulation_stage or (
        "hist" if args.scenario == "historical" else "proj"
    )
    args.output_root = args.output_root or default_output_root(
        args.model, args.scenario, simulation_stage, list(args.regions)
    )
    simulation_start = args.simulation_start
    simulation_end = args.simulation_end
    if args.scenario == "historical":
        simulation_start = simulation_start or "1993"
        simulation_end = simulation_end or "2014"
    if args.tile_workers < 1 or args.threads_per_worker < 1:
        parser.error("tile workers and threads per worker must both be positive")

    manifest_records = []
    region_specs = resolve_regions(
        args.reference_root, list(args.regions), args.variables[0]
    )
    global_spec = resolve_regions(
        args.reference_root, ["global"], args.variables[0]
    )["global"]
    adjustment_tile_lat = args.tile_lat_degrees or 5
    target_specs = [region_specs[region] for region in args.regions]
    required_cells = required_adjustment_mask(global_spec, target_specs)
    regular_adjustment_tiles = required_adjustment_tiles(
        global_spec,
        target_specs,
        adjustment_tile_lat,
        args.tile_lon_degrees,
    )

    for variable in args.variables:
        adjusted_path = adjusted_store_path(
            args.adjusted_root,
            args.model,
            args.scenario,
            simulation_stage,
            variable,
        )
        coverage_path = adjusted_path.with_name(f"{variable}.coverage.zarr")
        if "adjust" in args.stages:
            reference_coarse = open_variable(
                args.reference_root / "coarse" / f"{variable}.zarr", variable
            )
            simulation = open_variable(
                simulation_path(
                    args.canonical_root,
                    args.model,
                    args.scenario,
                    simulation_stage,
                    variable,
                ),
                variable,
            ).sel(lat=reference_coarse.lat, lon=reference_coarse.lon)
            simulation = select_simulation_period(
                simulation, simulation_start, simulation_end
            )
            initialize_adjusted_store(simulation, adjusted_path)
            initialize_coverage_store(simulation, coverage_path)
            if args.seed_adjusted_from is not None:
                seeded = seed_adjusted_store(
                    adjusted_path,
                    coverage_path,
                    args.seed_adjusted_from,
                    region_specs,
                    list(args.regions),
                    variable,
                )
                print(f"SEEDED {variable} adjusted cells: {seeded}", flush=True)
            coverage = open_variable(coverage_path, "coverage").compute().values
            missing_cells = required_cells & ~coverage
            if args.seed_adjusted_from is None and not coverage.any():
                pending_adjustment = regular_adjustment_tiles
            else:
                pending_adjustment = missing_cell_tiles(
                    missing_cells, args.tile_lon_degrees
                )
            print(
                f"START {variable} global-context adjustment tiles: "
                f"{len(pending_adjustment)}; missing cells: "
                f"{int(missing_cells.sum())}",
                flush=True,
            )
            with ProcessPoolExecutor(max_workers=args.tile_workers) as executor:
                futures = [
                    executor.submit(
                        run_adjustment_tile,
                        model=args.model,
                        scenario=args.scenario,
                        simulation_stage=simulation_stage,
                        simulation_start=simulation_start,
                        simulation_end=simulation_end,
                        global_spec=global_spec,
                        variable=variable,
                        tile=tile,
                        reference_root=str(args.reference_root),
                        canonical_root=str(args.canonical_root),
                        adjusted_root=str(args.adjusted_root),
                        coverage_path=str(coverage_path),
                        threads_per_worker=args.threads_per_worker,
                    )
                    for tile in pending_adjustment
                ]
                for index, future in enumerate(as_completed(futures), start=1):
                    record = future.result()
                    manifest_records.append(record)
                    print(
                        f"DONE {variable} global-context adjustment tile "
                        f"{index}/{len(pending_adjustment)}: {record.get('tile')}",
                        flush=True,
                    )
        elif not adjusted_path.exists():
            parser.error(f"shared adjusted store does not exist: {adjusted_path}")
        if "spatial" in args.stages:
            if not coverage_path.exists():
                parser.error(f"adjustment coverage store does not exist: {coverage_path}")
            coverage = open_variable(coverage_path, "coverage").compute().values
            missing_context = int((required_cells & ~coverage).sum())
            if missing_context:
                parser.error(
                    f"{variable} is missing {missing_context} adjusted "
                    "global-context cells; run the adjust stage first"
                )

    for region in args.regions:
        region_spec = region_specs[region]
        if "spatial" not in args.stages:
            continue
        coarse_lat = region_spec["coarse_lat"]
        tile_lat_degrees = args.tile_lat_degrees or (
            5 if region == "global" else coarse_lat.stop - coarse_lat.start
        )
        mask_path = ensure_spatial_valid_mask(
            model=args.model,
            scenario=args.scenario,
            simulation_stage=simulation_stage,
            simulation_start=simulation_start,
            simulation_end=simulation_end,
            region=region,
            region_spec=region_spec,
            reference_root=args.reference_root,
            canonical_root=args.canonical_root,
            output_root=args.output_root,
            variables=tuple(args.variables),
        )
        manifest_records.append({"spatial_valid_mask": str(mask_path)})
        for variable in args.variables:
            tiles = tile_specs(
                region_spec, tile_lat_degrees, args.tile_lon_degrees
            )
            adjusted_path = adjusted_store_path(
                args.adjusted_root,
                args.model,
                args.scenario,
                simulation_stage,
                variable,
            )
            adjusted = open_variable(adjusted_path, variable)
            fine_reference = _region_fine(
                open_variable(
                    args.reference_root / "fine" / f"{variable}.zarr", variable
                ),
                region_spec,
            )
            initialize_output_store(
                adjusted,
                fine_reference,
                args.output_root / region / f"{variable}_downscaled.zarr",
                iterations=args.iterations,
                quantiles=args.quantiles,
            )
            pending = [
                tile
                for tile in tiles
                if not _tile_already_written(
                    args.output_root, region, variable, tile, region_spec
                )
            ]
            print(
                f"START {variable} {region} MBCnSD tiles: {len(pending)}/{len(tiles)}",
                flush=True,
            )
            with ProcessPoolExecutor(max_workers=args.tile_workers) as executor:
                futures = [
                    executor.submit(
                        run_tile,
                        model=args.model,
                        scenario=args.scenario,
                        region=region,
                        region_spec=region_spec,
                        variable=variable,
                        tile=tile,
                        reference_root=str(args.reference_root),
                        adjusted_path=str(adjusted_path),
                        global_spec=global_spec,
                        output_root=str(args.output_root),
                        iterations=args.iterations,
                        quantiles=args.quantiles,
                        threads_per_worker=args.threads_per_worker,
                    )
                    for tile in pending
                ]
                for index, future in enumerate(as_completed(futures), start=1):
                    record = future.result()
                    manifest_records.append(record)
                    print(
                        f"DONE {variable} {region} tile {index}/{len(pending)}: "
                        f"{record.get('tile')}",
                        flush=True,
                    )

    if "spatial" in args.stages:
        for region in args.regions:
            record = apply_static_sentinel_mask_to_region(args.output_root, region)
            manifest_records.append({"finalize_static_sentinels": record})
            print(f"DONE {region} static sentinel finalization: {record}", flush=True)

    manifest = {
        "model": args.model,
        "scenario": args.scenario,
        "simulation_stage": simulation_stage,
        "simulation_start": simulation_start,
        "simulation_end": simulation_end,
        "stages": list(args.stages),
        "adjusted_root": str(args.adjusted_root),
        "spatial_context": "global 1-degree halo, cropped after MBCnSD",
        "regions": {key: region_specs[key]["description"] for key in args.regions},
        "tile_lat_degrees": args.tile_lat_degrees,
        "tile_lon_degrees": args.tile_lon_degrees,
        "tile_workers": args.tile_workers,
        "threads_per_worker": args.threads_per_worker,
        "execution_slots": args.tile_workers * args.threads_per_worker,
        "records": manifest_records,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "downscale-tiles-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
