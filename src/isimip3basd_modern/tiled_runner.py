"""Regional or global MBCnSD as restartable two-dimensional tiles.

The ``scripts/run_*_downscale_tiles.py`` entry points are thin wrappers around
:func:`main`. Tile workers are spawned processes that import this module by
name, so it has to live in the package rather than beside the scripts.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from concurrent.futures import Executor, ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict
import hashlib
import json
import multiprocessing
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
import zarr

warnings.filterwarnings("ignore", message="invalid value encountered in divide")
warnings.filterwarnings(
    "ignore", message="All-nan slice encountered in interp_on_quantiles"
)
warnings.filterwarnings("ignore", message="Increasing number of chunks by factor")
warnings.filterwarnings(
    "ignore", message="QDM method can now perform the adjustment step"
)
warnings.filterwarnings("ignore", message="keys will default to True in jsonpickle")

from . import __version__
from .downscaling import (
    CIL_PRECIPITATION_CEILING,
    CIL_TEMPERATURE_VALID_RANGE,
    DOWNSCALING_BOUNDS,
    apply_downscaled_value_controls,
    downscale_variable,
)
from .pipeline import adjust_variable, bias_adjustment_metadata
from .presets import get_preset
from .publication import packing_encoding
from .validation import validate_variable


DEFAULT_VARIABLES = ("hurs", "pr", "sfcWind", "tas")
# Bias-adjustment fits are trained on this historical window. It is part of
# the fit-cache fingerprint, so both uses must come from the same constant.
TRAINING_PERIOD = ("1993", "2014")
# Nesting factor assumed for the named regions, which predate inferred factors.
DEFAULT_DOWNSCALING_FACTOR = 10
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
    "socal": {
        "fine_lat": slice(880, 940),
        "fine_lon": slice(2380, 2470),
        "coarse_lat": slice(88, 94),
        "coarse_lon": slice(238, 247),
        "description": "31.05-36.95N, 121.95-113.05W",
    },
    "spokane": {
        "fine_lat": slice(1020, 1070),
        "fine_lon": slice(2390, 2460),
        "coarse_lat": slice(102, 107),
        "coarse_lon": slice(239, 246),
        "description": "45.05-49.95N, 120.95-114.05W",
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


def _digest_optional_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def adjustment_fit_cache_key(
    *,
    model: str,
    variable: str,
    tile: dict[str, int],
    reference_root: Path,
    canonical_root: Path,
    quantiles: int,
) -> str:
    """Fingerprint the inputs and settings that define one trained fit."""
    reference_store = reference_root / "coarse" / f"{variable}.zarr"
    historical_store = (
        canonical_root / model / "historical" / "hist" / f"{variable}.zarr"
    )
    payload = {
        "schema": 1,
        "model": model,
        "variable": variable,
        "tile": {key: int(value) for key, value in sorted(tile.items())},
        "training_period": list(TRAINING_PERIOD),
        "preset": asdict(get_preset(variable)),
        "quantiles": quantiles,
        "reference_root": str(reference_root.resolve()),
        "historical_store": str(historical_store.resolve()),
        "reference_manifest": _digest_optional_file(
            reference_root / "reference-manifest.json"
        ),
        "reference_group_metadata": _digest_optional_file(
            reference_store / "zarr.json"
        ),
        "reference_array_metadata": _digest_optional_file(
            reference_store / variable / "zarr.json"
        ),
        "historical_group_metadata": _digest_optional_file(
            historical_store / "zarr.json"
        ),
        "historical_array_metadata": _digest_optional_file(
            historical_store / variable / "zarr.json"
        ),
        "isimip3basd_modern": __version__,
        "xsdba": version("xsdba"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def adjustment_fit_cache_path(
    root: Path, model: str, variable: str, tile: dict[str, int]
) -> Path:
    return root / model / variable / f"{_tile_name(tile)}.zarr"


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
        return (
            Path("/nas/dat1/cmip6_fwi/processing/downscaled_0p1deg")
            / model
            / scenario
            / simulation_stage
        )
    return Path("/data1/access_europe_downscale_full")


def success_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.success")


def is_complete(path: Path) -> bool:
    return success_path(path).exists()


def write_zarr_atomic(
    data: xr.DataArray, path: Path, *, packed: bool = False
) -> None:
    partial = path.with_name(f"{path.name}.partial")
    shutil.rmtree(partial, ignore_errors=True)
    data.to_dataset().to_zarr(
        partial,
        mode="w",
        consolidated=False,
        zarr_format=3,
        encoding={data.name: packing_encoding(data.name)} if packed else None,
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


def write_adjusted_region(
    adjusted: xr.DataArray,
    path: Path,
    region: dict[str, slice],
) -> None:
    """Write one disjoint adjustment tile after aligning it to store chunks."""
    variable_only_dataset(adjusted).to_zarr(
        path,
        mode="r+",
        region=region,
        consolidated=False,
        align_chunks=True,
    )


def initialize_output_store(
    adjusted: xr.DataArray,
    fine_reference: xr.DataArray,
    path: Path,
    *,
    iterations: int,
    quantiles: int,
    invalidate: Callable[[], None] | None = None,
    spatial_chunks: tuple[int, int] = (
        DEFAULT_DOWNSCALING_FACTOR,
        DEFAULT_DOWNSCALING_FACTOR,
    ),
) -> bool:
    expected_revision = get_preset(adjusted.name).revision
    adjustment_attrs = {
        **adjusted.attrs,
        **bias_adjustment_metadata(adjusted.name, quantiles=quantiles),
    }
    if path.exists():
        physical_dtype = zarr.open_group(path, mode="r")[adjusted.name].dtype
        if physical_dtype != np.dtype("int16"):
            raise ValueError(
                f"existing output is {physical_dtype}, expected scaled int16: {path}"
            )
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
        stored_revision = int(
            existing.attrs.get("bias_adjustment_preset_revision", 1)
        )
        stale = stored_revision != expected_revision
        if stale and invalidate is not None:
            # Drop checkpoints before recording the new revision. Stamping
            # first would let a crash in between leave stale tiles looking
            # complete under the current revision.
            invalidate()
        zarr.open_group(path, mode="a")[adjusted.name].attrs.update(
            adjustment_attrs
        )
        return stale
    path.parent.mkdir(parents=True, exist_ok=True)
    dims = ("time", "lat", "lon")
    shape = (
        adjusted.sizes["time"],
        fine_reference.sizes["lat"],
        fine_reference.sizes["lon"],
    )
    # One stored chunk per coarse cell: tiles are coarse-cell aligned, so
    # concurrent tile workers can never write into the same chunk.
    chunks = (adjusted.sizes["time"], *spatial_chunks)
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
            **adjustment_attrs,
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
            "storage_format": "scaled int16 Zarr v3",
            "storage_compressor": "Blosc Zstd level 3 with bitshuffle",
        },
    )
    template.to_dataset().to_zarr(
        path,
        mode="w",
        compute=False,
        consolidated=False,
        zarr_format=3,
        encoding={adjusted.name: packing_encoding(adjusted.name)},
    )
    return False


def initialize_adjusted_store(
    simulation: xr.DataArray,
    path: Path,
    *,
    quantiles: int = 50,
    spatial_chunks: tuple[int, int] = (1, 1),
    invalidate: Callable[[], None] | None = None,
) -> bool:
    expected_revision = get_preset(simulation.name).revision
    adjustment_attrs = {
        **simulation.attrs,
        **bias_adjustment_metadata(simulation.name, quantiles=quantiles),
    }
    if path.exists():
        existing = open_variable(path, simulation.name)
        if dict(existing.sizes) != dict(simulation.sizes) or any(
            not np.array_equal(existing[dim].values, simulation[dim].values)
            for dim in simulation.dims
        ):
            raise ValueError(
                f"existing adjusted store does not match requested simulation: {path}"
            )
        stored_revision = int(
            existing.attrs.get("bias_adjustment_preset_revision", 1)
        )
        stale = stored_revision != expected_revision
        if stale and invalidate is not None:
            # Drop checkpoints before recording the new revision. Stamping
            # first would let a crash in between leave stale tiles looking
            # complete under the current revision.
            invalidate()
        zarr.open_group(path, mode="a")[simulation.name].attrs.update(
            adjustment_attrs
        )
        return stale
    path.parent.mkdir(parents=True, exist_ok=True)
    chunks = (
        simulation.sizes["time"],
        min(spatial_chunks[0], simulation.sizes["lat"]),
        min(spatial_chunks[1], simulation.sizes["lon"]),
    )
    template = xr.DataArray(
        da.empty(simulation.shape, chunks=chunks, dtype=simulation.dtype),
        dims=simulation.dims,
        coords={dim: simulation[dim] for dim in simulation.dims},
        name=simulation.name,
        attrs={
            **adjustment_attrs,
        },
    )
    template.to_dataset().to_zarr(
        path,
        mode="w",
        compute=False,
        consolidated=False,
        zarr_format=3,
        encoding={simulation.name: {"_FillValue": float("nan")}},
    )
    return False


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


def _reference_support_key(
    reference_root: Path, variables: tuple[str, ...], grid: str
) -> str:
    """Fingerprint the reference stores that define a support footprint."""
    stores = {}
    for variable in variables:
        store = reference_root / grid / f"{variable}.zarr"
        array_metadata = store / variable / "zarr.json"
        stores[variable] = {
            "group_metadata": _digest_optional_file(store / "zarr.json"),
            "array_metadata": _digest_optional_file(array_metadata),
            "array_metadata_mtime_ns": (
                array_metadata.stat().st_mtime_ns
                if array_metadata.is_file()
                else None
            ),
            "qc": _digest_optional_file(reference_root / f"{variable}.qc.json"),
        }
    payload = {
        "schema": 1,
        "grid": grid,
        "stores": stores,
        "reference_manifest": _digest_optional_file(
            reference_root / "reference-manifest.json"
        ),
        "reference_final_qc": _digest_optional_file(
            reference_root / "reference-final-qc.json"
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def reference_support(
    reference_root: Path,
    variables: tuple[str, ...],
    grid: str,
    *,
    cache: bool = True,
) -> xr.DataArray:
    """Return cells whose reference series is complete for every variable.

    The footprint depends only on the prepared reference, yet computing it
    reads every time step of every variable. It is therefore cached beside
    the reference and shared by all models, scenarios, and periods.
    """
    if grid not in {"fine", "coarse"}:
        raise ValueError(f"unknown reference grid: {grid}")
    if not variables:
        raise ValueError("at least one reference variable is required")
    ordered = tuple(sorted(variables))
    key = _reference_support_key(reference_root, ordered, grid)
    path = reference_root / "support" / f"{grid}-{'-'.join(ordered)}.zarr"
    if cache and path.exists():
        try:
            cached = open_variable(path, "support")
            if cached.attrs.get("support_cache_key") == key:
                return cached.compute()
        except Exception as error:  # a damaged cache is rebuilt, never trusted
            print(f"IGNORED unreadable support cache {path}: {error}", flush=True)

    support = None
    for variable in ordered:
        data = open_variable(reference_root / grid / f"{variable}.zarr", variable)
        valid = data.notnull().all("time").compute()
        support = valid if support is None else support & valid
    assert support is not None
    support = xr.DataArray(
        np.asarray(support.values, dtype=bool),
        dims=("lat", "lon"),
        coords={"lat": support.lat, "lon": support.lon},
        name="support",
        attrs={
            "long_name": f"complete {grid} reference support",
            "variables": ",".join(ordered),
            "support_cache_key": key,
        },
    )
    if cache:
        partial = path.with_name(f".{path.name}.partial-{os.getpid()}")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.rmtree(partial, ignore_errors=True)
            support.to_dataset().to_zarr(
                partial, mode="w", consolidated=False, zarr_format=3
            )
            shutil.rmtree(path, ignore_errors=True)
            partial.rename(path)
        except OSError as error:
            shutil.rmtree(partial, ignore_errors=True)
            print(f"SKIPPED support cache write {path}: {error}", flush=True)
    return support


def common_coarse_reference_support(
    reference_root: Path,
    variables: tuple[str, ...],
    global_spec: dict[str, object],
    *,
    cache: bool = True,
) -> np.ndarray:
    """Return cells with complete coarse reference data for every variable."""
    support = reference_support(reference_root, variables, "coarse", cache=cache)
    return np.asarray(_region_coarse(support, global_spec).values)


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


def tiles_intersecting_mask(
    tiles: list[dict[str, int]], mask: np.ndarray
) -> list[dict[str, int]]:
    """Retain regular tiles containing at least one selected coarse cell."""
    return [
        tile
        for tile in tiles
        if mask[
            tile["coarse_lat_start"] : tile["coarse_lat_stop"],
            tile["coarse_lon_start"] : tile["coarse_lon_stop"],
        ].any()
    ]


def missing_adjustment_cells(
    required: np.ndarray,
    coverage: np.ndarray,
    endpoint_values: np.ndarray,
) -> np.ndarray:
    """Find required coarse cells absent from state or stored endpoint data."""
    if endpoint_values.ndim != 3 or endpoint_values.shape[0] != 2:
        raise ValueError("endpoint values must have shape (2, lat, lon)")
    available = np.isfinite(endpoint_values).all(axis=0)
    if required.shape != coverage.shape or required.shape != available.shape:
        raise ValueError("adjustment coverage arrays must share a spatial shape")
    return required & (~coverage | ~available)


def spatial_tiles_intersecting_mask(
    tiles: list[dict[str, int]], mask: np.ndarray
) -> list[dict[str, int]]:
    """Retain active tiles and crop empty margins on coarse-cell boundaries."""
    selected: list[dict[str, int]] = []
    for tile in tiles:
        fine = mask[
            tile["fine_lat_start"] : tile["fine_lat_stop"],
            tile["fine_lon_start"] : tile["fine_lon_stop"],
        ]
        if not fine.any():
            continue
        required = {
            "coarse_lat_start",
            "coarse_lat_stop",
            "coarse_lon_start",
            "coarse_lon_stop",
        }
        if not required.issubset(tile):
            selected.append(tile)
            continue
        lat_factor = (tile["fine_lat_stop"] - tile["fine_lat_start"]) // (
            tile["coarse_lat_stop"] - tile["coarse_lat_start"]
        )
        lon_factor = (tile["fine_lon_stop"] - tile["fine_lon_start"]) // (
            tile["coarse_lon_stop"] - tile["coarse_lon_start"]
        )
        active_lat, active_lon = np.where(fine)
        relative_coarse_lat_start = int(active_lat.min()) // lat_factor
        relative_coarse_lat_stop = int(active_lat.max()) // lat_factor + 1
        relative_coarse_lon_start = int(active_lon.min()) // lon_factor
        relative_coarse_lon_stop = int(active_lon.max()) // lon_factor + 1
        cropped = dict(tile)
        for dimension in ("lat", "lon"):
            cropped[f"marker_coarse_{dimension}_start"] = tile[
                f"coarse_{dimension}_start"
            ]
            cropped[f"marker_coarse_{dimension}_stop"] = tile[
                f"coarse_{dimension}_stop"
            ]
        cropped.update(
            coarse_lat_start=(
                tile["coarse_lat_start"] + relative_coarse_lat_start
            ),
            coarse_lat_stop=tile["coarse_lat_start"] + relative_coarse_lat_stop,
            coarse_lon_start=(
                tile["coarse_lon_start"] + relative_coarse_lon_start
            ),
            coarse_lon_stop=tile["coarse_lon_start"] + relative_coarse_lon_stop,
            fine_lat_start=tile["fine_lat_start"]
            + relative_coarse_lat_start * lat_factor,
            fine_lat_stop=tile["fine_lat_start"]
            + relative_coarse_lat_stop * lat_factor,
            fine_lon_start=tile["fine_lon_start"]
            + relative_coarse_lon_start * lon_factor,
            fine_lon_stop=tile["fine_lon_start"]
            + relative_coarse_lon_stop * lon_factor,
        )
        selected.append(cropped)
    return selected


def initialize_coverage_store(
    simulation: xr.DataArray,
    path: Path,
    *,
    spatial_chunks: tuple[int, int] = (1, 1),
) -> None:
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
            chunks=(
                min(spatial_chunks[0], simulation.sizes["lat"]),
                min(spatial_chunks[1], simulation.sizes["lon"]),
            ),
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
    cache_support: bool = True,
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
    fine_support = _region_fine(
        reference_support(reference_root, variables, "fine", cache=cache_support),
        region_spec,
    )
    model_support = None
    for variable in variables:
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
        model_valid = valid_model.all("time").compute()
        model_support = (
            model_valid if model_support is None else model_support & model_valid
        )

    assert model_support is not None
    fine_template = fine_support
    reference_support_values = np.asarray(fine_support.values)
    lat_factor = int(region_spec.get("lat_factor", DEFAULT_DOWNSCALING_FACTOR))
    lon_factor = int(region_spec.get("lon_factor", DEFAULT_DOWNSCALING_FACTOR))
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
        expanded & reference_support_values,
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
    lat_factor = int(region_spec.get("lat_factor", DEFAULT_DOWNSCALING_FACTOR))
    lon_factor = int(region_spec.get("lon_factor", DEFAULT_DOWNSCALING_FACTOR))
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
    fit_cache_root: str | None = None,
    quantiles: int = 50,
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
        .sel(time=slice(*TRAINING_PERIOD))
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
    cache_path = None
    cache_key = None
    if fit_cache_root is not None:
        cache_path = adjustment_fit_cache_path(
            Path(fit_cache_root), model, variable, tile
        )
        cache_key = adjustment_fit_cache_key(
            model=model,
            variable=variable,
            tile=tile,
            reference_root=reference,
            canonical_root=canonical,
            quantiles=quantiles,
        )
    adjusted = adjust_variable(
        obs_coarse,
        historical,
        simulation,
        variable=variable,
        quantiles=quantiles,
        chunks={"lat": 1, "lon": 1},
        fit_cache_path=cache_path,
        fit_cache_key=cache_key,
    )
    if variable in {"pr", "sfcWind"}:
        adjusted = apply_downscaled_value_controls(adjusted, variable)
    write_adjusted_region(
        adjusted,
        adjusted_path,
        {
            "time": slice(0, adjusted.sizes["time"]),
            "lat": slice(local_lat_start, local_lat_stop),
            "lon": slice(local_lon_start, local_lon_stop),
        },
    )
    tile_marker.parent.mkdir(parents=True, exist_ok=True)
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
        "fit_cache": str(cache_path) if cache_path is not None else None,
        "fit_cache_hit": bool(
            adjusted.attrs.get("bias_adjustment_fit_cache_hit", False)
        ),
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
    tile_marker.touch()
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
    lat_factor = int(spec.get("lat_factor", DEFAULT_DOWNSCALING_FACTOR))
    lon_factor = int(spec.get("lon_factor", DEFAULT_DOWNSCALING_FACTOR))
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
        f"lat{tile.get('marker_coarse_lat_start', tile['coarse_lat_start']):03d}-"
        f"{tile.get('marker_coarse_lat_stop', tile['coarse_lat_stop']):03d}_"
        f"lon{tile.get('marker_coarse_lon_start', tile['coarse_lon_start']):03d}-"
        f"{tile.get('marker_coarse_lon_stop', tile['coarse_lon_stop']):03d}"
    )


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


def tile_report_matches_mask(marker: Path, expected_active_cells: int) -> bool:
    """Reject checkpoints created against an older, smaller support mask."""
    if not tile_complete(marker):
        return False
    try:
        report = json.loads(report_path(marker).read_text())
    except (OSError, ValueError, TypeError):
        return False
    return bool(report.get("valid")) and report.get("active_cells") == int(
        expected_active_cells
    )


def spatial_tile_already_written(
    output_root: Path,
    region: str,
    variable: str,
    tile: dict[str, int],
    spatial_valid_mask: np.ndarray,
) -> bool:
    """Validate a spatial checkpoint against the current support footprint."""
    marker = marker_path(output_root, region, variable, tile)
    expected_active_cells = int(
        np.count_nonzero(
            spatial_valid_mask[
                tile["fine_lat_start"] : tile["fine_lat_stop"],
                tile["fine_lon_start"] : tile["fine_lon_stop"],
            ]
        )
    )
    if tile_report_matches_mask(marker, expected_active_cells):
        return True
    marker.unlink(missing_ok=True)
    report_path(marker).unlink(missing_ok=True)
    return False


def configure_worker_runtime(threads_per_worker: int) -> None:
    if threads_per_worker < 1:
        raise ValueError("threads_per_worker must be at least one")
    scheduler = "threads" if threads_per_worker > 1 else "synchronous"
    dask.config.set(scheduler=scheduler, num_workers=threads_per_worker)


def tile_qc(
    written_tile: xr.DataArray,
    reference_tile: xr.DataArray,
    variable: str,
    *,
    min_valid_fraction: float = 0.95,
) -> dict[str, object]:
    """Check one written tile over its complete time axis.

    Output chunks span the whole period, so a leading sample would decompress
    the same bytes as the full record while leaving later years unchecked.
    """
    qc_steps = written_tile.sizes["time"]
    written_sample = written_tile
    valid_count = np.isfinite(written_sample).sum("time")
    active = valid_count > 0
    partial = active & (valid_count / qc_steps < min_valid_fraction)
    reference_sample = reference_tile.isel(
        time=slice(0, min(reference_tile.sizes["time"], 366))
    )
    reference_active = reference_sample.notnull().any("time")
    written_active = written_sample.notnull().any("time")
    static_floor = xr.zeros_like(active)
    if variable == "tas":
        floor = float(
            convert_units_to(DOWNSCALING_BOUNDS["tas"].lower_bound, written_sample)
        )
        static_floor = (written_sample == floor).all("time")
    (
        active_cells,
        partial_cells,
        missing_reference_cells,
        extra_cells,
        has_inf,
        minimum,
        maximum,
        static_floor_cells,
    ) = dask.compute(
        active.sum(),
        partial.sum(),
        (reference_active & ~written_active).sum(),
        (written_active & ~reference_active).sum(),
        np.isinf(written_sample).any(),
        written_sample.min(skipna=True),
        written_sample.max(skipna=True),
        static_floor.sum(),
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
        static_floor_cells = int(static_floor_cells)
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
        write_zarr_atomic(cleaned, temporary, packed=True)
        success_path(temporary).unlink(missing_ok=True)
        shutil.rmtree(path)
        temporary.rename(path)
        success_path(path).touch()
        report[variable] = str(path)
    return report


def coarse_row_strips(
    context_rows: int, fine_center: slice, factor: int
) -> list[dict[str, slice]]:
    """Split a haloed tile into single coarse rows with their own halos.

    The bilinear first guess of a coarse row depends only on the rows directly
    above and below it, so each strip reproduces the whole-tile result exactly.
    Slices index the tile context, except ``tile_rows`` which indexes the
    unhaloed tile.
    """
    if fine_center.start % factor or fine_center.stop % factor:
        raise ValueError("tile center must align with the downscaling factor")
    first_row = fine_center.start // factor
    strips = []
    for row in range(first_row, fine_center.stop // factor):
        lower = max(row - 1, 0)
        upper = min(row + 2, context_rows)
        strips.append(
            {
                "coarse_context": slice(lower, upper),
                "fine_context": slice(lower * factor, upper * factor),
                "fine_core": slice((row - lower) * factor, (row - lower + 1) * factor),
                "tile_rows": slice(
                    (row - first_row) * factor, (row - first_row + 1) * factor
                ),
            }
        )
    return strips


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
    spatial_mask_path: str | None = None,
) -> dict[str, object]:
    configure_worker_runtime(threads_per_worker)
    started = time.perf_counter()
    reference = Path(reference_root)
    output = Path(output_root)
    tile_marker = marker_path(output, region, variable, tile)
    downscaled_path = output / region / f"{variable}_downscaled.zarr"
    if tile_complete(tile_marker):
        current_mask = open_variable(
            Path(spatial_mask_path)
            if spatial_mask_path is not None
            else output / region / "spatial_valid_mask.zarr",
            "spatial_valid_mask",
        ).isel(
            lat=slice(tile["fine_lat_start"], tile["fine_lat_stop"]),
            lon=slice(tile["fine_lon_start"], tile["fine_lon_stop"]),
        )
        expected_active_cells = int(current_mask.sum().compute())
        if tile_report_matches_mask(tile_marker, expected_active_cells):
            return {
                "variable": variable,
                "region": region,
                "tile": _tile_name(tile),
                "skipped": True,
            }
        tile_marker.unlink(missing_ok=True)
        report_path(tile_marker).unlink(missing_ok=True)

    adjusted = open_variable(Path(adjusted_path), variable)
    lat_factor = int(region_spec.get("lat_factor", DEFAULT_DOWNSCALING_FACTOR))
    lon_factor = int(region_spec.get("lon_factor", DEFAULT_DOWNSCALING_FACTOR))
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
    spatial_mask = open_variable(
        Path(spatial_mask_path)
        if spatial_mask_path is not None
        else output / region / "spatial_valid_mask.zarr",
        "spatial_valid_mask",
    ).isel(
        lat=slice(local_lat_start, local_lat_stop),
        lon=slice(local_lon_start, local_lon_stop),
    )
    # A tile fits in memory, so it is downscaled eagerly: the lazy graph for a
    # single tile runs to about a million Dask tasks and costs several times
    # the numerics to schedule. One coarse row is processed at a time, with
    # the neighbouring rows as its interpolation halo, which keeps a worker's
    # footprint near one row of the period instead of the whole tile.
    sim = sim.load()
    obs_fine_context = obs_fine_context.load()
    spatial_mask = spatial_mask.load()
    obs_fine = obs_fine_context.isel(
        lat=fine_lat_center,
        lon=fine_lon_center,
    ).where(spatial_mask)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="invalid value encountered in divide"
        )
        for strip in coarse_row_strips(
            sim.sizes["lat"], fine_lat_center, lat_factor
        ):
            downscaled = downscale_variable(
                obs_fine_context.isel(lat=strip["fine_context"]),
                sim.isel(lat=strip["coarse_context"]),
                variable=variable,
                iterations=iterations,
                quantiles=quantiles,
                core={"lat": strip["fine_core"], "lon": fine_lon_center},
                eager=True,
            )
            downscaled = apply_downscaled_value_controls(downscaled, variable)
            downscaled = downscaled.where(spatial_mask.isel(lat=strip["tile_rows"]))
            variable_only_dataset(downscaled).to_zarr(
                downscaled_path,
                mode="r+",
                region={
                    "time": slice(0, downscaled.sizes["time"]),
                    "lat": slice(
                        local_lat_start + strip["tile_rows"].start,
                        local_lat_start + strip["tile_rows"].stop,
                    ),
                    "lon": slice(local_lon_start, local_lon_stop),
                },
                consolidated=False,
            )

    written_tile = open_variable(downscaled_path, variable).isel(
        lat=slice(local_lat_start, local_lat_stop),
        lon=slice(local_lon_start, local_lon_stop),
    )
    qc = tile_qc(written_tile, obs_fine, variable)
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
    tile_marker.parent.mkdir(parents=True, exist_ok=True)
    report_path(tile_marker).write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    tile_marker.touch()
    return record


def _spawn_pool(workers: int) -> Executor:
    return ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
    )


def run_tile_pool(
    label: str,
    function: Callable[..., dict[str, object]],
    tasks: list[dict[str, object]],
    *,
    workers: int,
    executor_factory: Callable[[int], Executor] = _spawn_pool,
) -> list[dict[str, object]]:
    """Run tile tasks, reporting every failure as soon as it happens.

    Remaining tiles still run after a failure because each one is an
    independent checkpoint, but the error is printed immediately instead of
    surfacing only once the whole pool has drained. A dead worker process
    breaks the pool for every pending tile, so that case stops at once.
    """
    records: list[dict[str, object]] = []
    failures: list[str] = []
    if not tasks:
        return records
    with executor_factory(workers) as executor:
        futures = {
            executor.submit(function, **task): _tile_name(task["tile"])
            for task in tasks
        }
        for index, future in enumerate(as_completed(futures), start=1):
            name = futures[future]
            try:
                record = future.result()
            except BrokenProcessPool:
                print(
                    f"FAILED {label} tile {index}/{len(tasks)}: {name}: "
                    "a worker process died; stopping this pool",
                    flush=True,
                )
                raise
            except Exception as error:
                failures.append(name)
                print(
                    f"FAILED {label} tile {index}/{len(tasks)}: {name}: "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )
                continue
            records.append(record)
            print(
                f"DONE {label} tile {index}/{len(tasks)}: {record.get('tile')}",
                flush=True,
            )
    if failures:
        raise RuntimeError(
            f"{len(failures)} of {len(tasks)} {label} tiles failed: "
            + ", ".join(sorted(failures))
        )
    return records


def main(
    argv: Sequence[str] | None = None,
    *,
    default_regions: Sequence[str] | None = None,
) -> None:
    parser = argparse.ArgumentParser(
        description="Run regional or global MBCnSD as restartable tiles."
    )
    parser.add_argument("--model", default="ACCESS-CM2")
    parser.add_argument("--scenario", default="ssp245")
    parser.add_argument(
        "--simulation-stage",
        choices=("hist", "projection"),
        default=None,
        help=(
            "canonical input stage; defaults to hist for historical and projection "
            "otherwise"
        ),
    )
    parser.add_argument("--simulation-start", default=None)
    parser.add_argument("--simulation-end", default=None)
    parser.add_argument(
        "--regions",
        nargs="+",
        choices=(*REGIONS, "global"),
        default=list(default_regions or REGIONS),
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
        default=Path("/nas/dat1/cmip6_fwi/reference/prepared_local_noon"),
    )
    parser.add_argument(
        "--canonical-root",
        type=Path,
        default=Path("/nas/dat1/cmip6_fwi/inputs/standardized_1deg"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--adjusted-root",
        type=Path,
        default=Path("/nas/dat1/cmip6_fwi/processing/bias_adjusted_1deg"),
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
        "--fit-cache-root",
        type=Path,
        default=None,
        help=(
            "optional model/variable/tile adjustment-fit cache shared across "
            "historical and projection periods"
        ),
    )
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
    parser.add_argument(
        "--spatial-tile-names",
        nargs="+",
        default=None,
        help="run only these spatial tile names (intended for benchmarks)",
    )
    parser.add_argument(
        "--spatial-valid-mask-store",
        type=Path,
        default=None,
        help="reuse an existing spatial_valid_mask.zarr store",
    )
    args = parser.parse_args(argv)

    simulation_stage = args.simulation_stage or (
        "hist" if args.scenario == "historical" else "projection"
    )
    args.output_root = args.output_root or default_output_root(
        args.model, args.scenario, simulation_stage, list(args.regions)
    )
    simulation_start = args.simulation_start
    simulation_end = args.simulation_end
    if args.scenario == "historical":
        simulation_start = simulation_start or "1989"
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
    required_cells &= common_coarse_reference_support(
        args.reference_root, tuple(args.variables), global_spec
    )
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
            def invalidate_adjustment(
                coverage_path: Path = coverage_path,
                state: Path = adjusted_path.parent / "state" / variable,
            ) -> None:
                shutil.rmtree(coverage_path, ignore_errors=True)
                shutil.rmtree(state, ignore_errors=True)

            stale_adjustment = initialize_adjusted_store(
                simulation,
                adjusted_path,
                quantiles=args.quantiles,
                spatial_chunks=(adjustment_tile_lat, args.tile_lon_degrees),
                invalidate=invalidate_adjustment,
            )
            if stale_adjustment:
                print(
                    f"INVALIDATED {variable} adjustment checkpoints: "
                    "bias-adjustment preset revision changed",
                    flush=True,
                )
            initialize_coverage_store(
                simulation,
                coverage_path,
                spatial_chunks=(adjustment_tile_lat, args.tile_lon_degrees),
            )
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
            endpoints = (
                open_variable(adjusted_path, variable)
                .isel(time=[0, -1])
                .compute()
                .values
            )
            missing_cells = missing_adjustment_cells(
                required_cells, coverage, endpoints
            )
            if args.seed_adjusted_from is None:
                pending_adjustment = tiles_intersecting_mask(
                    regular_adjustment_tiles, missing_cells
                )
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
            for tile in pending_adjustment:
                stale_marker = adjustment_marker_path(
                    args.adjusted_root,
                    args.model,
                    args.scenario,
                    simulation_stage,
                    variable,
                    tile,
                )
                stale_marker.unlink(missing_ok=True)
                report_path(stale_marker).unlink(missing_ok=True)
            manifest_records.extend(
                run_tile_pool(
                    f"{variable} global-context adjustment",
                    run_adjustment_tile,
                    [
                        {
                            "model": args.model,
                            "scenario": args.scenario,
                            "simulation_stage": simulation_stage,
                            "simulation_start": simulation_start,
                            "simulation_end": simulation_end,
                            "global_spec": global_spec,
                            "variable": variable,
                            "tile": tile,
                            "reference_root": str(args.reference_root),
                            "canonical_root": str(args.canonical_root),
                            "adjusted_root": str(args.adjusted_root),
                            "coverage_path": str(coverage_path),
                            "threads_per_worker": args.threads_per_worker,
                            "fit_cache_root": (
                                str(args.fit_cache_root)
                                if args.fit_cache_root is not None
                                else None
                            ),
                            "quantiles": args.quantiles,
                        }
                        for tile in pending_adjustment
                    ],
                    workers=args.tile_workers,
                )
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
        if args.spatial_valid_mask_store is not None:
            if len(args.regions) != 1:
                parser.error(
                    "--spatial-valid-mask-store requires exactly one region"
                )
            mask_path = args.spatial_valid_mask_store
            if not mask_path.exists():
                parser.error(f"spatial valid mask does not exist: {mask_path}")
        else:
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
        spatial_valid_mask = np.asarray(
            open_variable(mask_path, "spatial_valid_mask").compute().values
        )
        for variable in args.variables:
            all_tiles = tile_specs(
                region_spec, tile_lat_degrees, args.tile_lon_degrees
            )
            tiles = spatial_tiles_intersecting_mask(
                all_tiles, spatial_valid_mask
            )
            if args.spatial_tile_names is not None:
                requested_names = set(args.spatial_tile_names)
                tiles = [tile for tile in tiles if _tile_name(tile) in requested_names]
                missing_names = requested_names - {_tile_name(tile) for tile in tiles}
                if missing_names:
                    parser.error(
                        "unknown or inactive spatial tile names: "
                        + ", ".join(sorted(missing_names))
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
            downscaled_store = (
                args.output_root / region / f"{variable}_downscaled.zarr"
            )

            def invalidate_spatial(
                store: Path = downscaled_store,
                state: Path = (
                    args.output_root
                    / region
                    / "state_spatial_global_context"
                    / variable
                ),
            ) -> None:
                shutil.rmtree(state, ignore_errors=True)
                success_path(store).unlink(missing_ok=True)

            stale_spatial = initialize_output_store(
                adjusted,
                fine_reference,
                downscaled_store,
                iterations=args.iterations,
                quantiles=args.quantiles,
                invalidate=invalidate_spatial,
                spatial_chunks=(
                    int(region_spec.get("lat_factor", DEFAULT_DOWNSCALING_FACTOR)),
                    int(region_spec.get("lon_factor", DEFAULT_DOWNSCALING_FACTOR)),
                ),
            )
            if stale_spatial:
                print(
                    f"INVALIDATED {variable} spatial checkpoints: "
                    "bias-adjustment preset revision changed",
                    flush=True,
                )
            pending = [
                tile
                for tile in tiles
                if not spatial_tile_already_written(
                    args.output_root,
                    region,
                    variable,
                    tile,
                    spatial_valid_mask,
                )
            ]
            print(
                f"START {variable} {region} MBCnSD tiles: {len(pending)}/{len(tiles)}",
                flush=True,
            )
            manifest_records.extend(
                run_tile_pool(
                    f"{variable} {region}",
                    run_tile,
                    [
                        {
                            "model": args.model,
                            "scenario": args.scenario,
                            "region": region,
                            "region_spec": region_spec,
                            "variable": variable,
                            "tile": tile,
                            "reference_root": str(args.reference_root),
                            "adjusted_path": str(adjusted_path),
                            "global_spec": global_spec,
                            "output_root": str(args.output_root),
                            "iterations": args.iterations,
                            "quantiles": args.quantiles,
                            "threads_per_worker": args.threads_per_worker,
                            "spatial_mask_path": str(mask_path),
                        }
                        for tile in pending
                    ],
                    workers=args.tile_workers,
                )
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
        "spatial_tile_land_cropping": (
            "coarse-cell-aligned bounding box of valid fine-grid cells"
        ),
        "tile_workers": args.tile_workers,
        "threads_per_worker": args.threads_per_worker,
        "execution_slots": args.tile_workers * args.threads_per_worker,
        "fit_cache_root": (
            str(args.fit_cache_root) if args.fit_cache_root is not None else None
        ),
        "spatial_tile_names": args.spatial_tile_names,
        "spatial_valid_mask_store": (
            str(args.spatial_valid_mask_store)
            if args.spatial_valid_mask_store is not None
            else None
        ),
        "records": manifest_records,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "downscale-tiles-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
