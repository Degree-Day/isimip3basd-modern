"""Faithful xarray/Dask implementation of ISIMIP3BASD MBCnSD."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from importlib.metadata import version

import dask
import numpy as np
import scipy.interpolate
import scipy.linalg
import xarray as xr
from xclim.core.units import convert_units_to

from . import __version__
from .presets import get_preset


@dataclass(frozen=True)
class DownscalingBounds:
    lower_bound: str | None = None
    lower_threshold: str | None = None
    upper_bound: str | None = None
    upper_threshold: str | None = None
    if_all_invalid_use: float | None = None


@dataclass(frozen=True)
class GridInfo:
    spatial_dims: tuple[str, ...]
    factors: tuple[int, ...]
    ascending: tuple[bool, ...]
    circular: tuple[bool, ...]


DOWNSCALING_BOUNDS: dict[str, DownscalingBounds] = {
    "hurs": DownscalingBounds("0 %", "0.01 %", "100 %", "99.99 %"),
    "pr": DownscalingBounds("0 mm d-1", "0.1 mm d-1"),
    "prsnratio": DownscalingBounds("0", "0.0001", "1", "0.9999", 0.0),
    "ps": DownscalingBounds(),
    "rlds": DownscalingBounds(),
    "rsds": DownscalingBounds("0 W m-2", "0.01 W m-2"),
    "sfcWind": DownscalingBounds("0 m s-1", "0.01 m s-1"),
    "tas": DownscalingBounds("150 K", "150 K", "350 K", "350 K"),
    "tasrange": DownscalingBounds("0 K", "0.01 K"),
    "tasskew": DownscalingBounds("0", "0.0001", "1", "0.9999"),
}

DOWNSCALING_CONSERVATION_TOLERANCE = {"prsnratio": 0.15}
DOWNSCALING_MIN_VALID_FRACTION = {"prsnratio": 0.5}
CIL_PRECIPITATION_CEILING = "3000 mm d-1"
CIL_TEMPERATURE_VALID_RANGE = ("130 K", "377 K")


def analyze_input_grids(
    simulation: xr.DataArray,
    observations: xr.DataArray,
) -> GridInfo:
    """Validate the nested regular grids required by MBCnSD."""
    if "time" not in simulation.dims or "time" not in observations.dims:
        raise ValueError("simulation and observations require a time dimension")
    spatial_dims = tuple(
        dimension for dimension in simulation.dims if dimension != "time"
    )
    observation_dims = tuple(
        dimension for dimension in observations.dims if dimension != "time"
    )
    if observation_dims != spatial_dims:
        raise ValueError("simulation and observation spatial dimensions differ")
    if not spatial_dims:
        raise ValueError("MBCnSD requires at least one spatial dimension")

    factors: list[int] = []
    ascending: list[bool] = []
    circular: list[bool] = []
    for dimension in spatial_dims:
        if (
            dimension not in simulation.coords
            or dimension not in observations.coords
        ):
            raise ValueError(f"{dimension} requires a coordinate on both grids")
        coarse = np.asarray(simulation[dimension].values)
        fine = np.asarray(observations[dimension].values)
        if coarse.ndim != 1 or fine.ndim != 1:
            raise ValueError(
                "MBCnSD supports one-dimensional regular-grid coordinates"
            )
        if coarse.size < 1 or fine.size % coarse.size:
            raise ValueError(f"invalid downscaling factor for {dimension}")
        factor = fine.size // coarse.size
        if factor <= 1:
            raise ValueError(f"{dimension} is not finer than the simulation grid")

        coarse_delta = np.diff(coarse)
        fine_delta = np.diff(fine)
        if coarse.size == 1:
            increasing = bool(np.all(fine_delta > 0))
            decreasing = bool(np.all(fine_delta < 0))
        else:
            increasing = bool(np.all(coarse_delta > 0) and np.all(fine_delta > 0))
            decreasing = bool(np.all(coarse_delta < 0) and np.all(fine_delta < 0))
        if not increasing and not decreasing:
            raise ValueError(
                f"{dimension} coordinates must be monotonic in the same direction"
            )

        if factor > 1:
            within = np.delete(
                fine_delta, np.arange(factor - 1, fine_delta.size, factor)
            ).reshape(factor - 1, coarse.size)
            if not np.allclose(within, within[:, :1]):
                raise ValueError(
                    f"fine cells are not uniformly spaced within {dimension}"
                )

        if coarse.size == 1:
            widths = np.asarray([fine_delta[0] * factor])
        else:
            widths = 0.5 * (
                np.concatenate((coarse_delta[:1], coarse_delta))
                + np.concatenate((coarse_delta, coarse_delta[-1:]))
            )
        fractions = np.arange(1, factor + 1) / factor
        expected = np.repeat(coarse - 0.5 * widths, factor) + np.repeat(
            widths, factor
        ) * np.tile(fractions - 0.5 * fractions[0], coarse.size)
        if not np.allclose(fine, expected):
            raise ValueError(f"fine {dimension} cells are not nested in coarse cells")

        signed_period = 360 if increasing else -360
        is_circular = bool(
            coarse.size > 1
            and np.allclose(coarse[:1] - coarse_delta[:1] + signed_period, coarse[-1:])
        )
        factors.append(factor)
        ascending.append(increasing)
        circular.append(is_circular)

    return GridInfo(
        spatial_dims=spatial_dims,
        factors=tuple(factors),
        ascending=tuple(ascending),
        circular=tuple(circular),
    )


def _quantity(quantity: str | None, target: xr.DataArray) -> float | None:
    if quantity is None:
        return None
    return float(convert_units_to(quantity, target, context="infer"))


def apply_downscaled_value_controls(
    data: xr.DataArray,
    variable: str | None = None,
    *,
    precipitation_ceiling: str = CIL_PRECIPITATION_CEILING,
    mask_static_temperature_floor: bool = True,
) -> xr.DataArray:
    """Apply production value controls inspired by the CIL CMIP6 pipeline.

    CIL caps precipitation above 3000 mm/day and validates tasmin/tasmax against
    a permissive [130, 377] K range. Temperature values outside that range are
    treated as invalid here, while static MBCnSD temperature-floor cells are
    masked to avoid keeping coastal mask sentinels as plausible data.
    """
    variable = variable or data.name
    if not variable:
        raise ValueError("a variable name is required")
    result = data

    if variable == "pr":
        ceiling = convert_units_to(precipitation_ceiling, result, context="hydro")
        result = result.clip(min=0, max=float(ceiling))
        result.attrs.update(
            cil_precipitation_ceiling=precipitation_ceiling,
            cil_precipitation_ceiling_native_units=float(ceiling),
        )
    elif variable == "sfcWind":
        result = result.clip(min=0)
    elif variable == "hurs":
        result = result.clip(min=0, max=100)
    elif variable in {"tas", "tasmin", "tasmax"}:
        lower = float(convert_units_to(CIL_TEMPERATURE_VALID_RANGE[0], result))
        upper = float(convert_units_to(CIL_TEMPERATURE_VALID_RANGE[1], result))
        result = result.where((result >= lower) & (result <= upper))
        if variable == "tas" and mask_static_temperature_floor and "time" in result.dims:
            floor = _quantity(DOWNSCALING_BOUNDS["tas"].lower_bound, result)
            if floor is not None:
                static_floor = (result == floor).all("time")
                result = result.where(~static_floor)
                result.attrs["static_temperature_floor_cells_masked"] = "true"
        result.attrs.update(
            cil_temperature_valid_min=CIL_TEMPERATURE_VALID_RANGE[0],
            cil_temperature_valid_max=CIL_TEMPERATURE_VALID_RANGE[1],
        )
    elif variable == "tasrange":
        result = result.clip(min=0)
    elif variable in {"tasskew", "prsnratio"}:
        result = result.clip(min=0, max=1)

    result.attrs.update(
        downscaled_value_controls=(
            "CIL-style precipitation ceiling; CIL temperature validation range; "
            "ISIMIP variable physical bounds"
        )
    )
    return result


def _resolved_bounds(variable: str, target: xr.DataArray) -> tuple[float | None, ...]:
    bounds = DOWNSCALING_BOUNDS[variable]
    return (
        _quantity(bounds.lower_bound, target),
        _quantity(bounds.lower_threshold, target),
        _quantity(bounds.upper_bound, target),
        _quantity(bounds.upper_threshold, target),
    )


def _periodic_extension(data: xr.DataArray, dimension: str) -> xr.DataArray:
    ordered = data.sortby(dimension)
    first = float(ordered[dimension][0])
    last = float(ordered[dimension][-1])
    lower = ordered.isel({dimension: [-1]}).assign_coords({dimension: [last - 360]})
    upper = ordered.isel({dimension: [0]}).assign_coords({dimension: [first + 360]})
    return xr.concat((lower, ordered, upper), dim=dimension)


def bilinear_broadcast(
    simulation: xr.DataArray,
    observations: xr.DataArray,
    grid: GridInfo | None = None,
) -> xr.DataArray:
    """Broadcast coarse data to the fine grid as in MBCnSD step 1."""
    grid = grid or analyze_input_grids(simulation, observations)
    source = simulation
    for dimension, circular in zip(grid.spatial_dims, grid.circular, strict=True):
        if circular and simulation.sizes[dimension] > 1:
            source = _periodic_extension(source, dimension)
    targets = {dimension: observations[dimension] for dimension in grid.spatial_dims}
    indexers = {
        dimension: xr.DataArray(
            np.repeat(np.arange(simulation.sizes[dimension]), factor),
            dims=dimension,
        )
        for dimension, factor in zip(grid.spatial_dims, grid.factors, strict=True)
    }
    central = simulation.isel(indexers).assign_coords(targets)
    interpolated = source.interp(
        {
            dimension: coordinate
            for dimension, coordinate in targets.items()
            if simulation.sizes[dimension] > 1
        },
        method="linear",
        assume_sorted=False,
    )
    single_cell_dimensions = [
        dimension
        for dimension in grid.spatial_dims
        if simulation.sizes[dimension] == 1 and dimension in interpolated.dims
    ]
    if single_cell_dimensions:
        interpolated = interpolated.isel(
            {dimension: 0 for dimension in single_cell_dimensions}, drop=True
        )
    interpolated = interpolated.broadcast_like(central).assign_coords(targets)
    result = interpolated.where(np.isfinite(interpolated), central)
    result = result.astype(simulation.dtype)
    result.attrs = simulation.attrs
    return result.transpose("time", *grid.spatial_dims)


def grid_cell_weights(observations: xr.DataArray, grid: GridInfo) -> xr.DataArray:
    """Return the upstream cosine-latitude area weights."""
    shape = tuple(observations.sizes[dimension] for dimension in grid.spatial_dims)
    values = np.ones(shape, dtype=np.float64)
    latitude_names = [
        name for name in ("lat", "latitude", "rlat") if name in grid.spatial_dims
    ]
    if len(latitude_names) > 1:
        raise ValueError("found more than one latitude coordinate")
    if latitude_names:
        latitude_name = latitude_names[0]
        latitudes = np.asarray(observations[latitude_name].values)
        if np.any(latitudes < -90) or np.any(latitudes > 90):
            raise ValueError("latitude coordinates lie outside [-90, 90]")
        axis = grid.spatial_dims.index(latitude_name)
        reshape = (1,) * axis + latitudes.shape + (1,) * (len(shape) - axis - 1)
        values *= np.cos(np.deg2rad(latitudes)).reshape(reshape)
    return xr.DataArray(
        values,
        coords={dimension: observations[dimension] for dimension in grid.spatial_dims},
        dims=grid.spatial_dims,
        name="cell_area_weight",
    )


def aggregate_to_coarse_grid(
    downscaled: xr.DataArray,
    coarse: xr.DataArray,
) -> xr.DataArray:
    """Area-average MBCnSD output back to its source grid."""
    grid = analyze_input_grids(coarse, downscaled)
    weights = grid_cell_weights(downscaled, grid)
    windows = {
        dimension: factor
        for dimension, factor in zip(grid.spatial_dims, grid.factors, strict=True)
    }
    numerator = (downscaled * weights).coarsen(windows, boundary="exact").sum()
    denominator = weights.where(downscaled.notnull()).coarsen(
        windows, boundary="exact"
    ).sum()
    result = numerator / denominator
    return result.assign_coords(
        {dimension: coarse[dimension] for dimension in grid.spatial_dims}
    ).transpose("time", *grid.spatial_dims)


def coarse_scale_conservation(
    downscaled: xr.DataArray,
    coarse: xr.DataArray,
) -> dict[str, float | bool | str]:
    """Measure the approximate coarse-scale conservation promised by MBCnSD."""
    aggregated = aggregate_to_coarse_grid(downscaled, coarse)
    difference = aggregated - coarse.transpose(*aggregated.dims)
    (
        mean_absolute_value,
        maximum_absolute_value,
        mean_square_value,
        mean_bias_value,
        standard_deviation_value,
        magnitude_value,
    ) = dask.compute(
        abs(difference).mean(skipna=True),
        abs(difference).max(skipna=True),
        (difference**2).mean(skipna=True),
        difference.mean(skipna=True),
        coarse.std(skipna=True),
        abs(coarse).mean(skipna=True),
    )
    mean_absolute = float(mean_absolute_value)
    maximum_absolute = float(maximum_absolute_value)
    root_mean_square = float(np.sqrt(mean_square_value))
    mean_bias = float(mean_bias_value)
    standard_deviation = float(standard_deviation_value)
    magnitude = float(magnitude_value)
    scale = standard_deviation if standard_deviation > 0 else magnitude
    normalized_rmse = root_mean_square / scale if scale > 0 else root_mean_square
    tolerance = DOWNSCALING_CONSERVATION_TOLERANCE.get(coarse.name or "", 0.05)
    return {
        "valid": bool(np.isfinite(normalized_rmse) and normalized_rmse <= tolerance),
        "mean_absolute_error": mean_absolute,
        "maximum_absolute_error": maximum_absolute,
        "root_mean_square_error": root_mean_square,
        "mean_bias": mean_bias,
        "normalized_rmse": normalized_rmse,
        "tolerance": tolerance,
        "units": coarse.attrs.get("units", ""),
    }


def _percentile_1d(values: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    count = values.size - 1
    ordered = np.sort(values)
    positions = count * probabilities
    lower = np.floor(positions).astype(int)
    upper_weight = positions - lower
    return (
        ordered[lower] * (1 - upper_weight)
        + ordered[lower + (lower < count)] * upper_weight
    )


def _quantile_map(
    values: np.ndarray,
    source_quantiles: np.ndarray,
    target_quantiles: np.ndarray,
) -> np.ndarray:
    below = values < source_quantiles[0]
    above = values > source_quantiles[-1]
    result = np.interp(values, source_quantiles, target_quantiles)
    result[below] = values[below] + target_quantiles[0] - source_quantiles[0]
    result[above] = values[above] + target_quantiles[-1] - source_quantiles[-1]
    return result


def _fixed_first_axis_rotation(weights: np.ndarray) -> np.ndarray:
    basis = np.diag(np.ones_like(weights))
    basis[:, 0] = weights
    orthogonal, _ = scipy.linalg.qr(basis)
    return -orthogonal


def generate_rotation_matrices(
    cells: int,
    iterations: int = 20,
    random_seed: int | None = 0,
) -> tuple[np.ndarray, ...]:
    """Generate the circular-real-ensemble rotations used by MBCnSD."""
    if iterations < 1:
        raise ValueError("iterations must be positive")
    random = np.random.RandomState(random_seed)
    matrices: list[np.ndarray] = []
    for _ in range(iterations):
        normal = random.randn(cells, cells)
        orthogonal, triangular = scipy.linalg.qr(normal)
        diagonal = np.diagonal(triangular)
        matrices.append(orthogonal * (diagonal / np.abs(diagonal)))
    return tuple(matrices)


@lru_cache(maxsize=256)
def _cached_rotation_matrices(
    cells: int, iterations: int, random_seed: int | None
) -> tuple[np.ndarray, ...]:
    return generate_rotation_matrices(cells, iterations, random_seed)


def weighted_sum_preserving_mbcn(
    observations: np.ndarray,
    coarse_simulation: np.ndarray,
    fine_simulation: np.ndarray,
    weights: np.ndarray,
    rotation_matrices: Sequence[np.ndarray],
    n_quantiles: int = 50,
) -> np.ndarray:
    """Apply the archived weighted-sum-preserving MBCn core."""
    if n_quantiles < 2:
        raise ValueError("n_quantiles must be at least 2")
    result = fine_simulation.copy()
    target = observations.copy()
    sum_weights = weights / np.sqrt(np.sum(np.square(weights)))
    coarse = coarse_simulation * np.sum(sum_weights)
    total_rotation = np.eye(weights.size)
    probabilities = np.linspace(0, 1, n_quantiles + 1)

    for index in range(len(rotation_matrices) + 2):
        if index == 0:
            rotation = _fixed_first_axis_rotation(sum_weights)
        elif index == len(rotation_matrices) + 1:
            rotation = total_rotation.T
        else:
            rotation = rotation_matrices[index - 1]
        total_rotation = total_rotation @ rotation
        result = result @ rotation
        target = target @ rotation
        sum_weights = sum_weights @ rotation

        if index == 0:
            result[:, 0] = coarse
            simulated_quantiles = _percentile_1d(coarse, probabilities)
            observed_quantiles = _percentile_1d(target[:, 0], probabilities)
            target[:, 0] = _quantile_map(
                target[:, 0], observed_quantiles, simulated_quantiles
            )
            continue

        previous = result.copy()
        for cell in range(weights.size):
            simulated_quantiles = _percentile_1d(result[:, cell], probabilities)
            observed_quantiles = _percentile_1d(target[:, cell], probabilities)
            result[:, cell] = _quantile_map(
                result[:, cell], simulated_quantiles, observed_quantiles
            )
        if index < len(rotation_matrices) + 1:
            result -= np.outer((result - previous) @ sum_weights, sum_weights)
    return result


def _average_valid(
    values: np.ndarray,
    fallback: float,
    lower_bound: float | None,
    lower_threshold: float | None,
    upper_bound: float | None,
    upper_threshold: float | None,
) -> np.ndarray | float:
    result = values.copy()
    if lower_bound is not None and lower_threshold is not None:
        result = np.where(result <= lower_threshold, lower_bound, result)
    if upper_bound is not None and upper_threshold is not None:
        result = np.where(result >= upper_threshold, upper_bound, result)
    if result.ndim == 1:
        valid = np.isfinite(result)
        return float(np.mean(result[valid])) if valid.any() else fallback
    averages = np.empty(result.shape[1], dtype=np.float64)
    for column in range(result.shape[1]):
        valid = np.isfinite(result[:, column])
        averages[column] = (
            np.mean(result[valid, column]) if valid.any() else fallback
        )
    return averages


def _sample_invalid_1d(
    values: np.ndarray,
    fallback: float,
    random_seed: int | None,
) -> np.ndarray:
    invalid = ~np.isfinite(values)
    if not invalid.any():
        return values.copy()
    valid_values = values[~invalid]
    if not valid_values.size:
        if np.isnan(fallback):
            raise ValueError("found no valid values")
        return np.full_like(values, fallback)
    random = np.random.RandomState(random_seed)
    sampled = _percentile_1d(valid_values, random.random_sample(invalid.sum()))
    result = values.copy()
    if valid_values.size == 1:
        result[invalid] = sampled
    else:
        valid_indices = np.where(~invalid)[0]
        valid_ranks = np.argsort(np.argsort(valid_values))
        rank_interpolator = scipy.interpolate.interp1d(
            valid_indices, valid_ranks, fill_value="extrapolate"
        )
        sampled_indices = np.where(invalid)[0]
        sampled_ranks = np.argsort(np.argsort(rank_interpolator(sampled_indices)))
        result[invalid] = np.sort(sampled)[sampled_ranks]
    return result


def _sample_invalid(
    values: np.ndarray,
    fallback: np.ndarray | float,
    random_seed: int | None,
) -> np.ndarray:
    if values.ndim == 1:
        return _sample_invalid_1d(values, float(fallback), random_seed)
    result = np.empty_like(values)
    for column in range(values.shape[1]):
        result[:, column] = _sample_invalid_1d(
            values[:, column], float(np.asarray(fallback)[column]), random_seed
        )
    return result


def _random_tie_ranks(values: np.ndarray, random: np.random.RandomState) -> np.ndarray:
    permutation = random.permutation(values.size)
    shuffled = values[permutation]
    ranks_shuffled = np.argsort(
        np.argsort(shuffled, kind="stable"), kind="stable"
    )
    ranks = np.empty(values.size, dtype=int)
    ranks[permutation] = ranks_shuffled
    return ranks


def _randomize_censored(
    values: np.ndarray,
    lower_bound: float | None,
    lower_threshold: float | None,
    upper_bound: float | None,
    upper_threshold: float | None,
    random_seed: int | None,
    *,
    inverse: bool = False,
) -> np.ndarray:
    result = values.copy()
    random = np.random.RandomState(random_seed)
    specifications = (
        (lower_bound, lower_threshold, True),
        (upper_bound, upper_threshold, False),
    )
    for bound, threshold, lower in specifications:
        if bound is None or threshold is None:
            continue
        selected = result <= threshold if lower else result >= threshold
        if inverse:
            result[selected] = bound
            continue
        count = int(selected.sum())
        if not count:
            continue
        fractions = np.power(random.uniform(0, 1, count), 10)
        replacements = bound + fractions * (threshold - bound)
        selected_values = result[selected]
        ranks = _random_tie_ranks(selected_values, random)
        result[selected] = np.sort(replacements)[ranks]
    return result


def _downscale_cell(
    observations: np.ndarray,
    coarse_simulation: np.ndarray,
    fine_simulation: np.ndarray,
    weights: np.ndarray,
    observation_months: np.ndarray,
    simulation_months: np.ndarray,
    *,
    rotation_matrices: Sequence[np.ndarray],
    n_quantiles: int,
    random_seed: int | None,
    lower_bound: float | None,
    lower_threshold: float | None,
    upper_bound: float | None,
    upper_threshold: float | None,
    if_all_invalid_use: float,
) -> np.ndarray:
    output = fine_simulation.copy()
    if np.isnan(if_all_invalid_use) and np.all(~np.isfinite(coarse_simulation)):
        return np.full_like(output, np.nan)

    active_cells = (
        np.isfinite(weights)
        & np.any(np.isfinite(observations), axis=0)
        & np.any(np.isfinite(fine_simulation), axis=0)
    )
    if not active_cells.any():
        return np.full_like(output, np.nan)
    if int(active_cells.sum()) == 1:
        output[:] = np.nan
        output[:, active_cells] = coarse_simulation[:, np.newaxis]
        return output
    if not active_cells.all():
        active_output = _downscale_cell(
            observations[:, active_cells],
            coarse_simulation,
            fine_simulation[:, active_cells],
            weights[active_cells],
            observation_months,
            simulation_months,
            rotation_matrices=_cached_rotation_matrices(
                int(active_cells.sum()), len(rotation_matrices), random_seed
            ),
            n_quantiles=n_quantiles,
            random_seed=random_seed,
            lower_bound=lower_bound,
            lower_threshold=lower_threshold,
            upper_bound=upper_bound,
            upper_threshold=upper_threshold,
            if_all_invalid_use=if_all_invalid_use,
        )
        output[:] = np.nan
        output[:, active_cells] = active_output
        return output

    finite_column_arrays = (observations, fine_simulation)
    if np.isnan(if_all_invalid_use) and any(
        np.any(np.all(~np.isfinite(array), axis=0)) for array in finite_column_arrays
    ):
        return np.full_like(output, np.nan)

    means = [
        _average_valid(
            array,
            if_all_invalid_use,
            lower_bound,
            lower_threshold,
            upper_bound,
            upper_threshold,
        )
        for array in (observations, coarse_simulation, fine_simulation)
    ]
    for month in range(1, 13):
        observation_mask = observation_months == month
        simulation_mask = simulation_months == month
        if not observation_mask.any() or not simulation_mask.any():
            raise ValueError(f"no data found for calendar month {month}")
        monthly = (
            observations[observation_mask],
            coarse_simulation[simulation_mask],
            fine_simulation[simulation_mask],
        )
        prepared = []
        for values, mean in zip(monthly, means, strict=True):
            sampled = _sample_invalid(values, mean, random_seed)
            prepared.append(
                _randomize_censored(
                    sampled,
                    lower_bound,
                    lower_threshold,
                    upper_bound,
                    upper_threshold,
                    random_seed,
                )
            )
        adjusted = weighted_sum_preserving_mbcn(
            prepared[0],
            prepared[1],
            prepared[2],
            weights,
            rotation_matrices,
            n_quantiles,
        )
        output[simulation_mask] = _randomize_censored(
            adjusted,
            lower_bound,
            lower_threshold,
            upper_bound,
            upper_threshold,
            random_seed,
            inverse=True,
        )
    return output


def _group_fine_cells(
    data: xr.DataArray,
    grid: GridInfo,
    *,
    time_dimension: str | None,
) -> tuple[xr.DataArray, tuple[str, ...], tuple[str, ...]]:
    work = data.rename({"time": time_dimension}) if time_dimension else data
    work = work.drop_vars(list(grid.spatial_dims))
    coarse_dims = tuple(f"coarse_{dimension}" for dimension in grid.spatial_dims)
    within_dims = tuple(f"within_{dimension}" for dimension in grid.spatial_dims)
    constructors = {
        dimension: (coarse, within)
        for dimension, coarse, within in zip(
            grid.spatial_dims, coarse_dims, within_dims, strict=True
        )
    }
    grouped = work.coarsen(
        {
            dimension: factor
            for dimension, factor in zip(
                grid.spatial_dims, grid.factors, strict=True
            )
        },
        boundary="exact",
    ).construct(**constructors)
    grouped = grouped.stack(cell=within_dims)
    chunking = {"cell": -1}
    if time_dimension:
        chunking[time_dimension] = -1
    return grouped.chunk(chunking), coarse_dims, within_dims


def downscale_variable(
    observations: xr.DataArray,
    simulation: xr.DataArray,
    *,
    variable: str | None = None,
    iterations: int = 20,
    quantiles: int = 50,
    random_seed: int | None = 0,
    chunks: Mapping[str, int] | None = None,
    if_all_invalid_use: float | None = None,
) -> xr.DataArray:
    """Downscale one adjusted variable using faithful ISIMIP3 MBCnSD."""
    variable = variable or simulation.name
    if not variable:
        raise ValueError("a variable name is required")
    get_preset(variable)
    calendars = (str(observations.time.dt.calendar), str(simulation.time.dt.calendar))
    supported_calendars = {"proleptic_gregorian", "noleap"}
    if calendars[0] != calendars[1] or calendars[0] not in supported_calendars:
        raise ValueError(
            "MBCnSD requires matching noleap or proleptic_gregorian observation "
            f"and simulation calendars; found {calendars}"
        )
    expected_months = set(range(1, 13))
    for label, data in (("observations", observations), ("simulation", simulation)):
        months = set(np.unique(data.time.dt.month.values).tolist())
        if months != expected_months:
            raise ValueError(f"{label} do not contain all calendar months")
    if quantiles < 2:
        raise ValueError("quantiles must be at least 2")
    grid = analyze_input_grids(simulation, observations)
    observations = convert_units_to(observations, simulation)
    observations = observations.transpose("time", *grid.spatial_dims)
    simulation = simulation.transpose("time", *grid.spatial_dims)
    if chunks:
        fine_chunks = {**chunks, "time": -1}
        coarse_chunks = {**chunks, "time": -1}
        for dimension, factor in zip(
            grid.spatial_dims, grid.factors, strict=True
        ):
            requested = chunks.get(dimension)
            if requested in (None, -1):
                continue
            if requested < factor or requested % factor:
                raise ValueError(
                    f"fine-grid chunk size for {dimension} must be a positive "
                    f"multiple of its downscaling factor {factor}"
                )
            coarse_chunks[dimension] = requested // factor
        simulation = simulation.chunk(coarse_chunks)
        observations = observations.chunk(fine_chunks)

    initial = bilinear_broadcast(simulation, observations, grid)
    weights = grid_cell_weights(observations, grid)
    grouped_observations, coarse_dims, within_dims = _group_fine_cells(
        observations, grid, time_dimension="observation_time"
    )
    grouped_initial, _, _ = _group_fine_cells(
        initial, grid, time_dimension="simulation_time"
    )
    grouped_weights, _, _ = _group_fine_cells(weights, grid, time_dimension=None)

    coarse = simulation.rename(
        {
            "time": "simulation_time",
            **dict(zip(grid.spatial_dims, coarse_dims, strict=True)),
        }
    ).chunk({"simulation_time": -1})
    for original, renamed in zip(grid.spatial_dims, coarse_dims, strict=True):
        coordinate = simulation[original].rename({original: renamed})
        grouped_observations = grouped_observations.assign_coords({renamed: coordinate})
        grouped_initial = grouped_initial.assign_coords({renamed: coordinate})
        grouped_weights = grouped_weights.assign_coords({renamed: coordinate})

    rotations = generate_rotation_matrices(
        int(np.prod(grid.factors)), iterations, random_seed
    )
    lower_bound, lower_threshold, upper_bound, upper_threshold = _resolved_bounds(
        variable, simulation
    )
    configured_fallback = DOWNSCALING_BOUNDS[variable].if_all_invalid_use
    fallback = configured_fallback if if_all_invalid_use is None else if_all_invalid_use
    fallback = np.nan if fallback is None else float(fallback)
    observation_months = observations.time.dt.month.rename(
        {"time": "observation_time"}
    )
    simulation_months = simulation.time.dt.month.rename({"time": "simulation_time"})

    result = xr.apply_ufunc(
        _downscale_cell,
        grouped_observations,
        coarse,
        grouped_initial,
        grouped_weights,
        observation_months,
        simulation_months,
        input_core_dims=[
            ["observation_time", "cell"],
            ["simulation_time"],
            ["simulation_time", "cell"],
            ["cell"],
            ["observation_time"],
            ["simulation_time"],
        ],
        output_core_dims=[["simulation_time", "cell"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[simulation.dtype],
        dask_gufunc_kwargs={"allow_rechunk": True},
        kwargs={
            "rotation_matrices": rotations,
            "n_quantiles": quantiles,
            "random_seed": random_seed,
            "lower_bound": lower_bound,
            "lower_threshold": lower_threshold,
            "upper_bound": upper_bound,
            "upper_threshold": upper_threshold,
            "if_all_invalid_use": fallback,
        },
    )
    result = result.unstack("cell")
    order: list[str] = ["simulation_time"]
    for coarse_dimension, within_dimension in zip(
        coarse_dims, within_dims, strict=True
    ):
        order.extend((coarse_dimension, within_dimension))
    values = result.transpose(*order).data.reshape(
        (simulation.sizes["time"],)
        + tuple(observations.sizes[dimension] for dimension in grid.spatial_dims)
    )
    output = xr.DataArray(
        values,
        coords={
            "time": simulation.time,
            **{dimension: observations[dimension] for dimension in grid.spatial_dims},
        },
        dims=("time", *grid.spatial_dims),
        name=variable,
        attrs=dict(simulation.attrs),
    )
    output.attrs.update(
        statistical_downscaling_method="MBCnSD",
        statistical_downscaling_iterations=iterations,
        statistical_downscaling_quantiles=quantiles,
        statistical_downscaling_random_seed=random_seed,
        statistical_downscaling_software=(
            f"isimip3basd-modern/{__version__}; xarray/{version('xarray')}; "
            f"scipy/{version('scipy')}"
        ),
        statistical_downscaling_source=(
            "ISIMIP3BASD/3.0.2; https://doi.org/10.5281/zenodo.7151476"
        ),
        statistical_downscaling_created_utc=datetime.now(timezone.utc).isoformat(),
    )
    return output
