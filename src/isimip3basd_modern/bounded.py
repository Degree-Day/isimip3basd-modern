"""Bounded relative-humidity adjustment used by ISIMIP3BASD v3.

The numerical steps are ported from the AGPL-3.0 ISIMIP3BASD v3.0.2
``bias_adjustment.py`` and ``utility_functions.py`` implementation.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.stats import rankdata


def _percentile(values: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    values = np.sort(values)
    n = values.size - 1
    positions = n * probabilities
    below = np.floor(positions).astype(int)
    above_weight = positions - below
    return (
        values[below] * (1 - above_weight)
        + values[below + (below < n)] * above_weight
    )


def _bounded_signal_transfer(
    observed: np.ndarray,
    historical: np.ndarray,
    simulation: np.ndarray,
    lower: float,
    upper: float,
) -> np.ndarray:
    negative_bias = historical < observed
    zero_bias = historical == observed
    positive_bias = historical > observed
    additive = (negative_bias & (simulation < historical)) | (
        positive_bias & (simulation > historical)
    )
    result = np.empty_like(observed)
    result[negative_bias] = upper - (
        (upper - observed[negative_bias])
        * (upper - simulation[negative_bias])
        / (upper - historical[negative_bias])
    )
    result[zero_bias] = simulation[zero_bias]
    result[positive_bias] = lower + (
        (observed[positive_bias] - lower)
        * (simulation[positive_bias] - lower)
        / (historical[positive_bias] - lower)
    )
    result[additive] = (
        observed[additive] + simulation[additive] - historical[additive]
    )
    return np.clip(result, lower, upper)


def _bounded_qdm_target(
    observed: np.ndarray,
    historical: np.ndarray,
    simulation: np.ndarray,
    *,
    quantiles: int,
    lower: float,
    upper: float,
) -> np.ndarray:
    count = min(
        quantiles + 1,
        observed.size,
        historical.size,
        simulation.size,
    )
    if count < 2:
        return observed.copy()
    probabilities = np.linspace(0, 1, count)
    observed_q = _percentile(observed, probabilities)
    historical_q = _percentile(historical, probabilities)
    simulation_q = _percentile(simulation, probabilities)
    percent_points = np.interp(observed, observed_q, probabilities)
    future_at_p = np.interp(percent_points, probabilities, simulation_q)
    historical_at_p = np.interp(percent_points, probabilities, historical_q)
    observed_at_p = np.interp(percent_points, probabilities, observed_q)
    return _bounded_signal_transfer(
        observed_at_p,
        historical_at_p,
        future_at_p,
        lower,
        upper,
    )


def _constant_extrapolation_map(
    values: np.ndarray,
    source_quantiles: np.ndarray,
    target_quantiles: np.ndarray,
) -> np.ndarray:
    result = np.interp(values, source_quantiles, target_quantiles)
    below = values < source_quantiles[0]
    above = values > source_quantiles[-1]
    result[below] = values[below] + target_quantiles[0] - source_quantiles[0]
    result[above] = values[above] + target_quantiles[-1] - source_quantiles[-1]
    return result


def _randomize_censored(
    values: np.ndarray,
    *,
    lower: float,
    lower_threshold: float,
    upper: float,
    upper_threshold: float,
    seed: int,
) -> np.ndarray:
    result = values.copy()
    rng = np.random.RandomState(seed)
    for selected, bound, threshold in (
        (result <= lower_threshold, lower, lower_threshold),
        (result >= upper_threshold, upper, upper_threshold),
    ):
        count = int(selected.sum())
        if not count:
            continue
        random_values = bound + rng.uniform(0, 1, count) * (threshold - bound)
        permutation = rng.choice(count, size=count, replace=False)
        shuffled_ranks = rankdata(
            result[selected][permutation], method="ordinal"
        ).astype(int) - 1
        ranks = np.empty(count, dtype=int)
        ranks[permutation] = shuffled_ranks
        result[selected] = np.sort(random_values)[ranks]
    return result


def _adjust_window(
    observed: np.ndarray,
    historical: np.ndarray,
    simulation: np.ndarray,
    *,
    quantiles: int,
    seed: int,
) -> np.ndarray:
    lower, lower_threshold = 0.0, 0.01
    upper, upper_threshold = 100.0, 99.99
    observed = _randomize_censored(
        observed,
        lower=lower,
        lower_threshold=lower_threshold,
        upper=upper,
        upper_threshold=upper_threshold,
        seed=seed,
    )
    historical = _randomize_censored(
        historical,
        lower=lower,
        lower_threshold=lower_threshold,
        upper=upper,
        upper_threshold=upper_threshold,
        seed=seed,
    )
    simulation = _randomize_censored(
        simulation,
        lower=lower,
        lower_threshold=lower_threshold,
        upper=upper,
        upper_threshold=upper_threshold,
        seed=seed,
    )

    target = _bounded_qdm_target(
        observed,
        historical,
        simulation,
        quantiles=quantiles,
        lower=lower,
        upper=upper,
    )
    lower_probability = np.mean(observed <= lower_threshold)
    upper_probability = np.mean(observed >= upper_threshold)
    source_selected = np.ones(simulation.shape, dtype=bool)
    target_selected = np.ones(target.shape, dtype=bool)
    result = simulation.copy()

    lower_source = (
        _percentile(simulation, np.array([lower_probability]))[0]
        if lower_probability > 0
        else lower - 1e-8
    )
    is_lower = simulation <= lower_source
    source_selected &= ~is_lower
    target_selected &= target > lower_threshold
    result[is_lower] = lower

    upper_source = (
        _percentile(simulation, np.array([1 - upper_probability]))[0]
        if upper_probability > 0
        else upper + 1e-8
    )
    is_upper = simulation >= upper_source
    source_selected &= ~is_upper
    target_selected &= target < upper_threshold
    result[is_upper] = upper

    if source_selected.any() and target_selected.any():
        probabilities = np.linspace(0, 1, quantiles + 1)
        source_q = _percentile(simulation[source_selected], probabilities)
        target_q = _percentile(target[target_selected], probabilities)
        result[source_selected] = _constant_extrapolation_map(
            simulation[source_selected], source_q, target_q
        )
    result[result <= lower_threshold] = lower
    result[result >= upper_threshold] = upper
    return np.clip(result, lower, upper)


def _circular_distance(labels: np.ndarray, center: int, period: int) -> np.ndarray:
    direct = np.abs(labels - center)
    return np.minimum(direct, period - direct)


def _adjust_series(
    observed: np.ndarray,
    historical: np.ndarray,
    simulation: np.ndarray,
    observed_groups: np.ndarray,
    historical_groups: np.ndarray,
    simulation_groups: np.ndarray,
    *,
    quantiles: int,
    window: int,
    period: int,
    seed: int,
) -> np.ndarray:
    result = np.full(simulation.shape, np.nan, dtype=np.float64)
    if not (
        np.isfinite(observed).all()
        and np.isfinite(historical).all()
        and np.isfinite(simulation).all()
    ):
        return result
    half_window = window // 2
    for center in np.unique(simulation_groups):
        if period:
            observed_window = _circular_distance(
                observed_groups, center, period
            ) <= half_window
            historical_window = _circular_distance(
                historical_groups, center, period
            ) <= half_window
            simulation_window = _circular_distance(
                simulation_groups, center, period
            ) <= half_window
        else:
            observed_window = observed_groups == center
            historical_window = historical_groups == center
            simulation_window = simulation_groups == center
        adjusted = _adjust_window(
            observed[observed_window],
            historical[historical_window],
            simulation[simulation_window],
            quantiles=quantiles,
            seed=seed,
        )
        keep = simulation_groups[simulation_window] == center
        result[simulation_window.nonzero()[0][keep]] = adjusted[keep]
    return result


def adjust_relative_humidity(
    reference: xr.DataArray,
    historical: xr.DataArray,
    simulation: xr.DataArray,
    *,
    group: str = "time.dayofyear",
    window: int = 31,
    quantiles: int = 50,
    random_seed: int = 0,
) -> xr.DataArray:
    """Apply the ISIMIP3BASD bounded-QDM humidity configuration."""
    coordinate = group.removeprefix("time.")
    if coordinate not in {"dayofyear", "month"}:
        raise ValueError(f"unsupported humidity grouping: {group}")
    period = 365 if coordinate == "dayofyear" else 0
    ref = reference.rename(time="reference_time")
    hist = historical.rename(time="historical_time")
    sim = simulation.rename(time="simulation_time")
    ref_groups = getattr(reference.time.dt, coordinate).rename(
        time="reference_time"
    )
    hist_groups = getattr(historical.time.dt, coordinate).rename(
        time="historical_time"
    )
    sim_groups = getattr(simulation.time.dt, coordinate).rename(
        time="simulation_time"
    )
    output = xr.apply_ufunc(
        _adjust_series,
        ref,
        hist,
        sim,
        ref_groups,
        hist_groups,
        sim_groups,
        input_core_dims=[
            ["reference_time"],
            ["historical_time"],
            ["simulation_time"],
            ["reference_time"],
            ["historical_time"],
            ["simulation_time"],
        ],
        output_core_dims=[["simulation_time"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64],
        kwargs={
            "quantiles": quantiles,
            "window": window,
            "period": period,
            "seed": random_seed,
        },
    )
    output = output.rename(simulation_time="time").transpose(*simulation.dims)
    output = output.assign_coords(time=simulation.time)
    output.name = simulation.name
    return output
