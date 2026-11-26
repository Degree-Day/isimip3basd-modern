"""Bounded area-to-area kriging for one humidity parent and its neighborhood."""

import numpy as np
from scipy.linalg import LinAlgWarning, solve
from scipy.ndimage import uniform_filter1d
import warnings


def downscale_humidity(reference, parent, rows, cols, factors):
    """Predict the central parent on finite reference support, in percent RH."""
    fine = reference.values.astype("float64")
    coarse = parent.values.astype("float64")
    if reference.time.dt.calendar != "noleap" or parent.time.dt.calendar != "noleap":
        raise ValueError("Humidity kriging requires noleap calendars")
    if len(fine) % 365 or int(reference.time.dt.dayofyear[0]) != 1:
        raise ValueError("Humidity kriging requires complete reference years")
    lat_factor, lon_factor = factors
    blocks = (coarse.shape[1], lat_factor, coarse.shape[2], lon_factor)
    support = np.isfinite(fine[0])
    if not np.all(np.isfinite(fine) == support):
        raise ValueError("Humidity reference support changes by day")
    area = np.cos(np.deg2rad(reference.lat.values))[:, None] * support
    block_area = area.reshape(blocks).sum(axis=(1, 3))
    active = block_area > 0
    center = (rows.start // lat_factor, cols.start // lon_factor)
    target = coarse[:, center[0], center[1]]
    if not np.isfinite(coarse[:, active]).all() or np.any((target < 0) | (target > 100)):
        raise ValueError("Missing or out-of-bounds adjusted humidity")
    weights = area[rows, cols][support[rows, cols]]
    weights /= weights.sum()
    fine[:, ~support] = 0

    def aggregate(values):
        totals = (values * area).reshape(-1, *blocks).sum(axis=(2, 4))
        return totals[:, active] / block_area[active]

    cycle = uniform_filter1d(fine.reshape(-1, 365, *support.shape).mean(axis=0),
                             31, axis=0, mode="wrap")
    ref_day = reference.time.dt.dayofyear.values - 1
    sim_day = parent.time.dt.dayofyear.values - 1
    ref_month = reference.time.dt.month.values
    sim_month = parent.time.dt.month.values
    center_support = support[rows, cols]
    center_cycle = cycle[:, rows, cols][:, center_support]
    coarse_cycle = aggregate(cycle)
    coarse_anomaly = aggregate(fine) - coarse_cycle[ref_day]
    fine_anomaly = fine[:, rows, cols][:, center_support] - center_cycle[ref_day]
    simulation = coarse[:, active] - coarse_cycle[sim_day]
    output = center_cycle[sim_day].copy()
    count = int(active.sum())
    for month in np.unique(sim_month):
        train = ref_month == month
        predict = sim_month == month
        coarse_train = coarse_anomaly[train]
        fine_train = fine_anomaly[train]
        coarse_train -= coarse_train.mean(axis=0)
        fine_train -= fine_train.mean(axis=0)
        matrix = np.ones((count + 1, count + 1))
        matrix[:count, :count] = coarse_train.T @ coarse_train / train.sum()
        matrix[-1, -1] = 0
        rhs = np.vstack((coarse_train.T @ fine_train / train.sum(), np.ones(len(weights))))
        with warnings.catch_warnings():
            warnings.simplefilter("error", LinAlgWarning)
            try:
                coefficients = solve(matrix, rhs, assume_a="sym")[:-1]
            except (LinAlgWarning, np.linalg.LinAlgError):
                matrix[:count, :count] += np.eye(count) * 1e-8
                coefficients = solve(matrix, rhs, assume_a="sym")[:-1]
        output[predict] += simulation[predict] @ coefficients

    output += (target - output @ weights)[:, None]
    extremes = (target == 0) | (target == 100)
    output[extremes] = target[extremes, None]
    repair = np.any((output < 0) | (output > 100), axis=1)
    values = output[repair]
    if len(values):
        low = -values.max(axis=1)
        high = 100 - values.min(axis=1)
        for _ in range(45):
            midpoint = (low + high) / 2
            below = np.clip(values + midpoint[:, None], 0, 100) @ weights < target[repair]
            low = np.where(below, midpoint, low)
            high = np.where(below, high, midpoint)
        output[repair] = np.clip(values + ((low + high) / 2)[:, None], 0, 100)
    if (not np.isfinite(output).all() or np.any((output < 0) | (output > 100))
            or not np.allclose(output @ weights, target, atol=1e-6, rtol=0)):
        raise ValueError("Bounded kriging failed humidity validation")
    result = np.full((len(target), lat_factor, lon_factor), np.nan)
    result[:, center_support] = output
    return result, int(repair.sum())
