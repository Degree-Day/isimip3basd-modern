"""Screen and limit daily temperature triplets before publication packing."""

import numpy as np
import xarray as xr

from .publication import PACKING_SPECS
from .validation import derive_variables


LOWER_K = 183.15
UPPER_K = 343.15
COMPONENTS = ("tas", "tasrange", "tasskew")
EPSILON_K = 1e-9


def _range_limit(mean, skew):
    cold_limit = (mean - LOWER_K) / skew.where(skew > 0)
    warm_limit = (UPPER_K - mean) / (1 - skew).where(skew < 1)
    return np.minimum(
        np.minimum(cold_limit.fillna(np.inf), warm_limit.fillna(np.inf)),
        PACKING_SPECS["tasrange"].maximum,
    ).clip(min=0)


def _round_to_packing_grid(field):
    spec = PACKING_SPECS[field.name]
    return np.rint((field - spec.add_offset) / spec.scale_factor) * spec.scale_factor + spec.add_offset


def cap_triplet(data):
    """Keep the mean and skew and limit the valid diurnal range."""
    data = data[list(COMPONENTS)].astype("float64")
    mean, spread, skew = (data[name] for name in COMPONENTS)
    valid_components = (
        np.isfinite(mean) & np.isfinite(spread) & np.isfinite(skew)
        & (spread >= 0) & (skew >= 0) & (skew <= 1)
    )
    mean_outside = (mean < LOWER_K) | (mean > UPPER_K)
    valid = valid_components & ~mean_outside
    limit = _range_limit(mean, skew)
    capped = valid & (spread > limit + EPSILON_K)
    result = data.assign(tasrange=np.minimum(spread, limit)).where(valid)
    return result, capped, mean_outside, ~valid_components


def prepare_packed_triplet(data):
    """Round to the int16 grids while keeping reconstructed extrema in bounds."""
    rounded = {name: _round_to_packing_grid(data[name]) for name in COMPONENTS}
    limit = _range_limit(rounded["tas"], rounded["tasskew"])
    spec = PACKING_SPECS["tasrange"]
    safe_range = (
        np.floor((limit - spec.add_offset + EPSILON_K) / spec.scale_factor) * spec.scale_factor
        + spec.add_offset
    )
    extra_step = rounded["tasrange"] > limit + EPSILON_K
    rounded["tasrange"] = xr.where(extra_step, safe_range, rounded["tasrange"])
    result = derive_variables(data.assign({
        name: data[name].copy(data=rounded[name].data) for name in COMPONENTS
    }))
    return result, extra_step
