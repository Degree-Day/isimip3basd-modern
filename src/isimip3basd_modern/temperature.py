"""Direct regional spatial downscaling of a complete, nested temperature triplet."""

import warnings

import numpy as np
import xarray as xr
from scipy.linalg import LinAlgWarning, solve
from scipy.ndimage import uniform_filter1d

from .downscaling import analyze_input_grids, apply_downscaled_value_controls, downscale_variable
from .validation import derive_variables

COMPONENTS = ("tas", "tasrange", "tasskew")


def downscale_temperature(reference, parent, method="joint-kriging"):
    """Downscale daily noleap temperature components; return unpacked fields in kelvin."""
    reference = reference[list(COMPONENTS)].transpose("time", "lat", "lon")
    parent = parent[list(COMPONENTS)].transpose("time", "lat", "lon")
    if method == "mbcnsd":
        fields = {}
        for name in COMPONENTS:
            field = downscale_variable(reference[name], parent[name], variable=name, eager=True)
            fields[name] = apply_downscaled_value_controls(field, name)
        return derive_variables(xr.Dataset(fields)).assign_attrs(spatial_method=method)
    if method != "joint-kriging":
        raise ValueError(f"Unknown temperature downscaling method: {method}")
    grid = analyze_input_grids(parent.tas, reference.tas)
    ny, nx = parent.tas.shape[1:]
    fy, fx = grid.factors
    for dim, factor in zip(("lat", "lon"), (fy, fx)):
        if not np.allclose(reference[dim].values.reshape(-1, factor).mean(axis=1),
                           parent[dim].values, atol=1e-7, rtol=0):
            raise ValueError(f"Misaligned parent and reference {dim} coordinates")
    shape = reference.tas.shape[1:]
    blocks = (ny, fy, nx, fx)
    land = np.isfinite(reference.tas.values[0])
    active = land.reshape(blocks).any(axis=(1, 3))
    if not active.any():
        raise ValueError("No land in the selected neighborhood")
    for ds, support in ((reference, land), (parent, active)):
        if ds.sizes["time"] == 0 or ds.time.dt.calendar != "noleap":
            raise ValueError("Expected consecutive noleap days")
        expected = xr.date_range(start=ds.time.values[0], periods=ds.sizes["time"], freq="D",
                                 calendar="noleap", use_cftime=True)
        if not ds.indexes["time"].equals(expected):
            raise ValueError("Expected consecutive noleap days")
        if not all(np.isfinite(ds[name].values[:, support]).all() for name in COMPONENTS):
            raise ValueError("Regional joint kriging requires complete component fields")
        if ds is reference and any(np.isfinite(ds[v].values[:, ~land]).any() for v in COMPONENTS):
            raise ValueError("Reference coverage changes by day or variable")
        if ds.tasrange.min() < 0 or ds.tasskew.min() < 0 or ds.tasskew.max() > 1:
            raise ValueError("Invalid input temperature range or skew")
    if reference.time.dt.dayofyear[0] != 1 or reference.time.dt.dayofyear[-1] != 365:
        raise ValueError("The reference must span whole calendar years")

    area = np.cos(np.deg2rad(reference.lat.values))[:, None] * land
    reference = reference.where(land, 0)
    block_area = area.reshape(blocks).sum(axis=(1, 3))

    def aggregate(field):
        return (field * area).reshape(-1, *blocks).sum(axis=(2, 4)) / np.where(active, block_area, 1)

    ref_days = reference.time.dt.dayofyear.values - 1
    sim_days = parent.time.dt.dayofyear.values - 1
    cycles = {
        v: uniform_filter1d(reference[v].values.astype("float64").reshape(-1, 365, *shape).mean(0),
                            31, axis=0, mode="wrap") for v in COMPONENTS
    }
    output = {v: cycles[v][sim_days].copy() for v in COMPONENTS}
    for month in np.unique(parent.time.dt.month):
        ref = reference.time.dt.month.values == month
        sim = parent.time.dt.month.values == month
        fine = np.hstack([(reference[v].values[ref] - cycles[v][ref_days[ref]]).reshape(ref.sum(), -1)
                          for v in COMPONENTS])
        coarse = np.hstack([aggregate(a.reshape(-1, *shape))[:, active]
                            for a in np.hsplit(fine, len(COMPONENTS))])
        simulation = np.hstack([(parent[v].values[sim] - aggregate(cycles[v])[sim_days[sim]])[:, active]
                                for v in COMPONENTS])
        fine -= fine.mean(0)
        coarse -= coarse.mean(0)
        scale = coarse.std(0)
        scale[scale < 1e-12] = 1.0
        coarse /= scale
        # One ordinary-cokriging unbiasedness constraint per component.
        design = np.repeat(np.eye(3), active.sum(), axis=0) / scale[:, None]
        matrix = np.block([[coarse.T @ coarse / len(coarse), design], [design.T, np.zeros((3, 3))]])
        rhs = np.vstack([coarse.T @ fine / len(coarse), np.repeat(np.eye(3), np.prod(shape), axis=1)])
        regularized = False
        with warnings.catch_warnings():
            warnings.simplefilter("error", LinAlgWarning)
            try:
                weights = solve(matrix, rhs, assume_a="sym")[:-3]
            except (np.linalg.LinAlgError, LinAlgWarning):
                # Identical predictors can occur on isolated coastal support.
                k = coarse.shape[1]
                matrix[:k, :k] += np.eye(k) * 1e-8
                weights = solve(matrix, rhs, assume_a="sym")[:-3]
                regularized = True
        for v, anomaly in zip(COMPONENTS, np.hsplit((simulation / scale) @ weights, 3)):
            output[v][sim] += anomaly.reshape(-1, *shape)
            if regularized:
                residual = np.where(active, parent[v].values[sim] - aggregate(output[v][sim]), 0)
                output[v][sim] += residual.repeat(fy, axis=1).repeat(fx, axis=2)

    invalid = (output["tasrange"] < 0) | (output["tasskew"] < 0) | (output["tasskew"] > 1)
    for name, upper in (("tasrange", np.inf), ("tasskew", 1)):
        values = output[name]
        bad = ((values < 0) | (values > upper)) & land
        for t, j, i in np.argwhere(bad.reshape(-1, *blocks).any(axis=(2, 4))):
            region = np.s_[j*fy:(j+1)*fy, i*fx:(i+1)*fx]
            cell = values[t][region]
            local_area = area[region]
            mean = np.average(cell, weights=local_area)
            hi = 2 * (np.max(abs(cell)) + 1)
            lo = -hi
            for _ in range(50):
                shift = (lo + hi) / 2
                if np.average(np.clip(cell + shift, 0, upper), weights=local_area) < mean:
                    lo = shift
                else:
                    hi = shift
            cell[:] = np.clip(cell + (lo + hi) / 2, 0, upper)
    for name in COMPONENTS:
        if not np.allclose(aggregate(output[name])[:, active], parent[name].values[:, active], atol=1e-6, rtol=0):
            raise ValueError(f"Regional cokriging failed to conserve {name}")
        output[name][:, ~land] = np.nan
    result = xr.Dataset({v: (("time", "lat", "lon"), output[v], reference[v].attrs) for v in COMPONENTS},
                        coords={"time": parent.time, "lat": reference.lat, "lon": reference.lon})
    return derive_variables(result).assign_attrs(spatial_method=method, component_bound_repairs=int((invalid & land).sum()))
