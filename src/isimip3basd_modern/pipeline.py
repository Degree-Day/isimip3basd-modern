"""xarray/xsdba bias-adjustment pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import shutil
from typing import Literal
from uuid import uuid4

import dask.array as da
import numpy as np
import xarray as xr
from xclim.core.units import convert_units_to
from xclim.indices import (
    clearness_index,
    shortwave_downwelling_radiation_from_clearness_index,
)
from xsdba.adjustment import (
    DetrendedQuantileMapping,
    QuantileDeltaMapping,
    Scaling,
)
from xsdba.base import Grouper
from xsdba.processing import from_additive_space, to_additive_space

from . import __version__
from .presets import VariablePreset, get_preset

Method = Literal["qdm", "dqm", "scaling"]
Kind = Literal["additive", "multiplicative"]


def _prepare(
    reference: xr.DataArray,
    historical: xr.DataArray,
    simulation: xr.DataArray,
    chunks: Mapping[str, int] | None,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    if "time" not in reference.dims:
        raise ValueError("reference variable has no time dimension")
    if set(reference.dims) != set(historical.dims):
        raise ValueError("reference and historical variables have different dimensions")
    if set(historical.dims) != set(simulation.dims):
        raise ValueError("historical and simulation variables have different dimensions")

    order = ("time", *(dim for dim in reference.dims if dim != "time"))
    reference = reference.transpose(*order)
    historical = historical.transpose(*order)
    simulation = simulation.transpose(*order)

    reference, historical = xr.align(reference, historical, join="exact")
    for dimension in order[1:]:
        if historical.sizes[dimension] != simulation.sizes[dimension]:
            raise ValueError(f"historical and simulation sizes differ for {dimension}")
        if (
            dimension in historical.coords
            and dimension in simulation.coords
            and not historical[dimension].equals(simulation[dimension])
        ):
            raise ValueError(
                f"historical and simulation coordinates differ for {dimension}"
            )
    historical = convert_units_to(historical, reference)
    simulation = convert_units_to(simulation, reference)

    requested_chunks = dict(chunks or {})
    requested_chunks["time"] = -1
    if requested_chunks:
        reference = reference.chunk(requested_chunks)
        historical = historical.chunk(requested_chunks)
        simulation = simulation.chunk(requested_chunks)
    return reference, historical, simulation


def adjust(
    reference: xr.DataArray,
    historical: xr.DataArray,
    simulation: xr.DataArray,
    *,
    method: Method = "qdm",
    kind: Kind = "additive",
    group: str | None = None,
    window: int = 1,
    quantiles: int = 50,
    interpolation: str = "nearest",
    extrapolation: str = "constant",
    chunks: Mapping[str, int] | None = None,
    adapt_freq_thresh: str | None = None,
    random_seed: int | None = 0,
    fit_cache_path: str | Path | None = None,
    fit_cache_key: str | None = None,
) -> xr.DataArray:
    """Train an xsdba adjustment and apply it to a simulation."""
    if quantiles < 2:
        raise ValueError("quantiles must be at least 2")
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")
    if adapt_freq_thresh is not None and random_seed is not None:
        np.random.seed(random_seed)
        da.random.seed(random_seed)

    reference, historical, simulation = _prepare(
        reference, historical, simulation, chunks
    )
    if group is None:
        group = "time.dayofyear" if method == "dqm" else "time.month"
    adjustment_kind = "+" if kind == "additive" else "*"
    grouper = Grouper(group, window=window)

    if fit_cache_path is not None and not fit_cache_key:
        raise ValueError("fit_cache_key is required when fit_cache_path is set")

    if method == "qdm":
        adjustment_class = QuantileDeltaMapping
        train_kwargs = {
            "nquantiles": quantiles,
            "kind": adjustment_kind,
            "group": grouper,
            "adapt_freq_thresh": adapt_freq_thresh,
            "jitter_under_thresh_value": adapt_freq_thresh,
        }
    elif method == "dqm":
        adjustment_class = DetrendedQuantileMapping
        train_kwargs = {
            "nquantiles": quantiles,
            "kind": adjustment_kind,
            "group": grouper,
            "adapt_freq_thresh": adapt_freq_thresh,
            "jitter_under_thresh_value": adapt_freq_thresh,
        }
    elif method == "scaling":
        adjustment_class = Scaling
        train_kwargs = {"kind": adjustment_kind, "group": grouper}
    else:
        raise ValueError(f"unknown method: {method}")

    cache_path = Path(fit_cache_path) if fit_cache_path is not None else None
    cache_hit = False
    trained = None
    if cache_path is not None and cache_path.exists():
        cached = xr.open_zarr(cache_path, consolidated=False)
        if cached.attrs.get("isimip3basd_fit_cache_key") == fit_cache_key:
            trained = adjustment_class.from_dataset(cached)
            cache_hit = True
        else:
            cached.close()
            shutil.rmtree(cache_path)

    if trained is None:
        trained = adjustment_class.train(
            reference,
            historical,
            **train_kwargs,
        )
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_name(
                f".{cache_path.name}.tmp-{uuid4().hex}"
            )
            cache_dataset = trained.ds.copy()
            cache_dataset.attrs.update(
                isimip3basd_fit_cache_schema=1,
                isimip3basd_fit_cache_key=fit_cache_key,
                isimip3basd_fit_cache_method=method,
                isimip3basd_fit_cache_created_utc=(
                    datetime.now(timezone.utc).isoformat()
                ),
            )
            cache_dataset.to_zarr(
                temporary, mode="w", consolidated=False, zarr_format=3
            )
            if cache_path.exists():
                shutil.rmtree(temporary)
            else:
                temporary.rename(cache_path)

    if method == "qdm":
        result = trained.adjust(
            simulation,
            interp=interpolation,
            extrapolation=extrapolation,
        )
    elif method == "dqm":
        result = trained.adjust(
            simulation,
            interp=interpolation,
            extrapolation=extrapolation,
        )
    elif method == "scaling":
        result = trained.adjust(simulation, interp=interpolation)

    result = result.transpose(*simulation.dims)
    result.name = simulation.name
    provenance = {
        "bias_adjustment_method": method,
        "bias_adjustment_kind": kind,
        "bias_adjustment_group": group,
        "bias_adjustment_software": (
            f"isimip3basd-modern/{__version__}; "
            f"xclim/{version('xclim')}; xsdba/{version('xsdba')}"
        ),
        "bias_adjustment_created_utc": datetime.now(timezone.utc).isoformat(),
    }
    if method in {"qdm", "dqm"}:
        provenance["bias_adjustment_quantiles"] = quantiles
    if adapt_freq_thresh is not None:
        provenance["bias_adjustment_adapt_frequency_threshold"] = (
            adapt_freq_thresh
        )
        provenance["bias_adjustment_random_seed"] = random_seed
    if cache_path is not None:
        provenance["bias_adjustment_fit_cache"] = str(cache_path)
        provenance["bias_adjustment_fit_cache_hit"] = cache_hit
        provenance["bias_adjustment_fit_cache_key"] = fit_cache_key
    result.attrs.update(provenance)
    return result


def _quantity_in_units(quantity: str | None, target: xr.DataArray) -> str | None:
    if quantity is None:
        return None
    value = convert_units_to(quantity, target, context="infer")
    return f"{float(value):.17g} {target.attrs.get('units', '')}".strip()


def _threshold_to_bound(
    data: xr.DataArray,
    *,
    lower_bound: str | None,
    lower_threshold: str | None,
    upper_bound: str | None,
    upper_threshold: str | None,
) -> xr.DataArray:
    if lower_bound is not None:
        bound = convert_units_to(lower_bound, data, context="infer")
        threshold = convert_units_to(
            lower_threshold or lower_bound, data, context="infer"
        )
        data = data.where(data >= threshold, bound)
        data = data.clip(min=bound)
    if upper_bound is not None:
        bound = convert_units_to(upper_bound, data, context="infer")
        threshold = convert_units_to(
            upper_threshold or upper_bound, data, context="infer"
        )
        data = data.where(data <= threshold, bound)
        data = data.clip(max=bound)
    return data


def _to_logit(data: xr.DataArray, preset: VariablePreset) -> xr.DataArray:
    lower = _quantity_in_units(preset.lower_bound, data)
    upper = _quantity_in_units(preset.upper_bound, data)
    if lower is None or upper is None:
        raise ValueError("logit presets require lower and upper bounds")
    lower_threshold = convert_units_to(
        preset.lower_threshold or preset.lower_bound, data, context="infer"
    )
    upper_threshold = convert_units_to(
        preset.upper_threshold or preset.upper_bound, data, context="infer"
    )
    data = data.clip(min=lower_threshold, max=upper_threshold)
    return to_additive_space(
        data,
        lower_bound=lower,
        upper_bound=upper,
        trans="logit",
        clip_next_to_bounds="strict",
    )


def _restore_boundary_masks(
    result: xr.DataArray,
    source: xr.DataArray,
    preset: VariablePreset,
) -> xr.DataArray:
    if preset.lower_bound is not None:
        source_threshold = convert_units_to(
            preset.lower_threshold or preset.lower_bound, source, context="infer"
        )
        result_bound = convert_units_to(
            preset.lower_bound, result, context="infer"
        )
        result = result.where(source > source_threshold, result_bound)
    if preset.upper_bound is not None:
        source_threshold = convert_units_to(
            preset.upper_threshold or preset.upper_bound, source, context="infer"
        )
        result_bound = convert_units_to(
            preset.upper_bound, result, context="infer"
        )
        result = result.where(source < source_threshold, result_bound)
    return result


def adjust_variable(
    reference: xr.DataArray,
    historical: xr.DataArray,
    simulation: xr.DataArray,
    *,
    variable: str | None = None,
    method: Method | None = None,
    kind: Kind | None = None,
    group: str | None = None,
    window: int | None = None,
    quantiles: int = 50,
    interpolation: str = "nearest",
    extrapolation: str = "constant",
    chunks: Mapping[str, int] | None = None,
    random_seed: int | None = 0,
    fit_cache_path: str | Path | None = None,
    fit_cache_key: str | None = None,
) -> xr.DataArray:
    """Adjust one of the ten supported ISIMIP variables using its preset."""
    variable = variable or simulation.name
    if not variable:
        raise ValueError("a variable name is required to select a preset")
    preset = get_preset(variable)
    selected_window = preset.window if window is None else window

    reference, historical, simulation = _prepare(
        reference, historical, simulation, chunks
    )
    original_simulation = simulation
    original_units = simulation.attrs.get("units", "")
    boundary_source = simulation

    if preset.transform == "clearness_index":
        reference = clearness_index(reference)
        historical = clearness_index(historical)
        simulation = clearness_index(simulation)
        boundary_source = simulation
        reference = _to_logit(reference, preset)
        historical = _to_logit(historical, preset)
        simulation = _to_logit(simulation, preset)
    elif preset.transform == "logit":
        reference = _to_logit(reference, preset)
        historical = _to_logit(historical, preset)
        simulation = _to_logit(simulation, preset)

    adapt_freq_thresh = None
    if preset.adapt_frequency:
        adapt_freq_thresh = _quantity_in_units(
            preset.lower_threshold, reference
        )

    result = adjust(
        reference,
        historical,
        simulation,
        method=method or preset.method,
        kind=kind or preset.kind,
        group=group or preset.group,
        window=selected_window,
        quantiles=quantiles,
        interpolation=interpolation,
        extrapolation=extrapolation,
        chunks=chunks,
        adapt_freq_thresh=adapt_freq_thresh,
        random_seed=random_seed,
        fit_cache_path=fit_cache_path,
        fit_cache_key=fit_cache_key,
    )
    adjustment_attrs = dict(result.attrs)

    if preset.transform in {"logit", "clearness_index"}:
        result = from_additive_space(
            result,
            lower_bound=preset.lower_bound,
            upper_bound=preset.upper_bound,
            trans="logit",
            units=reference.attrs.get("xsdba_transform_units", "1"),
        )
        result = _threshold_to_bound(
            result,
            lower_bound=preset.lower_bound,
            lower_threshold=preset.lower_threshold,
            upper_bound=preset.upper_bound,
            upper_threshold=preset.upper_threshold,
        )
        result = _restore_boundary_masks(result, boundary_source, preset)

    if preset.transform == "clearness_index":
        result = shortwave_downwelling_radiation_from_clearness_index(result)
        result = convert_units_to(result, original_units)
    elif preset.transform is None:
        result = _threshold_to_bound(
            result,
            lower_bound=preset.lower_bound,
            lower_threshold=preset.lower_threshold,
            upper_bound=preset.upper_bound,
            upper_threshold=preset.upper_threshold,
        )

    result = result.transpose(*original_simulation.dims)
    result.name = variable
    result.attrs.update(adjustment_attrs)
    result.attrs["units"] = original_units
    result.attrs["bias_adjustment_preset"] = variable
    result.attrs["bias_adjustment_window"] = selected_window
    if preset.transform:
        result.attrs["bias_adjustment_transform"] = preset.transform
    return result
