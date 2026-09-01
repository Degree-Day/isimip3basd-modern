"""Preflight, physical, statistical, and cross-variable quality control."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import dask
import numpy as np
import xarray as xr
from xclim.core.dataflags import (
    data_flags,
    negative_accumulation_values,
    percentage_values_outside_of_bounds,
    wind_values_outside_of_bounds,
)
from xclim.core.units import convert_units_to
from xclim.indices import clearness_index, specific_humidity

from .presets import get_preset


STANDARD_NAMES: dict[str, tuple[str, ...]] = {
    "hurs": ("relative_humidity",),
    "pr": ("precipitation_flux",),
    "prsnratio": ("snowfall_precipitation_ratio",),
    "ps": ("surface_air_pressure",),
    "rlds": (
        "surface_downwelling_longwave_flux_in_air",
        "surface_downwelling_longwave_flux",
    ),
    "rsds": (
        "surface_downwelling_shortwave_flux_in_air",
        "surface_downwelling_shortwave_flux",
    ),
    "sfcWind": ("wind_speed",),
    "tas": ("air_temperature",),
    "tasrange": ("air_temperature_range",),
    "tasskew": ("air_temperature_skewness",),
}


@dataclass(frozen=True)
class ValidationReport:
    variable: str
    valid: bool
    finite: bool
    physical_bounds: bool
    minimum: float
    maximum: float
    units: str
    valid_fraction: float = 1.0
    all_missing_cells: int = 0
    partial_missing_cells: int = 0
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    checks: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QCReport:
    label: str
    valid: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    checks: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _time_calendar(data: xr.DataArray) -> str:
    try:
        return str(data.time.dt.calendar)
    except (AttributeError, TypeError):
        return "unknown"


def _time_checks(
    data: xr.DataArray, min_years: int | None
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    checks: dict[str, Any] = {}
    if "time" not in data.dims:
        return ["missing time dimension"], checks

    index = data.indexes.get("time")
    if index is None:
        return ["time has no coordinate index"], checks
    monotonic = bool(index.is_monotonic_increasing)
    unique = bool(index.is_unique)
    checks.update(time_monotonic=monotonic, time_unique=unique)
    if not monotonic:
        errors.append("time is not strictly increasing")
    if not unique:
        errors.append("time contains duplicate values")

    try:
        frequency = xr.infer_freq(index)
    except (TypeError, ValueError):
        frequency = None
    daily = frequency in {"D", "1D"}
    checks.update(time_frequency=frequency, daily_without_gaps=daily)
    if not daily:
        errors.append(f"time is not a complete daily sequence (inferred {frequency!r})")

    calendar = _time_calendar(data)
    checks["calendar"] = calendar
    if min_years is not None:
        days_per_year = 360 if calendar == "360_day" else 365
        training_days = int(data.sizes["time"])
        enough = training_days >= min_years * days_per_year
        checks.update(
            training_days=training_days,
            minimum_training_years=min_years,
            training_period_sufficient=enough,
        )
        if not enough:
            errors.append(
                f"training period has {training_days} days; at least "
                f"{min_years * days_per_year} are required"
            )
    return errors, checks


def _metadata_checks(
    data: xr.DataArray, variable: str
) -> tuple[list[str], list[str], dict[str, Any]]:
    errors: list[str] = []
    warnings: list[str] = []
    units = data.attrs.get("units", "")
    if not units:
        errors.append("variable has no units attribute")

    expected = STANDARD_NAMES[variable]
    standard_name = data.attrs.get("standard_name", "")
    if standard_name not in expected:
        warnings.append(
            f"standard_name is {standard_name!r}; expected one of {expected!r}"
        )

    coordinate_metadata: dict[str, bool] = {}
    coordinates = (
        (("lat", "latitude"), {"degrees_north", "degree_north", "degrees_N"}),
        (("lon", "longitude"), {"degrees_east", "degree_east", "degrees_E"}),
    )
    for names, expected_units in coordinates:
        name = next(
            (candidate for candidate in names if candidate in data.coords), None
        )
        if name is None:
            warnings.append(f"no {names[1]} coordinate found")
            continue
        coordinate = data[name]
        units_ok = coordinate.attrs.get("units") in expected_units
        if coordinate.ndim == 1:
            index = coordinate.to_index()
            unique = bool(index.is_unique)
            monotonic = bool(
                index.is_monotonic_increasing or index.is_monotonic_decreasing
            )
        else:
            unique = monotonic = True
        coordinate_metadata[f"{name}_units"] = units_ok
        coordinate_metadata[f"{name}_unique"] = unique
        coordinate_metadata[f"{name}_monotonic"] = monotonic
        if not units_ok:
            warnings.append(f"{name} has missing or non-CF units")
        if not unique or not monotonic:
            errors.append(f"{name} coordinate must be unique and monotonic")
        if name in {"lat", "latitude"}:
            lower, upper = dask.compute(coordinate.min(), coordinate.max())
            if float(lower) < -90 or float(upper) > 90:
                errors.append(f"{name} coordinate lies outside [-90, 90]")

    checks = {
        "units_present": bool(units),
        "standard_name": standard_name,
        "standard_name_recognized": standard_name in expected,
        **coordinate_metadata,
    }
    return errors, warnings, checks


def _coverage(
    data: xr.DataArray, min_valid_fraction: float
) -> tuple[bool, float, int, int, bool]:
    if not 0 < min_valid_fraction <= 1:
        raise ValueError("min_valid_fraction must be in (0, 1]")
    finite = np.isfinite(data)
    valid_count = finite.sum("time")
    active = valid_count > 0
    all_missing = valid_count == 0
    partial = active & (valid_count / data.sizes["time"] < min_valid_fraction)
    finite_fraction, all_missing_count, partial_count, has_inf = dask.compute(
        finite.sum() / data.size,
        all_missing.sum(),
        partial.sum(),
        np.isinf(data).any(),
    )
    return (
        not bool(has_inf) and int(partial_count) == 0,
        float(finite_fraction),
        int(all_missing_count),
        int(partial_count),
        bool(has_inf),
    )


def _physical_bounds(
    data: xr.DataArray,
    variable: str,
    minimum: float,
    maximum: float,
    allow_out_of_bounds_hurs: bool,
) -> bool:
    if variable == "hurs":
        if allow_out_of_bounds_hurs:
            return True
        return not bool(percentage_values_outside_of_bounds(data).any().compute())
    if variable == "pr":
        return not bool(negative_accumulation_values(data).any().compute())
    if variable == "sfcWind":
        return not bool(wind_values_outside_of_bounds(data).any().compute())
    if variable in {"prsnratio", "tasskew"}:
        return minimum >= 0 and maximum <= 1
    if variable == "rsds":
        index = clearness_index(data)
        lower, upper = dask.compute(index.min(skipna=True), index.max(skipna=True))
        return float(lower) >= 0 and float(upper) <= 1
    if variable == "tas":
        temperature_kelvin = convert_units_to(data, "K")
        return float(temperature_kelvin.min(skipna=True).compute()) >= 0
    if variable in {"ps", "rlds", "tasrange"}:
        return minimum >= 0
    return False  # pragma: no cover


def _statistical_warnings(
    data: xr.DataArray,
) -> tuple[list[str], dict[str, Any]]:
    try:
        flags = data_flags(data, dims="all")
        results: dict[str, Any] = {
            name: bool(flag.any().compute()) for name, flag in flags.data_vars.items()
        }
    except Exception as error:  # xclim flags depend on optional CF context
        return [f"xclim statistical flags could not run: {error}"], {}
    triggered = [
        f"xclim data flag triggered: {name}"
        for name, value in results.items()
        if value
    ]
    spatial_dims = [dimension for dimension in data.dims if dimension != "time"]
    series = data.mean(spatial_dims, skipna=True) if spatial_dims else data
    climatology = series.groupby("time.dayofyear").mean(skipna=True)
    changes = abs(climatology.diff("dayofyear")).compute().values
    maximum_jump = float(np.nanmax(changes))
    typical_jump = float(np.nanmedian(changes))
    jump_ratio = maximum_jump / typical_jump if typical_jump > 0 else 0.0
    discontinuity = bool(typical_jump > 0 and jump_ratio > 10)
    results["climatology_discontinuity"] = discontinuity
    results["climatology_max_to_median_jump_ratio"] = jump_ratio
    if discontinuity:
        triggered.append(
            "day-of-year climatology has an abrupt discontinuity "
            f"(maximum/median adjacent jump ratio {jump_ratio:.1f})"
        )
    return triggered, results


def validate_variable(
    data: xr.DataArray,
    variable: str,
    *,
    min_valid_fraction: float = 1.0,
    statistical: bool = True,
    allow_out_of_bounds_hurs: bool = False,
) -> ValidationReport:
    """Compute coverage, physical-bound, and xclim statistical checks."""
    get_preset(variable)
    coverage_ok, valid_fraction, all_missing, partial, has_inf = _coverage(
        data, min_valid_fraction
    )
    minimum, maximum = dask.compute(data.min(skipna=True), data.max(skipna=True))
    minimum_value = float(minimum)
    maximum_value = float(maximum)
    physical = _physical_bounds(
        data,
        variable,
        minimum_value,
        maximum_value,
        allow_out_of_bounds_hurs,
    )
    errors: list[str] = []
    if has_inf:
        errors.append("variable contains infinite values")
    if partial:
        errors.append(
            f"{partial} active spatial cells are below the required "
            f"{min_valid_fraction:.3f} valid fraction"
        )
    if not physical:
        errors.append("variable violates physical bounds")
    warnings, flags = _statistical_warnings(data) if statistical else ([], {})
    retained_out_of_bounds_hurs = variable == "hurs" and allow_out_of_bounds_hurs and (
        minimum_value < 0 or maximum_value > 100
    )
    if retained_out_of_bounds_hurs:
        warnings.append(
            "out-of-bounds model humidity retained for ISIMIP bounded-threshold "
            "bias adjustment"
        )
    return ValidationReport(
        variable=variable,
        valid=coverage_ok and physical,
        finite=coverage_ok,
        physical_bounds=physical,
        minimum=minimum_value,
        maximum=maximum_value,
        units=data.attrs.get("units", ""),
        valid_fraction=valid_fraction,
        all_missing_cells=all_missing,
        partial_missing_cells=partial,
        errors=tuple(errors),
        warnings=tuple(warnings),
        checks={
            "minimum_valid_fraction": min_valid_fraction,
            "out_of_bounds_hurs_retained": retained_out_of_bounds_hurs,
            "xclim_data_flags": flags,
        },
    )


def preflight_variable(
    data: xr.DataArray,
    variable: str,
    *,
    label: str,
    min_years: int | None = None,
    min_valid_fraction: float = 1.0,
) -> QCReport:
    """Validate one input before an expensive adjustment starts."""
    get_preset(variable)
    time_errors, time_checks = _time_checks(data, min_years)
    metadata_errors, metadata_warnings, metadata_checks = _metadata_checks(
        data, variable
    )
    variable_report = validate_variable(
        data, variable, min_valid_fraction=min_valid_fraction, statistical=True
    )
    errors = [*time_errors, *metadata_errors, *variable_report.errors]
    warnings = [*metadata_warnings, *variable_report.warnings]
    return QCReport(
        label=label,
        valid=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        checks={
            **time_checks,
            **metadata_checks,
            "variable": variable_report.to_dict(),
        },
    )


def validate_inputs(
    reference: xr.DataArray,
    historical: xr.DataArray,
    simulation: xr.DataArray,
    variable: str,
    *,
    min_training_years: int = 10,
    min_valid_fraction: float = 1.0,
) -> QCReport:
    """Run preflight checks and verify temporal and spatial compatibility."""
    reports = [
        preflight_variable(
            reference,
            variable,
            label="reference",
            min_years=min_training_years,
            min_valid_fraction=min_valid_fraction,
        ),
        preflight_variable(
            historical,
            variable,
            label="historical",
            min_years=min_training_years,
            min_valid_fraction=min_valid_fraction,
        ),
        preflight_variable(
            simulation,
            variable,
            label="simulation",
            min_valid_fraction=min_valid_fraction,
        ),
    ]
    errors = [
        f"{report.label}: {message}"
        for report in reports
        for message in report.errors
    ]
    warnings = [
        f"{report.label}: {message}"
        for report in reports
        for message in report.warnings
    ]
    calendars = [_time_calendar(array) for array in (reference, historical, simulation)]
    calendars_match = len(set(calendars)) == 1
    if not calendars_match:
        errors.append(f"input calendars differ: {calendars}")

    reference_historical_aligned = False
    try:
        xr.align(reference, historical, join="exact")
        reference_historical_aligned = True
    except ValueError:
        errors.append("reference and historical coordinates do not align exactly")

    spatial_grid_matches = True
    spatial_dims = tuple(dim for dim in historical.dims if dim != "time")
    if set(spatial_dims) != {dim for dim in simulation.dims if dim != "time"}:
        spatial_grid_matches = False
    else:
        for dim in spatial_dims:
            if historical.sizes[dim] != simulation.sizes[dim]:
                spatial_grid_matches = False
            elif (dim in historical.coords) != (dim in simulation.coords):
                spatial_grid_matches = False
            elif dim in historical.coords:
                spatial_grid_matches &= historical[dim].equals(simulation[dim])
        historical_spatial_coords = {
            name
            for name, coordinate in historical.coords.items()
            if "time" not in coordinate.dims
        }
        simulation_spatial_coords = {
            name
            for name, coordinate in simulation.coords.items()
            if "time" not in coordinate.dims
        }
        if historical_spatial_coords != simulation_spatial_coords:
            spatial_grid_matches = False
        else:
            spatial_grid_matches &= all(
                historical[name].equals(simulation[name])
                for name in historical_spatial_coords
            )
    if not spatial_grid_matches:
        errors.append("historical and simulation spatial grids differ")

    return QCReport(
        label="inputs",
        valid=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        checks={
            "calendars": calendars,
            "calendars_match": calendars_match,
            "reference_historical_aligned": reference_historical_aligned,
            "spatial_grid_matches": bool(spatial_grid_matches),
            "datasets": [report.to_dict() for report in reports],
        },
    )


def validate_output(
    output: xr.DataArray,
    simulation: xr.DataArray,
    variable: str,
    *,
    min_valid_fraction: float = 1.0,
) -> QCReport:
    """Validate output values and prove that adjustment preserved the input grid."""
    report = validate_variable(output, variable, min_valid_fraction=min_valid_fraction)
    grid_preserved = bool(
        output.dims == simulation.dims
        and output.sizes == simulation.sizes
        and set(output.coords) == set(simulation.coords)
        and all(
            output[coordinate].equals(simulation[coordinate])
            for coordinate in simulation.coords
        )
    )
    errors = list(report.errors)
    if not grid_preserved:
        errors.append("output dimensions or coordinates differ from simulation")
    return QCReport(
        label="output",
        valid=not errors,
        errors=tuple(errors),
        warnings=report.warnings,
        checks={"grid_preserved": grid_preserved, "variable": report.to_dict()},
    )


def derive_variables(dataset: xr.Dataset) -> xr.Dataset:
    """Derive physically linked variables when their required inputs are present."""
    result = dataset.copy()
    if {"tas", "tasrange", "tasskew"} <= set(result.data_vars):
        if "tasmin" not in result:
            result["tasmin"] = result.tas - result.tasskew * result.tasrange
            result.tasmin.attrs.update(
                units=result.tas.attrs.get("units", ""),
                standard_name="air_temperature",
                long_name="Daily Minimum Near-Surface Air Temperature",
            )
        if "tasmax" not in result:
            result["tasmax"] = result.tas + (1 - result.tasskew) * result.tasrange
            result.tasmax.attrs.update(
                units=result.tas.attrs.get("units", ""),
                standard_name="air_temperature",
                long_name="Daily Maximum Near-Surface Air Temperature",
            )
    if {"pr", "prsnratio"} <= set(result.data_vars):
        if "prsn" not in result:
            result["prsn"] = result.pr * result.prsnratio
            result.prsn.attrs.update(
                units=result.pr.attrs.get("units", ""), standard_name="snowfall_flux"
            )
    if {"tas", "hurs", "ps"} <= set(result.data_vars) and "huss" not in result:
        result["huss"] = specific_humidity(
            result.tas, result.hurs, result.ps, invalid_values="clip"
        )
    return result


def _all_close(
    actual: xr.DataArray,
    expected: xr.DataArray,
    *,
    rtol: float,
    atol: float,
) -> bool:
    tolerance = atol + rtol * abs(expected)
    return bool((abs(actual - expected) <= tolerance).all().compute())


def validate_dataset(dataset: xr.Dataset) -> QCReport:
    """Check linked temperature, snowfall, and humidity variables."""
    derived = derive_variables(dataset)
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}
    if {"tasmin", "tas", "tasmax"} <= set(derived.data_vars):
        ordered = bool(
            ((derived.tasmin <= derived.tas) & (derived.tas <= derived.tasmax))
            .all()
            .compute()
        )
        checks["tasmin_le_tas_le_tasmax"] = ordered
        if not ordered:
            errors.append("temperature ordering tasmin <= tas <= tasmax is violated")
    if {"tas", "tasrange", "tasskew", "tasmin", "tasmax"} <= set(dataset.data_vars):
        expected_min = dataset.tas - dataset.tasskew * dataset.tasrange
        expected_max = dataset.tas + (1 - dataset.tasskew) * dataset.tasrange
        consistent = _all_close(
            dataset.tasmin, expected_min, rtol=1e-6, atol=1e-6
        ) and _all_close(
            dataset.tasmax, expected_max, rtol=1e-6, atol=1e-6
        )
        checks["temperature_components_consistent"] = consistent
        if not consistent:
            errors.append("tasmin or tasmax is inconsistent with tasrange and tasskew")
    if {"prsn", "pr"} <= set(derived.data_vars):
        snowfall_bounded = bool(
            ((derived.prsn >= 0) & (derived.prsn <= derived.pr)).all().compute()
        )
        checks["snowfall_between_zero_and_precipitation"] = snowfall_bounded
        if not snowfall_bounded:
            errors.append("prsn must lie between zero and total precipitation")
    if {"pr", "prsnratio", "prsn"} <= set(dataset.data_vars):
        expected_snowfall = dataset.pr * dataset.prsnratio
        consistent = _all_close(
            dataset.prsn, expected_snowfall, rtol=1e-6, atol=1e-12
        )
        checks["snowfall_components_consistent"] = consistent
        if not consistent:
            errors.append("prsn is inconsistent with pr and prsnratio")
    if "huss" in derived:
        humidity_bounded = bool(
            ((derived.huss >= 0) & (derived.huss <= 1)).all().compute()
        )
        checks["specific_humidity_between_zero_and_one"] = humidity_bounded
        if not humidity_bounded:
            errors.append("huss must lie between zero and one")
    if {"tas", "hurs", "ps", "huss"} <= set(dataset.data_vars):
        expected_humidity = specific_humidity(
            dataset.tas, dataset.hurs, dataset.ps, invalid_values="clip"
        )
        supplied_humidity = convert_units_to(dataset.huss, expected_humidity)
        consistent = _all_close(
            supplied_humidity, expected_humidity, rtol=1e-4, atol=1e-8
        )
        checks["humidity_components_consistent"] = consistent
        if not consistent:
            errors.append("huss is inconsistent with tas, hurs, and ps")

    checks["derivable_variables"] = sorted(
        set(derived.data_vars) - set(dataset.data_vars)
    )
    if set(checks) == {"derivable_variables"}:
        warnings.append("no complete linked-variable groups were found")
    return QCReport(
        label="dataset",
        valid=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        checks=checks,
    )
