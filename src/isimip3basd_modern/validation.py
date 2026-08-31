"""Physical output validation using xclim data-quality routines."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import dask
import numpy as np
import xarray as xr
from xclim.core.units import convert_units_to
from xclim.core.dataflags import (
    negative_accumulation_values,
    percentage_values_outside_of_bounds,
    wind_values_outside_of_bounds,
)
from xclim.indices import clearness_index

from .presets import get_preset


@dataclass(frozen=True)
class ValidationReport:
    variable: str
    valid: bool
    finite: bool
    physical_bounds: bool
    minimum: float
    maximum: float
    units: str

    def to_dict(self) -> dict[str, str | bool | float]:
        return asdict(self)


def validate_variable(data: xr.DataArray, variable: str) -> ValidationReport:
    """Compute finite-value and variable-specific physical checks."""
    get_preset(variable)
    minimum, maximum, finite = dask.compute(
        data.min(skipna=False),
        data.max(skipna=False),
        np.isfinite(data).all(),
    )
    minimum_value = float(minimum)
    maximum_value = float(maximum)
    finite_value = bool(finite)

    if variable == "hurs":
        physical = not bool(
            percentage_values_outside_of_bounds(data).any().compute()
        )
    elif variable == "pr":
        physical = not bool(negative_accumulation_values(data).any().compute())
    elif variable == "sfcWind":
        physical = not bool(wind_values_outside_of_bounds(data).any().compute())
    elif variable in {"prsnratio", "tasskew"}:
        physical = minimum_value >= 0 and maximum_value <= 1
    elif variable == "rsds":
        index = clearness_index(data)
        index_minimum, index_maximum = dask.compute(index.min(), index.max())
        physical = float(index_minimum) >= 0 and float(index_maximum) <= 1
    elif variable == "tas":
        temperature_kelvin = convert_units_to(data, "K")
        physical = float(temperature_kelvin.min().compute()) >= 0
    elif variable in {"ps", "rlds", "tasrange"}:
        physical = minimum_value >= 0
    else:  # pragma: no cover - guarded by get_preset
        physical = False

    return ValidationReport(
        variable=variable,
        valid=finite_value and physical,
        finite=finite_value,
        physical_bounds=physical,
        minimum=minimum_value,
        maximum=maximum_value,
        units=data.attrs.get("units", ""),
    )
