"""Canonical-grid preprocessing for raw rectilinear climate-model output."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Literal

import numpy as np
import xarray as xr
import xarray_regrid  # noqa: F401 - registers the ``regrid`` accessor
from xclim.core.units import convert_units_to

from .validation import validate_variable

RegridMethod = Literal["linear", "conservative"]

TARGET_UNITS = {
    "hurs": "%",
    "pr": "kg m-2 s-1",
    "prsnratio": "1",
    "ps": "Pa",
    "rlds": "W m-2",
    "rsds": "W m-2",
    "sfcWind": "m s-1",
    "tas": "K",
    "tasrange": "K",
    "tasskew": "1",
}
REGRID_METHODS: dict[str, RegridMethod] = {
    "hurs": "linear",
    "pr": "conservative",
    "prsnratio": "linear",
    "ps": "linear",
    "rlds": "linear",
    "rsds": "linear",
    "sfcWind": "linear",
    "tas": "linear",
    "tasrange": "linear",
    "tasskew": "linear",
}
CLIP_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "pr": (0.0, None),
    "prsnratio": (0.0, 1.0),
    "ps": (0.0, None),
    "rlds": (0.0, None),
    "rsds": (0.0, None),
    "sfcWind": (0.0, None),
    "tasrange": (0.0, None),
    "tasskew": (0.0, 1.0),
}
CANONICAL_DIMS = ("time", "lat", "lon")
CANONICAL_COORDS = set(CANONICAL_DIMS)


@dataclass(frozen=True)
class PreprocessingReport:
    source: str
    output: str
    variable: str
    valid: bool
    source_calendar: str
    target_calendar: str
    calendar_day_delta: int
    regrid_method: str
    input_units: str
    interpreted_input_units: str
    output_units: str
    output_dtype: str
    output_shape: dict[str, int]
    precipitation_unit_repaired: bool
    physical_bound_correction: str
    validation: dict[str, object]
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def canonical_grid(resolution: float = 1.0) -> xr.Dataset:
    """Return a global cell-centered regular grid nested within 0..360."""
    cells_lat = round(180 / resolution)
    cells_lon = round(360 / resolution)
    if not np.isclose(cells_lat * resolution, 180) or not np.isclose(
        cells_lon * resolution, 360
    ):
        raise ValueError("resolution must divide both 180 and 360 exactly")
    latitude = -90 + resolution / 2 + resolution * np.arange(cells_lat)
    longitude = resolution / 2 + resolution * np.arange(cells_lon)
    return xr.Dataset(
        coords={
            "lat": xr.DataArray(
                latitude,
                dims="lat",
                attrs={"standard_name": "latitude", "units": "degrees_north"},
            ),
            "lon": xr.DataArray(
                longitude,
                dims="lon",
                attrs={"standard_name": "longitude", "units": "degrees_east"},
            ),
        }
    )


def _standardize_coordinates(data: xr.DataArray) -> xr.DataArray:
    renames = {}
    for source, target in (("latitude", "lat"), ("longitude", "lon")):
        if source in data.dims or source in data.coords:
            renames[source] = target
    data = data.rename(renames)
    missing = {"time", "lat", "lon"} - set(data.dims)
    if missing:
        raise ValueError(f"input is missing required dimensions: {sorted(missing)}")
    extra = [dimension for dimension in data.dims if dimension not in {"time", "lat", "lon"}]
    for dimension in extra:
        if data.sizes[dimension] != 1:
            raise ValueError(f"unsupported non-scalar dimension {dimension!r}")
        data = data.isel({dimension: 0}, drop=True)
    longitude = np.mod(np.asarray(data.lon.values, dtype=np.float64), 360)
    data = data.assign_coords(lon=longitude).sortby("lon").sortby("lat")
    if np.unique(data.lon).size != data.sizes["lon"]:
        raise ValueError("longitude normalization produced duplicate coordinates")
    data.lat.attrs.update(standard_name="latitude", units="degrees_north")
    data.lon.attrs.update(standard_name="longitude", units="degrees_east")
    return _drop_cruft_coordinates(data).transpose(*CANONICAL_DIMS)


def _drop_cruft_coordinates(data: xr.DataArray) -> xr.DataArray:
    """Keep only the canonical time/lat/lon coordinates on a climate variable."""
    drop_names = [name for name in data.coords if name not in CANONICAL_COORDS]
    if drop_names:
        data = data.drop_vars(drop_names)
    return data


def _extend_periodic_longitude(data: xr.DataArray) -> xr.DataArray:
    lower = data.isel(lon=[-1]).assign_coords(lon=[float(data.lon[-1]) - 360])
    upper = data.isel(lon=[0]).assign_coords(lon=[float(data.lon[0]) + 360])
    return xr.concat((lower, data, upper), dim="lon")


def _extend_poles(data: xr.DataArray) -> xr.DataArray:
    pieces = []
    if float(data.lat[0]) > -90:
        pieces.append(data.isel(lat=[0]).assign_coords(lat=[-90.0]))
    pieces.append(data)
    if float(data.lat[-1]) < 90:
        pieces.append(data.isel(lat=[-1]).assign_coords(lat=[90.0]))
    return xr.concat(pieces, dim="lat") if len(pieces) > 1 else data


def _regrid(
    data: xr.DataArray,
    target: xr.Dataset,
    method: RegridMethod,
    spatial_chunk: int,
) -> xr.DataArray:
    source = _extend_periodic_longitude(data)
    output_chunks = {"lat": spatial_chunk, "lon": spatial_chunk}
    if method == "linear":
        source = _extend_poles(source)
        result = source.regrid.linear(target)
    else:
        result = source.regrid.conservative(
            target,
            latitude_coord="lat",
            skipna=False,
            output_chunks=output_chunks,
        )
    return result.chunk({"time": 365, **output_chunks})


def _normalize_calendar(
    data: xr.DataArray,
    variable: str,
) -> tuple[xr.DataArray, str, int]:
    source_calendar = str(data.time.dt.calendar)
    if source_calendar == "noleap":
        converted = data
    else:
        conversion_source = data
        if variable == "pr" and source_calendar != "360_day":
            leap_day = (data.time.dt.month == 2) & (data.time.dt.day == 29)
            leap_accumulation = data.where(leap_day, 0).shift(
                time=-1, fill_value=0
            )
            conversion_source = data + leap_accumulation
        converted = conversion_source.convert_calendar(
            "noleap",
            align_on="year",
            missing=np.nan,
            use_cftime=True,
        )
    day_delta = converted.sizes["time"] - data.sizes["time"]
    if day_delta > 0:
        converted = converted.chunk({"time": -1}).interpolate_na(
            "time", method="linear"
        )
    if variable == "pr" and source_calendar == "360_day" and day_delta != 0:
        source_totals = data.groupby("time.year").sum("time", skipna=False)
        target_totals = converted.groupby("time.year").sum("time", skipna=False)
        ratios = xr.where(target_totals != 0, source_totals / target_totals, 1)
        converted = converted.groupby("time.year") * ratios
    first = converted.time.dt
    start = (
        f"{int(first.year[0]):04d}-{int(first.month[0]):02d}-"
        f"{int(first.day[0]):02d}"
    )
    canonical_time = xr.date_range(
        start,
        periods=converted.sizes["time"],
        freq="D",
        calendar="noleap",
        use_cftime=True,
    )
    converted = converted.assign_coords(time=canonical_time)
    return converted, source_calendar, day_delta


def preprocess_variable(
    data: xr.DataArray,
    variable: str,
    *,
    source_path: str = "",
    input_units_override: str | None = None,
    resolution: float = 1.0,
    spatial_chunk: int = 20,
) -> tuple[xr.DataArray, dict[str, object]]:
    """Standardize, regrid, normalize time, and encode one model variable."""
    if variable not in TARGET_UNITS:
        raise ValueError(f"unsupported preprocessing variable {variable!r}")
    source = _standardize_coordinates(data)
    original_units = str(source.attrs.get("units", ""))
    interpreted_units = input_units_override or original_units
    source.attrs["units"] = interpreted_units
    source = convert_units_to(
        source,
        TARGET_UNITS[variable],
        context="hydro" if variable == "pr" else None,
    )
    bounds = CLIP_BOUNDS.get(variable)
    if bounds is not None:
        source = source.clip(min=bounds[0], max=bounds[1])
    method = REGRID_METHODS[variable]
    regridded = _regrid(source, canonical_grid(resolution), method, spatial_chunk)
    normalized, source_calendar, day_delta = _normalize_calendar(regridded, variable)
    output = _drop_cruft_coordinates(normalized).astype("float32").chunk(
        {"time": 365, "lat": spatial_chunk, "lon": spatial_chunk}
    )
    output.lat.attrs.update(standard_name="latitude", units="degrees_north")
    output.lon.attrs.update(standard_name="longitude", units="degrees_east")
    output.name = variable
    output.attrs.update(data.attrs)
    output.attrs.update(
        units=TARGET_UNITS[variable],
        preprocessing_grid_resolution_degrees=resolution,
        preprocessing_regrid_method=method,
        preprocessing_calendar="noleap",
        preprocessing_physical_bound_correction=(
            f"clip to [{bounds[0]}, {bounds[1]}]" if bounds else "none"
        ),
        preprocessing_source=source_path,
        preprocessing_created_utc=datetime.now(timezone.utc).isoformat(),
    )
    if input_units_override:
        output.attrs.update(
            preprocessing_original_units=original_units,
            preprocessing_interpreted_input_units=input_units_override,
        )
    diagnostics = {
        "source_calendar": source_calendar,
        "calendar_day_delta": day_delta,
        "method": method,
        "input_units": original_units,
        "interpreted_input_units": interpreted_units,
        "unit_repaired": bool(input_units_override and input_units_override != original_units),
        "physical_bound_correction": (
            f"clip to [{bounds[0]}, {bounds[1]}]" if bounds else "none"
        ),
    }
    return output, diagnostics


def validate_preprocessed(
    data: xr.DataArray,
    variable: str,
    diagnostics: dict[str, object],
    *,
    source: str,
    output: str,
) -> PreprocessingReport:
    """Run semantic checks against the canonical preprocessing contract."""
    errors: list[str] = []
    grid = canonical_grid()
    if not np.array_equal(data.lat.values, grid.lat.values) or not np.array_equal(
        data.lon.values, grid.lon.values
    ):
        errors.append("output coordinates do not equal the canonical 1-degree grid")
    calendar = str(data.time.dt.calendar)
    if calendar != "noleap":
        errors.append(f"output calendar is {calendar!r}")
    frequency = xr.infer_freq(data.time)
    if frequency not in {"D", "1D"}:
        errors.append(f"output chronology is not complete daily data ({frequency!r})")
    if data.dtype != np.dtype("float32"):
        errors.append(f"output dtype is {data.dtype}, expected float32")
    variable_report = validate_variable(
        data,
        variable,
        min_valid_fraction=0.5 if variable == "prsnratio" else 1.0,
        statistical=False,
        allow_out_of_bounds_hurs=variable == "hurs",
    )
    errors.extend(variable_report.errors)
    return PreprocessingReport(
        source=source,
        output=output,
        variable=variable,
        valid=not errors,
        source_calendar=str(diagnostics["source_calendar"]),
        target_calendar=calendar,
        calendar_day_delta=int(diagnostics["calendar_day_delta"]),
        regrid_method=str(diagnostics["method"]),
        input_units=str(diagnostics["input_units"]),
        interpreted_input_units=str(diagnostics["interpreted_input_units"]),
        output_units=str(data.attrs.get("units", "")),
        output_dtype=str(data.dtype),
        output_shape=dict(data.sizes),
        precipitation_unit_repaired=bool(diagnostics["unit_repaired"]),
        physical_bound_correction=str(diagnostics["physical_bound_correction"]),
        validation=variable_report.to_dict(),
        errors=tuple(errors),
    )
