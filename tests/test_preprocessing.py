import numpy as np
import xarray as xr

from isimip3basd_modern.preprocessing import (
    _normalize_calendar,
    canonical_grid,
    preprocess_variable,
    validate_preprocessed,
)


def source_data(variable: str, units: str, values: np.ndarray) -> xr.DataArray:
    time = xr.date_range("2000-02-27", periods=4, freq="D", calendar="noleap", use_cftime=True)
    return xr.DataArray(
        np.broadcast_to(values, (4, 2, 2)).copy(),
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": [-45.0, 45.0], "lon": [0.0, 180.0]},
        name=variable,
        attrs={
            "units": units,
            "standard_name": {
                "tas": "air_temperature",
                "pr": "precipitation_flux",
            }[variable],
        },
    )


def test_canonical_grid_is_global_and_nested():
    grid = canonical_grid()

    assert grid.sizes == {"lat": 180, "lon": 360}
    assert float(grid.lat[0]) == -89.5
    assert float(grid.lat[-1]) == 89.5
    assert float(grid.lon[0]) == 0.5
    assert float(grid.lon[-1]) == 359.5


def test_preprocess_linear_variable_normalizes_grid_calendar_and_dtype():
    source = source_data("tas", "K", np.array([[280.0, 282.0], [284.0, 286.0]]))

    result, diagnostics = preprocess_variable(source, "tas", spatial_chunk=20)

    assert result.sizes == {"time": 4, "lat": 180, "lon": 360}
    assert str(result.time.dt.calendar) == "noleap"
    assert result.dtype == np.dtype("float32")
    assert diagnostics["calendar_day_delta"] == 0
    assert np.isfinite(result).all().compute()


def test_supersaturated_humidity_is_retained_for_isimip_bias_adjustment():
    source = source_data("tas", "K", np.full((2, 2), 280.0)).rename("hurs")
    source.attrs.update(units="%", standard_name="relative_humidity")
    source[0, 0, 0] = -2
    source[1, 1, 1] = 150

    result, diagnostics = preprocess_variable(source, "hurs", spatial_chunk=20)

    assert float(result.max().compute()) > 100
    assert diagnostics["physical_bound_correction"] == "none"

    report = validate_preprocessed(
        result,
        "hurs",
        diagnostics,
        source="input.zarr",
        output="output.zarr",
    )
    assert report.valid
    assert report.validation["checks"]["out_of_bounds_hurs_retained"] is True


def test_precipitation_unit_repair_and_calendar_conversion_preserve_total():
    source = source_data("pr", "kg m-2 s-1", np.full((2, 2), 8.64))

    result, diagnostics = preprocess_variable(
        source,
        "pr",
        input_units_override="mm d-1",
        spatial_chunk=20,
    )

    expected = 8.64 / 86400
    assert np.isclose(float(result.sum("time").max()), expected * 4, rtol=1e-5)
    assert diagnostics["unit_repaired"] is True


def test_gregorian_leap_day_is_removed():
    time = xr.date_range(
        "2000-02-27",
        periods=4,
        freq="D",
        calendar="proleptic_gregorian",
        use_cftime=True,
    )
    source = xr.DataArray(np.arange(4.0), dims="time", coords={"time": time})

    result, source_calendar, day_delta = _normalize_calendar(source, "tas")

    assert source_calendar == "proleptic_gregorian"
    assert day_delta == -1
    assert str(result.time.dt.calendar) == "noleap"
    assert list(result.time.dt.day.values) == [27, 28, 1]
    np.testing.assert_array_equal(result.values, [0.0, 1.0, 3.0])


def test_gregorian_leap_day_precipitation_is_transferred_to_february_28():
    time = xr.date_range(
        "2000-02-27",
        periods=4,
        freq="D",
        calendar="proleptic_gregorian",
        use_cftime=True,
    )
    source = xr.DataArray(np.arange(4.0), dims="time", coords={"time": time})

    result, _, day_delta = _normalize_calendar(source, "pr")

    assert day_delta == -1
    np.testing.assert_array_equal(result.values, [0.0, 3.0, 3.0])
    assert float(result.sum()) == float(source.sum())


def test_360_day_precipitation_maps_to_noleap_and_preserves_annual_total():
    time = xr.date_range(
        "2001-01-01", periods=360, freq="D", calendar="360_day", use_cftime=True
    )
    source = xr.DataArray(np.ones(360), dims="time", coords={"time": time})

    result, source_calendar, day_delta = _normalize_calendar(source, "pr")

    assert source_calendar == "360_day"
    assert day_delta == 5
    assert str(result.time.dt.calendar) == "noleap"
    assert result.sizes["time"] == 365
    assert np.isclose(float(result.sum()), 360.0)


def test_preprocessed_semantic_qc_passes():
    source = source_data("tas", "K", np.full((2, 2), 280.0))
    result, diagnostics = preprocess_variable(source, "tas")

    report = validate_preprocessed(
        result,
        "tas",
        diagnostics,
        source="input.zarr",
        output="output.zarr",
    )

    assert report.valid
