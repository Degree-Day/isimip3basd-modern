import json

import numpy as np
import pandas as pd
import xarray as xr

from isimip3basd_modern.validation import (
    derive_variables,
    preflight_variable,
    validate_dataset,
    validate_output,
    validate_variable,
)


def tas_array(days=20, lon=(10.0, 11.0)):
    data = xr.DataArray(
        np.full((days, 1, len(lon)), 280.0),
        coords={
            "time": pd.date_range("2001-01-01", periods=days, freq="D"),
            "lat": [45.0],
            "lon": list(lon),
        },
        dims=("time", "lat", "lon"),
        name="tas",
        attrs={"units": "K", "standard_name": "air_temperature"},
    )
    data.lat.attrs["units"] = "degrees_north"
    data.lon.attrs["units"] = "degrees_east"
    return data


def test_all_missing_mask_is_allowed_but_partial_missing_is_not():
    data = tas_array()
    data[:, :, 1] = np.nan
    report = validate_variable(data, "tas", statistical=False)

    assert report.valid
    assert report.all_missing_cells == 1
    assert report.partial_missing_cells == 0

    data[0, :, 0] = np.nan
    report = validate_variable(data, "tas", statistical=False)
    assert not report.valid
    assert report.partial_missing_cells == 1


def test_preflight_rejects_short_and_gapped_training_data():
    data = tas_array(days=20).isel(time=[index for index in range(20) if index != 10])
    report = preflight_variable(data, "tas", label="reference", min_years=1)

    assert not report.valid
    assert any("complete daily sequence" in error for error in report.errors)
    assert any("training period" in error for error in report.errors)


def test_output_requires_exact_simulation_grid():
    simulation = tas_array()
    output = simulation.assign_coords(lon=[10.0, 12.0])
    report = validate_output(output, simulation, "tas")

    assert not report.valid
    assert report.checks["grid_preserved"] is False
    json.dumps(report.to_dict())


def test_derives_temperature_extremes_and_snowfall():
    tas = tas_array(days=3, lon=(10.0,))
    shape = tas.shape
    dataset = xr.Dataset(
        {
            "tas": tas,
            "tasrange": xr.full_like(tas, 10.0).assign_attrs(units="K"),
            "tasskew": xr.full_like(tas, 0.25).assign_attrs(units="1"),
            "pr": xr.full_like(tas, 4.0).assign_attrs(units="mm d-1"),
            "prsnratio": xr.DataArray(
                np.full(shape, 0.5),
                coords=tas.coords,
                dims=tas.dims,
                attrs={"units": "1"},
            ),
        }
    )

    derived = derive_variables(dataset)
    assert set(derived.data_vars) >= {"tasmin", "tasmax", "prsn"}
    np.testing.assert_allclose(derived.tasmin, 277.5)
    np.testing.assert_allclose(derived.tasmax, 287.5)
    np.testing.assert_allclose(derived.prsn, 2.0)
    assert validate_dataset(derived).valid


def test_dataset_validation_detects_inconsistent_temperature_components():
    tas = tas_array(days=3, lon=(10.0,))
    dataset = xr.Dataset(
        {
            "tas": tas,
            "tasrange": xr.full_like(tas, 10.0).assign_attrs(units="K"),
            "tasskew": xr.full_like(tas, 0.25).assign_attrs(units="1"),
            "tasmin": xr.full_like(tas, 279.0),
            "tasmax": xr.full_like(tas, 281.0),
        }
    )

    report = validate_dataset(dataset)
    assert not report.valid
    assert report.checks["temperature_components_consistent"] is False


def test_derives_specific_humidity_with_xclim():
    tas = tas_array(days=3, lon=(10.0,))
    dataset = xr.Dataset(
        {
            "tas": tas,
            "hurs": xr.full_like(tas, 50.0).assign_attrs(
                units="%", standard_name="relative_humidity"
            ),
            "ps": xr.full_like(tas, 101325.0).assign_attrs(
                units="Pa", standard_name="surface_air_pressure"
            ),
        }
    )

    derived = derive_variables(dataset)
    assert "huss" in derived
    assert bool(((derived.huss > 0) & (derived.huss < 1)).all())
