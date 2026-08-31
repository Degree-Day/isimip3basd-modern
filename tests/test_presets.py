import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xclim.core.units import convert_units_to
from xclim.indices import shortwave_downwelling_radiation_from_clearness_index

from isimip3basd_modern.pipeline import adjust_variable
from isimip3basd_modern.presets import VARIABLE_PRESETS, get_preset
from isimip3basd_modern.validation import validate_variable


def climate_array(values, variable, units, standard_name=None):
    values = np.asarray(values, dtype=np.float64)
    time = pd.date_range("2000-01-01", periods=values.size, freq="D")
    result = xr.DataArray(
        values[:, None, None],
        coords={"time": time, "lat": [45.0], "lon": [10.0]},
        dims=("time", "lat", "lon"),
        name=variable,
        attrs={"units": units},
    )
    result.lat.attrs["units"] = "degrees_north"
    if standard_name:
        result.attrs["standard_name"] = standard_name
    return result


def test_all_isimip_variables_have_presets():
    assert tuple(VARIABLE_PRESETS) == (
        "hurs",
        "pr",
        "prsnratio",
        "ps",
        "rlds",
        "rsds",
        "sfcWind",
        "tas",
        "tasrange",
        "tasskew",
    )
    with pytest.raises(ValueError, match="no preset"):
        get_preset("unknown")


def test_hurs_preset_handles_model_values_above_100_percent():
    seasonal = 65 + 25 * np.sin(np.arange(731) * 2 * np.pi / 365.25)
    reference = climate_array(seasonal, "hurs", "%", "relative_humidity")
    historical = climate_array(seasonal + 8, "hurs", "%", "relative_humidity")
    simulation_values = seasonal + 10
    simulation_values[0] = 105
    simulation = climate_array(
        simulation_values, "hurs", "%", "relative_humidity"
    )

    result = adjust_variable(
        reference,
        historical,
        simulation,
        variable="hurs",
        group="time.month",
        window=1,
        quantiles=10,
        chunks={"lat": 1, "lon": 1},
    ).compute()

    assert np.isfinite(result).all()
    assert float(result.min()) >= 0
    assert float(result.max()) <= 100
    assert result.isel(time=0, lat=0, lon=0).item() == 100
    assert result.attrs["bias_adjustment_transform"] == "logit"


def test_pr_preset_adapts_dry_values_and_remains_nonnegative():
    wave = np.maximum(0, np.sin(np.arange(731) * 2 * np.pi / 30)) * 2e-4
    reference = climate_array(wave, "pr", "kg m-2 s-1", "precipitation_flux")
    historical = climate_array(
        wave * 0.7, "pr", "kg m-2 s-1", "precipitation_flux"
    )
    simulation = climate_array(
        wave * 0.9, "pr", "kg m-2 s-1", "precipitation_flux"
    )

    result = adjust_variable(
        reference,
        historical,
        simulation,
        variable="pr",
        group="time.month",
        window=1,
        quantiles=10,
        chunks={"lat": 1, "lon": 1},
    ).compute()

    threshold = convert_units_to("0.1 mm d-1", result, context="infer")
    assert np.isfinite(result).all()
    assert float(result.min()) >= 0
    assert bool(((result == 0) | (result >= threshold)).all())
    assert "bias_adjustment_adapt_frequency_threshold" in result.attrs
    assert result.attrs["bias_adjustment_random_seed"] == 0

    repeated = adjust_variable(
        reference,
        historical,
        simulation,
        variable="pr",
        group="time.month",
        window=1,
        quantiles=10,
        chunks={"lat": 1, "lon": 1},
        random_seed=0,
    ).compute()
    np.testing.assert_array_equal(result.values, repeated.values)


def test_rsds_preset_adjusts_in_clearness_index_space():
    phase = np.arange(731) * 2 * np.pi / 365.25
    ci_ref = climate_array(0.45 + 0.1 * np.sin(phase), "ci", "")
    ci_hist = climate_array(0.38 + 0.1 * np.sin(phase), "ci", "")
    ci_sim = climate_array(0.42 + 0.1 * np.sin(phase), "ci", "")
    reference = shortwave_downwelling_radiation_from_clearness_index(ci_ref)
    historical = shortwave_downwelling_radiation_from_clearness_index(ci_hist)
    simulation = shortwave_downwelling_radiation_from_clearness_index(ci_sim)
    for data in (reference, historical, simulation):
        data.name = "rsds"
        data.attrs["standard_name"] = "surface_downwelling_shortwave_flux_in_air"

    result = adjust_variable(
        reference,
        historical,
        simulation,
        variable="rsds",
        group="time.month",
        window=1,
        quantiles=10,
        chunks={"lat": 1, "lon": 1},
    ).compute()

    assert np.isfinite(result).all()
    assert float(result.min()) >= 0
    assert result.attrs["bias_adjustment_transform"] == "clearness_index"


def test_validation_rejects_out_of_bounds_humidity():
    valid = climate_array([0, 50, 100], "hurs", "%", "relative_humidity")
    invalid = climate_array([-1, 50, 101], "hurs", "%", "relative_humidity")

    assert validate_variable(valid, "hurs").valid
    report = validate_variable(invalid, "hurs")
    assert not report.valid
    assert not report.physical_bounds


def test_validation_accepts_temperature_below_zero_celsius():
    temperature = climate_array(
        [-40, 0, 35], "tas", "degC", "air_temperature"
    )
    assert validate_variable(temperature, "tas").valid
