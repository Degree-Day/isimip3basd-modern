import numpy as np
import pandas as pd
import pytest
import xarray as xr

from isimip3basd_modern.io import open_dataset, parse_chunks, write_zarr
from isimip3basd_modern.pipeline import adjust


def climate_array(values, name="tas", start="2001-01-01"):
    time = pd.date_range(start, periods=len(values), freq="D")
    result = xr.DataArray(
        np.asarray(values, dtype=np.float64).reshape(-1, 1, 1),
        coords={"time": time, "lat": [45.0], "lon": [10.0]},
        dims=("time", "lat", "lon"),
        name=name,
    )
    result.attrs["units"] = "K"
    return result


def test_scaling_removes_additive_bias():
    reference = climate_array(np.arange(365) / 20 + 273.15)
    historical = reference + 2
    historical.attrs["units"] = "K"
    simulation = historical + 1
    simulation.attrs["units"] = "K"

    result = adjust(
        reference,
        historical,
        simulation,
        method="scaling",
        kind="additive",
        group="time.month",
    ).compute()

    xr.testing.assert_allclose(result, reference + 1)
    assert result.attrs["bias_adjustment_method"] == "scaling"
    assert "bias_adjustment_quantiles" not in result.attrs


@pytest.mark.parametrize(
    ("method", "group"),
    (("qdm", "time.month"), ("dqm", None)),
)
def test_quantile_methods_are_lazy_and_finite(method, group):
    reference = climate_array(
        np.sin(np.arange(731) / 30) + 273.15,
        start="2000-01-01",
    )
    historical = reference + 2
    historical.attrs["units"] = "K"
    simulation = historical + 1
    simulation.attrs["units"] = "K"

    result = adjust(
        reference,
        historical,
        simulation,
        method=method,
        kind="additive",
        group=group,
        quantiles=20,
        chunks={"lat": 1, "lon": 1},
    )

    assert result.chunks is not None
    assert np.isfinite(result.compute()).all()


@pytest.mark.parametrize("method", ("qdm", "dqm"))
def test_adjustment_fit_cache_round_trip_is_bitwise_identical(tmp_path, method):
    reference = climate_array(
        np.sin(np.arange(731) / 30) + 273.15,
        start="2000-01-01",
    )
    historical = reference + 2
    historical.attrs["units"] = "K"
    simulation = historical + 1
    simulation.attrs["units"] = "K"
    cache = tmp_path / "fit.zarr"

    uncached = adjust(
        reference,
        historical,
        simulation,
        method=method,
        kind="additive",
        group="time.month",
        quantiles=20,
        chunks={"lat": 1, "lon": 1},
        fit_cache_path=cache,
        fit_cache_key="inputs-v1",
    ).compute()
    cached = adjust(
        reference,
        historical,
        simulation,
        method=method,
        kind="additive",
        group="time.month",
        quantiles=20,
        chunks={"lat": 1, "lon": 1},
        fit_cache_path=cache,
        fit_cache_key="inputs-v1",
    ).compute()

    np.testing.assert_array_equal(cached.values, uncached.values)
    assert not uncached.attrs["bias_adjustment_fit_cache_hit"]
    assert cached.attrs["bias_adjustment_fit_cache_hit"]


def test_adjustment_fit_cache_retrains_for_new_key(tmp_path):
    reference = climate_array(np.arange(365) / 20 + 273.15)
    historical = reference + 2
    historical.attrs["units"] = "K"
    simulation = historical + 1
    simulation.attrs["units"] = "K"
    cache = tmp_path / "fit.zarr"

    for key in ("inputs-v1", "inputs-v2"):
        result = adjust(
            reference,
            historical,
            simulation,
            method="scaling",
            kind="additive",
            group="time.month",
            fit_cache_path=cache,
            fit_cache_key=key,
        ).compute()
        assert not result.attrs["bias_adjustment_fit_cache_hit"]
        with xr.open_zarr(cache, consolidated=False) as stored:
            assert stored.attrs["isimip3basd_fit_cache_key"] == key


@pytest.mark.parametrize("zarr_format", (2, 3))
def test_zarr_round_trip(tmp_path, zarr_format):
    source = climate_array(np.arange(31) + 273.15).to_dataset()
    store = tmp_path / f"sample-v{zarr_format}.zarr"
    write_zarr(source, store, zarr_format=zarr_format)
    with open_dataset(store, {"time": -1}) as restored:
        xr.testing.assert_identical(restored.load(), source)


def test_parse_chunks():
    assert parse_chunks("time=-1,lat=8,lon=16") == {
        "time": -1,
        "lat": 8,
        "lon": 16,
    }


@pytest.mark.parametrize("chunks", ("time=0", "lat=-2", "=4", "lat"))
def test_parse_chunks_rejects_invalid_values(chunks):
    with pytest.raises(ValueError):
        parse_chunks(chunks)
