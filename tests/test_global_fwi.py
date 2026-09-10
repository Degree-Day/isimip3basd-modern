from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import xarray as xr


SCRIPT = Path(__file__).parents[1] / "scripts" / "calc_global_fwi.py"
SPEC = importlib.util.spec_from_file_location("calc_global_fwi", SCRIPT)
assert SPEC and SPEC.loader
FWI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FWI)


def _weather_arrays(start: str, years: int, temperature: float = 295.0):
    time = xr.date_range(
        start,
        periods=years * 365,
        freq="D",
        calendar="noleap",
        use_cftime=True,
    )
    template = xr.DataArray(
        np.ones((time.size, 1, 1), dtype="float32"),
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": [-30.0], "lon": [150.0]},
    )
    return {
        "tas": (template * temperature).assign_attrs(units="K"),
        "hurs": (template * 35).assign_attrs(units="%"),
        "pr": (template * 0).assign_attrs(units="mm/day"),
        "sfcWind": (template * 5).assign_attrs(units="m s-1"),
    }


def test_fwi_tiles_cover_global_grid_once():
    coverage = np.zeros((91, 179), dtype=np.uint8)
    for tile in FWI.tile_specs(91, 179, 40):
        coverage[
            tile["lat_start"] : tile["lat_stop"],
            tile["lon_start"] : tile["lon_stop"],
        ] += 1
    assert np.all(coverage == 1)


def test_compute_indices_has_clean_metadata_and_dimension_order():
    time = xr.date_range(
        "2001-01-01", periods=3 * 365, freq="D", calendar="noleap", use_cftime=True
    )
    template = xr.DataArray(
        np.ones((time.size, 1, 1), dtype="float32"),
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": [45.0], "lon": [10.0]},
    )
    arrays = {
        "tas": (template * 290).assign_attrs(units="K"),
        "hurs": (template * 40).assign_attrs(units="%"),
        "pr": (template * 0.00001).assign_attrs(units="kg m-2 s-1"),
        "sfcWind": (template * 5).assign_attrs(units="m s-1"),
    }
    result = FWI.compute_indices(arrays, "2003-01-01", "2003-12-31")

    assert set(result.data_vars) == set(FWI.INDEX_METADATA)
    assert all(result[name].dims == ("time", "lat", "lon") for name in result)
    assert all(result[name].dtype == np.dtype("float32") for name in result)
    assert all(result[name].attrs["units"] == "1" for name in result)
    assert all("air_temperature" not in result[name].attrs.values() for name in result)
    assert result.fwi.notnull().any().compute().item()
    assert result.fwi.attrs["fwi_dry_start"] == "none"


def test_concatenate_history_requires_adjacent_matching_periods():
    history = _weather_arrays("2001-01-01", 1)["tas"]
    simulation = _weather_arrays("2002-01-01", 1)["tas"]

    combined = FWI.concatenate_history(history, simulation)

    assert combined.sizes["time"] == 730
    assert str(combined.time.values[364])[:10] == "2001-12-31"
    assert str(combined.time.values[365])[:10] == "2002-01-01"

    gap = _weather_arrays("2002-01-02", 1)["tas"]
    with np.testing.assert_raises_regex(ValueError, "not consecutive"):
        FWI.concatenate_history(history, gap)

    shifted = simulation.assign_coords(lon=[151.0])
    with np.testing.assert_raises_regex(ValueError, "lon coordinates differ"):
        FWI.concatenate_history(history, shifted)


def test_historical_context_changes_projection_initial_state():
    history = _weather_arrays("2001-01-01", 1)
    projection = _weather_arrays("2002-01-01", 1)
    continuous_inputs = {
        name: FWI.concatenate_history(history[name], projection[name])
        for name in FWI.INPUT_VARIABLES
    }

    continuous = FWI.compute_indices(
        continuous_inputs, "2002-01-01", "2002-12-31"
    )
    restarted = FWI.compute_indices(
        projection, "2002-01-01", "2002-12-31"
    )

    assert not np.isclose(
        continuous.dc.isel(time=0).compute().item(),
        restarted.dc.isel(time=0).compute().item(),
    )


def test_pack_indices_reserves_fill_and_preserves_values():
    values = np.array([[[0.0, 10.0, np.nan]]], dtype="float32")
    dataset = xr.Dataset(
        {name: (("time", "lat", "lon"), values.copy()) for name in FWI.INDEX_METADATA}
    )

    packed = FWI.pack_indices(dataset)

    for name, result in packed.items():
        spec = FWI.PACKING_SPECS[name]
        assert result.dtype == np.dtype("int16")
        assert result[0, 0, 2] == FWI.PACKED_FILL_VALUE
        decoded = result[0, 0, :2] * spec.scale_factor + spec.add_offset
        np.testing.assert_allclose(decoded, values[0, 0, :2], atol=spec.scale_factor / 2)


def test_pack_indices_rejects_infinity():
    values = np.array([[[np.inf]]], dtype="float32")
    dataset = xr.Dataset(
        {name: (("time", "lat", "lon"), values.copy()) for name in FWI.INDEX_METADATA}
    )

    with np.testing.assert_raises_regex(ValueError, "contains infinite values"):
        FWI.pack_indices(dataset)
