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
