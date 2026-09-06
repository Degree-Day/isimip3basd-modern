from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
import xarray as xr


SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_era5land_reference.py"
SPEC = importlib.util.spec_from_file_location("prepare_reference", SCRIPT)
assert SPEC and SPEC.loader
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


def test_era5_relative_humidity_fraction_is_converted_to_percent():
    source = xr.DataArray(
        np.array([[[0.25, 1.00001]]], dtype="float32"),
        dims=("time", "lat", "lon"),
        coords={"time": [0], "lat": [0.0], "lon": [-180.0, 179.75]},
        name="hurs",
        attrs={"units": "dimensionless"},
    )

    result = PREPARE.normalize_era5_daily(source, "hurs")

    assert result.attrs["units"] == "%"
    np.testing.assert_allclose(result.values, [[[100.0, 25.0]]])
    np.testing.assert_allclose(result.lon, [179.75, 180.0])


def test_era5_precipitation_roundoff_is_clipped_at_zero():
    source = xr.DataArray(
        np.array([[[-0.001, 8.64]]], dtype="float32"),
        dims=("time", "lat", "lon"),
        coords={"time": [0], "lat": [0.0], "lon": [0.0, 0.25]},
        name="pr",
    )

    result = PREPARE.normalize_era5_daily(source, "pr")

    assert result.attrs["units"] == "kg m-2 s-1"
    np.testing.assert_allclose(result.values, [[[0.0, 0.0001]]])


def test_lulc_land_area_is_aggregated_and_wrapped_to_zero_360(tmp_path):
    path = tmp_path / "land.tif"
    values = np.zeros((24, 24), dtype="float32")
    values[:12, 12:] = 1.0
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=24,
        height=24,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(-0.1, 0.1, 1 / 120, 1 / 120),
    ) as target:
        target.write(values, 1)
    grid = xr.Dataset(coords={"lat": [-0.05, 0.05], "lon": [0.05, 359.95]})

    result = PREPARE.build_lulc_land_mask(path, grid)

    np.testing.assert_array_equal(result.values, [[False, False], [True, False]])
