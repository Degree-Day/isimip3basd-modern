import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).parents[1] / "scripts" / "rebuild_era5_local_noon_reference.py"
)
SPEC = importlib.util.spec_from_file_location("rebuild_era5_local_noon_reference", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_local_noon_utc_offset_preserves_dateline_date():
    longitude = np.array([0.0, 90.0, 179.9, 180.1, 270.0])
    np.testing.assert_array_equal(
        MODULE.local_noon_utc_offset(longitude), np.array([12, 6, 0, 24, 18])
    )


def test_relative_humidity_is_saturated_at_the_dewpoint():
    temperature = np.array([253.15, 273.15, 303.15])
    np.testing.assert_allclose(
        MODULE.relative_humidity(temperature, temperature), 100.0
    )


def test_aggregate_block_is_area_weighted_and_ignores_missing_values():
    block = np.ones((2, 20, 10), dtype="float32")
    block[:, :10] = 2.0
    block[0, 0, 0] = np.nan
    latitude = np.arange(20, dtype=float)
    result = MODULE.aggregate_block(block, latitude)
    assert result.shape == (2, 2, 1)
    np.testing.assert_allclose(result[:, 0, 0], 2.0)
    np.testing.assert_allclose(result[:, 1, 0], 1.0)


def test_group_spatial_chunks_keeps_cell_indices():
    groups = MODULE.group_spatial_chunks(
        np.array([0, 49, 50, 99]), np.array([0, 49, 50, 10]), 50
    )
    np.testing.assert_array_equal(groups[(0, 0)], np.array([0, 1]))
    np.testing.assert_array_equal(groups[(1, 1)], np.array([2]))
    np.testing.assert_array_equal(groups[(1, 0)], np.array([3]))
