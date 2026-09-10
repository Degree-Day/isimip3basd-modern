import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "compare_arco_local_noon.py"
SPEC = importlib.util.spec_from_file_location("compare_arco_local_noon", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_local_noon_utc_hour_wraps_longitudes():
    longitude = np.array([0.0, 90.0, 180.0, 270.0])
    np.testing.assert_array_equal(
        MODULE.local_noon_utc_hour(longitude), np.array([12, 6, 0, 18])
    )


def test_bilinear_samples_periodic_global_grid():
    latitude = np.arange(90.0, -90.01, -0.25)
    longitude = np.arange(0.0, 360.0, 0.25)
    field = latitude[:, None] + np.cos(np.deg2rad(longitude))[None, :]
    sampled = MODULE.bilinear(
        field,
        np.array([12.125, -40.375]),
        np.array([359.875, 120.125]),
    )
    expected = np.array(
        [
            12.125 + np.cos(np.deg2rad(359.875)),
            -40.375 + np.cos(np.deg2rad(120.125)),
        ]
    )
    np.testing.assert_allclose(sampled, expected, atol=2e-5)


def test_relative_humidity_is_saturated_when_temperature_equals_dewpoint():
    temperature = np.array([273.15, 293.15, 303.15])
    np.testing.assert_allclose(
        MODULE.relative_humidity(temperature, temperature), 100.0
    )


def test_comparison_reports_fraction_improved():
    result = MODULE.comparison(
        np.array([4.0, 8.0]),
        np.array([2.0, 7.0]),
        np.array([0.0, 10.0]),
    )
    assert result["fraction_of_join_observations_improved"] == 0.5
