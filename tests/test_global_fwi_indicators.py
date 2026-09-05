from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "calc_global_fwi_indicators.py"
SPEC = importlib.util.spec_from_file_location("calc_global_fwi_indicators", SCRIPT)
assert SPEC and SPEC.loader
FWI_INDICATORS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = FWI_INDICATORS
SPEC.loader.exec_module(FWI_INDICATORS)


def test_annual_fwi_indicator_definitions():
    years = np.repeat([2001, 2002], 365)
    annual = np.arange(365, dtype="float32") % 100
    values = np.tile(annual, 2).reshape(730, 1, 1)
    q95 = np.array([[94.0]], dtype="float32")
    midrange = np.array([[49.5]], dtype="float32")

    result = FWI_INDICATORS._annual_values(values, years, q95, midrange)

    assert result["fwixx"].shape == (2, 1, 1)
    np.testing.assert_array_equal(result["fwixx"][:, 0, 0], [99, 99])
    np.testing.assert_array_equal(result["fwixd"][:, 0, 0], [15, 15])
    np.testing.assert_array_equal(result["fwils"][:, 0, 0], [165, 165])
    assert np.all(result["fwisa"][:, 0, 0] > 50)


def test_annual_indicator_packing_round_trip():
    values = np.array([0.0, 12.34, np.nan], dtype="float32")
    packed = FWI_INDICATORS._pack(values, scale=0.04, offset=1000.0)

    assert packed.dtype == np.dtype("int16")
    assert packed[-1] == FWI_INDICATORS.FILL
    decoded = packed[:2] * 0.04 + 1000.0
    np.testing.assert_allclose(decoded, values[:2], atol=0.02)


def test_support_mask_preserves_seasonal_missing_values_and_masks_ocean():
    values = np.array(
        [
            [[np.nan, np.nan], [3.0, np.nan]],
            [[2.0, np.nan], [np.nan, np.nan]],
        ],
        dtype="float32",
    )
    support = np.array([[True, False], [True, True]])

    result = FWI_INDICATORS._apply_support(values, support)

    np.testing.assert_array_equal(result[:, 0, 0], [np.nan, 2.0])
    np.testing.assert_array_equal(result[:, 1, 0], [3.0, np.nan])
    assert np.isnan(result[:, 1, 1]).all()
    assert np.isnan(result[:, 0, 1]).all()


def test_no_active_season_can_be_encoded_as_zero_on_supported_land():
    values = np.array([[[np.nan, np.nan]]], dtype="float32")
    support = np.array([[True, False]])

    result = np.where(
        support[None, :, :] & np.isnan(values), 0.0, values
    )

    assert result[0, 0, 0] == 0
    assert np.isnan(result[0, 0, 1])


def test_missing_reference_threshold_is_zero_only_on_supported_land():
    values = np.array([[np.nan, np.nan, 4.0]], dtype="float32")
    support = np.array([[True, False, True]])

    result = FWI_INDICATORS._fill_missing_threshold(values, support)

    np.testing.assert_array_equal(result[0, [0, 2]], [0.0, 4.0])
    assert np.isnan(result[0, 1])
