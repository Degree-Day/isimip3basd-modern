from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import xarray as xr


SCRIPT = Path(__file__).parents[1] / "scripts" / "verify_downscaled_land_support.py"
SPEC = importlib.util.spec_from_file_location("verify_downscaled_land_support", SCRIPT)
VERIFY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(VERIFY)


def test_coverage_report_requires_every_mapped_land_cell():
    land = xr.DataArray(
        np.array([[True, True], [False, True]]), dims=("lat", "lon")
    )
    complete = xr.DataArray(
        np.array([[True, True], [True, True]]), dims=("lat", "lon")
    )
    incomplete = complete.copy()
    incomplete[0, 1] = False

    assert VERIFY.coverage_report(complete, land)["valid"]
    report = VERIFY.coverage_report(incomplete, land)
    assert not report["valid"]
    assert report["mapped_land_cells_missing_support"] == 1
