from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import xarray as xr


SCRIPT = Path(__file__).parents[1] / "scripts" / "qc_reference_dataset.py"
SPEC = importlib.util.spec_from_file_location("qc_reference_dataset", SCRIPT)
QC = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(QC)


def test_coordinate_report_recognizes_complete_noleap_grid():
    time = xr.date_range(
        "2001-01-01", periods=365, calendar="noleap", use_cftime=True
    )
    data = xr.DataArray(
        np.ones((365, 2, 3)),
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": [-0.05, 0.05], "lon": [0.05, 0.15, 0.25]},
    )

    report = QC.coordinate_report(data, 0.1)

    assert report["grid_regular"]
    assert report["calendar"] == "noleap"
    assert report["year_day_counts"] == [365]
    assert not report["contains_february_29"]


def test_boundary_differences_separates_source_transitions():
    field = np.array([[0.0, 1.0, 11.0, 12.0]])
    source = np.array([[1, 1, 3, 3]])

    report = QC.boundary_differences(field, source)

    assert report["boundary"]["edge_count"] == 2
    assert report["boundary"]["median"] == 11.0
    assert report["same_source"]["median"] == 1.0
    assert report["boundary_to_same_source_p95_ratio"] > 1
    assert report["by_source_pair"]["era5_land--era5"]["edge_count"] == 2


def test_distribution_ignores_missing_values():
    report = QC.distribution(np.array([1.0, 2.0, np.nan, np.inf]))

    assert report["count"] == 2
    assert report["quantiles"]["min"] == 1.0
    assert report["quantiles"]["max"] == 2.0
