from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import xarray as xr


SCRIPT = Path(__file__).parents[1] / "scripts" / "qc_coastal_model_outputs.py"
SPEC = importlib.util.spec_from_file_location("qc_coastal_model_outputs", SCRIPT)
QC = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(QC)


def test_sampled_edges_separates_source_transitions_and_controls():
    source = np.array([[1, 1, 2], [1, 3, 2]], dtype="uint8")

    groups = QC.sampled_edges(source, maximum_per_group=100, seed=42)

    assert groups["era5_land--era5_land_coastal_repair"]["available_edges"] > 0
    assert groups["era5_land--era5_extension"]["available_edges"] > 0
    assert groups["era5_land_coastal_repair--era5_extension"]["available_edges"] > 0
    assert groups["same--era5_land"]["available_edges"] > 0


def test_sampled_edges_is_deterministic_and_bounded():
    source = np.ones((10, 10), dtype="uint8")

    first = QC.sampled_edges(source, maximum_per_group=5, seed=7)
    second = QC.sampled_edges(source, maximum_per_group=5, seed=7)

    assert first == second
    assert len(first["same--era5_land"]["first_lat"]) == 5


def test_audit_array_distinguishes_seasonal_missing_from_empty_edges():
    data = xr.DataArray(
        np.array(
            [
                [[np.nan, np.nan, np.nan]],
                [[1.0, 2.0, np.nan]],
            ]
        ),
        dims=("time", "lat", "lon"),
    )
    groups = {
        "sample": {
            "available_edges": 2,
            "first_lat": [0, 0],
            "first_lon": [0, 1],
            "second_lat": [0, 0],
            "second_lon": [1, 2],
        }
    }

    report = QC.audit_array(data, groups, 1.0)["sample"]

    assert report["missing_endpoint_observations"] > 0
    assert report["edges_without_any_paired_observation"] == 1
    assert report["edges_with_either_endpoint_never_valid"] == 1
