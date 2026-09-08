from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import xarray as xr
import zarr


SCRIPT = Path(__file__).parents[1] / "scripts" / "merge_projection_segments.py"
SPEC = importlib.util.spec_from_file_location("merge_projection_segments", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write_segment(
    path: Path, variable: str, start: str, end: str, code: int
) -> None:
    time = xr.date_range(
        start=start,
        end=end,
        freq="D",
        calendar="noleap",
        use_cftime=True,
    )
    data = xr.DataArray(
        np.full((time.size, 2, 3), code, dtype="int16"),
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": [-0.5, 0.5], "lon": [0.5, 1.5, 2.5]},
        name=variable,
        attrs={"scale_factor": 0.1, "add_offset": 200.0},
    )
    data.to_dataset().to_zarr(
        path,
        mode="w",
        zarr_format=3,
        encoding={variable: {"chunks": (365, 2, 3)}},
    )


def test_merge_store_preserves_packed_values_and_is_restartable(tmp_path: Path) -> None:
    scenario = tmp_path / "MODEL" / "ssp126"
    periods = (
        ("2015-01-01", "2020-12-31"),
        ("2021-01-01", "2033-12-31"),
        ("2034-01-01", "2095-12-31"),
        ("2096-01-01", "2100-12-31"),
    )
    for stage, (start, end), code in zip(
        MODULE.SEGMENTS, periods, (1, 2, 3, 4), strict=True
    ):
        _write_segment(scenario / stage / "tas.zarr", "tas", start, end, code)

    record = MODULE.merge_store(scenario, "tas", overwrite=False)

    assert record["status"] == "merged"
    group = zarr.open_group(
        scenario / "projection" / "tas.zarr",
        mode="r",
        use_consolidated=False,
    )
    assert group["tas"].dtype == np.dtype("int16")
    assert group["tas"].shape == (31_390, 2, 3)
    assert group["tas"][[0, 2_189, 2_190, 6_934, 6_935, 29_564, 29_565, 31_389], 0, 0].tolist() == [
        1,
        1,
        2,
        2,
        3,
        3,
        4,
        4,
    ]

    for stage in MODULE.SEGMENTS:
        (scenario / stage / "tas.zarr").rename(
            scenario / stage / "removed.zarr"
        )
    assert MODULE.merge_store(scenario, "tas", overwrite=False)["status"] == "existing"
