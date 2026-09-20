"""Run the tiled production workflow end to end with spawned tile workers."""

from __future__ import annotations

import json

import numpy as np
import xarray as xr
import zarr

from isimip3basd_modern import tiled_runner


COARSE_LAT = np.array([-30.0, 0.0, 30.0])
COARSE_LON = np.array([45.0, 135.0, 225.0, 315.0])
FINE_LAT = np.array([-37.5, -22.5, -7.5, 7.5, 22.5, 37.5])
FINE_LON = np.arange(22.5, 360, 45.0)


def _series(start, end, lat, lon, *, offset, seed):
    time = xr.date_range(
        start, end, freq="D", calendar="noleap", use_cftime=True
    )
    random = np.random.RandomState(seed)
    season = 8 * np.sin(2 * np.pi * np.arange(time.size) / 365)[:, None, None]
    gradient = np.linspace(-3, 3, lat.size)[None, :, None]
    noise = random.normal(scale=2, size=(time.size, lat.size, lon.size))
    data = xr.DataArray(
        (285 + offset + season + gradient + noise).astype("float32"),
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lat, "lon": lon},
        name="tas",
        attrs={"units": "K"},
    )
    data.lat.attrs["units"] = "degrees_north"
    data.lon.attrs["units"] = "degrees_east"
    return data


def _write(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_dataset().to_zarr(path, mode="w", consolidated=False, zarr_format=3)


def test_global_tiles_run_restart_and_mask_ocean_with_spawned_workers(
    tmp_path, capsys
):
    reference = tmp_path / "reference"
    canonical = tmp_path / "canonical"
    adjusted = tmp_path / "adjusted"
    output = tmp_path / "output"
    fine = _series("1993-01-01", "2014-12-31", FINE_LAT, FINE_LON, offset=0, seed=1)
    fine[:, :2, :2] = np.nan  # one all-ocean coarse cell
    fine[:, 5, 7] = np.nan  # one ocean fine cell inside a land coarse cell
    coarse = _series(
        "1993-01-01", "2014-12-31", COARSE_LAT, COARSE_LON, offset=0, seed=2
    )
    coarse[:, 0, 0] = np.nan
    model = _series(
        "1989-01-01", "2014-12-31", COARSE_LAT, COARSE_LON, offset=2, seed=3
    )
    _write(fine, reference / "fine" / "tas.zarr")
    _write(coarse, reference / "coarse" / "tas.zarr")
    _write(model, canonical / "MODEL" / "historical" / "hist" / "tas.zarr")
    arguments = [
        "--model", "MODEL",
        "--scenario", "historical",
        "--variables", "tas",
        "--reference-root", str(reference),
        "--canonical-root", str(canonical),
        "--adjusted-root", str(adjusted),
        "--output-root", str(output),
        "--tile-lat-degrees", "2",
        "--tile-lon-degrees", "2",
        "--tile-workers", "2",
        "--iterations", "3",
    ]  # fmt: skip

    tiled_runner.main(arguments, default_regions=["global"])

    first = capsys.readouterr().out
    assert "START tas global MBCnSD tiles: 2/2" in first
    assert first.count("DONE tas global tile") == 2
    assert "FAILED" not in first
    assert (reference / "support" / "fine-tas.zarr").is_dir()
    assert (reference / "support" / "coarse-tas.zarr").is_dir()

    store = output / "global" / "tas_downscaled.zarr"
    assert zarr.open_group(store, mode="r")["tas"].dtype == np.dtype("int16")
    downscaled = xr.open_zarr(store, consolidated=False)["tas"].load()
    assert downscaled.sizes == {"time": 9490, "lat": 6, "lon": 8}
    ocean = np.zeros((6, 8), dtype=bool)
    ocean[:2, :2] = True
    ocean[5, 7] = True
    np.testing.assert_array_equal(downscaled.isnull().all("time").values, ocean)
    assert not np.isnan(downscaled.values[:, ~ocean]).any()
    # The +2 K model bias is removed by the adjustment before downscaling.
    assert abs(float(downscaled.mean()) - float(fine.mean())) < 0.5

    state = output / "global" / "state_spatial_global_context" / "tas"
    reports = sorted(state.glob("*.report.json"))
    assert len(reports) == len(list(state.glob("*.success"))) == 2
    for report in reports:
        record = json.loads(report.read_text())
        assert record["valid"] is True
        assert record["qc_time_steps"] == 9490  # the complete record is checked

    # A restart reuses every adjustment and spatial checkpoint.
    tiled_runner.main(arguments, default_regions=["global"])
    second = capsys.readouterr().out
    assert "adjustment tiles: 0; missing cells: 0" in second
    assert "START tas global MBCnSD tiles: 0/2" in second
    assert "DONE tas global tile" not in second
