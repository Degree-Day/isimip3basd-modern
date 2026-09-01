from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import xarray as xr


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_europe_downscale_tiles.py"
SPEC = importlib.util.spec_from_file_location("run_downscale_tiles", SCRIPT)
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_global_tile_specs_cover_domain_once_with_inferred_factors():
    region = {
        "coarse_lat": slice(0, 7),
        "coarse_lon": slice(0, 12),
        "lat_factor": 2,
        "lon_factor": 3,
    }
    tiles = RUNNER.tile_specs(region, 3, 5)
    coverage = np.zeros((14, 36), dtype=np.uint8)
    for tile in tiles:
        coverage[
            tile["fine_lat_start"] : tile["fine_lat_stop"],
            tile["fine_lon_start"] : tile["fine_lon_stop"],
        ] += 1

    assert np.all(coverage == 1)
    assert all(
        tile["coarse_lat_stop"] - tile["coarse_lat_start"] != 1 for tile in tiles
    )
    assert all(
        tile["coarse_lon_stop"] - tile["coarse_lon_start"] != 1 for tile in tiles
    )


def test_periodic_context_wraps_longitude_and_locates_center():
    data = xr.DataArray(
        np.arange(24).reshape(2, 12),
        dims=("lat", "lon"),
        coords={"lat": [-0.5, 0.5], "lon": np.arange(15.0, 360.0, 30.0)},
    )
    context, lat_center, lon_center = RUNNER._context_subset(
        data,
        lat_start=0,
        lat_stop=2,
        lon_start=0,
        lon_stop=2,
        lat_halo=1,
        lon_halo=1,
        periodic_lon=True,
    )

    np.testing.assert_allclose(context.lon, [-15.0, 15.0, 45.0, 75.0])
    xr.testing.assert_identical(
        context.isel(lat=lat_center, lon=lon_center), data[:, :2]
    )


def test_nonperiodic_context_clips_at_domain_edges():
    data = xr.DataArray(
        np.zeros((4, 5)),
        dims=("lat", "lon"),
        coords={"lat": np.arange(4), "lon": np.arange(5)},
    )
    context, lat_center, lon_center = RUNNER._context_subset(
        data,
        lat_start=0,
        lat_stop=2,
        lon_start=3,
        lon_stop=5,
        lat_halo=1,
        lon_halo=1,
        periodic_lon=False,
    )

    assert context.shape == (3, 3)
    assert context.isel(lat=lat_center, lon=lon_center).shape == (2, 2)


def test_spatial_mask_selects_canonical_model_by_coordinates(tmp_path):
    reference = tmp_path / "reference"
    canonical = tmp_path / "canonical"
    output = tmp_path / "output"
    coarse_lat = [0.5, 1.5]
    coarse_lon = [60.0, 180.0, 300.0]
    fine_lat = [0.25, 0.75, 1.25, 1.75]
    fine_lon = [30.0, 90.0, 150.0, 210.0, 270.0, 330.0]
    fine_values = np.full((1, 4, 6), 280.0, dtype=np.float32)
    fine_values[0, 0, 0] = np.nan
    fine = xr.DataArray(
        fine_values,
        dims=("time", "lat", "lon"),
        coords={"time": [0], "lat": fine_lat, "lon": fine_lon},
        name="tas",
        attrs={"units": "K"},
    )
    coarse_reference = xr.DataArray(
        np.full((1, 2, 3), 280.0, dtype=np.float32),
        dims=("time", "lat", "lon"),
        coords={"time": [0], "lat": coarse_lat, "lon": coarse_lon},
        name="tas",
        attrs={"units": "K"},
    )
    model_values = np.full((2, 4, 3), 285.0, dtype=np.float32)
    model_values[:, 2, 1] = 0.0
    model = xr.DataArray(
        model_values,
        dims=("time", "lat", "lon"),
        coords={
            "time": [0, 1],
            "lat": [-1.5, -0.5, 0.5, 1.5],
            "lon": coarse_lon,
        },
        name="tas",
        attrs={"units": "K"},
    )
    fine.to_dataset().to_zarr(reference / "fine" / "tas.zarr")
    coarse_reference.to_dataset().to_zarr(reference / "coarse" / "tas.zarr")
    model.to_dataset().to_zarr(
        canonical / "MODEL" / "ssp245" / "proj" / "tas.zarr"
    )
    region = {
        "fine_lat": slice(0, 4),
        "fine_lon": slice(0, 6),
        "coarse_lat": slice(0, 2),
        "coarse_lon": slice(0, 3),
        "lat_factor": 2,
        "lon_factor": 2,
    }

    path = RUNNER.ensure_spatial_valid_mask(
        model="MODEL",
        scenario="ssp245",
        simulation_stage="proj",
        simulation_start=None,
        simulation_end=None,
        region="global",
        region_spec=region,
        reference_root=reference,
        canonical_root=canonical,
        output_root=output,
    )
    mask = xr.open_zarr(path, consolidated=False)["spatial_valid_mask"]

    assert not bool(mask.isel(lat=0, lon=0).compute().item())
    assert not bool(
        mask.isel(lat=slice(0, 2), lon=slice(2, 4)).any().compute().item()
    )
    assert bool(mask.isel(lat=3, lon=5).compute().item())
