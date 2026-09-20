from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import zarr

import isimip3basd_modern.tiled_runner as RUNNER




def test_global_default_output_root_includes_model_scenario_and_stage():
    path = RUNNER.default_output_root(
        "ACCESS-CM2", "ssp245", "projection", ["global"]
    )

    expected_root = Path("/nas/dat1/cmip6_fwi/processing/downscaled_0p1deg")
    assert path == expected_root / "ACCESS-CM2" / "ssp245" / "projection"


def test_fit_cache_path_is_shared_across_application_periods(tmp_path):
    tile = {
        "coarse_lat_start": 10,
        "coarse_lat_stop": 15,
        "coarse_lon_start": 20,
        "coarse_lon_stop": 30,
    }

    assert RUNNER.adjustment_fit_cache_path(
        tmp_path, "ACCESS-CM2", "tas", tile
    ) == (
        tmp_path
        / "ACCESS-CM2"
        / "tas"
        / "lat010-015_lon020-030.zarr"
    )


def test_us_regions_are_nested_on_point_one_degree_grid():
    for name in ("socal", "spokane"):
        region = RUNNER.REGIONS[name]
        assert region["fine_lat"].start == region["coarse_lat"].start * 10
        assert region["fine_lat"].stop == region["coarse_lat"].stop * 10
        assert region["fine_lon"].start == region["coarse_lon"].start * 10
        assert region["fine_lon"].stop == region["coarse_lon"].stop * 10


def test_existing_adjusted_store_must_match_requested_coordinates(tmp_path):
    path = tmp_path / "tas.zarr"
    original = xr.DataArray(
        np.ones((2, 1, 1), dtype="float32"),
        dims=("time", "lat", "lon"),
        coords={"time": [0, 1], "lat": [0.5], "lon": [0.5]},
        name="tas",
    )
    RUNNER.initialize_adjusted_store(original, path)
    incompatible = original.assign_coords(lon=[1.5])

    with np.testing.assert_raises_regex(ValueError, "does not match"):
        RUNNER.initialize_adjusted_store(incompatible, path)


def test_adjusted_and_coverage_stores_use_processing_tile_chunks(tmp_path):
    simulation = xr.DataArray(
        np.ones((3, 7, 12), dtype="float32"),
        dims=("time", "lat", "lon"),
        coords={"time": range(3), "lat": range(7), "lon": range(12)},
        name="tas",
        attrs={"units": "K"},
    )
    adjusted = tmp_path / "tas.zarr"
    coverage = tmp_path / "tas.coverage.zarr"

    RUNNER.initialize_adjusted_store(
        simulation, adjusted, spatial_chunks=(5, 10)
    )
    RUNNER.initialize_coverage_store(
        simulation, coverage, spatial_chunks=(5, 10)
    )

    assert zarr.open_group(adjusted, mode="r")["tas"].chunks == (3, 5, 10)
    assert zarr.open_group(coverage, mode="r")["coverage"].chunks == (5, 10)


def test_adjustment_region_write_aligns_smaller_dask_chunks(tmp_path):
    simulation = xr.DataArray(
        np.zeros((3, 5, 10), dtype="float32"),
        dims=("time", "lat", "lon"),
        coords={"time": range(3), "lat": range(5), "lon": range(10)},
        name="tas",
        attrs={"units": "K"},
    )
    path = tmp_path / "tas.zarr"
    RUNNER.initialize_adjusted_store(
        simulation, path, spatial_chunks=(5, 10)
    )
    adjusted = xr.ones_like(simulation).chunk({"time": -1, "lat": 1, "lon": 1})

    RUNNER.write_adjusted_region(
        adjusted,
        path,
        {"time": slice(0, 3), "lat": slice(0, 5), "lon": slice(0, 10)},
    )

    written = xr.open_zarr(path, consolidated=False)["tas"].compute()
    np.testing.assert_array_equal(written.values, np.ones((3, 5, 10)))


def test_old_hurs_adjusted_store_is_marked_stale(tmp_path):
    path = tmp_path / "hurs.zarr"
    humidity = xr.DataArray(
        np.full((2, 1, 1), 80.0, dtype="float32"),
        dims=("time", "lat", "lon"),
        coords={"time": [0, 1], "lat": [0.5], "lon": [0.5]},
        name="hurs",
        attrs={"units": "%"},
    )
    humidity.to_dataset().to_zarr(path, zarr_format=3)

    assert RUNNER.initialize_adjusted_store(humidity, path)
    with xr.open_zarr(path, consolidated=False) as updated:
        assert updated.hurs.attrs["bias_adjustment_preset_revision"] == 4
        assert updated.hurs.attrs["bias_adjustment_kind"] == "bounded"
        assert updated.hurs.attrs["bias_adjustment_distribution"] == "nonparametric"
        assert updated.hurs.attrs["bias_adjustment_bound_frequency"] == (
            "fixed_to_reference"
        )


def test_current_hurs_adjusted_store_is_reusable(tmp_path):
    path = tmp_path / "hurs.zarr"
    humidity = xr.DataArray(
        np.full((2, 1, 1), 80.0, dtype="float32"),
        dims=("time", "lat", "lon"),
        coords={"time": [0, 1], "lat": [0.5], "lon": [0.5]},
        name="hurs",
        attrs={"units": "%"},
    )

    assert not RUNNER.initialize_adjusted_store(humidity, path)
    zarr.open_group(path, mode="a")["hurs"].attrs.update(
        bias_adjustment_kind="stale"
    )
    assert not RUNNER.initialize_adjusted_store(humidity, path)
    with xr.open_zarr(path, consolidated=False) as updated:
        assert updated.hurs.attrs["bias_adjustment_kind"] == "bounded"


def test_downscaled_store_is_physically_scaled_int16(tmp_path):
    path = tmp_path / "tas_downscaled.zarr"
    adjusted = xr.DataArray(
        np.full((2, 1, 1), 285.0, dtype="float32"),
        dims=("time", "lat", "lon"),
        coords={"time": [0, 1], "lat": [0.5], "lon": [0.5]},
        name="tas",
        attrs={"units": "K"},
    )

    RUNNER.initialize_output_store(
        adjusted,
        adjusted.isel(time=0, drop=True),
        path,
        iterations=20,
        quantiles=50,
    )

    metadata = json.loads((path / "tas" / "zarr.json").read_text())
    assert metadata["data_type"] == "int16"
    assert metadata["codecs"][1]["name"] == "blosc"
    RUNNER.variable_only_dataset(adjusted).to_zarr(
        path,
        mode="r+",
        region={"time": slice(0, 2), "lat": slice(0, 1), "lon": slice(0, 1)},
        consolidated=False,
    )
    with xr.open_zarr(path, consolidated=False) as decoded:
        assert decoded.tas.dtype.kind == "f"
        assert decoded.tas.attrs["storage_format"] == "scaled int16 Zarr v3"
        np.testing.assert_allclose(decoded.tas.values, adjusted.values, atol=0.0025)


def test_old_hurs_downscaled_store_is_marked_stale(tmp_path):
    path = tmp_path / "hurs_downscaled.zarr"
    adjusted = xr.DataArray(
        np.full((2, 1, 1), 80.0, dtype="float32"),
        dims=("time", "lat", "lon"),
        coords={"time": [0, 1], "lat": [0.5], "lon": [0.5]},
        name="hurs",
        attrs={"units": "%", "bias_adjustment_preset_revision": 4},
    )
    RUNNER.initialize_output_store(
        adjusted,
        adjusted.isel(time=0, drop=True),
        path,
        iterations=20,
        quantiles=50,
    )
    zarr.open_group(path, mode="a")["hurs"].attrs.update(
        bias_adjustment_preset_revision=1,
        bias_adjustment_kind="stale",
        bias_adjustment_distribution="stale",
    )

    assert RUNNER.initialize_output_store(
        adjusted,
        adjusted.isel(time=0, drop=True),
        path,
        iterations=20,
        quantiles=50,
    )
    with xr.open_zarr(path, consolidated=False) as updated:
        assert updated.hurs.attrs["bias_adjustment_kind"] == "bounded"
        assert updated.hurs.attrs["bias_adjustment_distribution"] == "nonparametric"
        assert updated.hurs.attrs["statistical_downscaling_method"] == "MBCnSD"


def test_existing_float_downscaled_store_is_rejected(tmp_path):
    path = tmp_path / "tas_downscaled.zarr"
    adjusted = xr.DataArray(
        np.full((2, 1, 1), 285.0, dtype="float32"),
        dims=("time", "lat", "lon"),
        coords={"time": [0, 1], "lat": [0.5], "lon": [0.5]},
        name="tas",
    )
    adjusted.to_dataset().to_zarr(path, zarr_format=3)

    with np.testing.assert_raises_regex(ValueError, "expected scaled int16"):
        RUNNER.initialize_output_store(
            adjusted,
            adjusted.isel(time=0, drop=True),
            path,
            iterations=20,
            quantiles=50,
        )


def test_tile_checkpoint_is_rejected_when_support_mask_expands(tmp_path):
    marker = tmp_path / "tile.success"
    marker.touch()
    marker.with_suffix(".report.json").write_text(
        json.dumps({"valid": True, "active_cells": 328})
    )

    assert RUNNER.tile_report_matches_mask(marker, 328)
    assert not RUNNER.tile_report_matches_mask(marker, 429)


def test_stale_spatial_checkpoint_is_removed_before_task_submission(tmp_path):
    tile = {
        "coarse_lat_start": 0,
        "coarse_lat_stop": 1,
        "coarse_lon_start": 0,
        "coarse_lon_stop": 1,
        "fine_lat_start": 0,
        "fine_lat_stop": 2,
        "fine_lon_start": 0,
        "fine_lon_stop": 2,
    }
    marker = RUNNER.marker_path(tmp_path, "global", "tas", tile)
    marker.parent.mkdir(parents=True)
    marker.touch()
    RUNNER.report_path(marker).write_text(
        json.dumps({"valid": True, "active_cells": 1})
    )

    current_mask = np.ones((2, 2), dtype=bool)
    assert not RUNNER.spatial_tile_already_written(
        tmp_path, "global", "tas", tile, current_mask
    )
    assert not marker.exists()
    assert not RUNNER.report_path(marker).exists()


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


def test_regional_tile_reads_spatial_halo_from_global_store():
    adjusted = xr.DataArray(
        np.arange(8).reshape(1, 2, 4),
        dims=("time", "lat", "lon"),
        coords={
            "time": [0],
            "lat": [-0.5, 0.5],
            "lon": [45.0, 135.0, 225.0, 315.0],
        },
    )
    fine = xr.DataArray(
        np.arange(32).reshape(1, 4, 8),
        dims=("time", "lat", "lon"),
        coords={
            "time": [0],
            "lat": [-0.75, -0.25, 0.25, 0.75],
            "lon": np.arange(22.5, 360.0, 45.0),
        },
    )
    region = {
        "coarse_lat": slice(0, 2),
        "coarse_lon": slice(3, 4),
        "fine_lat": slice(0, 4),
        "fine_lon": slice(6, 8),
        "lat_factor": 2,
        "lon_factor": 2,
    }
    global_region = {
        "coarse_lat": slice(0, 2),
        "coarse_lon": slice(0, 4),
        "fine_lat": slice(0, 4),
        "fine_lon": slice(0, 8),
        "lat_factor": 2,
        "lon_factor": 2,
        "periodic_lon": True,
    }
    tile = RUNNER.tile_specs(region, 2, 1)[0]

    simulation, observations, lat_center, lon_center = RUNNER.global_tile_contexts(
        adjusted, fine, region, global_region, tile
    )

    np.testing.assert_allclose(simulation.lon, [225.0, 315.0, 405.0])
    np.testing.assert_allclose(
        observations.lon,
        [202.5, 247.5, 292.5, 337.5, 382.5, 427.5],
    )
    xr.testing.assert_identical(
        observations.isel(lat=lat_center, lon=lon_center), fine.isel(lon=slice(6, 8))
    )


def test_adjustment_tiles_include_regional_halo_without_overlap():
    global_region = {
        "coarse_lat": slice(0, 4),
        "coarse_lon": slice(0, 8),
        "fine_lat": slice(0, 8),
        "fine_lon": slice(0, 16),
        "lat_factor": 2,
        "lon_factor": 2,
    }
    target = {
        "coarse_lat": slice(1, 3),
        "coarse_lon": slice(7, 8),
        "fine_lat": slice(2, 6),
        "fine_lon": slice(14, 16),
    }
    tiles = RUNNER.required_adjustment_tiles(global_region, [target], 2, 2)
    coverage = np.zeros((4, 8), dtype=np.uint8)
    for tile in tiles:
        coverage[
            tile["coarse_lat_start"] : tile["coarse_lat_stop"],
            tile["coarse_lon_start"] : tile["coarse_lon_stop"],
        ] += 1

    assert np.all(coverage[:, [0, 6, 7]] == 1)
    assert int(coverage.max()) == 1
    assert np.all(coverage[:, 2:6] == 0)


def test_partial_adjustment_restart_retains_regular_tiles():
    tiles = [
        {
            "coarse_lat_start": 0,
            "coarse_lat_stop": 2,
            "coarse_lon_start": start,
            "coarse_lon_stop": start + 2,
        }
        for start in (0, 2, 4)
    ]
    missing = np.zeros((2, 6), dtype=bool)
    missing[1, 3] = True

    pending = RUNNER.tiles_intersecting_mask(tiles, missing)

    assert pending == [tiles[1]]


def test_adjustment_restart_checks_stored_endpoints_as_well_as_coverage():
    required = np.array([[True, True], [False, True]])
    coverage = np.ones((2, 2), dtype=bool)
    endpoints = np.ones((2, 2, 2), dtype=float)
    endpoints[:, 0, 1] = np.nan

    missing = RUNNER.missing_adjustment_cells(required, coverage, endpoints)

    np.testing.assert_array_equal(
        missing, np.array([[False, True], [False, False]])
    )


def test_spatial_tiles_skip_empty_fine_regions():
    tiles = [
        {
            "fine_lat_start": 0,
            "fine_lat_stop": 2,
            "fine_lon_start": start,
            "fine_lon_stop": start + 2,
        }
        for start in (0, 2, 4)
    ]
    valid = np.zeros((2, 6), dtype=bool)
    valid[0, 4] = True

    selected = RUNNER.spatial_tiles_intersecting_mask(tiles, valid)

    assert selected == [tiles[2]]


def test_spatial_tiles_crop_empty_margins_to_parent_cells():
    tile = RUNNER.tile_specs(
        {
            "coarse_lat": slice(0, 4),
            "coarse_lon": slice(0, 5),
            "lat_factor": 2,
            "lon_factor": 2,
        },
        4,
        5,
    )[0]
    valid = np.zeros((8, 10), dtype=bool)
    valid[2:6, 4:8] = True

    cropped = RUNNER.spatial_tiles_intersecting_mask([tile], valid)[0]

    assert cropped["coarse_lat_start"] == 1
    assert cropped["coarse_lat_stop"] == 3
    assert cropped["coarse_lon_start"] == 2
    assert cropped["coarse_lon_stop"] == 4
    assert cropped["fine_lat_start"] == 2
    assert cropped["fine_lat_stop"] == 6
    assert cropped["fine_lon_start"] == 4
    assert cropped["fine_lon_stop"] == 8
    assert RUNNER._tile_name(cropped) == "lat000-004_lon000-005"


def test_adjustment_controls_cap_precipitation_before_qc():
    data = xr.DataArray(
        [-1.0, 0.001, 0.1],
        dims="time",
        name="pr",
        attrs={"units": "kg m-2 s-1"},
    )

    controlled = RUNNER.apply_downscaled_value_controls(data, "pr")

    assert float(controlled.min()) == 0.0
    assert float(controlled.max()) == 3000.0 / 86400.0


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
        variables=("tas",),
    )
    mask = xr.open_zarr(path, consolidated=False)["spatial_valid_mask"]

    assert not bool(mask.isel(lat=0, lon=0).compute().item())
    assert not bool(
        mask.isel(lat=slice(0, 2), lon=slice(2, 4)).any().compute().item()
    )
    assert bool(mask.isel(lat=3, lon=5).compute().item())


def _tile_series(values, name="tas", units="K"):
    return xr.DataArray(
        np.asarray(values, dtype=np.float32),
        dims=("time", "lat", "lon"),
        coords={
            "time": np.arange(np.shape(values)[0]),
            "lat": [0.25, 0.75],
            "lon": [0.25, 0.75],
        },
        name=name,
        attrs={"units": units},
    ).chunk({"time": -1})


def test_tile_qc_checks_years_beyond_the_first():
    reference = _tile_series(np.full((366, 2, 2), 280.0))
    written = np.full((4 * 365, 2, 2), 281.0)
    written[500:, 0, 0] = np.nan  # complete first year, then a hole

    report = RUNNER.tile_qc(_tile_series(written), reference, "tas")

    assert report["qc_time_steps"] == 4 * 365
    assert report["active_cells"] == 4
    assert report["partial_missing_cells"] == 1
    assert not report["valid"]


def test_tile_qc_accepts_a_complete_tile_and_reports_its_full_range():
    reference = _tile_series(np.full((366, 2, 2), 280.0))
    written = np.full((3 * 365, 2, 2), 281.0)
    written[-1, 1, 1] = 300.0

    report = RUNNER.tile_qc(_tile_series(written), reference, "tas")

    assert report["valid"], report["errors"]
    assert report["maximum"] == 300.0
    assert report["missing_reference_cells"] == 0
    assert report["extra_cells"] == 0


def test_tile_qc_flags_static_temperature_floor_from_the_bounds_table():
    reference = _tile_series(np.full((366, 2, 2), 280.0))
    written = np.full((730, 2, 2), 281.0)
    written[:, 0, 1] = 150.0

    report = RUNNER.tile_qc(_tile_series(written), reference, "tas")

    assert any("static" in error for error in report["errors"])


def _pool_task(*, tile, fail=False):
    if fail:
        raise ValueError(f"bad tile {tile['coarse_lat_start']}")
    return {"tile": RUNNER._tile_name(tile)}


def _pool_tile(index):
    return {
        "coarse_lat_start": index,
        "coarse_lat_stop": index + 1,
        "coarse_lon_start": 0,
        "coarse_lon_stop": 1,
    }


def test_tile_pool_reports_failures_immediately_and_finishes_other_tiles(capsys):
    from concurrent.futures import ThreadPoolExecutor

    tasks = [
        {"tile": _pool_tile(index), "fail": index == 1} for index in range(4)
    ]

    with pytest.raises(RuntimeError, match="1 of 4 tas global tiles failed"):
        RUNNER.run_tile_pool(
            "tas global",
            _pool_task,
            tasks,
            workers=2,
            executor_factory=lambda workers: ThreadPoolExecutor(workers),
        )

    output = capsys.readouterr().out
    assert output.count("DONE tas global tile") == 3
    assert "FAILED tas global tile" in output
    assert "ValueError: bad tile 1" in output
    assert "lat001-002_lon000-001" in output


def test_tile_pool_returns_records_and_skips_empty_work():
    from concurrent.futures import ThreadPoolExecutor

    factory = lambda workers: ThreadPoolExecutor(workers)  # noqa: E731
    assert RUNNER.run_tile_pool("x", _pool_task, [], workers=2) == []
    records = RUNNER.run_tile_pool(
        "x",
        _pool_task,
        [{"tile": _pool_tile(index)} for index in range(3)],
        workers=2,
        executor_factory=factory,
    )
    assert sorted(record["tile"] for record in records) == [
        "lat000-001_lon000-001",
        "lat001-002_lon000-001",
        "lat002-003_lon000-001",
    ]


def test_stale_adjusted_store_invalidates_checkpoints_before_restamping(tmp_path):
    path = tmp_path / "hurs.zarr"
    humidity = xr.DataArray(
        np.full((2, 1, 1), 80.0, dtype="float32"),
        dims=("time", "lat", "lon"),
        coords={"time": [0, 1], "lat": [0.5], "lon": [0.5]},
        name="hurs",
        attrs={"units": "%"},
    )
    humidity.to_dataset().to_zarr(path, zarr_format=3)
    revisions_seen_by_invalidate = []

    def invalidate():
        stored = zarr.open_group(path, mode="r", use_consolidated=False)["hurs"].attrs
        revisions_seen_by_invalidate.append(
            stored.get("bias_adjustment_preset_revision", 1)
        )

    assert RUNNER.initialize_adjusted_store(humidity, path, invalidate=invalidate)
    assert revisions_seen_by_invalidate == [1]
    assert (
        zarr.open_group(path, mode="r", use_consolidated=False)["hurs"].attrs[
            "bias_adjustment_preset_revision"
        ]
        == 4
    )

    # A current store is reusable and must not invalidate anything.
    assert not RUNNER.initialize_adjusted_store(
        humidity, path, invalidate=invalidate
    )
    assert revisions_seen_by_invalidate == [1]


def _write_reference(root, variable, values, grid="fine"):
    data = xr.DataArray(
        np.asarray(values, dtype=np.float32),
        dims=("time", "lat", "lon"),
        coords={
            "time": np.arange(np.shape(values)[0]),
            "lat": np.arange(np.shape(values)[1]) + 0.5,
            "lon": np.arange(np.shape(values)[2]) + 0.5,
        },
        name=variable,
    )
    data.to_dataset().to_zarr(
        root / grid / f"{variable}.zarr", mode="w", zarr_format=3
    )


def test_reference_support_is_cached_and_shared_across_variable_order(tmp_path):
    tas = np.full((3, 2, 3), 280.0)
    tas[1, 0, 0] = np.nan
    pr = np.full((3, 2, 3), 1.0)
    pr[2, 1, 2] = np.nan
    _write_reference(tmp_path, "tas", tas)
    _write_reference(tmp_path, "pr", pr)
    expected = np.ones((2, 3), dtype=bool)
    expected[0, 0] = expected[1, 2] = False

    first = RUNNER.reference_support(tmp_path, ("tas", "pr"), "fine")
    cache = tmp_path / "support" / "fine-pr-tas.zarr"
    assert cache.is_dir()
    np.testing.assert_array_equal(first.values, expected)

    # Prove the second call is served from the cache, in either order.
    marker = zarr.open_group(cache, mode="a")
    marker["support"][0, 1] = False
    second = RUNNER.reference_support(tmp_path, ("pr", "tas"), "fine")
    assert not bool(second.values[0, 1])

    uncached = RUNNER.reference_support(
        tmp_path, ("tas", "pr"), "fine", cache=False
    )
    np.testing.assert_array_equal(uncached.values, expected)


def test_reference_support_cache_is_rebuilt_when_the_reference_changes(tmp_path):
    tas = np.full((2, 2, 2), 280.0)
    _write_reference(tmp_path, "tas", tas)
    assert RUNNER.reference_support(tmp_path, ("tas",), "fine").values.all()

    tas[0, 1, 1] = np.nan
    _write_reference(tmp_path, "tas", tas)
    metadata = tmp_path / "fine" / "tas.zarr" / "tas" / "zarr.json"
    stamp = metadata.stat().st_mtime_ns + 10**9
    import os

    os.utime(metadata, ns=(stamp, stamp))

    rebuilt = RUNNER.reference_support(tmp_path, ("tas",), "fine")
    assert not bool(rebuilt.values[1, 1])
    assert rebuilt.values.sum() == 3


def test_common_coarse_support_uses_the_cached_footprint(tmp_path):
    tas = np.full((2, 2, 3), 280.0)
    tas[0, 0, 2] = np.nan
    _write_reference(tmp_path, "tas", tas, grid="coarse")
    spec = {"coarse_lat": slice(0, 2), "coarse_lon": slice(0, 3)}

    support = RUNNER.common_coarse_reference_support(tmp_path, ("tas",), spec)

    assert support.dtype == bool
    assert support.tolist() == [[True, True, False], [True, True, True]]
    assert (tmp_path / "support" / "coarse-tas.zarr").is_dir()


def test_coarse_row_strips_cover_the_tile_once_with_clipped_halos():
    # A tile of 3 coarse rows at the southern domain edge: no halo row below.
    strips = RUNNER.coarse_row_strips(4, slice(0, 30), 10)

    assert [strip["tile_rows"] for strip in strips] == [
        slice(0, 10),
        slice(10, 20),
        slice(20, 30),
    ]
    assert strips[0]["coarse_context"] == slice(0, 2)
    assert strips[0]["fine_core"] == slice(0, 10)
    assert strips[1]["coarse_context"] == slice(0, 3)
    assert strips[1]["fine_context"] == slice(0, 30)
    assert strips[1]["fine_core"] == slice(10, 20)
    assert strips[2]["coarse_context"] == slice(1, 4)

    # An interior tile keeps one halo row on both sides of every strip.
    interior = RUNNER.coarse_row_strips(7, slice(10, 60), 10)
    assert len(interior) == 5
    assert all(
        strip["coarse_context"].stop - strip["coarse_context"].start == 3
        for strip in interior
    )
    assert all(strip["fine_core"] == slice(10, 20) for strip in interior)

    with pytest.raises(ValueError, match="must align"):
        RUNNER.coarse_row_strips(4, slice(5, 30), 10)
