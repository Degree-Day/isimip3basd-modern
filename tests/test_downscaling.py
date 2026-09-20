import dask
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from isimip3basd_modern.downscaling import (
    aggregate_to_coarse_grid,
    analyze_input_grids,
    apply_downscaled_value_controls,
    bilinear_broadcast,
    coarse_scale_conservation,
    downscale_variable,
    generate_rotation_matrices,
    weighted_sum_preserving_mbcn,
)
from isimip3basd_modern.validation import validate_variable


COARSE_COORDINATES = [0.0, 2.0]
FINE_COORDINATES = [-0.5, 0.5, 1.5, 2.5]


def climate_data(
    values,
    coordinates,
    time,
    *,
    variable="tas",
    units="K",
):
    data = xr.DataArray(
        values,
        coords={"time": time, "lat": coordinates, "lon": coordinates},
        dims=("time", "lat", "lon"),
        name=variable,
        attrs={"units": units},
    )
    data.lat.attrs["units"] = "degrees_north"
    data.lon.attrs["units"] = "degrees_east"
    return data


def monthly_inputs(variable="tas", units="K"):
    observation_time = pd.date_range("2000-01-01", periods=36, freq="MS")
    simulation_time = pd.date_range("2065-01-01", periods=24, freq="MS")
    random = np.random.RandomState(8)
    if variable == "tas":
        observations = 280 + random.normal(size=(36, 4, 4))
        simulation = 281 + random.normal(size=(24, 2, 2))
    else:
        observations = np.maximum(0, random.gamma(2, 1, size=(36, 4, 4)))
        simulation = np.maximum(0, random.gamma(2, 1, size=(24, 2, 2)))
    return (
        climate_data(
            observations,
            FINE_COORDINATES,
            observation_time,
            variable=variable,
            units=units,
        ),
        climate_data(
            simulation,
            COARSE_COORDINATES,
            simulation_time,
            variable=variable,
            units=units,
        ),
    )


def test_grid_analysis_requires_nested_fine_cells():
    observations, simulation = monthly_inputs()
    grid = analyze_input_grids(simulation, observations)

    assert grid.factors == (2, 2)
    assert grid.ascending == (True, True)

    observations = observations.assign_coords(lon=[-0.4, 0.5, 1.5, 2.5])
    with pytest.raises(ValueError, match="fine cells"):
        analyze_input_grids(simulation, observations)


def test_grid_analysis_supports_single_coarse_cell_tiles():
    time = pd.date_range("2000-01-01", periods=2)
    simulation = climate_data(
        np.ones((2, 1, 1)),
        [0.0],
        time,
    )
    observations = climate_data(
        np.ones((2, 10, 10)),
        np.arange(-0.45, 0.55, 0.1),
        time,
    )

    grid = analyze_input_grids(simulation, observations)

    assert grid.factors == (10, 10)
    assert grid.ascending == (True, True)


def test_bilinear_broadcast_uses_central_value_at_outer_edges():
    time = pd.date_range("2001-01-01", periods=1)
    simulation = climate_data(
        np.array([[[0.0, 2.0], [4.0, 6.0]]]),
        COARSE_COORDINATES,
        time,
    )
    observations = climate_data(
        np.zeros((1, 4, 4)), FINE_COORDINATES, time
    )

    result = bilinear_broadcast(simulation, observations)

    assert result.dtype == simulation.dtype
    assert result.isel(time=0, lat=0, lon=0).item() == 0
    assert result.isel(time=0, lat=-1, lon=-1).item() == 6
    assert result.isel(time=0, lat=1, lon=1).item() == pytest.approx(1.5)


def test_bilinear_broadcast_supports_single_coarse_cell_tiles():
    time = pd.date_range("2001-01-01", periods=1)
    simulation = climate_data(np.array([[[7.0]]]), [0.0], time)
    observations = climate_data(
        np.zeros((1, 10, 10)),
        np.arange(-0.45, 0.55, 0.1),
        time,
    )

    result = bilinear_broadcast(simulation, observations)

    assert result.sizes == {"time": 1, "lat": 10, "lon": 10}
    assert np.all(result.values == 7.0)


def test_bilinear_broadcast_wraps_circular_longitude():
    time = pd.date_range("2001-01-01", periods=1)
    coarse_lon = [-135.0, -45.0, 45.0, 135.0]
    fine_lon = [-157.5, -112.5, -67.5, -22.5, 22.5, 67.5, 112.5, 157.5]
    simulation = xr.DataArray(
        np.broadcast_to(np.arange(1.0, 5.0), (1, 2, 4)),
        coords={"time": time, "lat": COARSE_COORDINATES, "lon": coarse_lon},
        dims=("time", "lat", "lon"),
        name="tas",
        attrs={"units": "K"},
    )
    observations = xr.DataArray(
        np.zeros((1, 4, 8)),
        coords={"time": time, "lat": FINE_COORDINATES, "lon": fine_lon},
        dims=("time", "lat", "lon"),
        name="tas",
        attrs={"units": "K"},
    )

    result = bilinear_broadcast(simulation, observations)

    assert result.isel(time=0, lat=1, lon=0).item() == pytest.approx(1.75)
    assert result.isel(time=0, lat=1, lon=-1).item() == pytest.approx(3.25)


def test_mbcnsd_core_matches_archived_v302_result():
    observations = np.array(
        [[1.0, 5.0], [2.0, 4.0], [3.0, 8.0], [4.0, 7.0], [5.0, 9.0], [6.0, 6.0]]
    )
    coarse = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    fine = np.array(
        [[1.5, 2.5], [2.5, 3.5], [3.5, 4.5], [4.5, 5.5], [5.5, 6.5]]
    )
    expected = np.array(
        [
            [0.00155682408, 2.78048780],
            [1.23028218, 3.97549408],
            [2.37329178, 5.25830433],
            [3.55867130, 6.86961402],
            [4.95583388, 7.65853659],
        ]
    )
    rotations = generate_rotation_matrices(2, iterations=2, random_seed=3)

    result = weighted_sum_preserving_mbcn(
        observations,
        coarse,
        fine,
        np.array([0.8, 1.0]),
        rotations,
        n_quantiles=3,
    )

    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-8)


def test_downscale_variable_is_lazy_reproducible_and_uses_fine_grid():
    observations, simulation = monthly_inputs()
    first = downscale_variable(
        observations.chunk({"time": -1, "lat": 2, "lon": 2}),
        simulation.chunk({"time": -1, "lat": 1, "lon": 1}),
        variable="tas",
        iterations=2,
        random_seed=4,
    )
    second = downscale_variable(
        observations,
        simulation,
        variable="tas",
        iterations=2,
        random_seed=4,
    )

    assert first.chunks is not None
    assert first.sizes == {"time": 24, "lat": 4, "lon": 4}
    assert first.lat.equals(observations.lat)
    assert first.lon.equals(observations.lon)
    assert first.attrs["statistical_downscaling_method"] == "MBCnSD"
    xr.testing.assert_allclose(first.compute(), second.compute())
    conservation = coarse_scale_conservation(first, simulation)
    assert np.isfinite(conservation["normalized_rmse"])


def test_downscale_variable_aligns_fine_and_coarse_chunks():
    observations, simulation = monthly_inputs()

    result = downscale_variable(
        observations,
        simulation,
        variable="tas",
        iterations=2,
        chunks={"lat": 2, "lon": 2},
    )

    assert result.chunksizes["lat"] == (2, 2)
    assert result.chunksizes["lon"] == (2, 2)
    assert result.data.npartitions == 4


def test_downscale_variable_keeps_land_when_coarse_block_contains_ocean():
    observations, simulation = monthly_inputs()
    observations[:, 0, 0] = np.nan

    result = downscale_variable(
        observations,
        simulation,
        variable="tas",
        iterations=2,
        random_seed=5,
    ).compute()

    coarse_block = result.isel(lat=slice(0, 2), lon=slice(0, 2))
    assert bool(coarse_block.isel(lat=0, lon=0).isnull().all())
    assert bool(np.isfinite(coarse_block.isel(lat=0, lon=1)).all())
    assert bool(np.isfinite(coarse_block.isel(lat=1, lon=0)).all())
    assert bool(np.isfinite(coarse_block.isel(lat=1, lon=1)).all())


def test_downscale_variable_preserves_coarse_series_for_one_active_fine_cell():
    observations, simulation = monthly_inputs()
    observations[:, :2, :2] = np.nan
    observations[:, 1, 1] = 282.0

    result = downscale_variable(
        observations,
        simulation,
        variable="tas",
        iterations=2,
        random_seed=5,
    ).compute()
    coarse_block = result.isel(lat=slice(0, 2), lon=slice(0, 2))

    xr.testing.assert_allclose(
        coarse_block.isel(lat=1, lon=1, drop=True),
        simulation.isel(lat=0, lon=0, drop=True),
    )
    assert int(coarse_block.notnull().sum(("lat", "lon")).max()) == 1


def haloed_inputs(variable="tas", units="K", *, daily=False):
    """A 4 x 5 coarse context whose 2 x 3 interior is the tile core.

    Daily series give every calendar month enough samples for the quantile
    mapping to amplify rounding-level input differences; monthly series are
    cheap but would hide them.
    """
    factor = 3
    coarse_lat = np.arange(40.5, 44.5)
    coarse_lon = np.arange(10.5, 15.5)
    fine_lat = 40 + (np.arange(coarse_lat.size * factor) + 0.5) / factor
    fine_lon = 10 + (np.arange(coarse_lon.size * factor) + 0.5) / factor
    frequency, observed_steps, simulated_steps = (
        ("D", 3 * 365, 2 * 365) if daily else ("MS", 48, 36)
    )
    observation_time = xr.date_range(
        "2000-01-01",
        periods=observed_steps,
        freq=frequency,
        calendar="noleap",
        use_cftime=True,
    )
    simulation_time = xr.date_range(
        "2065-01-01",
        periods=simulated_steps,
        freq=frequency,
        calendar="noleap",
        use_cftime=True,
    )
    random = np.random.RandomState(11)
    fine_shape = (observed_steps, fine_lat.size, fine_lon.size)
    coarse_shape = (simulated_steps, coarse_lat.size, coarse_lon.size)
    if variable == "tas":
        observations = 280 + random.normal(size=fine_shape)
        simulation = 281 + random.normal(size=coarse_shape)
    else:
        observations = random.gamma(0.6, 4, size=fine_shape) * (
            random.uniform(size=fine_shape) > 0.4
        )
        simulation = random.gamma(0.6, 4, size=coarse_shape) * (
            random.uniform(size=coarse_shape) > 0.4
        )
    # Ocean: one missing coarse cell in the halo and a partly missing core cell.
    observations[:, :factor, :factor] = np.nan
    observations[:, factor : factor + 2, factor : factor + 1] = np.nan

    def build(values, lat, lon, time):
        data = xr.DataArray(
            values.astype("float32"),
            coords={"time": time, "lat": lat, "lon": lon},
            dims=("time", "lat", "lon"),
            name=variable,
            attrs={"units": units},
        )
        data.lat.attrs["units"] = "degrees_north"
        data.lon.attrs["units"] = "degrees_east"
        return data

    core = {
        "lat": slice(factor, 3 * factor),
        "lon": slice(factor, 4 * factor),
    }
    return (
        build(observations, fine_lat, fine_lon, observation_time),
        build(simulation, coarse_lat, coarse_lon, simulation_time),
        core,
        factor,
    )


def test_selecting_a_tile_core_never_downscales_the_halo(monkeypatch):
    """The tiled runner relies on Dask culling halo blocks after ``isel``."""
    import isimip3basd_modern.downscaling as downscaling

    observations, simulation, core, factor = haloed_inputs()
    calls = []
    original = downscaling._downscale_cell

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(downscaling, "_downscale_cell", counting)
    settings = {
        "variable": "tas",
        "iterations": 2,
        "chunks": {"lat": factor, "lon": factor},
    }

    with dask.config.set(scheduler="synchronous"):
        calls.clear()
        cropped = downscale_variable(observations, simulation, **settings)
        cropped = cropped.isel(core).compute()
        core_calls = len(calls)
        calls.clear()
        complete = downscale_variable(observations, simulation, **settings).compute()
        context_calls = len(calls)

    core_cells, context_cells = 2 * 3, 4 * 5
    # xarray may add one zero-size call to infer output metadata.
    assert core_cells <= core_calls <= core_cells + 1
    assert context_cells <= context_calls <= context_cells + 1
    np.testing.assert_array_equal(cropped.values, complete.isel(core).values)


def _below(module, minimum):
    from importlib.metadata import version

    from packaging.version import Version

    return Version(version(module)) < Version(minimum)


@pytest.mark.skipif(
    _below("xarray", "2026.7") or _below("dask", "2026.8"),
    reason=(
        "lazy and eager first guesses only agree to the last bit on the pinned "
        "xarray and dask releases; older ones interpolate Dask-backed "
        "precipitation differently at float64 rounding level"
    ),
)
@pytest.mark.parametrize(
    ("variable", "units"), [("tas", "K"), ("pr", "mm d-1")]
)
def test_eager_core_downscaling_is_bitwise_identical_to_the_lazy_tile_path(
    variable, units, monkeypatch
):
    import isimip3basd_modern.downscaling as downscaling

    observations, simulation, core, factor = haloed_inputs(
        variable, units, daily=True
    )
    # Force several first-guess time blocks so blocking itself is exercised.
    monkeypatch.setattr(downscaling, "EAGER_FIRST_GUESS_TIME_BLOCK", 200)
    lazy = (
        downscale_variable(
            observations.chunk({"time": 365}),
            simulation.chunk({"time": -1}),
            variable=variable,
            iterations=3,
            chunks={"lat": factor, "lon": factor},
        )
        .isel(core)
        .compute()
    )

    eager = downscale_variable(
        observations.chunk({"time": 365}),
        simulation.chunk({"time": -1}),
        variable=variable,
        iterations=3,
        core=core,
        eager=True,
    )

    assert eager.chunks is None
    assert eager.dtype == lazy.dtype == np.dtype("float32")
    assert eager.lat.equals(lazy.lat) and eager.lon.equals(lazy.lon)
    assert eager.time.equals(lazy.time)
    assert np.isfinite(eager.values).any() and np.isnan(eager.values).any()
    np.testing.assert_array_equal(eager.values, lazy.values)


def test_lazy_core_matches_cropping_and_eager_matches_lazy_without_a_core():
    observations, simulation, core, factor = haloed_inputs()
    settings = {"variable": "tas", "iterations": 2}
    complete = downscale_variable(observations, simulation, **settings).compute()

    lazy_core = downscale_variable(
        observations,
        simulation,
        core=core,
        chunks={"lat": factor, "lon": factor},
        **settings,
    )
    eager_complete = downscale_variable(
        observations, simulation, eager=True, **settings
    )

    assert lazy_core.data.npartitions == 6
    np.testing.assert_array_equal(
        lazy_core.compute().values, complete.isel(core).values
    )
    np.testing.assert_array_equal(eager_complete.values, complete.values)


def test_single_row_strips_reproduce_the_whole_tile():
    """The tiled runner bounds memory by downscaling one coarse row at a time."""
    observations, simulation, core, factor = haloed_inputs("pr", "mm d-1")
    settings = {"variable": "pr", "iterations": 3, "eager": True}
    whole = downscale_variable(observations, simulation, core=core, **settings)

    strips = []
    for row in range(core["lat"].start // factor, core["lat"].stop // factor):
        first, last = max(row - 1, 0), min(row + 2, simulation.sizes["lat"])
        strips.append(
            downscale_variable(
                observations.isel(lat=slice(first * factor, last * factor)),
                simulation.isel(lat=slice(first, last)),
                core={
                    "lat": slice((row - first) * factor, (row - first + 1) * factor),
                    "lon": core["lon"],
                },
                **settings,
            )
        )

    np.testing.assert_array_equal(
        xr.concat(strips, dim="lat").values, whole.values
    )


def test_core_and_eager_arguments_are_validated():
    observations, simulation, core, factor = haloed_inputs()

    with pytest.raises(ValueError, match="must align"):
        downscale_variable(
            observations,
            simulation,
            variable="tas",
            core={**core, "lat": slice(core["lat"].start + 1, core["lat"].stop)},
        )
    with pytest.raises(ValueError, match="non-spatial"):
        downscale_variable(
            observations, simulation, variable="tas", core={"time": slice(0, 3)}
        )
    with pytest.raises(ValueError, match="does not accept Dask chunk"):
        downscale_variable(
            observations,
            simulation,
            variable="tas",
            eager=True,
            chunks={"lat": factor, "lon": factor},
        )


def test_downscale_variable_rejects_misaligned_fine_chunks():
    observations, simulation = monthly_inputs()

    with pytest.raises(ValueError, match="positive multiple"):
        downscale_variable(
            observations,
            simulation,
            variable="tas",
            iterations=2,
            chunks={"lat": 3, "lon": 2},
        )


def test_coarse_aggregation_uses_only_valid_fine_cell_weights():
    time = pd.date_range("2001-01-01", periods=1)
    fine = climate_data(
        np.array(
            [[[np.nan, np.nan, 2.0, 4.0], [np.nan, np.nan, 6.0, 8.0],
              [1.0, 3.0, 10.0, 12.0], [5.0, 7.0, 14.0, 16.0]]]
        ),
        FINE_COORDINATES,
        time,
    )
    coarse = climate_data(
        np.zeros((1, 2, 2)), COARSE_COORDINATES, time
    )

    result = aggregate_to_coarse_grid(fine, coarse)

    assert np.isnan(result.isel(time=0, lat=0, lon=0))
    assert result.isel(time=0, lat=0, lon=1).item() == pytest.approx(5.0)


def test_precipitation_downscaling_respects_lower_bound():
    observations, simulation = monthly_inputs("pr", "mm d-1")
    observations[:, 0, 0] = 0
    simulation[:, 0, 0] = 0

    result = downscale_variable(
        observations,
        simulation,
        variable="pr",
        iterations=2,
        random_seed=0,
    ).compute()

    assert float(result.min()) >= 0


def test_downscaled_value_controls_cap_precipitation_at_cil_ceiling():
    time = pd.date_range("2001-01-01", periods=2)
    data = climate_data(
        np.array(
            [
                [[0.0, 0.05], [0.001, 0.02]],
                [[0.04, 0.0], [0.1, 0.0001]],
            ],
            dtype=np.float32,
        ),
        COARSE_COORDINATES,
        time,
        variable="pr",
        units="kg m-2 s-1",
    )

    result = apply_downscaled_value_controls(data, "pr")

    assert float(result.max()) == pytest.approx(3000 / 86400)
    assert float(result.min()) == 0.0
    assert result.attrs["cil_precipitation_ceiling"] == "3000 mm d-1"


def test_downscaled_value_controls_cap_wind_at_xclim_ceiling():
    data = climate_data(
        np.array(
            [
                [[-1.0, 12.0], [46.0, 48.5]],
                [[0.0, 20.0], [45.0, 100.0]],
            ],
            dtype=np.float32,
        ),
        COARSE_COORDINATES,
        pd.date_range("2001-01-01", periods=2),
        variable="sfcWind",
        units="m s-1",
    )

    result = apply_downscaled_value_controls(data, "sfcWind")

    assert float(result.min()) == 0.0
    assert float(result.max()) == 46.0
    assert result.attrs["xclim_wind_ceiling"] == "46 m s-1"


def test_downscaled_value_controls_mask_static_temperature_floor_cells():
    time = pd.date_range("2001-01-01", periods=3)
    values = np.full((3, 2, 2), 280.0, dtype=np.float32)
    values[:, 0, 0] = 150.0
    values[0, 1, 1] = 120.0
    data = climate_data(values, COARSE_COORDINATES, time)

    result = apply_downscaled_value_controls(data, "tas")

    assert result[:, 0, 0].isnull().all()
    assert np.isnan(float(result[0, 1, 1]))
    assert float(result[1, 1, 1]) == 280.0
    assert result.attrs["static_temperature_floor_cells_masked"] == "true"


def test_validation_rejects_precipitation_above_cil_ceiling():
    time = pd.date_range("2001-01-01", periods=2)
    data = climate_data(
        np.full((2, 2, 2), 4000 / 86400, dtype=np.float32),
        COARSE_COORDINATES,
        time,
        variable="pr",
        units="kg m-2 s-1",
    )

    report = validate_variable(data, "pr", statistical=False)

    assert not report.physical_bounds
    assert not report.valid
