import numpy as np
import shapely
import xarray as xr

from isimip3basd_modern.coastal import build_coastal_fill_plan


def test_coastal_plan_adds_only_adjacent_land_intersections():
    valid = xr.DataArray(
        np.array(
            [[False, False, False], [False, True, False], [False, False, False]]
        ),
        dims=("lat", "lon"),
        coords={"lat": [0.0, 1.0, 2.0], "lon": [0.0, 1.0, 2.0]},
    )
    land = shapely.box(1.51, 0.51, 2.49, 1.49)

    plan = build_coastal_fill_plan(valid, land)

    expected = np.zeros((3, 3), dtype=bool)
    expected[1, 2] = True
    np.testing.assert_array_equal(plan.coastal_fill, expected)
    assert int(plan.source_lat_index[1, 2]) == 1
    assert int(plan.source_lon_index[1, 2]) == 1


def test_coastal_plan_wraps_neighbor_search_at_dateline():
    valid = xr.DataArray(
        np.array(
            [[False, False, False], [True, False, False], [False, False, False]]
        ),
        dims=("lat", "lon"),
        coords={"lat": [-1.0, 0.0, 1.0], "lon": [0.0, 120.0, 240.0]},
    )
    land = shapely.box(-121.0, -0.4, -119.0, 0.4)

    plan = build_coastal_fill_plan(valid, land)

    assert bool(plan.coastal_fill[1, 2])
    assert int(plan.source_lon_index[1, 2]) == 0
