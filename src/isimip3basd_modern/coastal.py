"""Conservative coastal support extensions for land-only reference grids."""

from __future__ import annotations

import numpy as np
import shapely
import xarray as xr
from scipy.ndimage import binary_dilation


def _neighbor_source_indices(valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic nearest valid neighbor for one-cell gaps."""
    source_lat = np.full(valid.shape, -1, dtype=np.int16)
    source_lon = np.full(valid.shape, -1, dtype=np.int16)
    rows, columns = np.indices(valid.shape)
    for dy, dx in (
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ):
        neighbor = np.roll(valid, shift=(-dy, -dx), axis=(0, 1))
        if dy < 0:
            neighbor[0, :] = False
        elif dy > 0:
            neighbor[-1, :] = False
        available = (source_lat < 0) & neighbor
        source_lat[available] = (rows[available] + dy).astype(np.int16)
        source_lon[available] = ((columns[available] + dx) % valid.shape[1]).astype(
            np.int16
        )
    return source_lat, source_lon


def build_coastal_fill_plan(
    valid_mask: xr.DataArray,
    land_geometry: object,
) -> xr.Dataset:
    """Select one-cell gaps whose grid footprints intersect mapped land."""
    if valid_mask.dims != ("lat", "lon"):
        raise ValueError("valid mask must have lat, lon dimensions")
    valid = np.asarray(valid_mask.values, dtype=bool)
    padded = np.pad(valid, ((1, 1), (0, 0)), constant_values=False)
    padded = np.concatenate((padded[:, -1:], padded, padded[:, :1]), axis=1)
    adjacent = binary_dilation(padded, structure=np.ones((3, 3), dtype=bool))[
        1:-1, 1:-1
    ]
    candidates = adjacent & ~valid
    candidate_lat, candidate_lon = np.where(candidates)

    latitudes = np.asarray(valid_mask.lat.values)
    longitudes = np.asarray(valid_mask.lon.values)
    lat_half = float(np.median(np.diff(latitudes))) / 2
    lon_half = float(np.median(np.diff(longitudes))) / 2
    candidate_x = ((longitudes[candidate_lon] + 180) % 360) - 180
    candidate_y = latitudes[candidate_lat]
    footprints = shapely.box(
        candidate_x - lon_half,
        candidate_y - lat_half,
        candidate_x + lon_half,
        candidate_y + lat_half,
    )
    intersects_land = shapely.intersects(footprints, land_geometry)

    coastal_fill = np.zeros(valid.shape, dtype=bool)
    coastal_fill[candidate_lat[intersects_land], candidate_lon[intersects_land]] = True
    source_lat, source_lon = _neighbor_source_indices(valid)
    if np.any((source_lat < 0) & coastal_fill):
        raise RuntimeError("coastal fill cell has no adjacent valid donor")
    source_lat[~coastal_fill] = -1
    source_lon[~coastal_fill] = -1

    coords = {"lat": valid_mask.lat, "lon": valid_mask.lon}
    return xr.Dataset(
        {
            "coastal_fill": xr.DataArray(
                coastal_fill, dims=("lat", "lon"), coords=coords
            ),
            "source_lat_index": xr.DataArray(
                source_lat, dims=("lat", "lon"), coords=coords
            ),
            "source_lon_index": xr.DataArray(
                source_lon, dims=("lat", "lon"), coords=coords
            ),
        },
        attrs={
            "method": "one-cell dilation intersected with land polygons",
            "donor_method": "nearest cardinal then diagonal valid land cell",
            "periodic_longitude": "true",
            "coastal_fill_cell_count": int(coastal_fill.sum()),
        },
    )
