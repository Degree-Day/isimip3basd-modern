import numpy as np
import xarray as xr

from isimip3basd_modern.cli import _parser
from isimip3basd_modern.temperature import COMPONENTS, downscale_temperature


def test_default_temperature_method_preserves_parent_and_order():
    args = _parser().parse_args([
        "downscale-temperature", "--observations-fine", "fine", "--simulation-coarse", "coarse",
        "--output", "result",
    ])
    assert args.method == "joint-kriging"
    dates = xr.date_range("2001-01-01", periods=730, calendar="noleap", use_cftime=True)
    lat = np.arange(4) + 40.5
    lon = np.arange(4) + 250.5
    rng = np.random.default_rng(5)
    reference = xr.Dataset(coords={"time": dates, "lat": lat, "lon": lon})
    for name, center, scale in zip(COMPONENTS, (280, 10, 0.5), (2, 0.5, 0.04)):
        reference[name] = (("time", "lat", "lon"), center + scale * rng.normal(size=(730, 4, 4)))
    weights = np.cos(np.deg2rad(reference.lat))
    window = dict(lat=2, lon=2)
    area = weights.broadcast_like(reference.tas.isel(time=0, drop=True))
    coarse = (reference * area).coarsen(**window).sum() / area.coarsen(**window).sum()
    parent = coarse.isel(time=slice(0, 20))
    result = downscale_temperature(reference, parent)
    recovered = (result * area).coarsen(**window).sum() / area.coarsen(**window).sum()
    for name in COMPONENTS:
        np.testing.assert_allclose(recovered[name], parent[name], atol=1e-7, rtol=0)
    assert (result.tasmin <= result.tas).all() and (result.tas <= result.tasmax).all()
