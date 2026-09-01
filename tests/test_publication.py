import json

import numpy as np
import pytest
import xarray as xr

from isimip3basd_modern.publication import PACKED_FILL_VALUE, pack_zarr


def sample_dataset() -> xr.Dataset:
    time = xr.date_range(
        "2001-01-01", periods=365, freq="D", calendar="noleap", use_cftime=True
    )
    values = np.linspace(260, 320, 365 * 4 * 5, dtype="float32").reshape(365, 4, 5)
    values[:, 0, 0] = np.nan
    return xr.Dataset(
        {
            "tas": xr.DataArray(
                values,
                dims=("time", "lat", "lon"),
                coords={"time": time, "lat": np.arange(4), "lon": np.arange(5)},
                attrs={"units": "K", "standard_name": "air_temperature"},
            )
        }
    )


def test_pack_zarr_writes_int16_and_decodes_with_bounded_error(tmp_path):
    source = tmp_path / "source.zarr"
    output = tmp_path / "packed.zarr"
    sample_dataset().to_zarr(source, zarr_format=3)

    report = pack_zarr(
        source,
        output,
        chunks={"time": 73, "lat": 2, "lon": 3},
    )

    assert report.valid
    metadata = json.loads((output / "tas" / "zarr.json").read_text())
    assert metadata["data_type"] == "int16"
    assert metadata["fill_value"] == int(PACKED_FILL_VALUE)
    assert metadata["chunk_grid"]["configuration"]["chunk_shape"] == [73, 2, 3]
    assert metadata["codecs"][1]["name"] == "blosc"
    with xr.open_zarr(output, consolidated=False) as decoded:
        assert decoded.tas.dtype.kind == "f"
        assert decoded.tas.isel(lat=0, lon=0).isnull().all()
        error = abs(decoded.tas - sample_dataset().tas).max(skipna=True)
        assert float(error) <= 0.0025 + np.finfo("float32").eps


def test_pack_zarr_rejects_saturation(tmp_path):
    source = tmp_path / "source.zarr"
    output = tmp_path / "packed.zarr"
    dataset = sample_dataset()
    dataset.tas[0, 1, 1] = 500
    dataset.to_zarr(source, zarr_format=3)

    with pytest.raises(ValueError, match="above packed range"):
        pack_zarr(source, output)


def test_pack_zarr_rejects_coordinate_only_store(tmp_path):
    source = tmp_path / "source.zarr"
    xr.Dataset(coords={"lat": [0.0], "lon": [0.0]}).to_zarr(source, zarr_format=3)

    with pytest.raises(ValueError, match="no data variables"):
        pack_zarr(source, tmp_path / "packed.zarr")
