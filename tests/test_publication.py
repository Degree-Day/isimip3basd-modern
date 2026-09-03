import json

import numpy as np
import pytest
import xarray as xr
import zarr

from isimip3basd_modern.publication import (
    PACKED_FILL_VALUE,
    pack_zarr,
    packing_encoding,
)


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


@pytest.mark.parametrize("zarr_format", (2, 3))
def test_packing_encoding_writes_physical_int16_for_supported_zarr_formats(
    tmp_path, zarr_format
):
    output = tmp_path / f"packed-v{zarr_format}.zarr"
    source = sample_dataset()

    source.to_zarr(
        output,
        zarr_format=zarr_format,
        encoding={"tas": packing_encoding("tas", zarr_format=zarr_format)},
    )

    assert zarr.open_group(output, mode="r")["tas"].dtype == np.dtype("int16")
    with xr.open_zarr(output) as decoded:
        error = abs(decoded.tas - source.tas).max(skipna=True)
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


def test_pack_zarr_handles_masked_cells(tmp_path):
    source = tmp_path / "source.zarr"
    output = tmp_path / "packed.zarr"
    values = np.array(
        [[[85.123, np.nan], [90.456, np.nan]]],
        dtype="float32",
    )
    xr.Dataset(
        {"ffmc": (("time", "lat", "lon"), values)},
        coords={"time": [0], "lat": [0.0, 0.1], "lon": [10.0, 10.1]},
    ).to_zarr(source, zarr_format=3)

    report = pack_zarr(source, output)

    assert report.valid
    with xr.open_zarr(output, consolidated=False) as decoded:
        assert np.isnan(decoded.ffmc.isel(lon=1)).all()


def test_pack_zarr_allows_float32_scale_offset_rounding(tmp_path):
    source = tmp_path / "source.zarr"
    output = tmp_path / "packed.zarr"
    values = np.linspace(0, 6000, 10_001, dtype="float32")
    xr.Dataset({"dmc": ("sample", values)}).to_zarr(source, zarr_format=3)

    report = pack_zarr(source, output)

    assert report.valid
    assert report.variables[0].maximum_absolute_error <= 0.104
