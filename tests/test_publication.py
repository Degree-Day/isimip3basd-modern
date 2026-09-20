import json

import numpy as np
import pytest
import xarray as xr
import zarr

import isimip3basd_modern.publication as publication
from isimip3basd_modern.publication import (
    PACKED_FILL_VALUE,
    PACKING_SPECS,
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


def test_pack_zarr_opens_source_with_native_chunks(tmp_path, monkeypatch):
    source = tmp_path / "source.zarr"
    output = tmp_path / "packed.zarr"
    sample_dataset().to_zarr(
        source,
        zarr_format=3,
        encoding={"tas": {"chunks": (365, 1, 1)}},
    )
    calls = []
    real_open_dataset = publication.open_dataset

    def tracked_open_dataset(path, chunks=None):
        calls.append(chunks)
        return real_open_dataset(path, chunks)

    monkeypatch.setattr(publication, "open_dataset", tracked_open_dataset)

    report = pack_zarr(
        source,
        output,
        chunks={"time": 73, "lat": 2, "lon": 3},
    )

    assert report.valid
    assert calls[0] is None


def test_pack_zarr_supports_read_optimized_annual_fwi_chunks(tmp_path):
    source = tmp_path / "annual-source.zarr"
    output = tmp_path / "annual-published.zarr"
    dataset = xr.Dataset(
        {
            "fwixd": xr.DataArray(
                np.arange(24, dtype="float32").reshape(2, 3, 4),
                dims=("time", "lat", "lon"),
                coords={"time": [2000, 2001], "lat": range(3), "lon": range(4)},
                attrs={"units": "d"},
            )
        }
    )
    dataset.to_zarr(source, zarr_format=3)

    report = pack_zarr(
        source,
        output,
        variables=["fwixd"],
        chunks={"time": -1, "lat": 3, "lon": 4},
    )

    assert report.valid
    array = zarr.open_group(output, mode="r")["fwixd"]
    assert array.dtype == np.dtype("int16")
    assert array.chunks == (2, 3, 4)


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
    assert (
        report.variables[0].maximum_absolute_error
        <= PACKING_SPECS["dmc"].scale_factor / 2 + 1e-4
    )


def _packed_source(tmp_path, *, chunks=(365, 2, 2)):
    """Write a production-style store: physical int16 with publication packing."""
    source = tmp_path / "packed-source.zarr"
    dataset = sample_dataset().chunk(dict(zip(("time", "lat", "lon"), chunks)))
    dataset.to_zarr(
        source,
        mode="w",
        consolidated=False,
        zarr_format=3,
        encoding={"tas": packing_encoding("tas")},
    )
    return source


def _array_metadata(store, name="tas"):
    metadata = json.loads((store / name / "zarr.json").read_text())
    return {
        key: metadata[key]
        for key in ("data_type", "fill_value", "codecs", "attributes")
    }


def test_pack_zarr_rechunks_already_packed_codes_without_requantizing(tmp_path):
    source = _packed_source(tmp_path)
    fast = tmp_path / "fast.zarr"
    slow = tmp_path / "slow.zarr"
    chunks = {"time": 73, "lat": 4, "lon": 5}

    fast_report = pack_zarr(source, fast, chunks=chunks)
    slow_report = pack_zarr(source, slow, chunks=chunks, requantize=True)

    assert fast_report.method == "rechunked packed codes"
    assert slow_report.method == "requantized"
    assert fast_report.valid and fast_report.chunks == chunks
    assert fast_report.variables[0].maximum_absolute_error == 0.0

    fast_codes = zarr.open_group(fast, mode="r")["tas"]
    slow_codes = zarr.open_group(slow, mode="r")["tas"]
    source_codes = zarr.open_group(source, mode="r")["tas"]
    assert fast_codes.dtype == np.dtype("int16")
    assert fast_codes.chunks == (73, 4, 5)
    np.testing.assert_array_equal(fast_codes[:], source_codes[:])
    np.testing.assert_array_equal(fast_codes[:], slow_codes[:])
    assert _array_metadata(fast) == _array_metadata(slow)

    with (
        xr.open_zarr(source, consolidated=False) as original,
        xr.open_zarr(fast, consolidated=False) as published,
    ):
        assert published.tas.dtype == original.tas.dtype
        assert bool(published.tas.isel(lat=0, lon=0).isnull().all())
        xr.testing.assert_identical(original.tas.load(), published.tas.load())
        assert published.attrs["publication_format"] == "scaled int16 Zarr v3"

    # Both paths describe the same physical range in their QC reports.
    fast_variable, slow_variable = fast_report.variables[0], slow_report.variables[0]
    assert fast_variable.packed_minimum == pytest.approx(slow_variable.packed_minimum)
    assert fast_variable.packed_maximum == pytest.approx(slow_variable.packed_maximum)
    assert json.loads((tmp_path / "fast.zarr.qc.json").read_text())["valid"] is True


def test_pack_zarr_requantizes_stores_with_a_different_packing(tmp_path):
    source = tmp_path / "other-packing.zarr"
    encoding = {**packing_encoding("tas"), "scale_factor": 0.01}
    sample_dataset().to_zarr(
        source,
        mode="w",
        consolidated=False,
        zarr_format=3,
        encoding={"tas": encoding},
    )

    report = pack_zarr(source, tmp_path / "packed.zarr")

    assert report.method == "requantized"
    published = zarr.open_group(tmp_path / "packed.zarr", mode="r")["tas"]
    assert published.attrs["scale_factor"] == PACKING_SPECS["tas"].scale_factor


def test_pack_zarr_rechunk_reports_all_missing_variables(tmp_path):
    source = tmp_path / "empty-packed.zarr"
    dataset = sample_dataset()
    dataset["tas"] = dataset.tas.where(False)
    dataset.to_zarr(
        source,
        mode="w",
        consolidated=False,
        zarr_format=3,
        encoding={"tas": packing_encoding("tas")},
    )

    report = pack_zarr(source, tmp_path / "packed.zarr")

    assert report.method == "rechunked packed codes"
    assert np.isnan(report.variables[0].source_minimum)
    with xr.open_zarr(tmp_path / "packed.zarr", consolidated=False) as published:
        assert bool(published.tas.isnull().all())
