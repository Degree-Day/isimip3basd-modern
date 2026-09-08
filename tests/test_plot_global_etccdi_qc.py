from pathlib import Path

import pytest

from scripts.plot_global_etccdi_qc import _variable_store


@pytest.mark.parametrize(
    "relative_path",
    ("tas.zarr", "tas_downscaled.zarr", "global/tas.zarr", "global/tas_downscaled.zarr"),
)
def test_variable_store_supports_flat_and_global_layouts(
    tmp_path: Path, relative_path: str
) -> None:
    store = tmp_path / relative_path
    store.mkdir(parents=True)

    assert _variable_store(tmp_path, "tas") == store
