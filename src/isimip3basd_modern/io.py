"""Input, chunking, and Zarr output helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import xarray as xr


def parse_chunks(value: str | None) -> dict[str, int]:
    """Parse ``dimension=size`` pairs accepted by the CLI."""
    if not value:
        return {}

    chunks: dict[str, int] = {}
    for item in value.split(","):
        try:
            dimension, size = item.split("=", maxsplit=1)
            dimension = dimension.strip()
            size = int(size)
        except ValueError as error:
            raise ValueError(
                f"invalid chunk specification {item!r}; expected dimension=size"
            ) from error
        if not dimension or size == 0 or size < -1:
            raise ValueError(
                f"invalid chunk specification {item!r}; size must be -1 or positive"
            )
        chunks[dimension] = size
    return chunks


def open_dataset(
    path: str | Path, chunks: Mapping[str, int] | None = None
) -> xr.Dataset:
    """Open a NetCDF file or Zarr store lazily."""
    path = Path(path)
    requested_chunks = dict(chunks or {}) or "auto"
    if path.suffix == ".zarr" or path.is_dir():
        consolidated = False if (path / "zarr.json").exists() else None
        return xr.open_zarr(
            path,
            chunks=requested_chunks,
            consolidated=consolidated,
        )
    return xr.open_dataset(path, chunks=requested_chunks, decode_cf=True)


def write_zarr(
    dataset: xr.Dataset,
    path: str | Path,
    *,
    zarr_format: int = 3,
    overwrite: bool = False,
) -> None:
    """Write a dataset to a consolidated Zarr store."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {path}")
    dataset.to_zarr(
        path,
        mode="w",
        consolidated=zarr_format == 2,
        zarr_format=zarr_format,
    )
