"""Analysis-friendly, quantized Zarr publication products."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
from typing import Mapping, Sequence

import dask
import numpy as np
import xarray as xr
import zarr
from zarr.codecs import BloscCodec

from .io import open_dataset


PACKED_FILL_VALUE = np.int16(-32768)
PACKED_MIN_CODE = -32767
PACKED_MAX_CODE = 32767


@dataclass(frozen=True)
class PackingSpec:
    """Linear int16 packing with one reserved missing-value code."""

    scale_factor: float
    add_offset: float

    @property
    def minimum(self) -> float:
        return self.add_offset + PACKED_MIN_CODE * self.scale_factor

    @property
    def maximum(self) -> float:
        return self.add_offset + PACKED_MAX_CODE * self.scale_factor


def _centered(lower: float, upper: float, resolution: float) -> PackingSpec:
    midpoint = (lower + upper) / 2
    if lower < midpoint + PACKED_MIN_CODE * resolution:
        raise ValueError("packing resolution does not cover the requested lower bound")
    if upper > midpoint + PACKED_MAX_CODE * resolution:
        raise ValueError("packing resolution does not cover the requested upper bound")
    return PackingSpec(resolution, midpoint)


PACKING_SPECS: dict[str, PackingSpec] = {
    # Primary and derived climate variables in canonical output units.
    "hurs": _centered(0, 100, 0.002),
    "pr": _centered(0, 0.04, 1e-6),  # kg m-2 s-1; covers 3456 mm/day.
    "prsn": _centered(0, 0.04, 1e-6),
    "prsnratio": _centered(0, 1, 2e-5),
    "ps": _centered(0, 120_000, 2.0),
    "rlds": _centered(0, 1_000, 0.02),
    "rsds": _centered(0, 1_500, 0.025),
    "sfcWind": _centered(0, 100, 0.002),
    "tas": _centered(130, 377, 0.005),
    "tasmin": _centered(130, 377, 0.005),
    "tasmax": _centered(130, 377, 0.005),
    "tasrange": _centered(0, 100, 0.002),
    "tasskew": _centered(0, 1, 2e-5),
    "huss": _centered(0, 0.1, 2e-6),
    # Canadian Forest Fire Weather Index System outputs.
    "ffmc": _centered(0, 101, 0.002),
    "dmc": _centered(0, 10_000, 0.2),
    "dc": _centered(0, 10_000, 0.2),
    "isi": _centered(0, 2_000, 0.04),
    "bui": _centered(0, 10_000, 0.2),
    "fwi": _centered(0, 2_000, 0.04),
}


def packing_encoding(variable: str) -> dict[str, object]:
    """Return the physical scaled-int16 Zarr encoding for a variable."""
    try:
        spec = PACKING_SPECS[variable]
    except KeyError as error:
        raise ValueError(f"no int16 packing specification for {variable!r}") from error
    return {
        "dtype": "int16",
        "_FillValue": PACKED_FILL_VALUE,
        "fill_value": PACKED_FILL_VALUE,
        "scale_factor": spec.scale_factor,
        "add_offset": spec.add_offset,
        "compressors": [
            BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")
        ],
    }


@dataclass(frozen=True)
class PackingVariableReport:
    variable: str
    source_minimum: float
    source_maximum: float
    packed_minimum: float
    packed_maximum: float
    scale_factor: float
    add_offset: float
    maximum_absolute_error: float
    maximum_allowed_error: float
    valid: bool


@dataclass(frozen=True)
class PackingReport:
    source: str
    output: str
    valid: bool
    chunks: dict[str, int]
    storage_bytes: int
    variables: tuple[PackingVariableReport, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _canonical_order(data: xr.DataArray) -> tuple[str, ...]:
    preferred = tuple(dim for dim in ("time", "lat", "lon") if dim in data.dims)
    return preferred + tuple(dim for dim in data.dims if dim not in preferred)


def _clean_data_array(data: xr.DataArray) -> xr.DataArray:
    ordered = data.transpose(*_canonical_order(data))
    return xr.DataArray(
        ordered.data,
        dims=ordered.dims,
        coords={dim: ordered[dim] for dim in ordered.dims},
        name=ordered.name,
        attrs=ordered.attrs,
    )


def _storage_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def pack_zarr(
    source: str | Path,
    output: str | Path,
    *,
    variables: Sequence[str] | None = None,
    chunks: Mapping[str, int] | None = None,
    overwrite: bool = False,
) -> PackingReport:
    """Pack finalized floating-point variables to decoded-on-read int16 Zarr."""
    source = Path(source)
    output = Path(output)
    partial = output.with_name(f"{output.name}.partial")
    requested_chunks = dict(chunks or {"time": 365, "lat": 128, "lon": 128})
    if output.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output}")
    shutil.rmtree(partial, ignore_errors=True)

    with open_dataset(source, requested_chunks) as opened:
        selected = list(variables or opened.data_vars)
        if not selected:
            raise ValueError("source contains no data variables to publish")
        missing = set(selected) - set(opened.data_vars)
        if missing:
            raise KeyError(f"variables not found in source: {sorted(missing)}")
        unsupported = set(selected) - set(PACKING_SPECS)
        if unsupported:
            raise ValueError(f"no int16 packing specification for {sorted(unsupported)}")

        cleaned = xr.Dataset(
            {name: _clean_data_array(opened[name]) for name in selected},
            attrs=opened.attrs,
        )
        effective_chunks = {
            dim: cleaned.sizes[dim] if size == -1 else min(size, cleaned.sizes[dim])
            for dim, size in requested_chunks.items()
            if dim in cleaned.dims
        }
        cleaned = cleaned.chunk(effective_chunks)
        reductions = []
        for name in selected:
            spec = PACKING_SPECS[name]
            decoded_quantized = (
                np.rint((cleaned[name] - spec.add_offset) / spec.scale_factor)
                * spec.scale_factor
                + spec.add_offset
            )
            reductions.extend(
                (
                    cleaned[name].min(skipna=True),
                    cleaned[name].max(skipna=True),
                    np.isinf(cleaned[name]).any(),
                    abs(decoded_quantized - cleaned[name]).max(skipna=True),
                    decoded_quantized.min(skipna=True),
                    decoded_quantized.max(skipna=True),
                )
            )

        encoding = {name: packing_encoding(name) for name in selected}
        cleaned.attrs.update(
            publication_format="scaled int16 Zarr v3",
            publication_compressor="Blosc Zstd level 3 with bitshuffle",
        )
        write = cleaned.to_zarr(
            partial,
            mode="w",
            consolidated=False,
            zarr_format=3,
            encoding=encoding,
            compute=False,
        )
        try:
            computed = dask.compute(write, *reductions)
        except Exception:
            shutil.rmtree(partial, ignore_errors=True)
            raise
        values = computed[1:]

        source_ranges: dict[str, tuple[float, float]] = {}
        reports = []
        for index, name in enumerate(selected):
            minimum, maximum, has_inf, error, packed_low, packed_high = values[
                index * 6 : index * 6 + 6
            ]
            if bool(has_inf):
                shutil.rmtree(partial, ignore_errors=True)
                raise ValueError(f"{name} contains infinite values")
            low, high = float(minimum), float(maximum)
            spec = PACKING_SPECS[name]
            tolerance = spec.scale_factor / 2
            if np.isfinite(low) and low < spec.minimum - tolerance:
                shutil.rmtree(partial, ignore_errors=True)
                raise ValueError(
                    f"{name} minimum {low} is below packed range {spec.minimum}"
                )
            if np.isfinite(high) and high > spec.maximum + tolerance:
                shutil.rmtree(partial, ignore_errors=True)
                raise ValueError(
                    f"{name} maximum {high} is above packed range {spec.maximum}"
                )
            source_ranges[name] = (low, high)
            packing_magnitude = max(
                1.0, abs(low), abs(high), abs(spec.minimum), abs(spec.maximum)
            )
            allowed_error = float(
                spec.scale_factor / 2
                + 4 * packing_magnitude * np.finfo("float32").eps
            )
            reports.append(
                PackingVariableReport(
                    variable=name,
                    source_minimum=low,
                    source_maximum=high,
                    packed_minimum=float(packed_low),
                    packed_maximum=float(packed_high),
                    scale_factor=spec.scale_factor,
                    add_offset=spec.add_offset,
                    maximum_absolute_error=float(error),
                    maximum_allowed_error=allowed_error,
                    valid=bool(float(error) <= allowed_error),
                )
            )

        with open_dataset(partial, {}) as decoded:
            variables_equal = set(decoded.data_vars) == set(selected)
            coordinates_equal = all(
                decoded[dim].equals(cleaned[dim]) for dim in cleaned.dims
            )
        group = zarr.open_group(partial, mode="r")
        dtypes_equal = all(group[name].dtype == np.dtype("int16") for name in selected)
        valid = (
            variables_equal
            and coordinates_equal
            and dtypes_equal
            and all(item.valid for item in reports)
        )

    if not valid:
        details = "; ".join(
            f"{item.variable}: error={item.maximum_absolute_error}, "
            f"allowed={item.maximum_allowed_error}"
            for item in reports
            if not item.valid
        )
        shutil.rmtree(partial, ignore_errors=True)
        raise RuntimeError(
            "packed Zarr failed round-trip quantization QC"
            + (f" ({details})" if details else " (coordinate mismatch)")
        )
    report = PackingReport(
        source=str(source),
        output=str(output),
        valid=True,
        chunks=effective_chunks,
        storage_bytes=_storage_bytes(partial),
        variables=tuple(reports),
    )
    report_json = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    shutil.rmtree(output, ignore_errors=True)
    partial.rename(output)
    Path(f"{output}.qc.json").write_text(report_json)
    return report
