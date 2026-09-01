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
            dim: min(size, cleaned.sizes[dim])
            for dim, size in requested_chunks.items()
            if dim in cleaned.dims
        }
        cleaned = cleaned.chunk(effective_chunks)
        reductions = []
        for name in selected:
            reductions.extend(
                (
                    cleaned[name].min(skipna=True),
                    cleaned[name].max(skipna=True),
                    np.isinf(cleaned[name]).any(),
                )
            )
        values = dask.compute(*reductions)

        source_ranges: dict[str, tuple[float, float]] = {}
        for index, name in enumerate(selected):
            minimum, maximum, has_inf = values[index * 3 : index * 3 + 3]
            if bool(has_inf):
                raise ValueError(f"{name} contains infinite values")
            low, high = float(minimum), float(maximum)
            spec = PACKING_SPECS[name]
            tolerance = spec.scale_factor / 2
            if np.isfinite(low) and low < spec.minimum - tolerance:
                raise ValueError(
                    f"{name} minimum {low} is below packed range {spec.minimum}"
                )
            if np.isfinite(high) and high > spec.maximum + tolerance:
                raise ValueError(
                    f"{name} maximum {high} is above packed range {spec.maximum}"
                )
            source_ranges[name] = (low, high)

        compressor = BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")
        encoding = {
            name: {
                "dtype": "int16",
                "_FillValue": PACKED_FILL_VALUE,
                "fill_value": PACKED_FILL_VALUE,
                "scale_factor": PACKING_SPECS[name].scale_factor,
                "add_offset": PACKING_SPECS[name].add_offset,
                "compressors": [compressor],
            }
            for name in selected
        }
        cleaned.attrs.update(
            publication_format="scaled int16 Zarr v3",
            publication_compressor="Blosc Zstd level 3 with bitshuffle",
        )
        cleaned.to_zarr(
            partial,
            mode="w",
            consolidated=False,
            zarr_format=3,
            encoding=encoding,
        )

        with open_dataset(partial, requested_chunks) as decoded:
            if set(decoded.data_vars) != set(selected):
                raise RuntimeError("published variables differ from source selection")
            error_tasks = [
                xr.DataArray(
                    abs(decoded[name].data - cleaned[name].data),
                    dims=cleaned[name].dims,
                ).max(skipna=True)
                for name in selected
            ]
            packed_ranges = [
                item
                for name in selected
                for item in (
                    decoded[name].min(skipna=True),
                    decoded[name].max(skipna=True),
                )
            ]
            computed = dask.compute(*(error_tasks + packed_ranges))
            errors = computed[: len(selected)]
            ranges = computed[len(selected) :]
            reports = []
            for index, name in enumerate(selected):
                error = float(errors[index])
                spec = PACKING_SPECS[name]
                packing_magnitude = max(
                    1.0,
                    abs(source_ranges[name][0]),
                    abs(source_ranges[name][1]),
                    abs(spec.minimum),
                    abs(spec.maximum),
                )
                # Scale/offset packing is evaluated through float32 source data.
                # Include a few ULPs at the full coding range in addition to the
                # unavoidable half-step quantization error.
                allowed_error = float(
                    spec.scale_factor / 2
                    + 4 * packing_magnitude * np.finfo("float32").eps
                )
                valid = bool(error <= allowed_error)
                reports.append(
                    PackingVariableReport(
                        variable=name,
                        source_minimum=source_ranges[name][0],
                        source_maximum=source_ranges[name][1],
                        packed_minimum=float(ranges[index * 2]),
                        packed_maximum=float(ranges[index * 2 + 1]),
                        scale_factor=spec.scale_factor,
                        add_offset=spec.add_offset,
                        maximum_absolute_error=error,
                        maximum_allowed_error=allowed_error,
                        valid=valid,
                    )
                )
            coordinates_equal = all(
                decoded[dim].equals(cleaned[dim]) for dim in cleaned.dims
            )
            valid = coordinates_equal and all(item.valid for item in reports)

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
