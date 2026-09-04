#!/usr/bin/env python3
"""Append a packed downscaled time segment without decoding its values."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cftime
import numpy as np
import zarr


VARIABLES = ("tas", "hurs", "pr", "sfcWind")
PACKING_KEYS = ("scale_factor", "add_offset", "_FillValue")


def store_path(root: Path, variable: str) -> Path:
    return root / f"{variable}_downscaled.zarr"


def decode_time(array: zarr.Array) -> np.ndarray:
    attrs = dict(array.attrs)
    return np.asarray(
        cftime.num2date(
            np.asarray(array[:]),
            attrs["units"],
            calendar=attrs["calendar"],
            only_use_cftime_datetimes=True,
        )
    )


def validate_pair(
    target: zarr.Group, source: zarr.Group, variable: str
) -> tuple[int, int, np.ndarray]:
    target_data = target[variable]
    source_data = source[variable]
    if target_data.dtype != np.dtype("int16") or source_data.dtype != np.dtype("int16"):
        raise ValueError(f"{variable}: both inputs must be physically stored as int16")
    if target_data.shape[1:] != source_data.shape[1:]:
        raise ValueError(f"{variable}: spatial shapes differ")
    for coordinate in ("lat", "lon"):
        if not np.array_equal(target[coordinate][:], source[coordinate][:]):
            raise ValueError(f"{variable}: {coordinate} coordinates differ")
    for key in PACKING_KEYS:
        if target_data.attrs.get(key) != source_data.attrs.get(key):
            raise ValueError(f"{variable}: packing attribute {key!r} differs")

    target_dates = decode_time(target["time"])
    source_dates = decode_time(source["time"])
    source_size = source_dates.size
    already_extended = target_dates[-1] == source_dates[-1]
    if already_extended:
        matches = np.flatnonzero(target_dates == source_dates[0])
        if matches.size != 1:
            raise ValueError(f"{variable}: cannot locate appended segment boundary")
        old_size = int(matches[0])
        if not np.array_equal(target_dates[old_size:], source_dates):
            raise ValueError(f"{variable}: existing appended time axis differs")
    else:
        old_size = target_dates.size
        if source_dates[0].year != target_dates[-1].year + 1:
            raise ValueError(
                f"{variable}: source does not begin in the year after the target"
            )
    target_time_attrs = dict(target["time"].attrs)
    rebased = np.asarray(
        cftime.date2num(
            source_dates.tolist(),
            target_time_attrs["units"],
            calendar=target_time_attrs["calendar"],
        ),
        dtype=target["time"].dtype,
    )
    boundary_value = np.asarray(target["time"][old_size - 1]).item()
    if rebased[0] != boundary_value + 1:
        raise ValueError(f"{variable}: time segments are not daily and contiguous")
    if np.any(np.diff(rebased) != 1):
        raise ValueError(f"{variable}: source time axis is not continuous daily data")
    return old_size, source_size, rebased


def append_variable(
    target_path: Path,
    source_path: Path,
    state_root: Path,
    variable: str,
    lat_block: int,
    lon_block: int,
) -> dict[str, object]:
    target = zarr.open_group(target_path, mode="a")
    source = zarr.open_group(source_path, mode="r")
    old_size, source_size, rebased_time = validate_pair(target, source, variable)
    total_size = old_size + source_size

    target_data = target[variable]
    target_time = target["time"]
    if target_data.shape[0] == old_size:
        target_data.resize((total_size, *target_data.shape[1:]))
    elif target_data.shape[0] != total_size:
        raise ValueError(f"{variable}: unexpected target data length {target_data.shape[0]}")
    if target_time.shape[0] == old_size:
        target_time.resize((total_size,))
        target_time[old_size:] = rebased_time
    elif target_time.shape[0] != total_size:
        raise ValueError(f"{variable}: unexpected target time length {target_time.shape[0]}")
    elif not np.array_equal(target_time[old_size:], rebased_time):
        raise ValueError(f"{variable}: existing appended time coordinates differ")

    variable_state = state_root / variable
    variable_state.mkdir(parents=True, exist_ok=True)
    source_data = source[variable]
    completed = 0
    written = 0
    for y0 in range(0, target_data.shape[1], lat_block):
        y1 = min(y0 + lat_block, target_data.shape[1])
        for x0 in range(0, target_data.shape[2], lon_block):
            x1 = min(x0 + lon_block, target_data.shape[2])
            marker = variable_state / f"lat{y0:04d}-{y1:04d}_lon{x0:04d}-{x1:04d}.success"
            if marker.exists():
                completed += 1
                continue
            target_data[old_size:total_size, y0:y1, x0:x1] = source_data[
                :, y0:y1, x0:x1
            ]
            marker.touch()
            completed += 1
            written += 1
            if completed % 25 == 0:
                print(f"{variable}: {completed} append blocks complete", flush=True)

    attrs = dict(target_data.attrs)
    attrs["time_coverage_start"] = str(decode_time(target_time)[0])
    attrs["time_coverage_end"] = str(decode_time(target_time)[-1])
    attrs["appended_source_store"] = str(source_path)
    attrs["appended_utc"] = datetime.now(timezone.utc).isoformat()
    attrs["time_segment_provenance"] = [
        {"period": "1989-2014", "source": str(target_path)},
        {"period": "2015-2020", "source": str(source_path)},
    ]
    target_data.attrs.update(attrs)
    return {
        "variable": variable,
        "target": str(target_path),
        "source": str(source_path),
        "old_days": old_size,
        "appended_days": source_size,
        "total_days": total_size,
        "blocks_written_this_run": written,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--lat-block", type=int, default=50)
    parser.add_argument("--lon-block", type=int, default=100)
    parser.add_argument("--variables", nargs="+", choices=VARIABLES, default=VARIABLES)
    args = parser.parse_args()

    records = []
    for variable in args.variables:
        records.append(
            append_variable(
                store_path(args.target_root, variable),
                store_path(args.source_root, variable),
                args.state_root,
                variable,
                args.lat_block,
                args.lon_block,
            )
        )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "raw packed-int16 time append",
        "records": records,
    }
    manifest_path = args.state_root / "append-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(manifest_path, flush=True)


if __name__ == "__main__":
    main()
