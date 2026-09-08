#!/usr/bin/env python3
"""Merge staged CMIP projection Zarr stores into one continuous period."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import cftime
import numpy as np
import zarr


SEGMENTS = ("ref", "gap", "proj", "tail")
VARIABLES = ("tas", "hurs", "pr", "sfcWind")


def _open_group(path: Path, mode: str) -> zarr.Group:
    return zarr.open_group(path, mode=mode, use_consolidated=False)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("root", type=Path)
    result.add_argument("--model", action="append", dest="models")
    result.add_argument("--workers", type=int, default=4)
    result.add_argument("--remove-segments", action="store_true")
    result.add_argument("--overwrite", action="store_true")
    return result


def _dates(group: zarr.Group) -> list[cftime.datetime]:
    time = group["time"]
    return list(
        cftime.num2date(
            time[:],
            units=time.attrs["units"],
            calendar=time.attrs.get("calendar", "standard"),
            only_use_cftime_datetimes=True,
        )
    )


def _same_array(left: zarr.Array, right: zarr.Array) -> bool:
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and dict(left.attrs) == dict(right.attrs)
        and np.array_equal(left[:], right[:])
    )


def _validate_inputs(paths: list[Path], variable: str) -> tuple[list[zarr.Group], list[list[cftime.datetime]]]:
    groups = [_open_group(path, "r") for path in paths]
    dates = [_dates(group) for group in groups]
    first = groups[0]
    for path, group, segment_dates in zip(paths, groups, dates, strict=True):
        if variable not in group:
            raise ValueError(f"{path} does not contain {variable}")
        array = group[variable]
        if array.ndim != 3 or array.shape[0] != len(segment_dates):
            raise ValueError(f"unexpected {variable} dimensions in {path}: {array.shape}")
        if array.dtype != np.dtype("int16"):
            raise ValueError(f"{path} is {array.dtype}, expected packed int16")
        if path != paths[0]:
            baseline = first[variable]
            if array.shape[1:] != baseline.shape[1:] or array.chunks != baseline.chunks:
                raise ValueError(f"incompatible shape or chunks in {path}")
            for coordinate in ("lat", "lon"):
                if not _same_array(first[coordinate], group[coordinate]):
                    raise ValueError(f"{coordinate} differs in {path}")
            for attribute in ("scale_factor", "add_offset", "_FillValue"):
                if array.attrs.get(attribute) != baseline.attrs.get(attribute):
                    raise ValueError(f"{attribute} differs in {path}")
    flat_dates = [date for segment in dates for date in segment]
    if any(right <= left for left, right in zip(flat_dates, flat_dates[1:], strict=False)):
        raise ValueError("segment dates overlap or are not strictly increasing")
    for left, right in zip(flat_dates, flat_dates[1:], strict=False):
        if cftime.date2num(right, "days since 0001-01-01", right.calendar) - cftime.date2num(
            left, "days since 0001-01-01", left.calendar
        ) != 1:
            raise ValueError(f"non-daily boundary between {left} and {right}")
    return groups, dates


def merge_store(scenario: Path, variable: str, overwrite: bool) -> dict[str, object]:
    sources = [scenario / segment / f"{variable}.zarr" for segment in SEGMENTS]
    output_dir = scenario / "projection"
    output = output_dir / f"{variable}.zarr"
    partial_dir = scenario / ".projection.partial"
    partial = partial_dir / f"{variable}.zarr"
    if output.exists() and not overwrite:
        group = _open_group(output, "r")
        dates = _dates(group)
        if str(dates[0])[:10] == "2015-01-01" and str(dates[-1])[:10] in {
            "2100-12-30",
            "2100-12-31",
        }:
            return {"variable": variable, "status": "existing", "days": len(dates)}
        raise FileExistsError(f"existing output failed coverage check: {output}")
    missing = [str(path) for path in sources if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"missing source stores: {missing}")
    groups, dates = _validate_inputs(sources, variable)
    if partial.exists():
        shutil.rmtree(partial)
    partial_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(sources[0], partial)
    target = _open_group(partial, "a")
    total = sum(len(item) for item in dates)
    target_array = target[variable]
    target_array.resize((total, *target_array.shape[1:]))
    target_time = target["time"]
    target_time.resize((total,))

    offset = len(dates[0])
    for group, segment_dates in zip(groups[1:], dates[1:], strict=True):
        source = group[variable]
        step = source.chunks[0]
        for start in range(0, source.shape[0], step):
            stop = min(start + step, source.shape[0])
            target_array[offset + start : offset + stop] = source[start:stop]
        offset += len(segment_dates)

    all_dates = [date for segment in dates for date in segment]
    target_time[:] = cftime.date2num(
        all_dates,
        units=target_time.attrs["units"],
        calendar=target_time.attrs.get("calendar", "standard"),
    ).astype(target_time.dtype)
    target.attrs.update(
        merged_projection_segments=list(SEGMENTS),
        projection_period="2015-2100",
        merged_utc=datetime.now(timezone.utc).isoformat(),
    )

    zarr.consolidate_metadata(partial)
    check = _open_group(partial, "r")
    check_dates = _dates(check)
    if len(check_dates) != total or check[variable].shape[0] != total:
        raise RuntimeError(f"written length mismatch for {partial}")
    for boundary in np.cumsum([len(item) for item in dates])[:-1]:
        for index in (boundary - 1, boundary):
            source_segment = next(
                i for i, cumulative in enumerate(np.cumsum([len(item) for item in dates])) if index < cumulative
            )
            previous = sum(len(item) for item in dates[:source_segment])
            expected = groups[source_segment][variable][index - previous]
            if not np.array_equal(check[variable][index], expected):
                raise RuntimeError(f"packed-value boundary check failed at {index}")

    dtype = str(check[variable].dtype)
    output_dir.mkdir(parents=True, exist_ok=True)
    if output.exists():
        shutil.rmtree(output)
    partial.rename(output)
    return {
        "variable": variable,
        "status": "merged",
        "days": total,
        "first": str(check_dates[0]),
        "last": str(check_dates[-1]),
        "dtype": dtype,
    }


def main() -> None:
    args = parser().parse_args()
    models = args.models or sorted(path.name for path in args.root.iterdir() if path.is_dir())
    scenarios = [
        path
        for model in models
        for path in sorted((args.root / model).glob("ssp*"))
        if path.is_dir()
    ]
    jobs = [(scenario, variable) for scenario in scenarios for variable in VARIABLES]
    records: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(merge_store, scenario, variable, args.overwrite): (scenario, variable)
            for scenario, variable in jobs
        }
        for future in as_completed(futures):
            scenario, variable = futures[future]
            record = future.result()
            record.update(model=scenario.parent.name, scenario=scenario.name)
            records.append(record)
            print(f"{scenario.parent.name}/{scenario.name}/{variable}: {record['status']}", flush=True)

    for scenario in scenarios:
        scenario_records = [
            record
            for record in records
            if record["model"] == scenario.parent.name and record["scenario"] == scenario.name
        ]
        if len(scenario_records) != len(VARIABLES):
            raise RuntimeError(f"incomplete merge for {scenario}")
        manifest = scenario / "projection" / "merge-manifest.json"
        manifest.write_text(json.dumps(sorted(scenario_records, key=lambda item: item["variable"]), indent=2) + "\n")
        partial_dir = scenario / ".projection.partial"
        if partial_dir.exists() and not any(partial_dir.iterdir()):
            partial_dir.rmdir()
        if args.remove_segments:
            for segment in SEGMENTS:
                shutil.rmtree(scenario / segment, ignore_errors=True)


if __name__ == "__main__":
    main()
