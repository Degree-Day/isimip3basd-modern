#!/usr/bin/env python3
"""Safely migrate one packed FWI array to the current packing specification."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import zarr
from zarr.codecs import BloscCodec

from isimip3basd_modern.publication import (
    PACKED_FILL_VALUE,
    PACKED_MAX_CODE,
    PACKED_MIN_CODE,
    PACKING_SPECS,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("store", type=Path)
    parser.add_argument("variable", choices=tuple(PACKING_SPECS))
    parser.add_argument("--keep-backup", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    group = zarr.open_group(str(args.store), mode="r")
    source = group[args.variable]
    if source.dtype != np.dtype("int16"):
        parser.error("source array is not physically int16")
    old_scale = float(source.attrs["scale_factor"])
    old_offset = float(source.attrs["add_offset"])
    old_fill = np.int16(source.attrs["_FillValue"])
    target = PACKING_SPECS[args.variable]
    if old_scale == target.scale_factor and old_offset == target.add_offset:
        print("packing already matches the current specification")
        return

    partial = args.store.with_name(f"{args.store.name}.{args.variable}.repack.partial")
    backup = args.store.with_name(f"{args.store.name}.{args.variable}.packing-backup")
    shutil.rmtree(partial, ignore_errors=True)
    if backup.exists():
        parser.error(f"backup already exists: {backup}")

    attrs = dict(source.attrs)
    attrs.update(
        packing_migration_utc=datetime.now(timezone.utc).isoformat(),
        previous_scale_factor=old_scale,
        previous_add_offset=old_offset,
        scale_factor=target.scale_factor,
        add_offset=target.add_offset,
        _FillValue=int(PACKED_FILL_VALUE),
    )
    partial_group = zarr.open_group(str(partial), mode="w")
    destination = partial_group.create_array(
        args.variable,
        shape=source.shape,
        chunks=source.chunks,
        dtype="int16",
        fill_value=PACKED_FILL_VALUE,
        compressors=[BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")],
        dimension_names=source.metadata.dimension_names,
        attributes=attrs,
    )

    chunk_root = args.store / args.variable / "c"
    chunk_indices = [
        tuple(int(part) for part in path.relative_to(chunk_root).parts)
        for path in chunk_root.rglob("*")
        if path.is_file()
    ]
    chunks_read = len(chunk_indices)
    chunks_written = 0
    valid_values = 0
    decoded_minimum = np.inf
    decoded_maximum = -np.inf
    def migrate_chunk(index: tuple[int, int, int]) -> tuple[int, float, float]:
        region = tuple(
            slice(
                chunk_index * chunk_size,
                min((chunk_index + 1) * chunk_size, dimension_size),
            )
            for chunk_index, chunk_size, dimension_size in zip(
                index, source.chunks, source.shape, strict=True
            )
        )
        raw = np.asarray(source[region])
        valid = raw != old_fill
        if not valid.any():
            return 0, np.inf, -np.inf
        decoded = raw[valid].astype("float64") * old_scale + old_offset
        codes = np.rint((decoded - target.add_offset) / target.scale_factor)
        if (codes < PACKED_MIN_CODE).any() or (codes > PACKED_MAX_CODE).any():
            raise ValueError(
                f"decoded range {decoded.min()}..{decoded.max()} exceeds target packing"
            )
        migrated = np.full(raw.shape, PACKED_FILL_VALUE, dtype="int16")
        migrated[valid] = codes.astype("int16")
        destination[region] = migrated
        return int(valid.sum()), float(decoded.min()), float(decoded.max())

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for batch_start in range(0, chunks_read, 500):
            batch = chunk_indices[batch_start : batch_start + 500]
            futures = [executor.submit(migrate_chunk, index) for index in batch]
            for future in as_completed(futures):
                count, minimum, maximum = future.result()
                if count:
                    chunks_written += 1
                    valid_values += count
                    decoded_minimum = min(decoded_minimum, minimum)
                    decoded_maximum = max(decoded_maximum, maximum)
            completed = min(batch_start + len(batch), chunks_read)
            print(
                f"migrated {completed}/{chunks_read} physical chunks",
                flush=True,
            )

    source_dir = args.store / args.variable
    migrated_dir = partial / args.variable
    source_dir.rename(backup)
    migrated_dir.rename(source_dir)
    shutil.rmtree(partial)

    reopened_group = zarr.open_group(str(args.store), mode="r")
    reopened = reopened_group[args.variable]
    valid = (
        reopened.dtype == np.dtype("int16")
        and float(reopened.attrs["scale_factor"]) == target.scale_factor
        and float(reopened.attrs["add_offset"]) == target.add_offset
    )
    report = {
        "store": str(args.store),
        "variable": args.variable,
        "old_scale_factor": old_scale,
        "old_add_offset": old_offset,
        "new_scale_factor": target.scale_factor,
        "new_add_offset": target.add_offset,
        "decoded_minimum": decoded_minimum,
        "decoded_maximum": decoded_maximum,
        "chunks_read": chunks_read,
        "chunks_written": chunks_written,
        "valid_values": valid_values,
        "backup": str(backup) if args.keep_backup else None,
        "valid": valid,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    report_path = args.store.with_name(f"{args.store.name}.{args.variable}.repack.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if not valid:
        raise RuntimeError("repacked array failed metadata verification")
    if not args.keep_backup:
        shutil.rmtree(backup)
    print(report_path)


if __name__ == "__main__":
    main()
