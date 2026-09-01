#!/usr/bin/env python3
"""Prepare Europe-only nested ERA5-Land references for 1 to 0.1 degree MBCnSD."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

from isimip3basd_modern.io import open_dataset
from isimip3basd_modern.validation import validate_variable

from prepare_era5land_reference import (
    METADATA,
    initialize_store,
    nested_grids,
    process_tile,
)


REGIONS = {
    "west": {
        "lat": (920, 1290),
        "lon": (3490, 3600),
        "description": "35.05-71.95N, 10.95W-0.05W",
    },
    "east": {
        "lat": (920, 1290),
        "lon": (0, 320),
        "description": "35.05-71.95N, 0.05-31.95E",
    },
}


def _tiles(region: str) -> list[tuple[int, int, int, int]]:
    bounds = REGIONS[region]
    lat_start, lat_stop = bounds["lat"]
    lon_start, lon_stop = bounds["lon"]
    return [
        (lat, min(lat + 50, lat_stop), lon, min(lon + 50, lon_stop))
        for lat in range(lat_start, lat_stop, 50)
        for lon in range(lon_start, lon_stop, 50)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--regions", nargs="+", choices=REGIONS, default=list(REGIONS))
    parser.add_argument("--variables", nargs="+", choices=METADATA, default=list(METADATA))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    fine_grid, coarse_grid = nested_grids()
    args.output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for variable in args.variables:
        fine_path = args.output / "fine" / f"{variable}.zarr"
        coarse_path = args.output / "coarse" / f"{variable}.zarr"
        initialize_store(fine_path, variable, fine_grid)
        initialize_store(coarse_path, variable, coarse_grid)

    for variable in args.variables:
        for region in args.regions:
            started = time.perf_counter()
            fine_path = args.output / "fine" / f"{variable}.zarr"
            coarse_path = args.output / "coarse" / f"{variable}.zarr"
            tiles = _tiles(region)
            state = args.output / "state" / variable
            pending = [
                tile
                for tile in tiles
                if not (state / f"{tile[0]:04d}-{tile[2]:04d}").exists()
            ]
            print(
                f"START {variable} {region}: {len(pending)}/{len(tiles)} tiles",
                flush=True,
            )
            if args.workers == 1:
                for index, tile in enumerate(pending, start=1):
                    marker = process_tile(
                        str(args.source),
                        str(args.output),
                        variable,
                        *tile,
                    )
                    if index % 10 == 0 or index == len(pending):
                        print(
                            f"{variable} {region} {index}/{len(pending)} tiles; "
                            f"last {marker}",
                            flush=True,
                        )
            else:
                with ProcessPoolExecutor(max_workers=args.workers) as executor:
                    futures = {
                        executor.submit(
                            process_tile,
                            str(args.source),
                            str(args.output),
                            variable,
                            *tile,
                        ): tile
                        for tile in pending
                    }
                    for index, future in enumerate(as_completed(futures), start=1):
                        marker = future.result()
                        if index % 10 == 0 or index == len(pending):
                            print(
                                f"{variable} {region} {index}/{len(pending)} tiles; "
                                f"last {marker}",
                                flush=True,
                            )

            fine_slice = {
                "lat": slice(*REGIONS[region]["lat"]),
                "lon": slice(*REGIONS[region]["lon"]),
            }
            coarse_slice = {
                "lat": slice(
                    REGIONS[region]["lat"][0] // 10,
                    REGIONS[region]["lat"][1] // 10,
                ),
                "lon": slice(
                    REGIONS[region]["lon"][0] // 10,
                    REGIONS[region]["lon"][1] // 10,
                ),
            }
            reports = {}
            for label, path, region_slice in (
                ("fine", fine_path, fine_slice),
                ("coarse", coarse_path, coarse_slice),
            ):
                with open_dataset(path, {"time": 365}) as written:
                    report = validate_variable(
                        written[variable].isel(region_slice),
                        variable,
                        statistical=False,
                    )
                if not report.valid:
                    raise RuntimeError(
                        f"{label} {variable} {region} QC failed: {report.errors}"
                    )
                reports[label] = report.to_dict()
            record = {
                "variable": variable,
                "region": region,
                "description": REGIONS[region]["description"],
                "valid": True,
                "elapsed_seconds": time.perf_counter() - started,
                "qc": reports,
            }
            records.append(record)
            qc_path = args.output / f"{variable}.{region}.qc.json"
            qc_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
            print(
                f"DONE {variable} {region} {record['elapsed_seconds']:.1f}s",
                flush=True,
            )

    manifest = {
        "source": str(args.source),
        "output": str(args.output),
        "regions": {key: REGIONS[key] for key in args.regions},
        "records": records,
    }
    (args.output / "reference-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
