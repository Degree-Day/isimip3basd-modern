#!/usr/bin/env python3
"""Publish finalized global float32 stores as restartable scaled-int16 Zarr."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from distributed import Client, LocalCluster

from isimip3basd_modern.io import parse_chunks
from isimip3basd_modern.publication import PACKING_SPECS, pack_zarr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--region", default="global")
    parser.add_argument("--variables", nargs="+", choices=tuple(PACKING_SPECS))
    parser.add_argument("--chunks", default="time=31,lat=256,lon=256")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument("--memory-limit", default="16GB")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_region = args.source_root / args.region
    output_region = args.output_root / args.region
    variables = args.variables or sorted(
        path.name.removesuffix("_downscaled.zarr")
        for path in source_region.glob("*_downscaled.zarr")
    )
    if not variables:
        raise SystemExit(f"no downscaled stores found under {source_region}")
    output_region.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []

    with LocalCluster(
        n_workers=args.workers,
        threads_per_worker=args.threads_per_worker,
        processes=True,
        memory_limit=args.memory_limit,
    ) as cluster:
        with Client(cluster):
            for index, variable in enumerate(variables, start=1):
                source = source_region / f"{variable}_downscaled.zarr"
                output = output_region / f"{variable}.zarr"
                qc_path = Path(f"{output}.qc.json")
                if output.exists() and qc_path.exists() and not args.overwrite:
                    existing = json.loads(qc_path.read_text())
                    if existing.get("valid"):
                        records.append(existing)
                        print(f"[{index}/{len(variables)}] SKIP {variable}", flush=True)
                        continue
                if not source.exists():
                    raise FileNotFoundError(source)
                print(f"[{index}/{len(variables)}] PACK {variable}", flush=True)
                report = pack_zarr(
                    source,
                    output,
                    variables=[variable],
                    chunks=parse_chunks(args.chunks),
                    overwrite=args.overwrite,
                )
                records.append(report.to_dict())
                print(
                    f"[{index}/{len(variables)}] DONE {variable}: "
                    f"{report.storage_bytes / 2**30:.2f} GiB",
                    flush=True,
                )

    manifest = {
        "source_root": str(args.source_root),
        "output_root": str(args.output_root),
        "region": args.region,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "chunks": parse_chunks(args.chunks),
        "variables": variables,
        "valid": len(records) == len(variables) and all(
            record.get("valid") for record in records
        ),
        "records": records,
    }
    manifest_path = args.output_root / "publication-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
