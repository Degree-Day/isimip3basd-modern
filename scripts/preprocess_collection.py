#!/usr/bin/env python3
"""Preprocess a tree of one-variable CMIP Zarr stores to a canonical grid."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

from distributed import Client, LocalCluster

from isimip3basd_modern.io import open_dataset, write_zarr
from isimip3basd_modern.preprocessing import (
    preprocess_variable,
    validate_preprocessed,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("source", type=Path)
    result.add_argument("output", type=Path)
    result.add_argument("--workers", type=int, default=12)
    result.add_argument("--threads-per-worker", type=int, default=1)
    result.add_argument("--memory-limit", default="12GB")
    result.add_argument("--spatial-chunk", type=int, default=20)
    result.add_argument("--model", action="append", dest="models")
    result.add_argument("--overwrite", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    stores = sorted(args.source.glob("*/*/*/*.zarr"))
    if args.models:
        selected = set(args.models)
        stores = [path for path in stores if path.relative_to(args.source).parts[0] in selected]
    if not stores:
        raise SystemExit("no input Zarr stores found")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "preprocessing-manifest.json"
    records: list[dict[str, object]] = []
    started_all = time.perf_counter()

    def write_manifest() -> None:
        manifest = {
            "source_root": str(args.source),
            "output_root": str(args.output),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "total_discovered": len(stores),
            "processed_records": len(records),
            "valid_records": sum(bool(item.get("valid")) for item in records),
            "elapsed_seconds": time.perf_counter() - started_all,
            "records": records,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    cluster = LocalCluster(
        n_workers=args.workers,
        threads_per_worker=args.threads_per_worker,
        processes=True,
        memory_limit=args.memory_limit,
    )
    with Client(cluster):
        for index, source in enumerate(stores, start=1):
            relative = source.relative_to(args.source)
            output = args.output / relative
            qc_path = output.with_suffix(".zarr.qc.json")
            if output.exists() and qc_path.exists() and not args.overwrite:
                existing = json.loads(qc_path.read_text())
                if existing.get("valid"):
                    records.append(existing)
                    print(f"[{index}/{len(stores)}] SKIP {relative}", flush=True)
                    write_manifest()
                    continue
            partial = output.with_name(f"{output.name}.partial")
            if partial.exists():
                shutil.rmtree(partial)
            if output.exists():
                if not args.overwrite:
                    raise FileExistsError(f"output exists without passing QC: {output}")
                shutil.rmtree(output)
            output.parent.mkdir(parents=True, exist_ok=True)
            variable = source.stem
            print(f"[{index}/{len(stores)}] START {relative}", flush=True)
            started = time.perf_counter()
            try:
                with open_dataset(source, {"time": 365}) as dataset:
                    if variable not in dataset:
                        raise KeyError(f"{variable!r} is not present in {source}")
                    result, diagnostics = preprocess_variable(
                        dataset[variable],
                        variable,
                        source_path=str(source),
                        input_units_override="mm d-1" if variable == "pr" else None,
                        spatial_chunk=args.spatial_chunk,
                    )
                    model, experiment, phase, _ = relative.parts
                    prepared = result.to_dataset()
                    prepared.attrs.update(
                        model_id=model,
                        experiment_id=experiment,
                        processing_phase=phase,
                        canonical_grid="global_1_degree_cell_centers",
                    )
                    write_zarr(prepared, partial, zarr_format=3)
                with open_dataset(partial, {"time": 365}) as written:
                    report = validate_preprocessed(
                        written[variable],
                        variable,
                        diagnostics,
                        source=str(source),
                        output=str(output),
                    )
                record = report.to_dict()
                record["elapsed_seconds"] = time.perf_counter() - started
                if not report.valid:
                    raise RuntimeError(f"semantic QC failed: {report.errors}")
                partial.rename(output)
                qc_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
                records.append(record)
                print(
                    f"[{index}/{len(stores)}] DONE {relative} "
                    f"{record['elapsed_seconds']:.1f}s",
                    flush=True,
                )
            except Exception as error:
                failure = {
                    "source": str(source),
                    "output": str(output),
                    "variable": variable,
                    "valid": False,
                    "error": f"{type(error).__name__}: {error}",
                    "elapsed_seconds": time.perf_counter() - started,
                }
                records.append(failure)
                qc_path.parent.mkdir(parents=True, exist_ok=True)
                qc_path.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
                print(f"[{index}/{len(stores)}] FAIL {relative}: {error}", flush=True)
                break
            finally:
                write_manifest()
    cluster.close()
    if len(records) != len(stores) or not all(item.get("valid") for item in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
