#!/usr/bin/env python3
"""Preprocess a tree of one-variable CMIP Zarr stores to a canonical grid."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

from distributed import Client, LocalCluster

from isimip3basd_modern.io import open_dataset, write_zarr
from isimip3basd_modern.preprocessing import (
    TARGET_UNITS,
    preprocess_variable,
    validate_preprocessed,
)


LEGACY_PROJECTION_STAGES = ("ref", "gap", "proj", "tail")


def discover_two_period_stores(
    source: Path,
    models: list[str] | None = None,
    required_variables: list[str] | None = None,
) -> list[Path]:
    selected = set(
        models
        or (
            path.name
            for path in source.iterdir()
            if path.is_dir() and (path / "historical" / "hist").is_dir()
        )
    )
    stores: list[Path] = []
    for model in sorted(selected):
        model_root = source / model
        if not model_root.is_dir():
            raise FileNotFoundError(f"selected model directory is missing: {model_root}")
        historical = sorted(
            path
            for path in (model_root / "historical" / "hist").glob("*.zarr")
            if path.stem in TARGET_UNITS
        )
        if not historical:
            raise ValueError(f"no supported historical stores found for {model}")
        required = set(required_variables or ())
        missing = required - {path.stem for path in historical}
        if missing:
            raise ValueError(
                f"{model}/historical/hist is missing required variables: "
                f"{sorted(missing)}"
            )
        stores.extend(historical)
        for scenario in sorted(model_root.glob("ssp*")):
            legacy = [stage for stage in LEGACY_PROJECTION_STAGES if (scenario / stage).exists()]
            if legacy:
                raise ValueError(
                    f"{scenario} still contains legacy projection stages: {legacy}; "
                    "run merge_projection_segments.py first"
                )
            projection = sorted(
                path
                for path in (scenario / "projection").glob("*.zarr")
                if path.stem in TARGET_UNITS
            )
            if not projection:
                raise ValueError(f"no supported projection stores found for {scenario}")
            missing = required - {path.stem for path in projection}
            if missing:
                raise ValueError(
                    f"{scenario}/projection is missing required variables: "
                    f"{sorted(missing)}"
                )
            stores.extend(projection)
    return stores


def expected_period(relative: Path) -> tuple[str, str]:
    _, experiment, phase, _ = relative.parts
    if experiment == "historical" and phase == "hist":
        return "1989-01-01", "2014-12-31"
    if experiment.startswith("ssp") and phase == "projection":
        return "2015-01-01", "2100-12-31"
    raise ValueError(f"unsupported two-period path: {relative}")


def model_license_attrs(source_root: Path, model: str) -> dict[str, str]:
    path = source_root / model / "LICENSE.json"
    if not path.exists():
        return {}
    record = json.loads(path.read_text())
    license_info = record["license"]
    return {
        "license": license_info["license"],
        "license_id": license_info["id"],
        "license_url": license_info["url"],
        "license_history": license_info["history"],
        "license_source": record["authoritative_registry_url"],
        "license_retrieved_utc": record["retrieved_utc"],
        "institution_id": ",".join(record["institution_id"]),
        "cmip6_terms_of_use": record["cmip6_terms_of_use_url"],
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("source", type=Path)
    result.add_argument("output", type=Path)
    result.add_argument("--workers", type=int, default=12)
    result.add_argument("--threads-per-worker", type=int, default=1)
    result.add_argument("--memory-limit", default="12GB")
    result.add_argument("--spatial-chunk", type=int, default=20)
    result.add_argument("--model", action="append", dest="models")
    result.add_argument("--require-variable", action="append", dest="required_variables")
    result.add_argument("--overwrite", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    stores = discover_two_period_stores(
        args.source, args.models, args.required_variables
    )
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
            "period_layout": {
                "historical/hist": "1989-2014",
                "ssp*/projection": "2015-2100",
            },
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
                    license_attrs = model_license_attrs(args.source, model)
                    result.attrs.update(license_attrs)
                    prepared = result.to_dataset()
                    prepared.attrs.update(
                        model_id=model,
                        experiment_id=experiment,
                        processing_phase=phase,
                        canonical_grid="global_1_degree_cell_centers",
                        **license_attrs,
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
                    expected_start, expected_end = expected_period(relative)
                    actual_start = str(written.time.values[0])[:10]
                    actual_end = str(written.time.values[-1])[:10]
                    if (actual_start, actual_end) != (expected_start, expected_end):
                        period_error = (
                            f"output period is {actual_start}..{actual_end}, expected "
                            f"{expected_start}..{expected_end}"
                        )
                        report = replace(
                            report,
                            valid=False,
                            errors=(*report.errors, period_error),
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
