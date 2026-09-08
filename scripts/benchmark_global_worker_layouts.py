#!/usr/bin/env python3
"""Benchmark global MBCnSD worker layouts on identical completed tiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np


def completed_tiles(state_root: Path, count: int) -> list[str]:
    records = []
    for path in sorted(state_root.glob("*.report.json")):
        record = json.loads(path.read_text())
        if record.get("valid") and int(record.get("active_cells", 0)) > 0:
            records.append(record)
    if len(records) < count:
        raise RuntimeError(f"only {len(records)} completed active tiles are available")
    indices = np.linspace(0, len(records) - 1, count, dtype=int)
    return [str(records[index]["tile"]) for index in indices]


def process_metrics(pid: int) -> tuple[float, float]:
    command = [
        "ps",
        "--ppid",
        str(pid),
        "-o",
        "rss=,pcpu=",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    rss_kib = 0.0
    cpu = 0.0
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) == 2:
            rss_kib += float(fields[0])
            cpu += float(fields[1])
    return rss_kib / 1024**2, cpu


def data_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if not item.is_file() or item.name == "zarr.json":
            continue
        digest.update(str(item.relative_to(path)).encode())
        with item.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--adjusted-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", default="ACCESS-CM2")
    parser.add_argument("--scenario", default="ssp245")
    parser.add_argument("--simulation-stage", default="projection")
    parser.add_argument("--simulation-start", default="2015")
    parser.add_argument("--simulation-end", default="2020")
    parser.add_argument("--variable", default="sfcWind")
    parser.add_argument("--tile-count", type=int, default=24)
    parser.add_argument(
        "--layouts", nargs="+", default=("12x3", "16x1", "24x1")
    )
    args = parser.parse_args()

    production_global = args.production_root / "global"
    state_root = (
        production_global
        / "state_spatial_global_context"
        / args.variable
    )
    tiles = completed_tiles(state_root, args.tile_count)
    mask = production_global / "spatial_valid_mask.zarr"
    results = []
    for layout in args.layouts:
        workers, threads = (int(value) for value in layout.split("x", 1))
        output = args.output_root / layout
        if output.exists():
            shutil.rmtree(output)
        log_path = args.output_root / f"{layout}.log"
        args.output_root.mkdir(parents=True, exist_ok=True)
        command = [
            str(Path(os.environ.get("DOWNSCALE_PYTHON", os.sys.executable))),
            str(args.repo / "scripts" / "run_global_downscale_tiles.py"),
            "--model", args.model,
            "--scenario", args.scenario,
            "--simulation-stage", args.simulation_stage,
            "--simulation-start", args.simulation_start,
            "--simulation-end", args.simulation_end,
            "--regions", "global",
            "--variables", args.variable,
            "--reference-root", str(args.reference_root),
            "--canonical-root", str(args.canonical_root),
            "--adjusted-root", str(args.adjusted_root),
            "--output-root", str(output),
            "--stages", "spatial",
            "--tile-lat-degrees", "5",
            "--tile-lon-degrees", "10",
            "--tile-workers", str(workers),
            "--threads-per-worker", str(threads),
            "--spatial-valid-mask-store", str(mask),
            "--spatial-tile-names", *tiles,
        ]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(args.repo / "src")
        started = time.perf_counter()
        peak_rss_gib = 0.0
        peak_cpu_percent = 0.0
        with log_path.open("w") as log:
            process = subprocess.Popen(
                command,
                cwd=args.repo,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            while process.poll() is None:
                rss_gib, cpu_percent = process_metrics(process.pid)
                peak_rss_gib = max(peak_rss_gib, rss_gib)
                peak_cpu_percent = max(peak_cpu_percent, cpu_percent)
                time.sleep(2)
        elapsed = time.perf_counter() - started
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command)
        reports = [
            json.loads(path.read_text())
            for path in (
                output
                / "global"
                / "state_spatial_global_context"
                / args.variable
            ).glob("*.report.json")
        ]
        results.append(
            {
                "layout": layout,
                "workers": workers,
                "threads_per_worker": threads,
                "tiles": len(reports),
                "wall_seconds": elapsed,
                "tiles_per_hour": len(reports) / elapsed * 3600,
                "median_tile_seconds": float(
                    np.median([record["elapsed_seconds"] for record in reports])
                ),
                "peak_worker_rss_gib": peak_rss_gib,
                "peak_worker_cpu_percent": peak_cpu_percent,
                "packed_data_sha256": data_digest(
                    output / "global" / f"{args.variable}_downscaled.zarr"
                ),
                "log": str(log_path),
            }
        )
        print(json.dumps(results[-1], indent=2), flush=True)

    hashes = {result["packed_data_sha256"] for result in results}
    summary = {
        "model": args.model,
        "period": f"{args.simulation_start}-{args.simulation_end}",
        "variable": args.variable,
        "tile_names": tiles,
        "identical_packed_outputs": len(hashes) == 1,
        "results": results,
    }
    summary_path = args.output_root / "worker-layout-benchmark.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(summary_path)


if __name__ == "__main__":
    main()
