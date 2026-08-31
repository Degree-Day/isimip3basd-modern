"""Command-line interface for the modern workflow."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
import json

from distributed import Client, LocalCluster

from .io import open_dataset, parse_chunks, write_zarr
from .pipeline import adjust_variable
from .presets import VARIABLE_PRESETS
from .validation import validate_variable


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="isimip3basd-modern")
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert = subparsers.add_parser("convert", help="convert NetCDF or Zarr to Zarr")
    convert.add_argument("input")
    convert.add_argument("output")
    convert.add_argument("--chunks", default="time=-1")
    convert.add_argument("--zarr-format", type=int, choices=(2, 3), default=3)
    convert.add_argument("--overwrite", action="store_true")

    validate = subparsers.add_parser(
        "validate", help="validate a bias-adjusted NetCDF file or Zarr store"
    )
    validate.add_argument("input")
    validate.add_argument("--variable", choices=tuple(VARIABLE_PRESETS), required=True)
    validate.add_argument("--chunks", default="time=-1")

    run = subparsers.add_parser("adjust", help="bias-adjust a climate simulation")
    run.add_argument("--reference", required=True)
    run.add_argument("--historical", required=True)
    run.add_argument("--simulation", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--variable", choices=tuple(VARIABLE_PRESETS), required=True)
    run.add_argument(
        "--method",
        choices=("qdm", "dqm", "scaling"),
        default=None,
        help="override the variable preset",
    )
    run.add_argument(
        "--kind",
        choices=("additive", "multiplicative"),
        default=None,
        help="override the variable preset",
    )
    run.add_argument(
        "--group",
        default=None,
        help="xsdba grouping (default: time.dayofyear for DQM, time.month otherwise)",
    )
    run.add_argument("--window", type=int, default=None)
    run.add_argument("--quantiles", type=int, default=50)
    run.add_argument("--random-seed", type=int, default=0)
    run.add_argument(
        "--interpolation",
        choices=("nearest", "linear", "cubic"),
        default="nearest",
    )
    run.add_argument(
        "--extrapolation", choices=("constant", "nan"), default="constant"
    )
    run.add_argument("--chunks", default="time=-1")
    execution = run.add_mutually_exclusive_group()
    execution.add_argument("--workers", type=int, default=0)
    execution.add_argument(
        "--scheduler-address",
        help="connect to an existing Dask scheduler, for example tcp://host:8786",
    )
    run.add_argument("--threads-per-worker", type=int, default=1)
    run.add_argument("--memory-limit", default="auto")
    run.add_argument("--zarr-format", type=int, choices=(2, 3), default=3)
    run.add_argument("--overwrite", action="store_true")
    return parser


@contextmanager
def _cluster_context(args: argparse.Namespace) -> Iterator[None]:
    if args.scheduler_address:
        with Client(args.scheduler_address):
            yield
        return
    if not args.workers:
        yield
        return
    with LocalCluster(
        n_workers=args.workers,
        threads_per_worker=args.threads_per_worker,
        memory_limit=args.memory_limit,
        processes=True,
    ) as cluster:
        with Client(cluster):
            yield


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    chunks = parse_chunks(args.chunks)

    if args.command == "convert":
        with open_dataset(args.input, chunks) as dataset:
            write_zarr(
                dataset.chunk(chunks),
                args.output,
                zarr_format=args.zarr_format,
                overwrite=args.overwrite,
            )
        print(f"wrote {args.output}")
        return

    if args.command == "validate":
        with open_dataset(args.input, chunks) as dataset:
            if args.variable not in dataset:
                raise KeyError(f"{args.variable!r} not found in input dataset")
            report = validate_variable(dataset[args.variable], args.variable)
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        if not report.valid:
            raise SystemExit(1)
        return

    with _cluster_context(args):
        with (
            open_dataset(args.reference, chunks) as reference,
            open_dataset(args.historical, chunks) as historical,
            open_dataset(args.simulation, chunks) as simulation,
        ):
            for label, dataset in (
                ("reference", reference),
                ("historical", historical),
                ("simulation", simulation),
            ):
                if args.variable not in dataset:
                    raise KeyError(f"{args.variable!r} not found in {label} dataset")

            result = adjust_variable(
                reference[args.variable],
                historical[args.variable],
                simulation[args.variable],
                variable=args.variable,
                method=args.method,
                kind=args.kind,
                group=args.group,
                window=args.window,
                quantiles=args.quantiles,
                interpolation=args.interpolation,
                extrapolation=args.extrapolation,
                chunks=chunks,
                random_seed=args.random_seed,
            )
            write_zarr(
                result.to_dataset(),
                args.output,
                zarr_format=args.zarr_format,
                overwrite=args.overwrite,
            )
    print(f"wrote {args.output} using the {args.variable} preset")


def supported_variables() -> tuple[str, ...]:
    """Return variable names supported by the preset CLI."""
    return tuple(VARIABLE_PRESETS)


if __name__ == "__main__":
    main()
