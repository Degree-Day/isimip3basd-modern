"""Command-line interface for the modern workflow."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
import json
from pathlib import Path

from distributed import Client, LocalCluster

from .io import open_dataset, parse_chunks, write_zarr
from .pipeline import adjust_variable
from .presets import VARIABLE_PRESETS
from .validation import (
    derive_variables,
    validate_dataset,
    validate_inputs,
    validate_output,
    validate_variable,
)


def _positive_integer(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return result


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
    validate.add_argument("--min-valid-fraction", type=float, default=1.0)

    validate_all = subparsers.add_parser(
        "validate-dataset", help="validate linked variables in a combined dataset"
    )
    validate_all.add_argument("input")
    validate_all.add_argument("--chunks", default="time=-1")

    derive = subparsers.add_parser(
        "derive", help="derive tasmin, tasmax, prsn, and huss when inputs are present"
    )
    derive.add_argument("input")
    derive.add_argument("output")
    derive.add_argument("--chunks", default="time=-1")
    derive.add_argument("--zarr-format", type=int, choices=(2, 3), default=3)
    derive.add_argument("--overwrite", action="store_true")

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
    run.add_argument(
        "--min-training-years",
        type=_positive_integer,
        default=10,
        help="minimum reference and historical record length",
    )
    run.add_argument("--min-valid-fraction", type=float, default=1.0)
    run.add_argument(
        "--qc-report",
        help="QC JSON path (default: OUTPUT.qc.json)",
    )
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
            report = validate_variable(
                dataset[args.variable],
                args.variable,
                min_valid_fraction=args.min_valid_fraction,
            )
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        if not report.valid:
            raise SystemExit(1)
        return

    if args.command == "validate-dataset":
        with open_dataset(args.input, chunks) as dataset:
            report = validate_dataset(dataset)
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        if not report.valid:
            raise SystemExit(1)
        return

    if args.command == "derive":
        with open_dataset(args.input, chunks) as dataset:
            result = derive_variables(dataset)
            added = sorted(set(result.data_vars) - set(dataset.data_vars))
            if not added:
                raise ValueError("input contains no complete derivation groups")
            write_zarr(
                result,
                args.output,
                zarr_format=args.zarr_format,
                overwrite=args.overwrite,
            )
        print(f"wrote {args.output}; derived {', '.join(added)}")
        return

    qc_path = Path(args.qc_report or f"{args.output}.qc.json")
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    qc_document: dict[str, object] = {}
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

            input_report = validate_inputs(
                reference[args.variable],
                historical[args.variable],
                simulation[args.variable],
                args.variable,
                min_training_years=args.min_training_years,
                min_valid_fraction=args.min_valid_fraction,
            )
            qc_document["preflight"] = input_report.to_dict()
            if not input_report.valid:
                qc_path.write_text(
                    json.dumps(qc_document, indent=2, sort_keys=True) + "\n"
                )
                print(json.dumps(qc_document, indent=2, sort_keys=True))
                raise SystemExit(1)

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
            with open_dataset(args.output, chunks) as written:
                output_report = validate_output(
                    written[args.variable],
                    simulation[args.variable],
                    args.variable,
                    min_valid_fraction=args.min_valid_fraction,
                )
            qc_document["output"] = output_report.to_dict()
            qc_document["valid"] = input_report.valid and output_report.valid

    qc_path.write_text(json.dumps(qc_document, indent=2, sort_keys=True) + "\n")
    print(json.dumps(qc_document, indent=2, sort_keys=True))
    if not qc_document["valid"]:
        raise SystemExit(1)
    print(
        f"wrote {args.output} using the {args.variable} preset; "
        f"QC report: {qc_path}"
    )


def supported_variables() -> tuple[str, ...]:
    """Return variable names supported by the preset CLI."""
    return tuple(VARIABLE_PRESETS)


if __name__ == "__main__":
    main()
