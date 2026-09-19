#!/usr/bin/env python3
"""Build a preflighted SSP2-4.5/SSP3-7.0 production queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cftime
import zarr


SCENARIOS = ("ssp245", "ssp370")
VARIABLES = ("tas", "hurs", "pr", "sfcWind")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--canonical-root", type=Path, required=True)
    result.add_argument("--published-root", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--model", action="append", dest="models")
    return result


def _valid_qc(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return bool(json.loads(path.read_text()).get("valid"))
    except (json.JSONDecodeError, OSError):
        return False


def _coverage(path: Path) -> tuple[str, str, int]:
    group = zarr.open_group(path, mode="r", use_consolidated=False)
    time = group["time"]
    values = time[[0, time.shape[0] - 1]]
    dates = cftime.num2date(
        values,
        units=time.attrs["units"],
        calendar=time.attrs.get("calendar", "standard"),
        only_use_cftime_datetimes=True,
    )
    return str(dates[0])[:10], str(dates[-1])[:10], int(time.shape[0])


def _period_ready(
    root: Path, start: str, end: str, days: int
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    for variable in VARIABLES:
        store = root / f"{variable}.zarr"
        if not store.is_dir():
            errors.append(f"missing {store}")
            continue
        try:
            actual_start, actual_end, actual_days = _coverage(store)
        except Exception as error:
            errors.append(
                f"cannot read {store}: {type(error).__name__}: {error}"
            )
            continue
        if (actual_start, actual_end, actual_days) != (start, end, days):
            errors.append(
                f"{store} covers {actual_start}..{actual_end} "
                f"({actual_days} days); expected {start}..{end} ({days} days)"
            )
    return not errors, errors


def _product_complete(published: Path, model: str, scenario: str) -> bool:
    model_root = published / model
    future = model_root / scenario / "projection"
    manifest_paths = (
        model_root / "historical/hist/weather/publication-manifest.json",
        future / "weather/publication-manifest.json",
    )
    qc_paths = (
        model_root
        / "historical/hist/fwi/global/"
        "daily_fire_weather_indices_1989-2014.zarr.qc.json",
        future / "fwi/global/daily_fire_weather_indices_2015-2100.zarr.qc.json",
        future / "fwi/annual/annual_fwi_indicators_1989_2100.zarr.qc.json",
        future / "fwi/annual/fwi_reference_thresholds_1995_2014.zarr.qc.json",
    )
    return all(path.is_file() for path in manifest_paths) and all(
        _valid_qc(path) for path in qc_paths
    )


def _historical_complete(published: Path, model: str) -> bool:
    root = published / model / "historical/hist"
    return (
        (root / "weather/publication-manifest.json").is_file()
        and _valid_qc(
            root
            / "fwi/global/"
            "daily_fire_weather_indices_1989-2014.zarr.qc.json"
        )
    )


def main() -> None:
    args = parser().parse_args()
    models = args.models or sorted(
        path.name
        for path in args.canonical_root.iterdir()
        if path.is_dir() and path.name != "logs"
    )
    records: list[dict[str, object]] = []
    for model in models:
        model_root = args.canonical_root / model
        historical_ready, historical_errors = _period_ready(
            model_root / "historical/hist",
            "1989-01-01",
            "2014-12-31",
            9490,
        )
        shared_history = _historical_complete(args.published_root, model)
        for scenario in SCENARIOS:
            scenario_root = model_root / scenario
            if not scenario_root.is_dir():
                continue
            projection_ready, projection_errors = _period_ready(
                scenario_root / "projection",
                "2015-01-01",
                "2100-12-31",
                31390,
            )
            complete = _product_complete(args.published_root, model, scenario)
            errors = [*historical_errors, *projection_errors]
            if complete:
                status = "complete"
            elif historical_ready and projection_ready:
                status = "ready"
            else:
                status = "blocked"
            records.append(
                {
                    "model": model,
                    "scenario": scenario,
                    "status": status,
                    "historical_product_reusable": shared_history,
                    "errors": errors,
                }
            )
    records.sort(
        key=lambda item: (
            {"ready": 0, "complete": 1, "blocked": 2}[str(item["status"])],
            not bool(item["historical_product_reusable"]),
            str(item["model"]),
            str(item["scenario"]),
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    for record in records:
        print(
            f"{record['status'].upper():8} {record['model']:16} "
            f"{record['scenario']} historical_reusable="
            f"{str(record['historical_product_reusable']).lower()}"
        )
        for error in record["errors"]:
            print(f"  {error}")


if __name__ == "__main__":
    main()
