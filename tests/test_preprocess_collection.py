from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "preprocess_collection.py"
SPEC = importlib.util.spec_from_file_location("preprocess_collection", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_model_license_attrs_reads_machine_record(tmp_path: Path) -> None:
    model_root = tmp_path / "MODEL"
    model_root.mkdir()
    (model_root / "LICENSE.json").write_text(
        json.dumps(
            {
                "institution_id": ["ONE", "TWO"],
                "license": {
                    "id": "CC BY 4.0",
                    "license": "Creative Commons Attribution 4.0 International",
                    "url": "https://creativecommons.org/licenses/by/4.0/",
                    "history": "relaxed to CC BY 4.0",
                },
                "authoritative_registry_url": "https://example.test/registry.json",
                "cmip6_terms_of_use_url": "https://example.test/terms",
                "retrieved_utc": "2026-09-08T00:00:00+00:00",
            }
        )
    )

    attrs = MODULE.model_license_attrs(tmp_path, "MODEL")

    assert attrs["license_id"] == "CC BY 4.0"
    assert attrs["institution_id"] == "ONE,TWO"
    assert attrs["cmip6_terms_of_use"] == "https://example.test/terms"


def test_model_license_attrs_is_optional(tmp_path: Path) -> None:
    assert MODULE.model_license_attrs(tmp_path, "MODEL") == {}


def test_discover_two_period_stores_uses_only_supported_layout(tmp_path: Path) -> None:
    for period in ("historical/hist", "ssp245/projection"):
        for variable in ("tas", "hurs", "pr", "sfcWind", "sfcWind_uv"):
            (tmp_path / "MODEL" / period / f"{variable}.zarr").mkdir(parents=True)

    stores = MODULE.discover_two_period_stores(
        tmp_path,
        ["MODEL"],
        ["tas", "hurs", "pr", "sfcWind"],
    )

    assert len(stores) == 8
    assert {path.stem for path in stores} == {"tas", "hurs", "pr", "sfcWind"}
    assert {path.parent.name for path in stores} == {"hist", "projection"}


def test_discover_two_period_stores_rejects_legacy_stage(tmp_path: Path) -> None:
    for period in ("historical/hist", "ssp245/projection", "ssp245/ref"):
        (tmp_path / "MODEL" / period / "tas.zarr").mkdir(parents=True)

    try:
        MODULE.discover_two_period_stores(tmp_path, ["MODEL"])
    except ValueError as error:
        assert "legacy projection stages" in str(error)
    else:
        raise AssertionError("legacy stage was accepted")


def test_expected_period_contract() -> None:
    assert MODULE.expected_period(Path("MODEL/historical/hist/tas.zarr")) == (
        "1989-01-01",
        "2014-12-31",
    )
    assert MODULE.expected_period(Path("MODEL/ssp245/projection/tas.zarr")) == (
        "2015-01-01",
        "2100-12-31",
    )
