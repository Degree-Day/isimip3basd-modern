"""Variable-specific bias-adjustment presets for ISIMIP climate data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Transform = Literal["logit", "clearness_index"]


@dataclass(frozen=True)
class VariablePreset:
    """Scientifically meaningful defaults for one climate variable."""

    method: Literal["qdm", "dqm", "scaling"]
    kind: Literal["additive", "multiplicative"]
    group: str = "time.dayofyear"
    window: int = 31
    transform: Transform | None = None
    lower_bound: str | None = None
    lower_threshold: str | None = None
    upper_bound: str | None = None
    upper_threshold: str | None = None
    adapt_frequency: bool = False


VARIABLE_PRESETS: dict[str, VariablePreset] = {
    "hurs": VariablePreset(
        method="qdm",
        kind="additive",
        transform="logit",
        lower_bound="0 %",
        lower_threshold="0.01 %",
        upper_bound="100 %",
        upper_threshold="99.99 %",
    ),
    "pr": VariablePreset(
        method="qdm",
        kind="multiplicative",
        lower_bound="0 mm d-1",
        lower_threshold="0.1 mm d-1",
        adapt_frequency=True,
    ),
    "prsnratio": VariablePreset(
        method="qdm",
        kind="additive",
        transform="logit",
        lower_bound="0",
        lower_threshold="0.0001",
        upper_bound="1",
        upper_threshold="0.9999",
    ),
    "ps": VariablePreset(method="dqm", kind="additive"),
    "rlds": VariablePreset(method="dqm", kind="additive"),
    "rsds": VariablePreset(
        method="qdm",
        kind="additive",
        transform="clearness_index",
        lower_bound="0",
        lower_threshold="0.0001",
        upper_bound="1",
        upper_threshold="0.9999",
    ),
    "sfcWind": VariablePreset(
        method="qdm",
        kind="multiplicative",
        lower_bound="0 m s-1",
        lower_threshold="0.01 m s-1",
    ),
    "tas": VariablePreset(method="dqm", kind="additive"),
    "tasrange": VariablePreset(
        method="qdm",
        kind="multiplicative",
        lower_bound="0 K",
        lower_threshold="0.01 K",
    ),
    "tasskew": VariablePreset(
        method="qdm",
        kind="additive",
        transform="logit",
        lower_bound="0",
        lower_threshold="0.0001",
        upper_bound="1",
        upper_threshold="0.9999",
    ),
}


def get_preset(variable: str) -> VariablePreset:
    """Return a preset or raise an error listing supported variables."""
    try:
        return VARIABLE_PRESETS[variable]
    except KeyError as error:
        supported = ", ".join(VARIABLE_PRESETS)
        raise ValueError(
            f"no preset for {variable!r}; supported presets: {supported}"
        ) from error

