from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "qc_global_fwi_products.py"
SPEC = importlib.util.spec_from_file_location("qc_global_fwi_products", SCRIPT)
assert SPEC and SPEC.loader
QC = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = QC
SPEC.loader.exec_module(QC)


def test_coverage_counts_distinguishes_land_gaps_from_ocean_leaks():
    fill = np.int16(-32768)
    raw = np.array([[10, fill], [fill, 20]], dtype="int16")
    support = np.array([[True, True], [False, False]])

    missing, outside = QC._coverage_counts(raw, support, fill)

    assert missing == 1
    assert outside == 1
