#!/usr/bin/env python3
"""Run MBCnSD over the full nested reference domain in restartable tiles."""

from __future__ import annotations

import sys

from run_europe_downscale_tiles import main


if __name__ == "__main__":
    if "--regions" not in sys.argv:
        sys.argv.extend(("--regions", "global"))
    if "--output-root" not in sys.argv:
        sys.argv.extend(("--output-root", "/data1/cmip6_downscaled_global"))
    main()
