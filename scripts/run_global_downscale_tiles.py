#!/usr/bin/env python3
"""Run MBCnSD over the full nested reference domain in restartable tiles."""

from __future__ import annotations

from isimip3basd_modern.tiled_runner import main


if __name__ == "__main__":
    main(default_regions=["global"])
