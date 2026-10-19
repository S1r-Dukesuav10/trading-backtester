"""Filesystem locations for the data layer.

Defaults are anchored to the *repository root*, not the current working
directory, so the project always uses one top-level ``data/`` directory no
matter where a command is run from.

Override the base directory with the ``BACKTESTER_DATA_DIR`` environment
variable (useful for tests, alternate datasets, or an external drive).
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_DATA_DIR = "BACKTESTER_DATA_DIR"

# backend/app/data/paths.py -> backend/app/data -> backend/app -> backend -> repo root
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def data_dir() -> Path:
    """Base data directory: ``$BACKTESTER_DATA_DIR`` if set, else <repo>/data."""
    override = os.environ.get(ENV_DATA_DIR)
    if override:
        return Path(override).expanduser()
    return PROJECT_ROOT / "data"


def candles_root() -> Path:
    """Root of canonical candle Parquet files: data/candles/."""
    return data_dir() / "candles"


def binance_raw_root() -> Path:
    """Root of cached Binance Vision zips: data/raw/binance/."""
    return data_dir() / "raw" / "binance"


def dukascopy_raw_root() -> Path:
    """Root of cached Dukascopy CSVs: data/raw/dukascopy/."""
    return data_dir() / "raw" / "dukascopy"
