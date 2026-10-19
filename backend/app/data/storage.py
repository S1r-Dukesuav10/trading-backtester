"""Parquet candle storage layout.

    data/candles/{market}/{symbol}/{timeframe}.parquet

``data/`` is gitignored (raw market data never gets committed) and the root
defaults to the repository's top-level data/ directory, see app.data.paths.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .paths import candles_root
from .schema import validate_candles


def candles_path(
    root: str | Path | None,
    market: str,
    symbol: str,
    timeframe: str,
) -> Path:
    """Return {root}/{market}/{symbol}/{timeframe}.parquet.

    ``root=None`` uses the default candles root (see app.data.paths).
    """
    base = candles_root() if root is None else Path(root)
    return base / str(market) / str(symbol) / f"{timeframe}.parquet"


def save_candles(
    df: pd.DataFrame,
    root: str | Path | None = None,
    *,
    market: str,
    symbol: str,
    timeframe: str,
) -> Path:
    """Validate and write candles to the canonical Parquet location."""
    candles = validate_candles(df)
    path = candles_path(root, market, symbol, timeframe)
    path.parent.mkdir(parents=True, exist_ok=True)
    candles.to_parquet(path, index=False)
    return path


def load_candles(
    root: str | Path | None = None,
    *,
    market: str,
    symbol: str,
    timeframe: str,
) -> pd.DataFrame:
    """Load candles from the canonical Parquet location, re-validated."""
    path = candles_path(root, market, symbol, timeframe)
    if not path.exists():
        raise FileNotFoundError(f"No parquet candles at {path}")
    return validate_candles(pd.read_parquet(path))
