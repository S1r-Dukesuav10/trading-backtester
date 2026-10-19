"""Canonical candle schema definition and validation.

Canonical candles:
- Columns: timestamp, open, high, low, close, volume (required)
  plus spread (optional, reserved for brokers/venues that publish it).
- timestamp: tz-aware, strictly UTC, sorted ascending, unique.
- All price/volume columns numeric.
- Timestamp resolution is normalized to nanoseconds, so two candles built
  from sources with different resolutions (seconds/ms/us) are identical.
"""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
)
OPTIONAL_COLUMNS: tuple[str, ...] = ("spread",)
CANDLE_COLUMNS: tuple[str, ...] = REQUIRED_COLUMNS + OPTIONAL_COLUMNS


def validate_candles(df: pd.DataFrame) -> pd.DataFrame:
    """Return a canonicalized copy of ``df``.

    Canonical means:
    - required columns present (others dropped unless optional/known)
    - timestamp tz-aware, converted to UTC, resolution normalized to ns
    - rows sorted by timestamp ascending
    - no duplicate timestamps (raises instead of silently dropping)
    - OHLCV (and spread, if present) numeric

    Naive (tz-less) timestamps are rejected: their timezone is unknowable,
    so callers must localize explicitly before validation.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required candle columns: {missing}")

    out = df.copy()

    ts = out["timestamp"]
    if not isinstance(ts.dtype, pd.DatetimeTZDtype):
        if pd.api.types.is_datetime64_any_dtype(ts.dtype):
            raise ValueError(
                "timestamp column is timezone-naive; localize it explicitly "
                "(e.g. to UTC) before validation"
            )
        raise TypeError("timestamp column must be a timezone-aware datetime dtype")

    out["timestamp"] = ts.dt.tz_convert("UTC").dt.as_unit("ns")

    for col in REQUIRED_COLUMNS[1:]:
        out[col] = pd.to_numeric(out[col], errors="raise")
    if "spread" in out.columns:
        out["spread"] = pd.to_numeric(out["spread"], errors="raise")

    if not out["timestamp"].is_monotonic_increasing:
        out = out.sort_values("timestamp", kind="stable")

    dup_mask = out["timestamp"].duplicated(keep=False)
    if dup_mask.any():
        examples = out.loc[dup_mask, "timestamp"].head(3).tolist()
        raise ValueError(f"Duplicate timestamps are not allowed: {examples}")

    out = out[[c for c in CANDLE_COLUMNS if c in out.columns]]
    return out.reset_index(drop=True)
