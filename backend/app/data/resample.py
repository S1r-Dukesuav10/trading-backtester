"""OHLC resampling with left-closed, left-labelled buckets.

Why left-closed/left-labelled: a 5m bar labelled 12:00 contains only data
from [12:00, 12:05). The label is the bucket's *first* source bar, so a bar
labelled T can never contain data from before T — no bar ever uses future
data, and downstream signals see the label timestamp as the bar's time.
"""

from __future__ import annotations

import re

import pandas as pd

from .schema import validate_candles

_SHORT_MINUTE = re.compile(r"^(\d+)m$")
_SHORT_DAY = re.compile(r"^(\d+)d$")

_OHLC_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}


def normalize_rule(rule: str) -> str:
    """Normalize human timeframe strings to pandas offset aliases.

    ``"5m"`` -> ``"5min"`` (bare ``m`` means month-end in pandas) and
    ``"1d"`` -> ``"1D"`` (lowercase ``d`` is deprecated in pandas).
    Already-valid aliases (``1min``, ``15min``, ``1h``, ``4h``, ``1D``)
    pass through unchanged.
    """
    rule = str(rule).strip()
    rule = _SHORT_MINUTE.sub(r"\1min", rule)
    return _SHORT_DAY.sub(r"\1D", rule)


def resample_candles(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample canonical candles to a larger timeframe.

    OHLC aggregation: open=first, high=max, low=min, close=last,
    volume=sum, spread=mean (if present). Empty buckets are dropped.
    """
    candles = validate_candles(df)
    if candles.empty:
        return candles

    agg = dict(_OHLC_AGG)
    if "spread" in candles.columns:
        agg["spread"] = "mean"

    resampled = (
        candles.set_index("timestamp")
        .resample(normalize_rule(rule), label="left", closed="left")
        .agg(agg)
    )
    # Buckets with no source rows come back all-NaN (open=first of nothing).
    resampled = resampled.dropna(subset=["open"]).reset_index()
    return validate_candles(resampled)
