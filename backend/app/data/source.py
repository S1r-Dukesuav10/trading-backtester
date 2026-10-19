"""Abstract data-source interface.

A DataSource answers: "give me candles for (symbol, timeframe, start, end)".
Concrete sources (local files now, exchange APIs later) implement this.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd

TimeLike = str | pd.Timestamp | datetime


class DataSource(ABC):
    @abstractmethod
    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        start: TimeLike | None = None,
        end: TimeLike | None = None,
    ) -> pd.DataFrame:
        """Return candles in canonical schema (see app.data.schema).

        Range semantics: ``[start, end)`` — start inclusive, end exclusive.
        Naive ``start``/``end`` values are interpreted as UTC.
        """


def to_utc_timestamp(value: TimeLike | None) -> pd.Timestamp | None:
    """Coerce a user-supplied bound to a UTC Timestamp. Naive -> assumed UTC."""
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def filter_range(
    df: pd.DataFrame,
    start: TimeLike | None,
    end: TimeLike | None,
) -> pd.DataFrame:
    """Filter canonical candles to [start, end) and reset the index."""
    lower = to_utc_timestamp(start)
    upper = to_utc_timestamp(end)
    mask = pd.Series(True, index=df.index)
    if lower is not None:
        mask &= df["timestamp"] >= lower
    if upper is not None:
        mask &= df["timestamp"] < upper
    return df.loc[mask].reset_index(drop=True)
