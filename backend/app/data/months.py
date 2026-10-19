"""Month-string helpers shared by the downloader sources."""

from __future__ import annotations

import re

import pandas as pd

from .source import TimeLike, to_utc_timestamp

MONTH_PATTERN = re.compile(r"^\d{4}-\d{2}$")


def is_month_string(value: object) -> bool:
    """True for strings like ``"2024-01"`` (and nothing else)."""
    return isinstance(value, str) and bool(MONTH_PATTERN.match(value.strip()))


def month_of(value: TimeLike) -> pd.Period:
    """Month period covering ``value`` ("2024-01" or any timestamp)."""
    if is_month_string(value):
        return pd.Period(str(value).strip(), freq="M")
    ts = to_utc_timestamp(value)
    assert ts is not None  # None handled by callers
    return ts.tz_localize(None).to_period("M")


def month_range(start: TimeLike, end: TimeLike) -> list[str]:
    """Inclusive list of "YYYY-MM" strings covering [start, end]."""
    start_period = month_of(start)
    end_period = month_of(end)
    if end_period < start_period:
        raise ValueError(f"end ({end}) is before start ({start})")
    return [
        str(period) for period in pd.period_range(start_period, end_period, freq="M")
    ]


def month_dates(month: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last *day* of ``month`` ("2024-01" -> 2024-01-01, 2024-01-31)."""
    period = pd.Period(month, freq="M")
    return period.start_time.normalize(), period.end_time.normalize()
