"""Tests for the canonical candle schema validation."""

from __future__ import annotations

import pandas as pd
import pytest

from app.data.schema import validate_candles


def make_df(
    timestamps, opens=None, highs=None, lows=None, closes=None, volumes=None
) -> pd.DataFrame:
    """Build a tz-aware canonical DataFrame from UTC timestamp strings."""
    n = len(timestamps)
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps, utc=True),
            "open": opens if opens is not None else [100.0 + i for i in range(n)],
            "high": highs if highs is not None else [101.0 + i for i in range(n)],
            "low": lows if lows is not None else [99.0 + i for i in range(n)],
            "close": closes if closes is not None else [100.5 + i for i in range(n)],
            "volume": volumes
            if volumes is not None
            else [10.0 * (i + 1) for i in range(n)],
        }
    )


def test_validate_passes_canonical_input():
    df = make_df(["2026-01-05 09:00:00", "2026-01-05 09:01:00"])
    out = validate_candles(df)
    assert list(out.columns) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert out["timestamp"].dt.tz is not None
    assert str(out["timestamp"].dt.tz) == "UTC"


def test_validate_sorts_unsorted_input():
    df = make_df(
        ["2026-01-05 09:02:00", "2026-01-05 09:00:00", "2026-01-05 09:01:00"],
        opens=[2.0, 0.0, 1.0],
    )
    out = validate_candles(df)
    assert out["timestamp"].is_monotonic_increasing
    assert out["open"].tolist() == [0.0, 1.0, 2.0]
    assert out.index.tolist() == [0, 1, 2]


def test_validate_rejects_duplicate_timestamps():
    df = make_df(["2026-01-05 09:00:00", "2026-01-05 09:00:00"])
    with pytest.raises(ValueError, match="Duplicate timestamps"):
        validate_candles(df)


def test_validate_rejects_naive_timestamps():
    df = make_df(["2026-01-05 09:00:00"])
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    with pytest.raises(ValueError, match="timezone-naive"):
        validate_candles(df)


def test_validate_rejects_missing_columns():
    df = make_df(["2026-01-05 09:00:00"]).drop(columns=["volume"])
    with pytest.raises(ValueError, match="Missing required candle columns"):
        validate_candles(df)


def test_validate_converts_other_timezones_to_utc():
    df = make_df(["2026-01-05 09:00:00", "2026-01-05 09:01:00"])
    df["timestamp"] = df["timestamp"].dt.tz_convert("Europe/Berlin")
    out = validate_candles(df)
    assert str(out["timestamp"].dt.tz) == "UTC"
    # 09:00 UTC == 10:00 Berlin in January; instants must be preserved.
    assert out["timestamp"].iloc[0] == pd.Timestamp("2026-01-05 09:00:00", tz="UTC")


def test_validate_keeps_optional_spread_drops_unknown_columns():
    df = make_df(["2026-01-05 09:00:00"])
    df["spread"] = 0.25
    df["venue_note"] = "synthetic"
    out = validate_candles(df)
    assert "spread" in out.columns
    assert "venue_note" not in out.columns
    assert out["spread"].tolist() == [0.25]


def test_validate_rejects_non_numeric_ohlc():
    df = make_df(["2026-01-05 09:00:00"])
    df["close"] = ["not-a-price"]
    with pytest.raises(ValueError):
        validate_candles(df)
