"""Tests for OHLC resampling using synthetic 1-minute CSV data.

Synthetic bar i (minute i, i = 0..9) at 2026-03-10 09:{i}:00 UTC:
    open=i, high=50+i, low=i-5, close=i, volume=10*(i+1)
Values are chosen so every aggregate (first/max/min/last/sum) is distinct
and checkable by hand.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.data.local_file import LocalFileSource
from app.data.resample import normalize_rule, resample_candles
from tests.helpers import write_csv

SYMBOL = "TEST"
TIMEFRAME = "1m"


def one_minute_rows(
    minutes: range | list[int], day: str = "2026-03-10", hour: int = 9
) -> list[str]:
    rows = []
    for m in minutes:
        rows.append(
            f"{day} {hour:02d}:{m:02d}:00+00:00,{m},{50 + m},{m - 5},{m},{10 * (m + 1)}"
        )
    return rows


def load_rows(tmp_path, rows) -> pd.DataFrame:
    path = write_csv(tmp_path, rows)
    src = LocalFileSource(path, symbol=SYMBOL, timeframe=TIMEFRAME)
    return src.get_candles(SYMBOL, TIMEFRAME)


def test_normalize_rule():
    assert normalize_rule("5m") == "5min"
    assert normalize_rule("15m") == "15min"
    assert normalize_rule("1min") == "1min"
    assert normalize_rule("1h") == "1h"
    assert normalize_rule("4h") == "4h"
    assert normalize_rule("1d") == "1D"
    assert normalize_rule("4d") == "4D"


def test_one_minute_to_five_minute_ohlc(tmp_path):
    candles = load_rows(tmp_path, one_minute_rows(range(10)))
    out = resample_candles(candles, "5m")

    assert len(out) == 2
    # Left-closed, left-labelled: first bucket is 09:00 (covering 09:00-09:04).
    assert out["timestamp"].iloc[0] == pd.Timestamp("2026-03-10 09:00:00", tz="UTC")
    assert out["timestamp"].iloc[1] == pd.Timestamp("2026-03-10 09:05:00", tz="UTC")

    first = out.iloc[0]
    assert first["open"] == 0  # first source bar of the bucket
    assert first["high"] == 54  # max(50..54)
    assert first["low"] == -5  # min(-5..-1)
    assert first["close"] == 4  # last source bar (09:04) -- no future data
    assert first["volume"] == 150  # 10+20+30+40+50

    second = out.iloc[1]
    assert second["open"] == 5
    assert second["high"] == 59  # max(55..59)
    assert second["low"] == 0  # min(0..4)
    assert second["close"] == 9  # source bar 09:09
    assert second["volume"] == 400  # 60+70+80+90+100


def test_resampled_bar_never_contains_future_data(tmp_path):
    """The 09:00 bar must close at the 09:04 source bar, not later."""
    candles = load_rows(tmp_path, one_minute_rows(range(10)))
    out = resample_candles(candles, "5m")

    bucket_0900 = out[out["timestamp"] == pd.Timestamp("2026-03-10 09:00:00", tz="UTC")]
    src_close_0904 = candles[
        candles["timestamp"] == pd.Timestamp("2026-03-10 09:04:00", tz="UTC")
    ]["close"].iloc[0]
    assert bucket_0900["close"].iloc[0] == src_close_0904


def test_empty_buckets_are_dropped(tmp_path):
    # 09:00-09:04, then a gap through 09:09, then 09:10-09:14.
    rows = one_minute_rows(list(range(5)) + list(range(10, 15)))
    candles = load_rows(tmp_path, rows)
    out = resample_candles(candles, "5m")

    labels = out["timestamp"].tolist()
    assert labels == [
        pd.Timestamp("2026-03-10 09:00:00", tz="UTC"),
        pd.Timestamp("2026-03-10 09:10:00", tz="UTC"),
    ]
    # Crucially: no bar labelled 09:05 despite the label being "missing".
    assert pd.Timestamp("2026-03-10 09:05:00", tz="UTC") not in labels


def test_one_minute_to_one_hour_boundaries(tmp_path):
    # 09:55-09:59 and 10:00-10:04 -> two hourly bars at 09:00 and 10:00.
    rows = [
        f"2026-03-10 09:{m:02d}:00+00:00,{m},{50 + m},{m - 5},{m},{10 * (m + 1)}"
        for m in range(55, 60)
    ] + [
        f"2026-03-10 10:0{m}:00+00:00,{m},{50 + m},{m - 5},{m},{10 * (m + 1)}"
        for m in range(5)
    ]
    candles = load_rows(tmp_path, rows)
    out = resample_candles(candles, "1h")

    assert out["timestamp"].tolist() == [
        pd.Timestamp("2026-03-10 09:00:00", tz="UTC"),
        pd.Timestamp("2026-03-10 10:00:00", tz="UTC"),
    ]
    assert len(out[out["timestamp"].dt.hour == 9]) == 1
    # Only the five 09:5x bars belong to the 09:00 bucket.
    assert out.iloc[0]["volume"] == 10 * (56 + 57 + 58 + 59 + 60)


def test_one_minute_to_one_day(tmp_path):
    rows = one_minute_rows(range(3))
    rows += one_minute_rows(range(4, 6))  # two more minutes, same day
    candles = load_rows(tmp_path, rows)
    out = resample_candles(candles, "1d")

    assert len(out) == 1
    assert out["timestamp"].iloc[0] == pd.Timestamp("2026-03-10 00:00:00", tz="UTC")
    assert out["open"].iloc[0] == 0
    assert out["close"].iloc[0] == 5
    assert out["volume"].iloc[0] == 10 + 20 + 30 + 50 + 60


def test_spread_column_averaged(tmp_path):
    rows = [
        f"2026-03-10 09:0{m}:00+00:00,{m},{50 + m},{m - 5},{m},{10 * (m + 1)},{m}"
        for m in range(5)
    ]
    path = write_csv(
        tmp_path, rows, header="timestamp,open,high,low,close,volume,spread"
    )
    src = LocalFileSource(path, symbol=SYMBOL, timeframe=TIMEFRAME)
    candles = src.get_candles(SYMBOL, TIMEFRAME)
    out = resample_candles(candles, "5m")

    assert len(out) == 1
    assert out["spread"].iloc[0] == pytest.approx(2.0)  # mean(0..4)


def test_resample_rejects_bad_input(tmp_path):
    candles = load_rows(tmp_path, one_minute_rows(range(3)))
    candles = candles.drop(columns=["high"])
    with pytest.raises(ValueError, match="Missing required"):
        resample_candles(candles, "5m")
