"""Tests for LocalFileSource using synthetic CSV/Parquet files."""

from __future__ import annotations

import pandas as pd
import pytest

from app.data.local_file import LocalFileSource
from app.data.schema import validate_candles
from tests.helpers import write_csv

SYMBOL = "TEST"
TIMEFRAME = "1m"


def test_imports_utc_millisecond_epoch(tmp_path):
    # 2026-03-10 09:00:00 UTC = 1773133200 s; 09:01:00 = +60 s.
    path = write_csv(
        tmp_path,
        ["1773133200000,100,101,99,100.5,10", "1773133260000,100.5,102,100,101,20"],
    )
    src = LocalFileSource(
        path, symbol=SYMBOL, timeframe=TIMEFRAME, timestamp_unit="milliseconds"
    )
    out = src.get_candles(SYMBOL, TIMEFRAME)
    assert str(out["timestamp"].dt.tz) == "UTC"
    assert out["timestamp"].iloc[0] == pd.Timestamp("2026-03-10 09:00:00", tz="UTC")
    assert len(out) == 2


def test_numeric_timestamps_without_unit_raise(tmp_path):
    path = write_csv(tmp_path, ["1773133200000,100,101,99,100.5,10"])
    src = LocalFileSource(path, symbol=SYMBOL, timeframe=TIMEFRAME)
    with pytest.raises(ValueError, match="timestamp_unit"):
        src.get_candles(SYMBOL, TIMEFRAME)


def test_unsorted_input_is_sorted(tmp_path):
    path = write_csv(
        tmp_path,
        [
            "2026-03-10 09:02:00+00:00,102,103,101,102.5,30",
            "2026-03-10 09:00:00+00:00,100,101,99,100.5,10",
            "2026-03-10 09:01:00+00:00,101,102,100,101.5,20",
        ],
    )
    src = LocalFileSource(path, symbol=SYMBOL, timeframe=TIMEFRAME)
    out = src.get_candles(SYMBOL, TIMEFRAME)
    assert out["timestamp"].is_monotonic_increasing
    assert out["volume"].tolist() == [10.0, 20.0, 30.0]


def test_duplicate_timestamps_raise_by_default(tmp_path):
    path = write_csv(
        tmp_path,
        [
            "2026-03-10 09:00:00+00:00,100,101,99,100.5,10",
            "2026-03-10 09:00:00+00:00,105,106,104,105.5,99",
        ],
    )
    src = LocalFileSource(path, symbol=SYMBOL, timeframe=TIMEFRAME)
    with pytest.raises(ValueError, match="Duplicate timestamps"):
        src.get_candles(SYMBOL, TIMEFRAME)


def test_duplicate_timestamps_keep_last_when_configured(tmp_path):
    path = write_csv(
        tmp_path,
        [
            "2026-03-10 09:00:00+00:00,100,101,99,100.5,10",
            "2026-03-10 09:00:00+00:00,105,106,104,105.5,99",
        ],
    )
    src = LocalFileSource(path, symbol=SYMBOL, timeframe=TIMEFRAME, on_duplicate="last")
    out = src.get_candles(SYMBOL, TIMEFRAME)
    assert len(out) == 1
    assert out["close"].iloc[0] == 105.5
    assert out["volume"].iloc[0] == 99.0


def test_non_utc_offsets_convert_to_utc(tmp_path):
    # Summer time in Berlin: +02:00. 12:00+02:00 must become 10:00 UTC.
    path = write_csv(
        tmp_path,
        [
            "2026-07-10 12:00:00+02:00,100,101,99,100.5,10",
            "2026-07-10 13:00:00+02:00,101,102,100,101.5,20",
        ],
    )
    src = LocalFileSource(path, symbol=SYMBOL, timeframe=TIMEFRAME)
    out = src.get_candles(SYMBOL, TIMEFRAME)
    assert str(out["timestamp"].dt.tz) == "UTC"
    assert out["timestamp"].iloc[0] == pd.Timestamp("2026-07-10 10:00:00", tz="UTC")


def test_mixed_offsets_same_instant_detected_as_duplicate(tmp_path):
    # 12:00+02:00 and 10:00+00:00 are the same instant -> duplicates in UTC.
    path = write_csv(
        tmp_path,
        [
            "2026-07-10 12:00:00+02:00,100,101,99,100.5,10",
            "2026-07-10 10:00:00+00:00,101,102,100,101.5,20",
        ],
    )
    src = LocalFileSource(path, symbol=SYMBOL, timeframe=TIMEFRAME)
    with pytest.raises(ValueError, match="Duplicate timestamps"):
        src.get_candles(SYMBOL, TIMEFRAME)


def test_mixed_offsets_convert_and_sort_in_utc(tmp_path):
    path = write_csv(
        tmp_path,
        [
            "2026-07-10 12:00:00+02:00,100,101,99,100.5,10",
            "2026-07-10 11:00:00+00:00,101,102,100,101.5,20",
        ],
    )
    src = LocalFileSource(path, symbol=SYMBOL, timeframe=TIMEFRAME)
    out = src.get_candles(SYMBOL, TIMEFRAME)
    assert str(out["timestamp"].dt.tz) == "UTC"
    assert out["timestamp"].iloc[0] == pd.Timestamp("2026-07-10 10:00:00", tz="UTC")
    assert out["timestamp"].iloc[1] == pd.Timestamp("2026-07-10 11:00:00", tz="UTC")
    assert out["timestamp"].is_monotonic_increasing


def test_naive_timestamps_use_input_timezone(tmp_path):
    # Naive winter time in Berlin (+01:00): 12:00 local -> 11:00 UTC.
    path = write_csv(
        tmp_path,
        [
            "2026-01-15 12:00:00,100,101,99,100.5,10",
            "2026-01-15 13:00:00,101,102,100,101.5,20",
        ],
    )
    src = LocalFileSource(
        path, symbol=SYMBOL, timeframe=TIMEFRAME, input_tz="Europe/Berlin"
    )
    out = src.get_candles(SYMBOL, TIMEFRAME)
    assert out["timestamp"].iloc[0] == pd.Timestamp("2026-01-15 11:00:00", tz="UTC")


def test_naive_timestamps_default_to_utc(tmp_path):
    path = write_csv(tmp_path, ["2026-03-10 09:00:00,100,101,99,100.5,10"])
    src = LocalFileSource(path, symbol=SYMBOL, timeframe=TIMEFRAME)
    out = src.get_candles(SYMBOL, TIMEFRAME)
    assert out["timestamp"].iloc[0] == pd.Timestamp("2026-03-10 09:00:00", tz="UTC")


def test_column_mapping(tmp_path):
    path = write_csv(
        tmp_path,
        ["2026-03-10 09:00:00,100,101,99,100.5,10"],
        header="Date,Open,High,Low,Close,Vol",
    )
    mapping = {
        "timestamp": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Vol",
    }
    src = LocalFileSource(
        path, symbol=SYMBOL, timeframe=TIMEFRAME, column_mapping=mapping
    )
    out = src.get_candles(SYMBOL, TIMEFRAME)
    assert list(out.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert out["close"].iloc[0] == 100.5


def test_column_mapping_missing_column_raises(tmp_path):
    path = write_csv(tmp_path, ["2026-03-10 09:00:00,100,101,99,100.5,10"])
    src = LocalFileSource(
        path, symbol=SYMBOL, timeframe=TIMEFRAME, column_mapping={"timestamp": "Date"}
    )
    with pytest.raises(ValueError, match="references columns not in file"):
        src.get_candles(SYMBOL, TIMEFRAME)


def test_range_filter_is_start_inclusive_end_exclusive(tmp_path):
    rows = [f"2026-03-10 09:0{m}:00+00:00,100,101,99,100.5,10" for m in range(5)]
    path = write_csv(tmp_path, rows)
    src = LocalFileSource(path, symbol=SYMBOL, timeframe=TIMEFRAME)
    out = src.get_candles(
        SYMBOL,
        TIMEFRAME,
        start="2026-03-10 09:01:00+00:00",
        end="2026-03-10 09:03:00+00:00",
    )
    assert len(out) == 2
    assert out["timestamp"].iloc[0] == pd.Timestamp("2026-03-10 09:01:00", tz="UTC")
    assert out["timestamp"].iloc[-1] == pd.Timestamp("2026-03-10 09:02:00", tz="UTC")


def test_symbol_mismatch_raises(tmp_path):
    path = write_csv(tmp_path, ["2026-03-10 09:00:00+00:00,100,101,99,100.5,10"])
    src = LocalFileSource(path, symbol=SYMBOL, timeframe=TIMEFRAME)
    with pytest.raises(ValueError, match="holds symbol"):
        src.get_candles("OTHER", TIMEFRAME)
    with pytest.raises(ValueError, match="holds timeframe"):
        src.get_candles(SYMBOL, "5m")


def test_parquet_import(tmp_path):
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-03-10 09:00:00", "2026-03-10 09:01:00"], utc=True
            ),
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [10.0, 20.0],
        }
    )
    path = tmp_path / "candles.parquet"
    validate_candles(df).to_parquet(path, index=False)

    src = LocalFileSource(path, symbol=SYMBOL, timeframe=TIMEFRAME)
    out = src.get_candles(SYMBOL, TIMEFRAME)
    assert len(out) == 2
    assert str(out["timestamp"].dt.tz) == "UTC"
    pd.testing.assert_series_equal(out["close"], df["close"])


def test_unsupported_extension_raises(tmp_path):
    path = tmp_path / "candles.json"
    path.write_text("[]", encoding="utf-8")
    src = LocalFileSource(path, symbol=SYMBOL, timeframe=TIMEFRAME)
    with pytest.raises(ValueError, match="Unsupported file type"):
        src.get_candles(SYMBOL, TIMEFRAME)


def test_missing_file_raises(tmp_path):
    src = LocalFileSource(tmp_path / "nope.csv", symbol=SYMBOL, timeframe=TIMEFRAME)
    with pytest.raises(FileNotFoundError):
        src.get_candles(SYMBOL, TIMEFRAME)
