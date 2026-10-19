"""Tests for the Parquet storage layout."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.data.local_file import LocalFileSource
from app.data.schema import validate_candles
from app.data.storage import candles_path, load_candles, save_candles


def make_candles() -> pd.DataFrame:
    return validate_candles(
        pd.DataFrame(
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
    )


def test_candles_path_layout():
    path = candles_path(Path("data/candles"), "futures", "ES", "5m")
    assert path == Path("data/candles/futures/ES/5m.parquet")

    path = candles_path("D:/data", "crypto", "BTCUSDT", "1h")
    assert path == Path("D:/data/crypto/BTCUSDT/1h.parquet")


def test_save_and_load_roundtrip_keeps_utc(tmp_path):
    df = make_candles()
    saved = save_candles(df, tmp_path, market="futures", symbol="ES", timeframe="5m")
    assert saved == tmp_path / "futures" / "ES" / "5m.parquet"
    assert saved.exists()

    loaded = load_candles(tmp_path, market="futures", symbol="ES", timeframe="5m")
    assert str(loaded["timestamp"].dt.tz) == "UTC"
    pd.testing.assert_frame_equal(loaded, df)


def test_save_rejects_naive_timestamps(tmp_path):
    df = make_candles()
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    with pytest.raises(ValueError, match="timezone-naive"):
        save_candles(df, tmp_path, market="futures", symbol="ES", timeframe="5m")


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No parquet candles"):
        load_candles(tmp_path, market="futures", symbol="ES", timeframe="5m")


def test_saved_file_readable_by_local_file_source(tmp_path):
    df = make_candles()
    saved = save_candles(df, tmp_path, market="futures", symbol="ES", timeframe="5m")

    src = LocalFileSource(saved, symbol="ES", timeframe="5m")
    out = src.get_candles("ES", "5m")
    pd.testing.assert_frame_equal(out, df)
