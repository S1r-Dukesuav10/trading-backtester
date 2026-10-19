"""Tests for anchored default data paths and the env override."""

from __future__ import annotations

import pandas as pd
import pytest

from app.data import paths
from app.data.binance_vision import BinanceVisionSource
from app.data.storage import load_candles, save_candles


def make_candles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-03-10 09:00:00"], utc=True),
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [10.0],
        }
    )


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv(paths.ENV_DATA_DIR, raising=False)


def test_project_root_points_at_repo_root():
    anchor = paths.PROJECT_ROOT / "backend" / "app" / "data" / "paths.py"
    assert anchor.exists()
    assert paths.PROJECT_ROOT.name == "Backtesting APP"


def test_default_data_dir_is_repo_root_data():
    assert paths.data_dir() == paths.PROJECT_ROOT / "data"
    assert paths.candles_root() == paths.PROJECT_ROOT / "data" / "candles"
    assert paths.binance_raw_root() == paths.PROJECT_ROOT / "data" / "raw" / "binance"


def test_env_var_overrides_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.ENV_DATA_DIR, str(tmp_path / "custom"))
    assert paths.data_dir() == tmp_path / "custom"
    assert paths.candles_root() == tmp_path / "custom" / "candles"
    assert paths.binance_raw_root() == tmp_path / "custom" / "raw" / "binance"


def test_save_and_load_use_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.ENV_DATA_DIR, str(tmp_path))

    saved = save_candles(
        make_candles(), market="crypto", symbol="BTCUSDT", timeframe="1h"
    )
    assert saved == tmp_path / "candles" / "crypto" / "BTCUSDT" / "1h.parquet"
    assert saved.exists()

    loaded = load_candles(market="crypto", symbol="BTCUSDT", timeframe="1h")
    assert len(loaded) == 1


def test_save_explicit_root_beats_env(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.ENV_DATA_DIR, str(tmp_path / "env"))
    explicit = tmp_path / "explicit"

    saved = save_candles(
        make_candles(),
        explicit,
        market="crypto",
        symbol="BTCUSDT",
        timeframe="1h",
    )
    assert saved == explicit / "crypto" / "BTCUSDT" / "1h.parquet"
    assert not (tmp_path / "env").exists()


def test_binance_source_defaults_to_env_root(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.ENV_DATA_DIR, str(tmp_path))
    source = BinanceVisionSource()
    assert (
        source.raw_month_path("BTCUSDT", "1h", "2023-01")
        == tmp_path
        / "raw"
        / "binance"
        / "spot"
        / "monthly"
        / "klines"
        / "BTCUSDT"
        / "1h"
        / "BTCUSDT-1h-2023-01.zip"
    )


def test_binance_source_default_root_is_anchored_when_env_unset():
    source = BinanceVisionSource()
    expected = paths.PROJECT_ROOT / "data" / "raw" / "binance"
    assert source.root == expected
