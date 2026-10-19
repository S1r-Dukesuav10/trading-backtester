"""Tests for the data-layer CLI (no network: HTTP injected)."""

from __future__ import annotations

import pandas as pd

from app.data import paths
from app.data.cli import main
from tests.test_binance_vision import FakeHttp, monthly_files


def test_fetch_binance_writes_canonical_parquet(tmp_path, capsys):
    files = monthly_files(["2023-01", "2023-02"])
    http = FakeHttp(files)
    data_root = tmp_path / "candles"

    exit_code = main(
        [
            "fetch-binance",
            "--symbol",
            "BTCUSDT",
            "--interval",
            "1h",
            "--start",
            "2023-01",
            "--end",
            "2023-02",
            "--raw-root",
            str(tmp_path / "raw"),
            "--data-root",
            str(data_root),
            "--delay",
            "0",
        ],
        http_get=http,
    )

    assert exit_code == 0
    expected = data_root / "crypto" / "BTCUSDT" / "1h.parquet"
    assert expected.exists()

    candles = pd.read_parquet(expected)
    assert len(candles) == 4  # two synthetic hours per month, two months
    assert str(candles["timestamp"].dt.tz) == "UTC"
    assert candles["timestamp"].is_monotonic_increasing

    output = capsys.readouterr().out
    assert "4 candles" in output
    assert str(expected) in output


def test_fetch_binance_reports_missing_month(tmp_path, capsys):
    files = monthly_files(["2023-01"])
    http = FakeHttp(files, missing={"BTCUSDT-1h-2023-02.zip"})

    exit_code = main(
        [
            "fetch-binance",
            "--symbol",
            "BTCUSDT",
            "--interval",
            "1h",
            "--start",
            "2023-01",
            "--end",
            "2023-02",
            "--raw-root",
            str(tmp_path / "raw"),
            "--data-root",
            str(tmp_path / "candles"),
            "--delay",
            "0",
        ],
        http_get=http,
    )

    assert exit_code == 1
    error = capsys.readouterr().err
    assert "error:" in error
    assert "2023-02" in error


def test_fetch_binance_reports_bad_range(tmp_path, capsys):
    exit_code = main(
        [
            "fetch-binance",
            "--symbol",
            "BTCUSDT",
            "--interval",
            "1h",
            "--start",
            "2024-01",
            "--end",
            "2023-01",
            "--raw-root",
            str(tmp_path / "raw"),
            "--data-root",
            str(tmp_path / "candles"),
        ],
        http_get=FakeHttp({}),
    )

    assert exit_code == 1
    assert "before start" in capsys.readouterr().err


def test_fetch_binance_defaults_honor_env_dir(monkeypatch, tmp_path, capsys):
    """Without --raw-root/--data-root the CLI uses $BACKTESTER_DATA_DIR."""
    monkeypatch.setenv(paths.ENV_DATA_DIR, str(tmp_path))
    files = monthly_files(["2023-01"])
    http = FakeHttp(files)

    exit_code = main(
        [
            "fetch-binance",
            "--symbol",
            "BTCUSDT",
            "--interval",
            "1h",
            "--start",
            "2023-01",
            "--end",
            "2023-01",
            "--delay",
            "0",
        ],
        http_get=http,
    )

    assert exit_code == 0
    assert (tmp_path / "candles" / "crypto" / "BTCUSDT" / "1h.parquet").exists()
    assert (
        tmp_path
        / "raw"
        / "binance"
        / "spot"
        / "monthly"
        / "klines"
        / "BTCUSDT"
        / "1h"
        / "BTCUSDT-1h-2023-01.zip"
    ).exists()
