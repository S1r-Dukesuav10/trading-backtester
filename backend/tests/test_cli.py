"""Tests for the data-layer CLI (no network: HTTP injected)."""

from __future__ import annotations

import pandas as pd

from app.data import paths
from app.data.cli import main
from tests.test_binance_vision import FakeHttp, monthly_files
from tests.test_dukascopy import FakeDukascopyCLI


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


# -- fetch-dukascopy ------------------------------------------------------


def test_fetch_dukascopy_writes_month_files(tmp_path, capsys):
    cli = FakeDukascopyCLI()

    exit_code = main(
        [
            "fetch-dukascopy",
            "--symbol",
            "XAUUSD",
            "--start",
            "2024-01",
            "--end",
            "2024-03",
            "--raw-root",
            str(tmp_path),
            "--delay",
            "0",
        ],
        run=cli,
    )

    assert exit_code == 0
    for month in ("2024-01", "2024-02", "2024-03"):
        assert (tmp_path / "XAUUSD" / f"XAUUSD-m1-bid-{month}.csv").exists()

    output = capsys.readouterr().out
    assert "3 month(s) requested" in output
    assert "downloaded: 3" in output
    assert "first 5 rows of XAUUSD-m1-bid-2024-01.csv" in output
    assert "last 3 rows of XAUUSD-m1-bid-2024-03.csv" in output


def test_fetch_dukascopy_is_resumable(tmp_path, capsys):
    cli = FakeDukascopyCLI()
    args = [
        "fetch-dukascopy",
        "--symbol",
        "XAUUSD",
        "--start",
        "2024-01",
        "--end",
        "2024-02",
        "--raw-root",
        str(tmp_path),
        "--delay",
        "0",
    ]

    assert main(args, run=cli) == 0
    assert len(cli.commands) == 2

    cli.commands.clear()
    assert main(args, run=cli) == 0  # second run: everything cached

    assert cli.commands == []
    assert "already present: 2" in capsys.readouterr().out


def test_fetch_dukascopy_reports_failures(tmp_path, capsys):
    cli = FakeDukascopyCLI(fail_months={"2024-02"})

    exit_code = main(
        [
            "fetch-dukascopy",
            "--start",
            "2024-01",
            "--end",
            "2024-02",
            "--raw-root",
            str(tmp_path),
            "--delay",
            "0",
            "--retry-wait",
            "0",
        ],
        run=cli,
    )

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "FAILED 2024-02" in captured.out
    assert "re-run to retry" in captured.err


def test_fetch_dukascopy_defaults_honor_env_dir(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.ENV_DATA_DIR, str(tmp_path))
    cli = FakeDukascopyCLI()

    exit_code = main(
        [
            "fetch-dukascopy",
            "--start",
            "2024-01",
            "--end",
            "2024-01",
            "--delay",
            "0",
        ],
        run=cli,
    )

    assert exit_code == 0
    assert (
        tmp_path / "raw" / "dukascopy" / "XAUUSD" / "XAUUSD-m1-bid-2024-01.csv"
    ).exists()
