"""Tests for the data-layer CLI (no network: HTTP injected)."""

from __future__ import annotations

import pandas as pd

from app.data import paths
from app.data.cli import main
from app.data.storage import candles_path, save_candles
from tests.test_binance_vision import FakeHttp, monthly_files
from tests.test_dukascopy import FakeDukascopyCLI
from tests.test_quality import (
    hourly_frame,
    weekend_frame,
    write_dataset,
    write_raw_dataset,
)


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


# -- report / list --------------------------------------------------------


def test_report_clean_dataset(tmp_path, capsys):
    write_dataset(tmp_path, frame=hourly_frame(5))

    exit_code = main(
        [
            "report",
            "--market",
            "crypto",
            "--symbol",
            "BTCUSDT",
            "--timeframe",
            "1h",
            "--data-root",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Quality report: crypto/BTCUSDT/1h" in output
    assert "rows: 5" in output
    assert "result: OK" in output


def test_report_flags_bad_data(tmp_path, capsys):
    # 00,01,02,04,05,05 -> a gap at 03:00 and a duplicate at 05:00
    frame = hourly_frame(7).drop(index=[3]).reset_index(drop=True)
    frame.loc[5, "timestamp"] = frame.loc[4, "timestamp"]
    write_raw_dataset(tmp_path, frame)

    exit_code = main(
        [
            "report",
            "--market",
            "crypto",
            "--symbol",
            "BTCUSDT",
            "--timeframe",
            "1h",
            "--data-root",
            str(tmp_path),
        ]
    )

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "ERROR" in output
    assert "duplicate timestamps" in output
    assert "gaps (1)" in output
    assert "start 2024-01-01T03:00:00Z" in output


def test_report_missing_dataset(tmp_path, capsys):
    exit_code = main(
        [
            "report",
            "--market",
            "crypto",
            "--symbol",
            "NOPE",
            "--timeframe",
            "1h",
            "--data-root",
            str(tmp_path),
        ]
    )

    assert exit_code == 1
    assert "no dataset" in capsys.readouterr().err


def test_report_corrupt_dataset(tmp_path, capsys):
    path = candles_path(tmp_path, "crypto", "BTCUSDT", "1h")
    path.parent.mkdir(parents=True)
    path.write_text("not parquet", encoding="utf-8")

    exit_code = main(
        [
            "report",
            "--market",
            "crypto",
            "--symbol",
            "BTCUSDT",
            "--timeframe",
            "1h",
            "--data-root",
            str(tmp_path),
        ]
    )

    assert exit_code == 1
    assert "could not analyse" in capsys.readouterr().err


def test_report_ignore_weekends_flag(tmp_path, capsys):
    save_candles(
        weekend_frame(), tmp_path, market="forex", symbol="XAUUSD", timeframe="1h"
    )
    args = [
        "report",
        "--market",
        "forex",
        "--symbol",
        "XAUUSD",
        "--timeframe",
        "1h",
        "--data-root",
        str(tmp_path),
    ]

    assert main(args) == 0  # gaps are warnings
    output = capsys.readouterr().out
    assert "gaps (2)" in output
    assert "weekend gaps ignored" not in output

    assert main([*args, "--ignore-weekends"]) == 0
    output = capsys.readouterr().out
    assert "gaps (1)" in output
    assert "weekend gaps ignored: 1" in output


def test_report_step_override(tmp_path, capsys):
    frame = hourly_frame(5).iloc[::2].reset_index(drop=True)  # 2h spacing
    save_candles(frame, tmp_path, market="crypto", symbol="BTCUSDT", timeframe="1h")
    args = [
        "report",
        "--market",
        "crypto",
        "--symbol",
        "BTCUSDT",
        "--timeframe",
        "1h",
        "--data-root",
        str(tmp_path),
    ]

    assert main(args) == 0
    output = capsys.readouterr().out
    assert "2h (inferred)" in output
    assert "gaps (" not in output

    assert main([*args, "--step", "1h"]) == 0
    output = capsys.readouterr().out
    assert "1h (given)" in output
    assert "gaps (2)" in output


def test_list_datasets(tmp_path, capsys):
    write_dataset(tmp_path, frame=hourly_frame(5))
    write_dataset(tmp_path, symbol="ETHUSDT", frame=hourly_frame(3))

    exit_code = main(["list", "--data-root", str(tmp_path)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Datasets under" in output
    assert "(2)" in output
    assert "crypto/BTCUSDT/1h" in output
    assert "crypto/ETHUSDT/1h" in output
    assert "5 rows" in output


def test_list_empty_root(tmp_path, capsys):
    assert main(["list", "--data-root", str(tmp_path)]) == 0
    assert "No datasets" in capsys.readouterr().out


def test_report_and_list_honor_env_dir(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.ENV_DATA_DIR, str(tmp_path))
    save_candles(
        hourly_frame(4), None, market="crypto", symbol="BTCUSDT", timeframe="1h"
    )

    assert (
        main(
            [
                "report",
                "--market",
                "crypto",
                "--symbol",
                "BTCUSDT",
                "--timeframe",
                "1h",
            ]
        )
        == 0
    )
    assert main(["list"]) == 0
