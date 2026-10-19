"""Tests for BinanceVisionSource. No network: HTTP is faked, zips are synthetic."""

from __future__ import annotations

import urllib.error
from urllib.request import Request  # noqa: F401  (kept for HTTPError shape parity)

import pandas as pd
import pytest

from app.data.binance_vision import (
    BinanceVisionError,
    BinanceVisionSource,
    MissingMonthError,
    detect_timestamp_unit,
    kline_filename,
    month_range,
    parse_kline_csv,
    parse_kline_zip,
    resolve_bounds,
)
from tests.helpers import binance_row, make_binance_zip

HOUR_MS = 3_600_000
HOUR_US = 3_600_000_000
HOUR_S = 3_600
JAN_2023_S = 1_672_531_200  # 2023-01-01 00:00:00 UTC


class FakeHttp:
    """Callable standing in for the HTTP layer; records every URL."""

    def __init__(self, files: dict[str, bytes], missing: set[str] | None = None):
        self.files = files
        self.missing = set(missing or ())
        self.calls: list[str] = []

    def __call__(self, url: str) -> bytes:
        self.calls.append(url)
        name = url.rsplit("/", 1)[-1]
        if name in self.missing:
            raise urllib.error.HTTPError(url, 404, "Not Found", None, None)
        if name not in self.files:
            raise AssertionError(f"unexpected URL requested: {url}")
        return self.files[name]


def monthly_files(
    months: list[str],
    *,
    unit: str = "ms",
    header: bool = False,
    symbol: str = "BTCUSDT",
    interval: str = "1h",
) -> dict[str, bytes]:
    """One synthetic monthly zip per month, with 2 hours of data each."""
    files: dict[str, bytes] = {}
    for index, month in enumerate(months):
        start_s = JAN_2023_S + index * 31 * 24 * 3600
        if unit == "s":
            base, step = start_s, HOUR_S
        elif unit == "ms":
            base, step = start_s * 1000, HOUR_MS
        elif unit == "us":
            base, step = start_s * 1_000_000, HOUR_US
        else:  # pragma: no cover
            raise ValueError(unit)
        rows = [binance_row(base), binance_row(base + step, close=101.5)]
        files[kline_filename(symbol, interval, month)] = make_binance_zip(
            rows, header=header
        )
    return files


def make_source(
    files: dict[str, bytes], tmp_path, **kwargs
) -> tuple[BinanceVisionSource, FakeHttp]:
    http = FakeHttp(files)
    source = BinanceVisionSource(
        tmp_path / "raw", http_get=http, sleep=lambda _seconds: None, **kwargs
    )
    return source, http


# -- unit detection --------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1_672_531_200, "s"),
        (1_672_531_200_000, "ms"),
        (1_672_531_200_000_000, "us"),
        (1_672_531_200_000_000_000, "ns"),
    ],
)
def test_detect_timestamp_unit(value, expected):
    assert detect_timestamp_unit(value) == expected


def test_detect_timestamp_unit_rejects_non_positive():
    with pytest.raises(ValueError, match="positive"):
        detect_timestamp_unit(0)


# -- parsing ---------------------------------------------------------------


def test_parse_zip_without_header():
    raw = make_binance_zip(
        [binance_row(JAN_2023_S * 1000), binance_row(JAN_2023_S * 1000 + HOUR_MS)]
    )
    candles = parse_kline_zip(raw)
    assert list(candles.columns) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert candles["timestamp"].iloc[0] == pd.Timestamp("2023-01-01 00:00:00", tz="UTC")
    assert str(candles["timestamp"].dt.tz) == "UTC"


def test_parse_zip_with_header():
    raw = make_binance_zip([binance_row(JAN_2023_S * 1000)], header=True)
    candles = parse_kline_zip(raw)
    assert len(candles) == 1
    assert candles["close"].iloc[0] == 100.5


def test_parse_zip_with_reordered_columns():
    """Inconsistent header order must still map by name, not position."""
    header = "volume,close_time,open_time,close,high,low,open,ignore"
    row = f"10.0,{JAN_2023_S * 1000 + HOUR_MS - 1},{JAN_2023_S * 1000},100.5,101.0,99.0,100.0,0"
    text = f"{header}\n{row}\n"
    import io
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("BTCUSDT-1h-2023-01.csv", text)

    candles = parse_kline_zip(buffer.getvalue())
    assert candles["timestamp"].iloc[0] == pd.Timestamp("2023-01-01 00:00:00", tz="UTC")
    assert candles["open"].iloc[0] == 100.0
    assert candles["close"].iloc[0] == 100.5
    assert candles["volume"].iloc[0] == 10.0


def test_parse_csv_rejects_empty():
    with pytest.raises(BinanceVisionError, match="Empty kline CSV"):
        parse_kline_csv("\n\n")


def test_parse_zip_rejects_corrupt_archive():
    with pytest.raises(BinanceVisionError, match="Corrupt zip"):
        parse_kline_zip(b"definitely not a zip")


def test_parse_zip_rejects_archive_without_csv():
    import io
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("readme.txt", "hi")
    with pytest.raises(BinanceVisionError, match="No CSV found"):
        parse_kline_zip(buffer.getvalue())


# -- unit handling end to end ---------------------------------------------


@pytest.mark.parametrize("unit", ["s", "ms", "us"])
def test_downloads_each_timestamp_unit(unit, tmp_path):
    files = monthly_files(["2023-01"], unit=unit)
    source, http = make_source(files, tmp_path)

    candles = source.get_candles("BTCUSDT", "1h", "2023-01", "2023-01")

    assert len(http.calls) == 1
    assert len(candles) == 2
    assert candles["timestamp"].iloc[0] == pd.Timestamp("2023-01-01 00:00:00", tz="UTC")
    assert candles["timestamp"].iloc[1] == pd.Timestamp("2023-01-01 01:00:00", tz="UTC")
    assert candles["timestamp"].is_monotonic_increasing


def test_all_units_produce_identical_candles(tmp_path):
    frames = []
    for unit in ("s", "ms", "us"):
        files = monthly_files(["2023-01"], unit=unit)
        source, _ = make_source(files, tmp_path / unit)
        frames.append(source.get_candles("BTCUSDT", "1h", "2023-01", "2023-01"))
    pd.testing.assert_frame_equal(frames[0], frames[1])
    pd.testing.assert_frame_equal(frames[1], frames[2])


def test_header_and_headerless_files_agree(tmp_path):
    without = monthly_files(["2023-01"], header=False)
    with_header = monthly_files(["2023-01"], header=True)

    src_a, _ = make_source(without, tmp_path / "a")
    src_b, _ = make_source(with_header, tmp_path / "b")

    pd.testing.assert_frame_equal(
        src_a.get_candles("BTCUSDT", "1h", "2023-01", "2023-01"),
        src_b.get_candles("BTCUSDT", "1h", "2023-01", "2023-01"),
    )


# -- URLs, caching, politeness -------------------------------------------


def test_url_and_cache_layout(tmp_path):
    source = BinanceVisionSource(tmp_path / "raw")
    url = source.month_url("BTCUSDT", "1h", "2023-01")
    assert url == (
        "https://data.binance.vision/data/spot/monthly/klines/"
        "BTCUSDT/1h/BTCUSDT-1h-2023-01.zip"
    )
    path = source.raw_month_path("BTCUSDT", "1h", "2023-01")
    assert path == (
        tmp_path
        / "raw"
        / "spot"
        / "monthly"
        / "klines"
        / "BTCUSDT"
        / "1h"
        / "BTCUSDT-1h-2023-01.zip"
    )


def test_zip_is_cached_and_second_run_skips_network(tmp_path):
    files = monthly_files(["2023-01", "2023-02"])
    source, http = make_source(files, tmp_path)

    first = source.get_candles("BTCUSDT", "1h", "2023-01", "2023-02")
    assert len(http.calls) == 2

    cached_path = source.raw_month_path("BTCUSDT", "1h", "2023-01")
    assert cached_path.exists()
    assert not (
        tmp_path
        / "raw"
        / "spot"
        / "monthly"
        / "klines"
        / "BTCUSDT"
        / "1h"
        / "BTCUSDT-1h-2023-01.zip.part"
    ).exists()

    second = source.get_candles("BTCUSDT", "1h", "2023-01", "2023-02")
    assert len(http.calls) == 2  # no new requests
    pd.testing.assert_frame_equal(first, second)


def test_delay_between_requests_only(tmp_path):
    slept: list[float] = []
    files = monthly_files(["2023-01", "2023-02", "2023-03"])
    http = FakeHttp(files)
    source = BinanceVisionSource(
        tmp_path / "raw", delay_seconds=0.25, http_get=http, sleep=slept.append
    )

    source.get_candles("BTCUSDT", "1h", "2023-01", "2023-03")

    # Three downloads -> two pauses (never before the first request).
    assert slept == [0.25, 0.25]


def test_missing_month_raises_clear_error(tmp_path):
    files = monthly_files(["2023-01"])
    http = FakeHttp(files, missing={"BTCUSDT-1h-2023-02.zip"})
    source = BinanceVisionSource(tmp_path / "raw", http_get=http, sleep=lambda _s: None)

    with pytest.raises(MissingMonthError) as excinfo:
        source.get_candles("BTCUSDT", "1h", "2023-01", "2023-02")

    message = str(excinfo.value)
    assert "BTCUSDT 1h 2023-02" in message
    assert "404" in message


def test_other_http_errors_wrapped(tmp_path):
    def http(url: str) -> bytes:
        raise urllib.error.HTTPError(url, 500, "Server Error", None, None)

    source = BinanceVisionSource(tmp_path / "raw", http_get=http, sleep=lambda _s: None)
    with pytest.raises(BinanceVisionError, match="HTTP 500"):
        source.get_candles("BTCUSDT", "1h", "2023-01", "2023-01")


# -- range handling --------------------------------------------------------


def test_month_range_inclusive():
    assert month_range("2023-01", "2023-03") == ["2023-01", "2023-02", "2023-03"]
    assert month_range("2023-01", "2023-01") == ["2023-01"]


def test_resolve_bounds_month_string_end_is_inclusive():
    lower, upper = resolve_bounds("2023-01", "2023-02")
    assert lower == pd.Timestamp("2023-01-01 00:00:00", tz="UTC")
    assert upper == pd.Timestamp("2023-03-01 00:00:00", tz="UTC")


def test_resolve_bounds_timestamp_end_is_exclusive():
    lower, upper = resolve_bounds("2023-01", "2023-02-15")
    assert lower == pd.Timestamp("2023-01-01 00:00:00", tz="UTC")
    assert upper == pd.Timestamp("2023-02-15 00:00:00", tz="UTC")


def test_get_candles_requires_both_bounds(tmp_path):
    source = BinanceVisionSource(tmp_path / "raw")
    with pytest.raises(ValueError, match="requires both start and end"):
        source.get_candles("BTCUSDT", "1h")
    with pytest.raises(ValueError, match="requires both start and end"):
        source.get_candles("BTCUSDT", "1h", start="2023-01")


def test_get_candles_rejects_reversed_range(tmp_path):
    source = BinanceVisionSource(tmp_path / "raw")
    with pytest.raises(ValueError, match="before start"):
        source.get_candles("BTCUSDT", "1h", "2023-05", "2023-01")


def test_get_candles_filters_within_month(tmp_path):
    files = monthly_files(["2023-01"])
    source, _ = make_source(files, tmp_path)

    # Both January rows are at 00:00 and 01:00 on 2023-01-01.
    candles = source.get_candles("BTCUSDT", "1h", "2023-01", "2023-01-01 01:00")
    assert len(candles) == 1
    assert candles["timestamp"].iloc[0] == pd.Timestamp("2023-01-01 00:00:00", tz="UTC")
