"""Tests for the data-quality report. Synthetic data only; no network."""

from __future__ import annotations

import pandas as pd
import pytest

from app.data.quality import (
    analyze_candles,
    check_dataset,
    find_datasets,
    format_dataset_list,
    format_quality_report,
)
from app.data.storage import candles_path, save_candles


def hourly_frame(count: int = 5, *, start: str = "2024-01-01") -> pd.DataFrame:
    """A clean hourly frame with ``count`` bars."""
    timestamps = pd.date_range(start, periods=count, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0] * count,
            "high": [101.0] * count,
            "low": [99.0] * count,
            "close": [100.5] * count,
            "volume": [10.0] * count,
        }
    )


def frame_from(timestamps: list[str]) -> pd.DataFrame:
    """An hourly-ish frame from explicit ISO timestamps."""
    stamps = pd.to_datetime(timestamps, utc=True)
    count = len(stamps)
    return pd.DataFrame(
        {
            "timestamp": stamps,
            "open": [100.0] * count,
            "high": [101.0] * count,
            "low": [99.0] * count,
            "close": [100.5] * count,
            "volume": [10.0] * count,
        }
    )


def weekend_frame() -> pd.DataFrame:
    """Friday close, Sunday reopen, plus a midweek gap."""
    return frame_from(
        [
            "2024-01-05T20:00:00Z",  # Friday
            "2024-01-05T21:00:00Z",  # last bar before the weekend closure
            "2024-01-07T22:00:00Z",  # Sunday reopen
            "2024-01-07T23:00:00Z",
            "2024-01-08T00:00:00Z",  # Monday
            "2024-01-08T03:00:00Z",  # 01:00 and 02:00 missing
        ]
    )


def write_dataset(
    root,
    *,
    market: str = "crypto",
    symbol: str = "BTCUSDT",
    timeframe: str = "1h",
    frame: pd.DataFrame | None = None,
):
    """Store a (valid) dataset under the candles root and return its path."""
    return save_candles(
        frame if frame is not None else hourly_frame(),
        root,
        market=market,
        symbol=symbol,
        timeframe=timeframe,
    )


def write_raw_dataset(root, frame: pd.DataFrame, *, symbol: str = "BTCUSDT"):
    """Store a frame *without* validation (needed for intentionally bad data)."""
    path = candles_path(root, "crypto", symbol, "1h")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


# -- clean data -----------------------------------------------------------


def test_clean_dataset_is_ok():
    report = analyze_candles(hourly_frame(5))

    assert report.rows == 5
    assert report.start == pd.Timestamp("2024-01-01T00:00:00Z")
    assert report.end == pd.Timestamp("2024-01-01T04:00:00Z")
    assert report.step == pd.Timedelta("1h")
    assert report.step_given is False
    assert report.duplicate_rows == 0
    assert report.unsorted_rows == 0
    assert report.invalid == []
    assert report.zero_volume_rows == 0
    assert report.missing_volume_rows == 0
    assert report.zero_range_rows == 0
    assert report.gaps == []
    assert report.ok
    assert report.errors == []
    assert report.warnings == []

    text = format_quality_report(report)
    assert "Quality report: dataset" in text
    assert "rows: 5" in text
    assert "step:  1h (inferred)" in text
    assert "result: OK" in text


# -- timestamps -----------------------------------------------------------


def test_duplicate_timestamps_are_errors():
    frame = hourly_frame(4)
    frame.loc[3, "timestamp"] = frame.loc[2, "timestamp"]

    report = analyze_candles(frame)

    assert report.duplicate_rows == 2
    assert report.duplicate_timestamp_count == 1
    assert report.duplicate_examples == [pd.Timestamp("2024-01-01T02:00:00Z")]
    assert not report.ok
    assert any("duplicate" in error for error in report.errors)

    text = format_quality_report(report)
    assert "ERROR" in text
    assert "2024-01-01T02:00:00Z" in text


def test_unsorted_rows_are_errors():
    frame = hourly_frame(5).iloc[[0, 1, 4, 2, 3]].reset_index(drop=True)

    report = analyze_candles(frame)

    assert report.unsorted_rows == 1
    assert report.unsorted_examples == [pd.Timestamp("2024-01-01T02:00:00Z")]
    assert report.gaps == []  # the gap check works on a sorted copy
    assert not report.ok


def test_out_of_order_data_still_reports_gaps():
    frame = hourly_frame(6).drop(index=[2, 3]).reset_index(drop=True)
    frame = frame.iloc[[0, 1, 3, 2]].reset_index(drop=True)  # last two swapped

    report = analyze_candles(frame)

    assert report.unsorted_rows == 1
    assert [gap.start for gap in report.gaps] == [pd.Timestamp("2024-01-01T02:00:00Z")]


# -- gaps -----------------------------------------------------------------


def test_gap_start_and_size():
    frame = hourly_frame(6).drop(index=[2, 3]).reset_index(drop=True)

    report = analyze_candles(frame)

    assert len(report.gaps) == 1
    gap = report.gaps[0]
    assert gap.start == pd.Timestamp("2024-01-01T02:00:00Z")
    assert gap.end == pd.Timestamp("2024-01-01T04:00:00Z")
    assert gap.missing == pd.Timedelta("2h")
    assert gap.bars == 2
    assert report.ok  # a gap is a warning, not an error

    text = format_quality_report(report)
    assert "start 2024-01-01T02:00:00Z" in text
    assert "2 bars missing (2h)" in text
    assert "result: 0 errors, 1 warning" in text


def test_every_gap_is_listed():
    frame = hourly_frame(7).drop(index=[1, 4]).reset_index(drop=True)

    report = analyze_candles(frame)

    assert [gap.start for gap in report.gaps] == [
        pd.Timestamp("2024-01-01T01:00:00Z"),
        pd.Timestamp("2024-01-01T04:00:00Z"),
    ]
    assert [gap.bars for gap in report.gaps] == [1, 1]

    text = format_quality_report(report)
    assert "gaps (2)" in text
    assert text.count("    - start ") == 2


def test_single_bar_gap_is_worded_singular():
    frame = hourly_frame(4).drop(index=[1]).reset_index(drop=True)

    report = analyze_candles(frame)

    assert "1 bar missing (1h)" in format_quality_report(report)


def test_weekend_gaps_can_be_ignored():
    frame = weekend_frame()

    plain = analyze_candles(frame)
    assert len(plain.gaps) == 2
    weekend_gap = plain.gaps[0]
    assert weekend_gap.start == pd.Timestamp("2024-01-05T22:00:00Z")
    assert weekend_gap.end == pd.Timestamp("2024-01-07T22:00:00Z")
    assert weekend_gap.missing == pd.Timedelta("48h")
    assert weekend_gap.bars == 48
    assert plain.weekend_gaps_ignored == 0

    filtered = analyze_candles(frame, ignore_weekends=True)
    assert filtered.weekend_gaps_ignored == 1
    assert len(filtered.gaps) == 1
    assert filtered.gaps[0].start == pd.Timestamp("2024-01-08T01:00:00Z")

    text = format_quality_report(filtered)
    assert "weekend gaps ignored: 1" in text
    # The weekday gap is still reported; only the weekend one was dropped.
    assert "2024-01-08T01:00:00Z" in text


def test_weekend_handling_only_suppresses_full_weekend_gaps():
    # Friday 23:00 -> Sunday 23:00 spills past the Sunday 22:00 reopen, so it
    # is *not* a pure weekend gap even with --ignore-weekends.
    frame = frame_from(
        [
            "2024-01-05T22:00:00Z",
            "2024-01-05T23:00:00Z",
            "2024-01-07T23:00:00Z",
            "2024-01-08T00:00:00Z",
        ]
    )

    report = analyze_candles(frame, ignore_weekends=True)

    assert report.weekend_gaps_ignored == 0
    assert len(report.gaps) == 1
    assert report.gaps[0].start == pd.Timestamp("2024-01-06T00:00:00Z")


def test_explicit_step_overrides_inference():
    frame = hourly_frame(5).iloc[::2].reset_index(drop=True)  # 00, 02, 04

    inferred = analyze_candles(frame)
    assert inferred.step == pd.Timedelta("2h")
    assert inferred.step_given is False
    assert inferred.gaps == []

    explicit = analyze_candles(frame, step="1h")
    assert explicit.step == pd.Timedelta("1h")
    assert explicit.step_given is True
    assert [gap.start for gap in explicit.gaps] == [
        pd.Timestamp("2024-01-01T01:00:00Z"),
        pd.Timestamp("2024-01-01T03:00:00Z"),
    ]
    assert explicit.gaps[0].bars == 1
    assert "1h (given)" in format_quality_report(explicit)


def test_no_gap_detection_with_a_single_bar():
    report = analyze_candles(hourly_frame(1))

    assert report.rows == 1
    assert report.step is None
    assert report.gaps == []


# -- candles --------------------------------------------------------------


def test_invalid_candles_are_detected():
    frame = hourly_frame(4)
    frame.loc[1, "low"] = 100.5  # low above open
    frame.loc[2, "high"] = 99.5  # high below open

    report = analyze_candles(frame)

    assert len(report.invalid) == 2
    assert report.invalid[0].timestamp == pd.Timestamp("2024-01-01T01:00:00Z")
    assert report.invalid[0].reasons == ("low above open/close",)
    assert report.invalid[1].reasons == ("high below open/close",)
    assert not report.ok

    text = format_quality_report(report)
    assert "invalid candles" in text
    assert "low above open/close" in text
    assert "high below open/close" in text


def test_missing_price_is_invalid():
    frame = hourly_frame(3)
    frame.loc[1, "open"] = float("nan")

    report = analyze_candles(frame)

    assert [candle.reasons for candle in report.invalid] == [("missing price data",)]
    assert not report.ok


def test_low_above_high_is_invalid():
    frame = hourly_frame(3)
    frame.loc[1, "low"] = 102.0  # above high (101) and above open/close

    report = analyze_candles(frame)

    assert "low above high" in report.invalid[0].reasons


def test_zero_range_candles_are_warnings():
    frame = hourly_frame(4)
    frame.loc[1, ["open", "high", "low", "close"]] = 100.0

    report = analyze_candles(frame)

    assert report.zero_range_rows == 1
    assert report.zero_range_examples == [pd.Timestamp("2024-01-01T01:00:00Z")]
    assert report.ok  # warning only
    assert any("zero-range" in warning for warning in report.warnings)
    assert "WARN" in format_quality_report(report)


def test_zero_and_missing_volume():
    frame = hourly_frame(4)
    frame.loc[1, "volume"] = 0.0
    frame.loc[2, "volume"] = float("nan")

    report = analyze_candles(frame)

    assert report.zero_volume_rows == 1
    assert report.zero_volume_examples == [pd.Timestamp("2024-01-01T01:00:00Z")]
    assert report.missing_volume_rows == 1
    assert report.missing_volume_examples == [pd.Timestamp("2024-01-01T02:00:00Z")]
    assert not report.ok  # missing volume is an error
    assert any("zero-volume" in warning for warning in report.warnings)
    assert any("missing volume" in error for error in report.errors)


def test_example_limit_caps_listed_timestamps():
    frame = hourly_frame(6)
    frame.loc[1, "volume"] = 0.0
    frame.loc[2, "volume"] = 0.0
    frame.loc[3, "volume"] = 0.0

    report = analyze_candles(frame, examples=2)

    assert report.zero_volume_rows == 3
    assert len(report.zero_volume_examples) == 2
    text = format_quality_report(report)
    assert "first 2 of 3" in text


# -- bad input ------------------------------------------------------------


def test_empty_dataset_is_an_error():
    report = analyze_candles(hourly_frame(0))

    assert report.rows == 0
    assert report.start is None
    assert not report.ok
    assert "no rows" in " ".join(report.errors)
    assert "rows: 0" in format_quality_report(report)


def test_missing_columns_raise():
    with pytest.raises(ValueError, match="missing required columns"):
        analyze_candles(hourly_frame(3).drop(columns=["volume"]))


def test_naive_timestamps_raise():
    frame = hourly_frame(3)
    frame["timestamp"] = frame["timestamp"].dt.tz_localize(None)

    with pytest.raises(ValueError, match="timezone-naive"):
        analyze_candles(frame)


def test_non_numeric_prices_raise():
    frame = hourly_frame(3)
    frame["close"] = ["a", "b", "c"]

    with pytest.raises(ValueError, match="not numeric"):
        analyze_candles(frame)


def test_invalid_step_raises():
    with pytest.raises(ValueError, match="invalid step"):
        analyze_candles(hourly_frame(3), step="nonsense")


# -- stored datasets ------------------------------------------------------


def test_check_dataset_reads_stored_parquet(tmp_path):
    frame = hourly_frame(4)
    frame.loc[3, "timestamp"] = frame.loc[2, "timestamp"]
    path = write_raw_dataset(tmp_path, frame)

    report = check_dataset(path, market="crypto", symbol="BTCUSDT", timeframe="1h")

    assert report.label == "crypto/BTCUSDT/1h"
    assert report.path == path
    assert report.size_bytes == path.stat().st_size
    assert report.duplicate_rows == 2
    assert not report.ok


def test_find_datasets_lists_and_sorts(tmp_path):
    write_dataset(tmp_path, symbol="ETHUSDT", frame=hourly_frame(3))
    write_dataset(tmp_path, symbol="BTCUSDT", frame=hourly_frame(5))
    write_dataset(tmp_path, market="forex", symbol="XAUUSD")

    datasets = find_datasets(tmp_path)

    assert [info.key for info in datasets] == [
        "crypto/BTCUSDT/1h",
        "crypto/ETHUSDT/1h",
        "forex/XAUUSD/1h",
    ]
    btc = datasets[0]
    assert btc.rows == 5
    assert btc.start == pd.Timestamp("2024-01-01T00:00:00Z")
    assert btc.end == pd.Timestamp("2024-01-01T04:00:00Z")
    assert btc.error is None


def test_find_datasets_reports_unreadable_files(tmp_path):
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")
    broken = tmp_path / "crypto" / "BROKEN" / "1h.parquet"
    broken.parent.mkdir(parents=True)
    broken.write_text("not parquet", encoding="utf-8")

    datasets = find_datasets(tmp_path)

    assert [info.key for info in datasets] == ["crypto/BROKEN/1h"]
    assert datasets[0].error is not None

    text = format_dataset_list(datasets, root=tmp_path)
    assert "UNREADABLE" in text


def test_find_datasets_empty_root(tmp_path):
    assert find_datasets(tmp_path) == []
    assert "No datasets" in format_dataset_list([], root=tmp_path)


def test_format_dataset_list_shows_range_rows_and_size(tmp_path):
    write_dataset(tmp_path, frame=hourly_frame(5))

    text = format_dataset_list(find_datasets(tmp_path), root=tmp_path)

    assert "crypto/BTCUSDT/1h" in text
    assert "5 rows" in text
    assert "2024-01-01T00:00:00Z .. 2024-01-01T04:00:00Z" in text
    assert "(1)" in text


def test_empty_dataset_row_is_shown_as_dash(tmp_path):
    write_dataset(tmp_path, frame=hourly_frame(0))

    text = format_dataset_list(find_datasets(tmp_path), root=tmp_path)

    assert "0 rows" in text
    assert "  -  " in text
