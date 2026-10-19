"""Data-quality checks for stored candle datasets.

Reads the canonical Parquet files *as stored* — no validation, no sorting —
so defects such as duplicate or out-of-order timestamps are still visible.

Severity:

- **errors** mean the data is not trustworthy as-is:
  duplicate timestamps, unsorted rows, impossible candles (low above
  open/close, high below open/close, missing prices), missing (NaN) volume,
  and empty datasets.
- **warnings** are worth knowing but do not necessarily invalidate the data:
  gaps, zero-volume candles, zero-range candles.

Gap checking:
the expected bar spacing is inferred from the most common timestamp
difference (or supplied explicitly). ``ignore_weekends`` suppresses only gaps
that lie *entirely* inside the weekend closure window (Friday 21:00 UTC to
Sunday 22:00 UTC by default) — the pattern used by forex/gold venues. Any gap
that spills into weekday trading is still reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import time
from pathlib import Path

import pandas as pd

from .paths import candles_root
from .resample import normalize_rule

REQUIRED_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
)
MAX_EXAMPLES = 5


# -- result types ---------------------------------------------------------


@dataclass(frozen=True)
class WeekendWindow:
    """A weekly trading closure, in UTC.

    Defaults match the typical forex/gold schedule: markets close on Friday
    at 21:00 UTC and reopen on Sunday at 22:00 UTC.
    """

    close_weekday: int = 4  # Monday=0 ... Friday=4
    close_time: time = time(21, 0)
    open_weekday: int = 6  # Sunday
    open_time: time = time(22, 0)

    def window_for(self, moment: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
        """The closure window belonging to ``moment``'s week."""
        days_since_close = (moment.weekday() - self.close_weekday) % 7
        close_day = moment.normalize() - pd.Timedelta(days=days_since_close)
        open_day = close_day + pd.Timedelta(
            days=(self.open_weekday - self.close_weekday) % 7
        )
        start = pd.Timestamp(f"{close_day.date()} {self.close_time}").tz_localize("UTC")
        end = pd.Timestamp(f"{open_day.date()} {self.open_time}").tz_localize("UTC")
        return start, end

    def covers(self, start: pd.Timestamp, end: pd.Timestamp) -> bool:
        """True when ``[start, end)`` is entirely inside one closure window."""
        window_start, window_end = self.window_for(start)
        return window_start <= start and end <= window_end


@dataclass(frozen=True)
class Gap:
    """A stretch of expected bars that is missing from the dataset."""

    start: pd.Timestamp  # expected timestamp of the first missing bar
    end: pd.Timestamp  # timestamp of the next bar that is present
    missing: pd.Timedelta  # total missing duration (end - start)
    bars: int  # number of whole bars missing at the expected step


@dataclass(frozen=True)
class InvalidCandle:
    timestamp: pd.Timestamp
    reasons: tuple[str, ...]


@dataclass
class QualityReport:
    """All findings for one dataset."""

    rows: int = 0
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    step: pd.Timedelta | None = None
    step_given: bool = False

    duplicate_rows: int = 0
    duplicate_timestamp_count: int = 0
    duplicate_examples: list[pd.Timestamp] = field(default_factory=list)

    unsorted_rows: int = 0
    unsorted_examples: list[pd.Timestamp] = field(default_factory=list)

    invalid: list[InvalidCandle] = field(default_factory=list)

    zero_volume_rows: int = 0
    zero_volume_examples: list[pd.Timestamp] = field(default_factory=list)
    missing_volume_rows: int = 0
    missing_volume_examples: list[pd.Timestamp] = field(default_factory=list)

    zero_range_rows: int = 0
    zero_range_examples: list[pd.Timestamp] = field(default_factory=list)

    gaps: list[Gap] = field(default_factory=list)
    weekend_gaps_ignored: int = 0

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    example_limit: int = MAX_EXAMPLES
    market: str = ""
    symbol: str = ""
    timeframe: str = ""
    path: Path | None = None
    size_bytes: int = 0

    @property
    def label(self) -> str:
        parts = [part for part in (self.market, self.symbol, self.timeframe) if part]
        return "/".join(parts) if parts else "dataset"

    @property
    def ok(self) -> bool:
        """True when no error-level problem was found."""
        return not self.errors


@dataclass(frozen=True)
class DatasetInfo:
    """A dataset found on disk (used by the ``list`` command)."""

    market: str
    symbol: str
    timeframe: str
    path: Path
    rows: int = 0
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    size_bytes: int = 0
    error: str | None = None

    @property
    def key(self) -> str:
        return f"{self.market}/{self.symbol}/{self.timeframe}"


# -- analysis -------------------------------------------------------------


def parse_step(value: str | pd.Timedelta) -> pd.Timedelta:
    """Parse an expected bar spacing such as ``"1h"`` or ``"15m"``."""
    if isinstance(value, pd.Timedelta):
        step = value
    else:
        try:
            step = pd.Timedelta(normalize_rule(str(value)))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid step {value!r}: {exc}") from exc
    if step <= pd.Timedelta(0):
        raise ValueError(f"step must be positive, got {value!r}")
    return step


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    try:
        return pd.to_numeric(df[column], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"column {column!r} is not numeric: {exc}") from exc


def _infer_step(ordered_ts: pd.Series) -> pd.Timedelta | None:
    """Most common positive spacing between consecutive timestamps.

    Ties are broken toward the *smaller* spacing, so a dataset that is mostly
    1h bars with a few 2h gaps reports 1h (and therefore the gaps) rather
    than silently accepting 2h.
    """
    diffs = ordered_ts.diff().dropna()
    positive = diffs[diffs > pd.Timedelta(0)]
    if positive.empty:
        return None
    counts = positive.value_counts()
    tied = counts[counts == counts.max()].index
    return pd.Timedelta(min(tied))


def analyze_candles(
    df: pd.DataFrame,
    *,
    step: str | pd.Timedelta | None = None,
    ignore_weekends: bool = False,
    weekend: WeekendWindow | None = None,
    examples: int = MAX_EXAMPLES,
) -> QualityReport:
    """Check an in-memory candle frame and return all findings.

    ``step`` forces the expected bar spacing; otherwise it is inferred from
    the most common timestamp difference.
    """
    weekend = weekend or WeekendWindow()
    limit = max(0, int(examples))

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    if not isinstance(df["timestamp"].dtype, pd.DatetimeTZDtype):
        if pd.api.types.is_datetime64_any_dtype(df["timestamp"].dtype):
            raise ValueError(
                "timestamp column is timezone-naive; localize it explicitly "
                "before analysing"
            )
        raise TypeError("timestamp column must be a datetime dtype")

    work = pd.DataFrame(
        {
            "timestamp": df["timestamp"].dt.tz_convert("UTC"),
            **{column: _numeric(df, column) for column in REQUIRED_COLUMNS[1:]},
        }
    ).reset_index(drop=True)
    rows = len(work)

    report = QualityReport(example_limit=limit)
    if rows == 0:
        report.errors.append("dataset has no rows")
        return report

    report.rows = rows
    report.start = work["timestamp"].min()
    report.end = work["timestamp"].max()

    if isinstance(step, str):
        step = parse_step(step)
    report.step_given = step is not None
    ordered_ts = work["timestamp"].sort_values(kind="stable").reset_index(drop=True)
    if step is None and len(ordered_ts) > 1:
        step = _infer_step(ordered_ts)
    report.step = step

    # -- timestamps ------------------------------------------------------
    ts = work["timestamp"]
    duplicates = ts.duplicated(keep=False)
    report.duplicate_rows = int(duplicates.sum())
    if report.duplicate_rows:
        duplicate_ts = ts[duplicates]
        report.duplicate_timestamp_count = int(duplicate_ts.nunique())
        report.duplicate_examples = list(duplicate_ts.drop_duplicates().head(limit))
        report.errors.append(
            f"{report.duplicate_rows} duplicate timestamp row(s) "
            f"across {report.duplicate_timestamp_count} timestamp(s)"
        )

    decreases = ts.diff() < pd.Timedelta(0)
    report.unsorted_rows = int(decreases.sum())
    if report.unsorted_rows:
        report.unsorted_examples = list(ts[decreases].head(limit))
        report.errors.append(f"{report.unsorted_rows} row(s) out of order")

    # -- prices ----------------------------------------------------------
    open_, high, low, close = work["open"], work["high"], work["low"], work["close"]
    prices = pd.concat([open_, close], axis=1)
    missing_price = open_.isna() | high.isna() | low.isna() | close.isna()
    low_above = low > prices.min(axis=1)
    high_below = high < prices.max(axis=1)
    low_above_high = low > high

    invalid_mask = missing_price | low_above | high_below | low_above_high
    if invalid_mask.any():
        for index in work.index[invalid_mask]:
            reasons = tuple(
                reason
                for reason, hit in (
                    ("missing price data", bool(missing_price.loc[index])),
                    ("low above open/close", bool(low_above.loc[index])),
                    ("high below open/close", bool(high_below.loc[index])),
                    ("low above high", bool(low_above_high.loc[index])),
                )
                if hit
            )
            report.invalid.append(
                InvalidCandle(timestamp=work.at[index, "timestamp"], reasons=reasons)
            )
        report.errors.append(f"{len(report.invalid)} invalid candle(s)")

    # -- volume ----------------------------------------------------------
    volume = work["volume"]
    zero_volume = volume == 0
    report.zero_volume_rows = int(zero_volume.sum())
    if report.zero_volume_rows:
        report.zero_volume_examples = list(
            work.loc[zero_volume, "timestamp"].head(limit)
        )
        report.warnings.append(f"{report.zero_volume_rows} zero-volume candle(s)")

    missing_volume = volume.isna()
    report.missing_volume_rows = int(missing_volume.sum())
    if report.missing_volume_rows:
        report.missing_volume_examples = list(
            work.loc[missing_volume, "timestamp"].head(limit)
        )
        report.errors.append(
            f"{report.missing_volume_rows} candle(s) with missing volume"
        )

    # -- zero-range ------------------------------------------------------
    zero_range = high == low
    report.zero_range_rows = int(zero_range.sum())
    if report.zero_range_rows:
        report.zero_range_examples = list(work.loc[zero_range, "timestamp"].head(limit))
        report.warnings.append(f"{report.zero_range_rows} zero-range candle(s)")

    # -- gaps ------------------------------------------------------------
    if step is not None:
        for index in range(1, len(ordered_ts)):
            previous = ordered_ts.iloc[index - 1]
            current = ordered_ts.iloc[index]
            delta = current - previous
            if delta <= step:
                continue
            gap_start = previous + step
            missing_duration = delta - step
            if ignore_weekends and weekend.covers(gap_start, current):
                report.weekend_gaps_ignored += 1
                continue
            report.gaps.append(
                Gap(
                    start=gap_start,
                    end=current,
                    missing=missing_duration,
                    bars=int(missing_duration // step),
                )
            )
        if report.gaps:
            report.warnings.append(f"{len(report.gaps)} gap(s)")

    return report


def check_dataset(
    path: str | Path,
    *,
    market: str,
    symbol: str,
    timeframe: str,
    step: str | pd.Timedelta | None = None,
    ignore_weekends: bool = False,
    weekend: WeekendWindow | None = None,
    examples: int = MAX_EXAMPLES,
) -> QualityReport:
    """Analyse one stored Parquet dataset (read as-is, without validation)."""
    path = Path(path)
    frame = pd.read_parquet(path)
    report = analyze_candles(
        frame,
        step=step,
        ignore_weekends=ignore_weekends,
        weekend=weekend,
        examples=examples,
    )
    return replace(
        report,
        market=market,
        symbol=symbol,
        timeframe=timeframe,
        path=path,
        size_bytes=path.stat().st_size,
    )


# -- dataset discovery ----------------------------------------------------


def find_datasets(root: str | Path | None = None) -> list[DatasetInfo]:
    """Find every {market}/{symbol}/{timeframe}.parquet under the candles root."""
    base = candles_root() if root is None else Path(root)
    if not base.is_dir():
        return []

    datasets: list[DatasetInfo] = []
    for path in sorted(base.glob("*/*/*.parquet")):
        info = DatasetInfo(
            market=path.parent.parent.name,
            symbol=path.parent.name,
            timeframe=path.stem,
            path=path,
            size_bytes=path.stat().st_size,
        )
        try:
            frame = pd.read_parquet(path, columns=["timestamp"])
            timestamps = frame["timestamp"]
            info = replace(
                info,
                rows=len(frame),
                start=timestamps.min() if len(frame) else None,
                end=timestamps.max() if len(frame) else None,
            )
        except Exception as exc:  # noqa: BLE001 - a bad file must not abort the list
            info = replace(info, error=f"{type(exc).__name__}: {exc}")
        datasets.append(info)
    return sorted(datasets, key=lambda info: info.key)


# -- formatting -----------------------------------------------------------


def format_timestamp(value: pd.Timestamp | None) -> str:
    """UTC timestamp as ``2024-01-01T00:00:00Z`` (or ``-``)."""
    if value is None or pd.isna(value):
        return "-"
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC").isoformat().replace("+00:00", "Z")


def format_duration(value: pd.Timedelta | None) -> str:
    """Compact duration such as ``2h`` or ``2d1h``."""
    if value is None or pd.isna(value):
        return "-"
    total = int(pd.Timedelta(value).total_seconds())
    sign = "-" if total < 0 else ""
    total = abs(total)
    parts: list[str] = []
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60), ("s", 1)):
        amount, total = divmod(total, size)
        if amount:
            parts.append(f"{amount}{unit}")
    return sign + ("".join(parts) or "0s")


def format_size(num_bytes: int) -> str:
    """Human-readable byte count."""
    size = float(num_bytes)
    if size < 1024:
        return f"{int(size)} B"
    for unit in ("KiB", "MiB", "GiB"):
        size /= 1024
        if size < 1024 or unit == "GiB":
            return f"{size:.1f} {unit}"
    return f"{size:.1f} GiB"  # pragma: no cover - unreachable


def _count(amount: int, noun: str) -> str:
    return f"{amount} {noun}{'' if amount == 1 else 's'}"


def _check_row(label: str, value: str, *, problem: bool, error: bool) -> str:
    marker = "ok" if not problem else ("ERROR" if error else "WARN")
    return f"    {label:<22}{value:<30}{marker}"


def _step_text(report: QualityReport) -> str:
    if report.step is None:
        return "unknown"
    origin = "given" if report.step_given else "inferred"
    return f"{format_duration(report.step)} ({origin})"


def _duplicate_text(report: QualityReport) -> str:
    if report.duplicate_rows == 0:
        return "0"
    return (
        f"{report.duplicate_rows} rows / {report.duplicate_timestamp_count} timestamps"
    )


def _example_section(
    lines: list[str],
    title: str,
    examples: list[pd.Timestamp],
    total: int,
    limit: int,
) -> None:
    if not examples:
        return
    shown = f"first {len(examples)} of {total}" if total > len(examples) else ""
    suffix = f" ({shown})" if shown else ""
    lines.append("")
    lines.append(f"  {title}{suffix}:")
    lines.extend(f"    - {format_timestamp(example)}" for example in examples[:limit])


def format_quality_report(report: QualityReport) -> str:
    """Render a full quality report as printable text."""
    lines: list[str] = [f"Quality report: {report.label}"]
    if report.path is not None:
        lines.append(f"  file:  {report.path} ({format_size(report.size_bytes)})")
    lines.append(f"  rows: {report.rows}")
    if report.rows:
        lines.append(
            f"  range: {format_timestamp(report.start)} -> "
            f"{format_timestamp(report.end)} (UTC)"
        )
    lines.append(f"  step:  {_step_text(report)}")

    checks = [
        (
            "duplicate timestamps",
            _duplicate_text(report),
            report.duplicate_rows > 0,
            True,
        ),
        ("unsorted rows", str(report.unsorted_rows), report.unsorted_rows > 0, True),
        ("invalid candles", str(len(report.invalid)), bool(report.invalid), True),
        (
            "zero volume",
            str(report.zero_volume_rows),
            report.zero_volume_rows > 0,
            False,
        ),
        (
            "missing volume",
            str(report.missing_volume_rows),
            report.missing_volume_rows > 0,
            True,
        ),
        (
            "zero-range candles",
            str(report.zero_range_rows),
            report.zero_range_rows > 0,
            False,
        ),
        ("gaps", str(len(report.gaps)), bool(report.gaps), False),
    ]

    lines.append("")
    lines.append("  checks:")
    for label, value, problem, is_error in checks:
        lines.append(_check_row(label, value, problem=problem, error=is_error))

    _example_section(
        lines,
        "duplicate timestamps",
        report.duplicate_examples,
        report.duplicate_timestamp_count,
        report.example_limit,
    )
    _example_section(
        lines,
        "unsorted rows",
        report.unsorted_examples,
        report.unsorted_rows,
        report.example_limit,
    )

    if report.invalid:
        shown = report.invalid[: report.example_limit]
        suffix = (
            f" (first {len(shown)} of {len(report.invalid)})"
            if len(report.invalid) > len(shown)
            else ""
        )
        lines.append("")
        lines.append(f"  invalid candles{suffix}:")
        lines.extend(
            f"    - {format_timestamp(candle.timestamp)}: {', '.join(candle.reasons)}"
            for candle in shown
        )

    _example_section(
        lines,
        "zero-volume candles",
        report.zero_volume_examples,
        report.zero_volume_rows,
        report.example_limit,
    )
    _example_section(
        lines,
        "missing-volume candles",
        report.missing_volume_examples,
        report.missing_volume_rows,
        report.example_limit,
    )
    _example_section(
        lines,
        "zero-range candles",
        report.zero_range_examples,
        report.zero_range_rows,
        report.example_limit,
    )

    if report.gaps:
        lines.append("")
        lines.append(f"  gaps ({len(report.gaps)}):")
        for gap in report.gaps:
            bars = _count(gap.bars, "bar")
            lines.append(
                f"    - start {format_timestamp(gap.start)}, "
                f"{bars} missing ({format_duration(gap.missing)}), "
                f"next bar {format_timestamp(gap.end)}"
            )

    if report.weekend_gaps_ignored:
        lines.append("")
        lines.append(
            f"  weekend gaps ignored: {report.weekend_gaps_ignored} "
            "(entirely inside the weekend closure)"
        )

    lines.append("")
    if report.ok and not report.warnings:
        lines.append("  result: OK")
    else:
        lines.append(
            f"  result: {_count(len(report.errors), 'error')}, "
            f"{_count(len(report.warnings), 'warning')}"
        )
    return "\n".join(lines)


def format_dataset_list(
    datasets: list[DatasetInfo],
    *,
    root: str | Path,
) -> str:
    """Render the ``list`` command's output."""
    if not datasets:
        return f"No datasets under {root}"

    width = max(len(info.key) for info in datasets)
    lines = [f"Datasets under {root} ({len(datasets)}):", ""]
    for info in datasets:
        if info.error:
            lines.append(f"  {info.key:<{width}}  UNREADABLE: {info.error}")
            continue
        span = (
            f"{format_timestamp(info.start)} .. {format_timestamp(info.end)}"
            if info.rows
            else "-"
        )
        lines.append(
            f"  {info.key:<{width}}  {info.rows:>8} rows  "
            f"{span}  {format_size(info.size_bytes)}"
        )
    return "\n".join(lines)
