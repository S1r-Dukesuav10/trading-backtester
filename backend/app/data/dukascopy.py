"""Dukascopy monthly candle downloader — download only, no importing.

Downloads 1-minute BID candles from Dukascopy's public datafeed through the
``dukascopy-node`` CLI (MIT, run transiently via npx; no API key). Files land
one-per-month under data/raw/dukascopy/<SYMBOL>/ and are never parsed here —
import is a separate concern.

Politeness:
- one month per CLI invocation, so a failure only costs that month;
- a pause between months;
- failed months are retried up to ``max_attempts`` with a longer wait each time;
- after several consecutive empty months the run stops early, since that
  pattern means the datafeed is unreachable rather than genuinely empty.

Note on empty downloads: Dukascopy (and its CDN) can answer with a 0-byte body
while dukascopy-node still reports success ("File saved ... 0 Bytes"). Such a
month is recorded as EMPTY, the empty file is deleted so a re-run retries it,
and the tool's own output is quoted in the report for diagnosis.

Resumability:
- a month whose file already exists and contains data is skipped, so the
  command can simply be re-run after an interruption.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .months import month_dates, month_range
from .paths import dukascopy_raw_root

DUKASCOPY_PACKAGE = "dukascopy-node"
DEFAULT_SYMBOL = "XAUUSD"
DEFAULT_TIMEFRAME = "m1"
DEFAULT_SIDE = "bid"
DEFAULT_DELAY_SECONDS = 3.0
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_RETRY_WAIT_SECONDS = 20.0
DEFAULT_TIMEOUT_SECONDS = 900.0
DEFAULT_ABORT_AFTER_EMPTY_MONTHS = 3

# dukascopy-node CLI flags, verified against `npx dukascopy-node --help`
# (v1.50.0). Note: `-dir` is the output directory; `-d` is --debug.
FLAG_INSTRUMENT = "-i"
FLAG_DATE_FROM = "-from"
FLAG_DATE_TO = "-to"
FLAG_TIMEFRAME = "-t"
FLAG_PRICE_TYPE = "-p"
FLAG_FORMAT = "-f"
FLAG_DIR = "-dir"
FLAG_UTC_OFFSET = "-utc"
FLAG_VOLUMES = "-v"

STATUS_DOWNLOADED = "downloaded"
STATUS_CACHED = "cached"
STATUS_EMPTY = "empty"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"


class DukascopyError(RuntimeError):
    """Raised for unrecoverable downloader problems (e.g. Node/npx missing)."""


@dataclass
class MonthResult:
    """Outcome for a single month."""

    month: str
    status: str
    path: Path | None = None
    size_bytes: int = 0
    attempts: int = 0
    error: str | None = None


@dataclass
class DownloadReport:
    """Outcome for a whole download run."""

    symbol: str
    timeframe: str
    side: str
    rows: list[MonthResult] = field(default_factory=list)

    @property
    def downloaded(self) -> list[MonthResult]:
        return [r for r in self.rows if r.status == STATUS_DOWNLOADED]

    @property
    def cached(self) -> list[MonthResult]:
        return [r for r in self.rows if r.status == STATUS_CACHED]

    @property
    def empty(self) -> list[MonthResult]:
        return [r for r in self.rows if r.status == STATUS_EMPTY]

    @property
    def failed(self) -> list[MonthResult]:
        return [r for r in self.rows if r.status == STATUS_FAILED]

    @property
    def skipped(self) -> list[MonthResult]:
        return [r for r in self.rows if r.status == STATUS_SKIPPED]

    @property
    def existing_files(self) -> list[MonthResult]:
        """Months that have usable data on disk (downloaded or cached)."""
        return self.downloaded + self.cached

    @property
    def total_size_bytes(self) -> int:
        return sum(r.size_bytes for r in self.existing_files)

    @property
    def ok(self) -> bool:
        return not self.failed and not self.empty and not self.skipped


def _default_runner(
    cmd: Sequence[str], *, timeout: float
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(cmd),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def resolve_npx() -> str:
    """Locate the npx executable (on Windows this is ``npx.cmd``)."""
    found = shutil.which("npx") or shutil.which("npx.cmd")
    if found is None:
        raise DukascopyError(
            "npx was not found on PATH. Install Node.js (which ships npm/npx) "
            "and re-run."
        )
    return found


def has_data_rows(path: Path) -> bool:
    """True when the CSV contains at least one data row.

    A header-only file (or an empty file) counts as *no data*: Dukascopy has
    no candles for that month, e.g. ahead of the datafeed's coverage.
    """
    if not path.exists() or path.stat().st_size == 0:
        return False
    lines = [
        line
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]
    if len(lines) >= 2:
        return True
    if not lines:
        return False
    first_cell = lines[0].split(",", 1)[0].strip().strip('"')
    try:
        float(first_cell)
    except ValueError:
        return False  # header only
    return True


def merge_csvs(files: Sequence[Path], target: Path) -> None:
    """Concatenate raw CSVs into ``target`` keeping a single header row."""
    header: str | None = None
    with target.open("w", encoding="utf-8", newline="\n") as out:
        for path in files:
            lines = [
                line
                for line in path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
                if line.strip()
            ]
            if not lines:
                continue
            if header is None:
                header = lines[0]
                out.write(header + "\n")
                body = lines[1:]
            elif lines[0] == header:
                body = lines[1:]
            else:
                body = lines
            for line in body:
                out.write(line + "\n")


def head_rows(path: Path, count: int) -> list[str]:
    """First ``count`` lines of a file (for reporting)."""
    lines: list[str] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                lines.append(line.rstrip("\n"))
            if len(lines) >= count:
                break
    return lines


def tail_rows(path: Path, count: int) -> list[str]:
    """Last ``count`` lines of a file (for reporting)."""
    from collections import deque

    buffer: deque[str] = deque(maxlen=count)
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                buffer.append(line.rstrip("\n"))
    return list(buffer)


class DukascopyDownloader:
    """Downloads Dukascopy months as raw CSV, one file per month.

    Parameters
    ----------
    root : cache dir; ``None`` uses data/raw/dukascopy (see app.data.paths).
    symbol : instrument, e.g. "XAUUSD" (lower-cased for the CLI).
    timeframe : dukascopy-node timeframe token, e.g. "m1".
    side : "bid" or "ask".
    volumes : include Dukascopy volume data (default True — the canonical
        schema requires a volume column, and re-downloading later is costly).
    run : injected subprocess runner (tests); default uses subprocess.
    sleep : injected sleep (tests); default time.sleep.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        symbol: str = DEFAULT_SYMBOL,
        timeframe: str = DEFAULT_TIMEFRAME,
        side: str = DEFAULT_SIDE,
        volumes: bool = True,
        delay_seconds: float = DEFAULT_DELAY_SECONDS,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        retry_wait_seconds: float = DEFAULT_RETRY_WAIT_SECONDS,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        run: Callable[..., subprocess.CompletedProcess[str]] | None = None,
        sleep: Callable[[float], None] | None = None,
        abort_after_empty_months: int = DEFAULT_ABORT_AFTER_EMPTY_MONTHS,
    ) -> None:
        if side not in ("bid", "ask"):
            raise ValueError(f"side must be 'bid' or 'ask', got {side!r}")
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")

        self.root = dukascopy_raw_root() if root is None else Path(root)
        self.symbol = symbol
        self.timeframe = timeframe
        self.side = side
        self.volumes = bool(volumes)
        self.delay_seconds = float(delay_seconds)
        self.max_attempts = int(max_attempts)
        self.retry_wait_seconds = float(retry_wait_seconds)
        self.timeout = float(timeout)
        self.abort_after_empty_months = int(abort_after_empty_months)
        self._run = run or _default_runner
        self._sleep = sleep or time.sleep
        self._npx: str | None = None

    # -- paths -------------------------------------------------------------

    @property
    def instrument(self) -> str:
        """Instrument id as dukascopy-node expects it (lower-case)."""
        return self.symbol.lower()

    def month_path(self, month: str) -> Path:
        """data/raw/dukascopy/<SYMBOL>/<SYMBOL>-<tf>-<side>-<YYYY-MM>.csv."""
        return (
            self.root
            / self.symbol
            / (f"{self.symbol}-{self.timeframe}-{self.side}-{month}.csv")
        )

    def build_command(self, month: str, out_dir: Path) -> list[str]:
        """Exact argv for one month (one process per month, by design)."""
        start, end = month_dates(month)
        if self._npx is None:
            self._npx = resolve_npx()
        command = [
            self._npx,
            "-y",
            DUKASCOPY_PACKAGE,
            FLAG_INSTRUMENT,
            self.instrument,
            FLAG_DATE_FROM,
            start.strftime("%Y-%m-%d"),
            FLAG_DATE_TO,
            end.strftime("%Y-%m-%d"),
            FLAG_TIMEFRAME,
            self.timeframe,
            FLAG_PRICE_TYPE,
            self.side,
            FLAG_FORMAT,
            "csv",
            # Explicitly UTC: never inherit the machine's local offset.
            FLAG_UTC_OFFSET,
            "0",
            FLAG_DIR,
            str(out_dir),
        ]
        if self.volumes:
            command.append(FLAG_VOLUMES)
        return command

    # -- downloading -------------------------------------------------------

    def download_months(self, months: Sequence[str]) -> DownloadReport:
        report = DownloadReport(
            symbol=self.symbol, timeframe=self.timeframe, side=self.side
        )
        consecutive_empty = 0
        for index, month in enumerate(months):
            result = self.download_month(month)
            report.rows.append(result)

            if result.status == STATUS_EMPTY:
                consecutive_empty += 1
                if (
                    self.abort_after_empty_months > 0
                    and consecutive_empty >= self.abort_after_empty_months
                ):
                    remaining = months[index + 1 :]
                    reason = (
                        f"aborted early: {consecutive_empty} consecutive months "
                        f"came back empty ({result.error or 'no candles returned'}). "
                        "The datafeed looks unreachable from this machine - check "
                        "network access / geo restrictions, or re-run to retry."
                    )
                    report.rows.extend(
                        MonthResult(m, STATUS_SKIPPED, error=reason) for m in remaining
                    )
                    break
            else:
                consecutive_empty = 0

            if index < len(months) - 1:
                self._sleep(self.delay_seconds)  # polite pause between months
        return report

    def download_month(self, month: str) -> MonthResult:
        target = self.month_path(month)
        if target.exists() and target.stat().st_size > 0 and has_data_rows(target):
            return MonthResult(
                month, STATUS_CACHED, target, target.stat().st_size, attempts=0
            )

        last_error = "unknown error"
        for attempt in range(1, self.max_attempts + 1):
            if attempt > 1:
                self._sleep(self.retry_wait_seconds * (attempt - 1))
            try:
                last_error, snippet = self._attempt_download(month, target)
            except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                last_error = f"{type(exc).__name__}: {exc}"
                snippet = ""
                continue
            if last_error:
                continue
            size = target.stat().st_size
            if not has_data_rows(target):
                target.unlink(missing_ok=True)  # so a re-run retries the month
                return MonthResult(
                    month,
                    STATUS_EMPTY,
                    target,
                    size,
                    attempts=attempt,
                    error=snippet or None,
                )
            return MonthResult(month, STATUS_DOWNLOADED, target, size, attempts=attempt)

        return MonthResult(
            month,
            STATUS_FAILED,
            target,
            0,
            attempts=self.max_attempts,
            error=last_error,
        )

    def _attempt_download(self, month: str, target: Path) -> tuple[str, str]:
        """One download attempt.

        Returns ``(error, tool_output)``: error is "" on success, while
        tool_output is a short tail of the CLI's own output — useful when the
        tool reports success but produced a 0-byte file.
        """
        with tempfile.TemporaryDirectory(prefix=f"dukascopy-{month}-") as scratch:
            scratch_dir = Path(scratch)
            cmd = self.build_command(month, scratch_dir)
            completed = self._run(cmd, timeout=self.timeout)
            output = _tail_text(completed.stdout, completed.stderr)
            if completed.returncode != 0:
                return (
                    f"dukascopy-node exited {completed.returncode}"
                    + (f": {output}" if output else ""),
                    output,
                )

            produced = sorted(scratch_dir.rglob("*.csv"))
            if not produced:
                return "dukascopy-node produced no CSV file", output

            target.parent.mkdir(parents=True, exist_ok=True)
            merge_csvs(produced, target)
        return "", output


def _tail_text(stdout: str | None, stderr: str | None, limit: int = 300) -> str:
    """Last ``limit`` characters of the tool's output, whitespace-collapsed."""
    combined = " ".join(part for part in (stdout, stderr) if part).strip()
    combined = " ".join(combined.split())
    return combined[-limit:]


def format_report(
    report: DownloadReport,
    *,
    root: Path,
    head_count: int = 5,
    tail_count: int = 3,
) -> str:
    """Human-readable summary, including first/last rows of the range."""
    months_requested = len(report.rows)
    files = report.existing_files
    lines = [
        (
            f"Dukascopy {report.symbol} {report.timeframe} {report.side}: "
            f"{months_requested} month(s) requested"
        ),
        (
            f"  downloaded: {len(report.downloaded)}, "
            f"already present: {len(report.cached)}, "
            f"empty: {len(report.empty)}, failed: {len(report.failed)}"
        ),
    ]
    if files:
        total_kb = report.total_size_bytes / 1024
        lines.append(
            f"  files: {len(files)}/{months_requested} "
            f"({total_kb:.1f} KB total) under {root}"
        )
        ordered = sorted(files, key=lambda r: r.month)
        first = ordered[0]
        last = ordered[-1]
        lines.append(f"  range on disk: {first.month} .. {last.month}")
        if first.path:
            lines.append(f"  first {head_count} rows of {first.path.name}:")
            lines.extend(f"    {row}" for row in head_rows(first.path, head_count))
        if last.path:
            lines.append(f"  last {tail_count} rows of {last.path.name}:")
            lines.extend(f"    {row}" for row in tail_rows(last.path, tail_count))
    for result in report.failed:
        lines.append(
            f"  FAILED {result.month} (attempts={result.attempts}): {result.error}"
        )
    for result in report.empty:
        detail = f" - tool said: {result.error}" if result.error else ""
        lines.append(f"  EMPTY  {result.month}: no candles returned{detail}")
    for result in report.skipped:
        lines.append(f"  SKIPPED {result.month}: {result.error}")
    return "\n".join(lines)


def download_range(
    start: str,
    end: str,
    *,
    root: str | Path | None = None,
    symbol: str = DEFAULT_SYMBOL,
    timeframe: str = DEFAULT_TIMEFRAME,
    side: str = DEFAULT_SIDE,
    volumes: bool = True,
    delay_seconds: float = DEFAULT_DELAY_SECONDS,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    retry_wait_seconds: float = DEFAULT_RETRY_WAIT_SECONDS,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    run: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    sleep: Callable[[float], None] | None = None,
    abort_after_empty_months: int = DEFAULT_ABORT_AFTER_EMPTY_MONTHS,
) -> DownloadReport:
    """Convenience wrapper: month range (inclusive) -> download report."""
    downloader = DukascopyDownloader(
        root,
        symbol=symbol,
        timeframe=timeframe,
        side=side,
        volumes=volumes,
        delay_seconds=delay_seconds,
        max_attempts=max_attempts,
        retry_wait_seconds=retry_wait_seconds,
        timeout=timeout,
        run=run,
        sleep=sleep,
        abort_after_empty_months=abort_after_empty_months,
    )
    return downloader.download_months(month_range(start, end))
