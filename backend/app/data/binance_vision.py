"""Binance Vision (data.binance.vision) monthly kline source.

Binance publishes public monthly ZIP archives of klines — no API key, no
authenticated endpoint. Archives are cached under data/raw/binance/ (which is
gitignored) and re-used on subsequent runs.

Only the standard library is used for HTTP (urllib.request): no extra
dependency, no secret.

Archive quirk handling:
- some files carry a header row, some do not;
- ``open_time`` may be seconds, milliseconds or microseconds since epoch —
  the unit is detected automatically from the value's magnitude;
- files may have a different column order.
"""

from __future__ import annotations

import io
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from functools import partial
from pathlib import Path

import pandas as pd

from .months import is_month_string, month_of, month_range
from .paths import binance_raw_root
from .schema import validate_candles
from .source import DataSource, TimeLike, filter_range, to_utc_timestamp

BASE_URL = "https://data.binance.vision"
DEFAULT_MARKET = "spot"
DEFAULT_DELAY_SECONDS = 0.5
DEFAULT_TIMEOUT_SECONDS = 30.0

_USER_AGENT = "trading-backtester/0.1 (local research tool)"

# Standard Binance kline CSV layout (positional, when there is no header).
KLINES_COLUMNS: tuple[str, ...] = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore",
)

_TIMESTAMP_COLUMN_CANDIDATES = ("open_time", "timestamp", "time", "date")
_PRICE_COLUMNS = ("open", "high", "low", "close", "volume")

# Magnitude thresholds for epoch epoch detection. Real market data sits in a
# range wide enough that these are unambiguous:
#   seconds      1.0e9  (2001) .. 4.1e9  (2100)
#   milliseconds 1.0e12        .. 4.1e12
#   microseconds 1.0e15        .. 4.1e15
_SECOND_MAX = 10**11
_MILLI_MAX = 10**14
_MICRO_MAX = 10**17


class BinanceVisionError(RuntimeError):
    """Raised when a Binance Vision archive cannot be fetched or parsed."""


class MissingMonthError(BinanceVisionError):
    """Raised when Binance has no archive for a requested month."""


def detect_timestamp_unit(value: int) -> str:
    """Detect the epoch unit of a raw kline timestamp from its magnitude.

    Returns one of ``"s"``, ``"ms"``, ``"us"`` or ``"ns"``.
    """
    value = int(value)
    if value <= 0:
        raise ValueError(f"Timestamp must be positive, got {value}")
    if value < _SECOND_MAX:
        return "s"
    if value < _MILLI_MAX:
        return "ms"
    if value < _MICRO_MAX:
        return "us"
    return "ns"


def kline_filename(symbol: str, interval: str, month: str) -> str:
    """``BTCUSDT-1h-2023-01.zip``."""
    return f"{symbol}-{interval}-{month}.zip"


def resolve_bounds(
    start: TimeLike | None,
    end: TimeLike | None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Turn user bounds into ``[lower, upper)`` UTC timestamps.

    ``end`` given as a bare month string ("2024-12") means *including* that
    whole month; a full timestamp is exclusive, matching the DataSource
    contract.
    """
    if start is None or end is None:
        raise ValueError("BinanceVisionSource requires both start and end")
    start_period = month_of(start)
    end_period = month_of(end)
    if end_period < start_period:
        raise ValueError(f"end ({end}) is before start ({start})")

    lower = (
        start_period.start_time.tz_localize("UTC")
        if is_month_string(start)
        else to_utc_timestamp(start)
    )
    if is_month_string(end):
        upper = (end_period + 1).start_time.tz_localize("UTC")
    else:
        upper = to_utc_timestamp(end)
    assert lower is not None and upper is not None
    return lower, upper


class BinanceVisionSource(DataSource):
    """Download and parse monthly kline archives from data.binance.vision.

    Parameters
    ----------
    root : cache directory for raw zips. ``None`` (default) uses
        data/raw/binance under the repository data dir, see app.data.paths.
    market : Binance market segment, e.g. "spot".
    delay_seconds : polite pause enforced *between* network requests.
    http_get : optional ``callable(url) -> bytes``; tests inject a fake.
    sleep : optional ``callable(seconds)``; tests inject a recorder.
    timeout : socket timeout for the default urllib fetch.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        market: str = DEFAULT_MARKET,
        delay_seconds: float = DEFAULT_DELAY_SECONDS,
        http_get: Callable[[str], bytes] | None = None,
        sleep: Callable[[float], None] | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.root = binance_raw_root() if root is None else Path(root)
        self.market = market
        self.delay_seconds = float(delay_seconds)
        self._http_get = http_get or partial(_urllib_get, timeout=timeout)
        self._sleep = sleep or time.sleep
        self._downloads = 0

    # -- DataSource --------------------------------------------------------

    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        start: TimeLike | None = None,
        end: TimeLike | None = None,
    ) -> pd.DataFrame:
        lower, upper = resolve_bounds(start, end)
        frames = [
            self._load_month(symbol, timeframe, month)
            for month in month_range(start, end)  # type: ignore[arg-type]
        ]
        candles = validate_candles(pd.concat(frames, ignore_index=True))
        return filter_range(candles, lower, upper)

    # -- paths -------------------------------------------------------------

    def raw_month_path(self, symbol: str, interval: str, month: str) -> Path:
        """Cache path mirroring the remote layout."""
        return (
            self.root
            / self.market
            / "monthly"
            / "klines"
            / symbol
            / interval
            / kline_filename(symbol, interval, month)
        )

    def month_url(self, symbol: str, interval: str, month: str) -> str:
        return (
            f"{BASE_URL}/data/{self.market}/monthly/klines/"
            f"{symbol}/{interval}/{kline_filename(symbol, interval, month)}"
        )

    # -- internals ---------------------------------------------------------

    def _load_month(self, symbol: str, interval: str, month: str) -> pd.DataFrame:
        path = self.raw_month_path(symbol, interval, month)
        if not path.exists():
            self._download_month(symbol, interval, month, path)
        return parse_kline_zip(path.read_bytes(), source_name=str(path))

    def _download_month(
        self,
        symbol: str,
        interval: str,
        month: str,
        path: Path,
    ) -> None:
        url = self.month_url(symbol, interval, month)
        if self._downloads > 0:
            self._sleep(self.delay_seconds)
        try:
            data = self._http_get(url)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise MissingMonthError(
                    f"No Binance Vision archive for {symbol} {interval} {month} "
                    f"(HTTP 404): {url}. The month may predate the symbol's "
                    f"listing, or the market/interval may be wrong."
                ) from exc
            raise BinanceVisionError(f"HTTP {exc.code} downloading {url}") from exc
        except urllib.error.URLError as exc:
            raise BinanceVisionError(
                f"Network error downloading {url}: {exc.reason}"
            ) from exc

        path.parent.mkdir(parents=True, exist_ok=True)
        partial_path = path.with_name(path.name + ".part")
        partial_path.write_bytes(data)
        partial_path.replace(path)  # never leave a half-written cache entry
        self._downloads += 1


def _urllib_get(url: str, *, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


# -- parsing ---------------------------------------------------------------


def parse_kline_zip(data: bytes, *, source_name: str = "<zip>") -> pd.DataFrame:
    """Parse a Binance Vision kline ZIP into canonical candles (UTC)."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
            if not names:
                raise BinanceVisionError(f"No CSV found inside {source_name}")
            if len(names) > 1:
                raise BinanceVisionError(
                    f"Expected one CSV inside {source_name}, found {names}"
                )
            text = archive.read(names[0]).decode("utf-8-sig")
    except zipfile.BadZipFile as exc:
        raise BinanceVisionError(f"Corrupt zip {source_name}: {exc}") from exc
    return parse_kline_csv(text, source_name=source_name)


def parse_kline_csv(text: str, *, source_name: str = "<csv>") -> pd.DataFrame:
    """Parse raw Binance kline CSV text (with or without header row)."""
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise BinanceVisionError(f"Empty kline CSV in {source_name}")

    has_header = not _first_cell_is_numeric(lines[0])
    frame = pd.read_csv(io.StringIO("\n".join(lines)), header=0 if has_header else None)

    if has_header:
        frame.columns = [str(col).strip().lower() for col in frame.columns]
        timestamp_column = next(
            (c for c in _TIMESTAMP_COLUMN_CANDIDATES if c in frame.columns), None
        )
        if timestamp_column is None:
            raise BinanceVisionError(
                f"{source_name}: no timestamp column found; "
                f"columns are {list(frame.columns)}"
            )
    else:
        if frame.shape[1] < len(_PRICE_COLUMNS) + 1:
            raise BinanceVisionError(
                f"{source_name}: expected at least 6 columns, got {frame.shape[1]}"
            )
        frame.columns = KLINES_COLUMNS[: frame.shape[1]]
        timestamp_column = "open_time"

    missing = [c for c in _PRICE_COLUMNS if c not in frame.columns]
    if missing:
        raise BinanceVisionError(f"{source_name}: missing columns {missing}")

    candles = pd.DataFrame(
        {
            "timestamp": _parse_timestamps(frame[timestamp_column], source_name),
        }
    )
    for column in _PRICE_COLUMNS:
        candles[column] = pd.to_numeric(frame[column], errors="raise")
    return validate_candles(candles)


def _parse_timestamps(series: pd.Series, source_name: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="raise")
    first = numeric.dropna()
    if first.empty:
        raise BinanceVisionError(f"{source_name}: timestamps column is empty")
    unit = detect_timestamp_unit(int(first.iloc[0]))
    return pd.to_datetime(numeric, unit=unit, utc=True)


def _first_cell_is_numeric(line: str) -> bool:
    cell = line.split(",", 1)[0].strip().strip('"')
    if not cell:
        return False
    try:
        float(cell)
    except ValueError:
        return False
    return True
