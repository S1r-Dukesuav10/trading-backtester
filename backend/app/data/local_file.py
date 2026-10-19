"""Local CSV / Parquet candle import source.

Imports a single file bound to one (symbol, timeframe) and converts
everything to the canonical UTC schema.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schema import validate_candles
from .source import DataSource, TimeLike, filter_range

_TIMESTAMP_UNITS = {
    "s": "s",
    "sec": "s",
    "second": "s",
    "seconds": "s",
    "ms": "ms",
    "millisecond": "ms",
    "milliseconds": "ms",
    "us": "us",
    "microsecond": "us",
    "microseconds": "us",
    "ns": "ns",
    "nanosecond": "ns",
    "nanoseconds": "ns",
}

_ON_DUPLICATE = ("raise", "first", "last")


class LocalFileSource(DataSource):
    """Reads candles from a local .csv or .parquet file.

    Parameters
    ----------
    path : path to the source file.
    symbol, timeframe : what this file represents; ``get_candles`` rejects
        requests for anything else.
    column_mapping : maps canonical name -> file column name, e.g.
        ``{"timestamp": "Date", "open": "Open"}``. Defaults to identity.
    input_tz : timezone of *naive* timestamps in the file. Aware timestamps
        (e.g. ``+02:00`` offsets) are always honored as-is and converted to
        UTC. Default "UTC".
    timestamp_unit : required when timestamps are epoch numbers — one of
        seconds/milliseconds/microseconds/nanoseconds (or s/ms/us/ns).
    on_duplicate : "raise" (default), "first" or "last" — what to do when
        the file contains duplicate timestamps.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        symbol: str,
        timeframe: str,
        column_mapping: dict[str, str] | None = None,
        input_tz: str = "UTC",
        timestamp_unit: str | None = None,
        on_duplicate: str = "raise",
    ) -> None:
        self.path = Path(path)
        self.symbol = symbol
        self.timeframe = timeframe
        self.column_mapping = dict(column_mapping) if column_mapping else {}
        self.input_tz = input_tz or "UTC"

        if timestamp_unit is not None:
            key = str(timestamp_unit).lower()
            if key not in _TIMESTAMP_UNITS:
                raise ValueError(
                    f"Unsupported timestamp_unit {timestamp_unit!r}; "
                    f"expected one of {sorted(set(_TIMESTAMP_UNITS))}"
                )
            timestamp_unit = _TIMESTAMP_UNITS[key]
        self.timestamp_unit = timestamp_unit

        if on_duplicate not in _ON_DUPLICATE:
            raise ValueError(
                f"on_duplicate must be one of {_ON_DUPLICATE}, got {on_duplicate!r}"
            )
        self.on_duplicate = on_duplicate

    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        start: TimeLike | None = None,
        end: TimeLike | None = None,
    ) -> pd.DataFrame:
        if symbol != self.symbol:
            raise ValueError(
                f"This source holds symbol {self.symbol!r}, requested {symbol!r}"
            )
        if timeframe != self.timeframe:
            raise ValueError(
                f"This source holds timeframe {self.timeframe!r}, requested {timeframe!r}"
            )
        candles = self._load()
        return filter_range(candles, start, end)

    # -- internals ---------------------------------------------------------

    def _load(self) -> pd.DataFrame:
        df = self._read_file()
        df = self._apply_mapping(df)
        df = self._parse_timestamps(df)
        if self.on_duplicate != "raise":
            df = df.drop_duplicates(subset="timestamp", keep=self.on_duplicate)
        return validate_candles(df)

    def _read_file(self) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(f"Source file not found: {self.path}")
        suffix = self.path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(self.path)
        if suffix == ".parquet":
            return pd.read_parquet(self.path)
        raise ValueError(
            f"Unsupported file type {suffix!r} for {self.path}; expected .csv or .parquet"
        )

    def _apply_mapping(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.column_mapping:
            return df
        missing = [src for src in self.column_mapping.values() if src not in df.columns]
        if missing:
            raise ValueError(
                f"column_mapping references columns not in file: {missing}; "
                f"available: {list(df.columns)}"
            )
        rename = {src: canonical for canonical, src in self.column_mapping.items()}
        return df.rename(columns=rename)

    def _parse_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        if "timestamp" not in df.columns:
            raise ValueError(
                f"No 'timestamp' column after mapping; available: {list(df.columns)}"
            )
        raw = df["timestamp"]

        if self.timestamp_unit is not None:
            numeric = pd.to_numeric(raw, errors="raise")
            parsed: pd.Series = pd.to_datetime(
                numeric, unit=self.timestamp_unit, utc=True
            )
        else:
            if pd.api.types.is_numeric_dtype(raw.dtype):
                raise ValueError(
                    "Numeric timestamps require timestamp_unit "
                    "(seconds/milliseconds/microseconds/nanoseconds)"
                )
            parsed = self._parse_datetime_strings(raw)

        out = df.copy()
        out["timestamp"] = parsed
        return out

    def _parse_datetime_strings(self, raw: pd.Series) -> pd.Series:
        try:
            parsed = pd.to_datetime(raw, errors="raise")
        except (ValueError, TypeError):
            # Mixed UTC offsets in one file -> parse element-wise as UTC.
            parsed = pd.to_datetime(raw, errors="raise", format="mixed", utc=True)

        if isinstance(parsed.dtype, pd.DatetimeTZDtype):
            # Aware input (single tz/offset): honor it, convert to UTC.
            return parsed.dt.tz_convert("UTC")
        if pd.api.types.is_datetime64_any_dtype(parsed.dtype):
            # Naive input: the file's timestamps are in input_tz.
            return parsed.dt.tz_localize(self.input_tz).dt.tz_convert("UTC")
        # Object dtype: mixed aware values (pandas couldn't unify them).
        return pd.to_datetime(parsed, utc=True)
