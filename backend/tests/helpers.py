"""Shared helpers for building small synthetic test data."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

BASIC_HEADER = "timestamp,open,high,low,close,volume"
SPREAD_HEADER = "timestamp,open,high,low,close,volume,spread"

BINANCE_HEADER = (
    "open_time,open,high,low,close,volume,close_time,quote_asset_volume,"
    "number_of_trades,taker_buy_base_asset_volume,"
    "taker_buy_quote_asset_volume,ignore"
)


def binance_row(
    open_time: int,
    *,
    open_: float = 100.0,
    high: float = 101.0,
    low: float = 99.0,
    close: float = 100.5,
    volume: float = 10.0,
    step_ms: int = 3_600_000,
) -> str:
    """One synthetic Binance kline CSV row (12 columns, positional layout)."""
    close_time = open_time + step_ms - 1
    return (
        f"{open_time},{open_},{high},{low},{close},{volume},{close_time},"
        f"{volume * close},1,0,0,0"
    )


def make_binance_zip(
    rows: list[str],
    *,
    header: bool = False,
    csv_name: str = "BTCUSDT-1h-2023-01.csv",
) -> bytes:
    """Build an in-memory ZIP shaped like a Binance Vision kline archive."""
    lines = [BINANCE_HEADER] if header else []
    lines.extend(rows)
    text = "\n".join(lines) + "\n"

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(csv_name, text)
    return buffer.getvalue()


def write_csv(
    tmp_path: Path,
    rows: list[str],
    name: str = "candles.csv",
    header: str = BASIC_HEADER,
) -> Path:
    """Write a small synthetic CSV inside the test's tmp dir."""
    path = tmp_path / name
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path
