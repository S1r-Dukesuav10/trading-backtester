"""Shared helpers for building small synthetic test data."""

from __future__ import annotations

from pathlib import Path

BASIC_HEADER = "timestamp,open,high,low,close,volume"
SPREAD_HEADER = "timestamp,open,high,low,close,volume,spread"


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
