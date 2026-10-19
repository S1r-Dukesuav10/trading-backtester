"""Small CLI for the data layer.

Usage:
    python -m app.data.cli fetch-binance \
        --symbol BTCUSDT --interval 1h --start 2023-01 --end 2024-12
    python -m app.data.cli report --market crypto --symbol BTCUSDT --timeframe 1h
    python -m app.data.cli list
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from .binance_vision import (
    DEFAULT_DELAY_SECONDS,
    DEFAULT_MARKET,
    BinanceVisionError,
    BinanceVisionSource,
)
from .dukascopy import (
    DEFAULT_ABORT_AFTER_EMPTY_MONTHS,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_RETRY_WAIT_SECONDS,
    DEFAULT_SIDE,
    DEFAULT_SYMBOL,
    DEFAULT_TIMEFRAME,
    DukascopyDownloader,
    DukascopyError,
    format_report,
)
from .months import month_range
from .paths import ENV_DATA_DIR, candles_root
from .quality import (
    MAX_EXAMPLES,
    check_dataset,
    find_datasets,
    format_dataset_list,
    format_quality_report,
)
from .storage import candles_path, save_candles

DEFAULT_STORAGE_MARKET = "crypto"

DATA_ROOT_HELP = (
    "root for canonical parquet "
    f"(default: ${ENV_DATA_DIR}/candles, else <repo>/data/candles)"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="app.data.cli",
        description="Data-layer utilities for the trading backtester.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch = subparsers.add_parser(
        "fetch-binance",
        help="Download Binance Vision monthly klines and cache canonical Parquet.",
    )
    fetch.add_argument("--symbol", required=True, help="e.g. BTCUSDT")
    fetch.add_argument("--interval", required=True, help="e.g. 1h")
    fetch.add_argument("--start", required=True, help="YYYY-MM (or an ISO timestamp)")
    fetch.add_argument(
        "--end",
        required=True,
        help="YYYY-MM (whole month included) or an ISO timestamp (exclusive)",
    )
    fetch.add_argument("--binance-market", default=DEFAULT_MARKET)
    fetch.add_argument(
        "--raw-root",
        default=None,
        help=(
            "cache directory for downloaded zips "
            f"(default: ${ENV_DATA_DIR}/raw/binance, else <repo>/data/raw/binance)"
        ),
    )
    fetch.add_argument(
        "--data-root",
        default=None,
        help=DATA_ROOT_HELP,
    )
    fetch.add_argument(
        "--storage-market",
        default=DEFAULT_STORAGE_MARKET,
        help="storage bucket for the canonical parquet",
    )
    fetch.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY_SECONDS,
        help="seconds to pause between downloads (politeness)",
    )

    fetch_duka = subparsers.add_parser(
        "fetch-dukascopy",
        help=(
            "Download Dukascopy monthly candles as raw CSV "
            "(download only - no import/parsing)."
        ),
    )
    fetch_duka.add_argument("--symbol", default=DEFAULT_SYMBOL, help="e.g. XAUUSD")
    fetch_duka.add_argument(
        "--start", required=True, help="first month, YYYY-MM (inclusive)"
    )
    fetch_duka.add_argument(
        "--end", required=True, help="last month, YYYY-MM (inclusive)"
    )
    fetch_duka.add_argument("--timeframe", default=DEFAULT_TIMEFRAME, help="e.g. m1")
    fetch_duka.add_argument(
        "--side",
        default=DEFAULT_SIDE,
        choices=["bid", "ask"],
        help="price side to download",
    )
    fetch_duka.add_argument(
        "--volumes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "include Dukascopy volume data (default: on; the canonical schema "
            "requires a volume column)"
        ),
    )
    fetch_duka.add_argument(
        "--raw-root",
        default=None,
        help=(
            "cache directory for downloaded CSVs "
            f"(default: ${ENV_DATA_DIR}/raw/dukascopy, else "
            "<repo>/data/raw/dukascopy)"
        ),
    )
    fetch_duka.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY_SECONDS,
        help="seconds to pause between months",
    )
    fetch_duka.add_argument(
        "--retry-wait",
        type=float,
        default=DEFAULT_RETRY_WAIT_SECONDS,
        help="base seconds to wait before retrying a failed month",
    )
    fetch_duka.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help="attempts per month before giving up",
    )
    fetch_duka.add_argument(
        "--abort-after-empty-months",
        type=int,
        default=DEFAULT_ABORT_AFTER_EMPTY_MONTHS,
        help=(
            "stop the run after this many consecutive empty months "
            "(guards against an unreachable datafeed); 0 disables"
        ),
    )

    report = subparsers.add_parser(
        "report",
        help="Data-quality report for one stored dataset.",
    )
    report.add_argument("--market", required=True, help="e.g. crypto")
    report.add_argument("--symbol", required=True, help="e.g. BTCUSDT")
    report.add_argument("--timeframe", required=True, help="e.g. 1h")
    report.add_argument("--data-root", default=None, help=DATA_ROOT_HELP)
    report.add_argument(
        "--step",
        default=None,
        help=("expected bar spacing, e.g. 1h or 15m (default: infer from the data)"),
    )
    report.add_argument(
        "--ignore-weekends",
        action="store_true",
        help=(
            "suppress gaps that lie entirely inside the weekend closure "
            "(Friday 21:00 -> Sunday 22:00 UTC), for forex/gold data"
        ),
    )
    report.add_argument(
        "--examples",
        type=int,
        default=MAX_EXAMPLES,
        help="example timestamps to print per issue (default: %(default)s)",
    )

    list_cmd = subparsers.add_parser(
        "list",
        help="List stored datasets with row counts and date ranges.",
    )
    list_cmd.add_argument("--data-root", default=None, help=DATA_ROOT_HELP)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    http_get: Callable[[str], bytes] | None = None,
    sleep: Callable[[float], None] | None = None,
    run: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "fetch-binance":
        return _fetch_binance(args, http_get=http_get, sleep=sleep)
    if args.command == "fetch-dukascopy":
        return _fetch_dukascopy(args, run=run, sleep=sleep)
    if args.command == "report":
        return _report(args)
    if args.command == "list":
        return _list(args)
    return 2  # pragma: no cover - argparse rejects unknown commands


def _fetch_binance(
    args: argparse.Namespace,
    *,
    http_get: Callable[[str], bytes] | None,
    sleep: Callable[[float], None] | None,
) -> int:
    source = BinanceVisionSource(
        args.raw_root,
        market=args.binance_market,
        delay_seconds=args.delay,
        http_get=http_get,
        sleep=sleep,
    )
    try:
        candles = source.get_candles(args.symbol, args.interval, args.start, args.end)
        path = save_candles(
            candles,
            args.data_root,
            market=args.storage_market,
            symbol=args.symbol,
            timeframe=args.interval,
        )
    except (BinanceVisionError, ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"{args.symbol} {args.interval} {args.start}..{args.end}: "
        f"{len(candles)} candles"
    )
    print(
        f"  range: {candles['timestamp'].iloc[0]} -> "
        f"{candles['timestamp'].iloc[-1]} (UTC)"
    )
    print(f"  saved: {path}")
    return 0


def _fetch_dukascopy(
    args: argparse.Namespace,
    *,
    run: Callable[..., subprocess.CompletedProcess[str]] | None,
    sleep: Callable[[float], None] | None,
) -> int:
    downloader = DukascopyDownloader(
        args.raw_root,
        symbol=args.symbol,
        timeframe=args.timeframe,
        side=args.side,
        volumes=args.volumes,
        delay_seconds=args.delay,
        max_attempts=args.max_attempts,
        retry_wait_seconds=args.retry_wait,
        abort_after_empty_months=args.abort_after_empty_months,
        run=run,
        sleep=sleep,
    )
    try:
        months = month_range(args.start, args.end)
        report = downloader.download_months(months)
    except (DukascopyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(format_report(report, root=downloader.root))
    if not report.ok:
        print(
            "error: some months failed or came back empty; re-run to retry "
            "the missing months",
            file=sys.stderr,
        )
        return 1
    return 0


def _report(args: argparse.Namespace) -> int:
    path = candles_path(args.data_root, args.market, args.symbol, args.timeframe)
    if not path.exists():
        print(f"error: no dataset at {path}", file=sys.stderr)
        return 1

    try:
        report = check_dataset(
            path,
            market=args.market,
            symbol=args.symbol,
            timeframe=args.timeframe,
            step=args.step,
            ignore_weekends=args.ignore_weekends,
            examples=args.examples,
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary, never traceback
        print(f"error: could not analyse {path}: {exc}", file=sys.stderr)
        return 1

    print(format_quality_report(report))
    return 0 if report.ok else 1


def _list(args: argparse.Namespace) -> int:
    root = candles_root() if args.data_root is None else Path(args.data_root)
    print(format_dataset_list(find_datasets(args.data_root), root=root))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
