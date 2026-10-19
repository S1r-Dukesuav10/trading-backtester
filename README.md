# trading-backtester

A local, solo research and learning tool for backtesting trading strategies.

**Not financial advice. Backtests are not predictions.** A backtest is a
reconstruction of how a rule set would have behaved on past data; it says
nothing about future results. Nothing in this repo promises or implies profit.

## Status

**Phase 1 — data layer: done.** What works today:

- Crypto data: Binance Vision monthly klines download to canonical Parquet
  (BTCUSDT 1h is verified end to end).
- A data-quality report and a dataset listing command.
- A local CSV/Parquet import reader that converts files to the canonical
  schema.
- OHLC resampling to larger timeframes.

**Gold (XAUUSD): blocked on the data source.** The local import reader handles
the CSV format, and a downloader for Dukascopy's public datafeed is
implemented — but Dukascopy's datafeed has been timing out (requests return
empty bodies) since July 2026, so no XAUUSD candles can be fetched yet. There
is no XAUUSD dataset in `data/` at the moment.

**Not started yet:** the backtest engine, the FastAPI backend, and the web
frontend. No strategy can be run or simulated yet.

## Principles

- Research and learning only. No advice, no performance claims.
- All timestamps are UTC internally; timezone-naive data is rejected rather
  than guessed at.
- No look-ahead: resampled bars are left-closed and left-labelled, so a bar
  never contains data from later than its own label.
- Market data is never committed to the repository.

## Setup (Windows)

Requires **Python 3.12**.

```powershell
cd "C:\Backtesting APP"

# Create the virtual environment once
py -3.12 -m venv .venv

# Loose list of direct dependencies
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
```

For an identical, fully pinned environment (the exact versions this repo was
tested with, including transitive dependencies), install the lock file
instead:

```powershell
.venv\Scripts\python.exe -m pip install -r backend\requirements.lock
```

`backend/requirements.txt` is the loose list you edit when adding a
dependency. `backend/requirements.lock` is generated from the environment with
`python -m pip freeze` and is what you install when you want reproducibility.
Keep both in sync: update `.txt` first, install, re-freeze into `.lock`.

## Data folder layout

Market data lives in a single top-level `data/` directory that is **not** in
the repository. Override its location with the `BACKTESTER_DATA_DIR`
environment variable; otherwise it defaults to `<repo>/data` no matter which
directory a command is run from.

```
data/
  candles/{market}/{symbol}/{timeframe}.parquet   # canonical, validated candles
  raw/binance/...                                # cached Binance Vision zips
  raw/dukascopy/{SYMBOL}/                        # downloaded Dukascopy CSVs
```

For example, the BTCUSDT 1h candles live at
`data/candles/crypto/BTCUSDT/1h.parquet`.

Canonical candles have the columns `timestamp, open, high, low, close, volume`
(plus an optional `spread`), are sorted by timestamp, have unique timestamps,
and are UTC.

## Command line

Run commands from the `backend/` directory so that the `app` package is
importable (from the repo root, prefix with
`$env:PYTHONPATH="backend"` instead).

```powershell
cd "C:\Backtesting APP\backend"
```

### `fetch-binance` — download crypto candles

Downloads Binance Vision monthly klines, converts them to the canonical
schema, and writes one Parquet file per dataset. Already-downloaded months are
reused.

```powershell
..\.venv\Scripts\python.exe -m app.data.cli fetch-binance `
    --symbol BTCUSDT --interval 1h --start 2023-01 --end 2024-12
```

`--end` given as a bare month (`2024-12`) includes that whole month; a full
timestamp is exclusive. Other options: `--binance-market` (default `spot`),
`--storage-market` (default `crypto`), `--raw-root`, `--data-root`, `--delay`.

### `fetch-dukascopy` — download raw Dukascopy CSVs

Downloads one calendar month per invocation, writes raw CSV only, and does not
import or parse anything. It is resumable: months whose file already exists
and contains data are skipped, so it can simply be re-run after an
interruption. It pauses between months and retries failed months with a longer
wait each time.

```powershell
..\.venv\Scripts\python.exe -m app.data.cli fetch-dukascopy `
    --symbol XAUUSD --start 2024-01 --end 2026-09
```

Both `--start` and `--end` are inclusive months. Other options:
`--timeframe` (default `m1`), `--side bid|ask` (default `bid`),
`--no-volumes`, `--raw-root`, `--delay`, `--retry-wait`, `--max-attempts`,
`--abort-after-empty-months`.

As of this writing this command returns empty responses for every month
because Dukascopy's datafeed is not serving data (see Status).

### `report` — data-quality report

```powershell
..\.venv\Scripts\python.exe -m app.data.cli report `
    --market crypto --symbol BTCUSDT --timeframe 1h
```

Checks row count and date range, duplicate timestamps, rows out of
chronological order, gaps (each gap is listed with its start time and size),
invalid candles (low must not exceed open/close, high must not fall below
them), zero or missing volume, and zero-range candles.

The exit code is `1` when an error-level problem is found (duplicates,
unsorted rows, invalid candles, missing volume, an empty dataset) and `0`
otherwise; gaps, zero-volume and zero-range candles are reported as warnings
and still exit `0`.

The expected bar spacing is inferred from the data; override it with
`--step 1h`. For markets that close over the weekend pass `--ignore-weekends`,
which suppresses gaps lying entirely inside the Friday 21:00 to Sunday 22:00
UTC closure (useful for forex/gold). `--examples N` controls how many example
timestamps are printed per issue, and `--data-root` points at a different
candles directory.

### `list` — list stored datasets

```powershell
..\.venv\Scripts\python.exe -m app.data.cli list
```

For each dataset under the candles root it prints the row count, the date
range, and the file size. `--data-root` points at a different candles
directory.

## Code layout

```
backend/
  app/data/
    schema.py          canonical schema + validation
    source.py          DataSource interface ([start, end) semantics)
    local_file.py      local CSV/Parquet import reader
    resample.py        OHLC resampling (left-closed, left-labelled)
    storage.py         canonical Parquet read/write
    quality.py         data-quality checks
    binance_vision.py  Binance Vision downloader
    dukascopy.py       Dukascopy downloader (raw CSV, no import)
    months.py          month-string helpers
    paths.py           data directory resolution
    cli.py             command-line entry point
  tests/               pytest suite
```

## Tests and linting

```powershell
cd "C:\Backtesting APP\backend"

# Tests (no network access; HTTP and subprocesses are mocked)
..\.venv\Scripts\python.exe -m pytest -q

# Lint and format
..\.venv\Scripts\python.exe -m ruff check .
..\.venv\Scripts\python.exe -m ruff format --check .
```

`ruff check --fix .` and `ruff format .` apply the fixes.

## Market data

**No market data is included in this repository**, and `data/` is gitignored.

Downloaded data comes from third-party public sources (Binance Vision,
Dukascopy). Each source has its own terms of use, and you are responsible for
complying with them. Download only what you need, be considerate of the
servers, and use the data for personal research only — do not redistribute
it.
