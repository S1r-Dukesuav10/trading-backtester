# Project: trading-backtester

Research and learning tool for backtesting trading strategies. Not financial advice. Never promise or imply profits.

## Stack
- Backend: Python 3.12, FastAPI, pandas, numpy, SQLite, Parquet for candles
- Frontend: Vite + React + TypeScript, TradingView Lightweight Charts v5 API (series via constructors, e.g. CandlestickSeries). Keep the TradingView attribution/logo visible.
- Tests: pytest. Lint/format: ruff.
- Data layer: a pluggable data-source interface. Local CSV/Parquet import is the first source. APIs get added later as separate sources.

## Hard rules
1. No look-ahead bias. A signal on bar N may only fill at the open of bar N+1. Indicators must only use data up to the current bar.
2. Fees, slippage and spread are ON by default and configurable. Never produce a backtest with zero costs unless explicitly asked.
3. If stop-loss and take-profit are both touched in one bar, assume the stop-loss hit first.
4. All timestamps are UTC internally.
5. Never hardcode API keys. Use environment variables and a gitignored .env file. Never print or log secrets.
6. Never commit raw market data. data/ is gitignored.
7. Do not copy or imitate proprietary code or assets from TradingView or Tradezella. Take inspiration from public features only.
8. Before adding any dependency: say what it is, its license, and when it was last released. Prefer MIT/Apache/BSD. Ask before adding anything GPL/AGPL.

## Working style
- Small steps. One feature per task. Write or update tests with each feature and run them.
- Explain what you changed in plain language at the end.
- Keep the README current.
- Do not make large refactors without asking.
