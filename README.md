# LS-ALGO

LS-ALGO is a Python pipeline for running a rules-driven long/short ETF strategy end to end. The system screens a configured universe, builds target pairs, optionally trades them through Interactive Brokers (IBKR), and monitors short borrow conditions for alerts.

## Components

### 1) ETF Screener — `etf_screener.py`
- Loads ETF metadata and historical CAGR values from `config/etf_cagr.csv`.
- Downloads the IBKR short-stock file from FTP (`IBKR_FTP_HOST`, `IBKR_FTP_USER`, `IBKR_FTP_PASS`, `IBKR_FTP_FILE`).
- Computes net borrow (`fee - rebate`), available shares, and screening flags.
- Applies rules driven by `BORROW_CAP` and `MIN_SHARES_AVAILABLE` to set `include_for_algo`.
- Writes the full table to `data/etf_screened_today.csv` for downstream steps.

### 2) Trading Logic — `ibkr_algo.py`
- Reads the screened universe (`SCREENED_CSV`, defaults to `data/etf_screened_today.csv`).
- Merges with `config/etf_cagr.csv` to map underlying/ETF pairs and leverage type.
- Connects to IBKR via `ib_insync` (`IB_HOST`, `IB_PORT`, `IB_CLIENT_ID`).
- If no positions exist, splits equity across covered-call (CC) and 2x leveraged pairs, sizes orders, and submits Adaptive Passive limit orders. Set `DRY_RUN=1` to log without placing orders.
- Saves intended positions to `data/positions_state.csv` and snapshots IB positions to `data/ib_positions_*.csv`.

### 3) Short-Stock Monitor — `short_stock_monitor.py`
- Loads the current ETF watchlist from the screened CSV (respects `include_for_algo` when present).
- Fetches the IBKR short-stock file, saves a dated snapshot under `data/shortstock_snapshots/`, and builds borrow/availability maps.
- Compares against the most recent prior snapshot to surface borrow spikes or supply drops based on `BORROW_ABS_THRESHOLD`, `BORROW_CHANGE_THRESHOLD`, `AVAIL_ABS_THRESHOLD`, and `AVAIL_CHANGE_THRESHOLD`.
- Prints alerts and optionally emails them (`EMAIL_FROM`, `EMAIL_TO`, `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASS`).

### Pipeline Wrapper — `scripts/run_daily_pipeline.sh`
Runs the three components in sequence using the same defaults as CI. Exports standard paths, installs dependencies, and then executes `etf_screener.py`, `ibkr_algo.py`, and `short_stock_monitor.py`.

## Repository Layout

- `config/`
  - `etf_cagr.csv` — ETF universe with CAGRs, underlying mapping, and leverage type.
  - `tickers.csv` — Additional ticker list used by older monitoring workflows.
- `data/` — Outputs and state (screened CSVs, IB position snapshots, short-stock snapshots).
- `scripts/run_daily_pipeline.sh` — Local wrapper mirroring the GitHub Actions sequence.
- `etf_screener.py` — Screener and borrow ingestion.
- `ibkr_algo.py` — IBKR portfolio construction and state sync.
- `short_stock_monitor.py` — Borrow/availability monitoring and alerting.
- `.github/workflows/` — CI schedules for the daily pipeline and standalone monitor runs.
- `requirements.txt` — Python dependencies for the core scripts.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Ensure access to the IBKR FTP short-stock feed and, if trading live, an IBKR TWS/Gateway session with API enabled.

## Running Locally

### Individual scripts
```bash
python etf_screener.py
python ibkr_algo.py
python short_stock_monitor.py
```

### Full daily sequence
Export any needed environment variables (see below), then run:
```bash
./scripts/run_daily_pipeline.sh
```

## Key Environment Variables

### Screener
- `CAGR_CSV` — Path to `etf_cagr.csv` (default `config/etf_cagr.csv`).
- `OUTPUT_DIR`, `OUTPUT_FILE` — Where to write the screened CSV (default `data/etf_screened_today.csv`).
- `IBKR_FTP_HOST`, `IBKR_FTP_USER`, `IBKR_FTP_PASS`, `IBKR_FTP_FILE` — FTP connection and filename (`usa.txt` by default).
- `BORROW_CAP` — Maximum acceptable borrow before exclusion (default `0.10`).
- `MIN_SHARES_AVAILABLE` — Minimum available shares required (default `1000`).

### Trading
- `SCREENED_CSV` — Input from the screener (`data/etf_screened_today.csv`).
- `IB_HOST`, `IB_PORT`, `IB_CLIENT_ID` — IBKR API connection details.
- `DRY_RUN` — Set to `1` to log actions without placing orders.
- `MAX_SHARES_PER_ORDER` — Clamp per-leg order size (default `500`).

### Monitoring & Alerts
- `SCREENED_CSV` — Watchlist source (defaults to screener output).
- `SNAPSHOT_DIR` — Directory for dated short-stock CSV snapshots (default `data/shortstock_snapshots`).
- `IBKR_FTP_FILE` — FTP filename to monitor (`usa.txt` default).
- `BORROW_ABS_THRESHOLD`, `BORROW_CHANGE_THRESHOLD` — Borrow-level alert thresholds.
- `AVAIL_ABS_THRESHOLD`, `AVAIL_CHANGE_THRESHOLD` — Availability alert thresholds.
- `EMAIL_FROM`, `EMAIL_TO`, `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASS` — Email settings.

## Automation

- **Daily ETF Pipeline** (`.github/workflows/daily_pipeline.yml`): runs screener → trading logic → short-stock monitor with secrets for FTP, IBKR, and email.
- **Short Stock Monitor** (`.github/workflows/monitor.yml`): legacy workflow to run the monitor independently on a cron schedule.

## Outputs

- `data/etf_screened_today.csv` — Screened ETF universe with borrow metrics and inclusion flags.
- `data/positions_state.csv` — Intended positions saved by the trading script.
- `data/ib_positions_*.csv` — Snapshots of current IBKR positions.
- `data/shortstock_snapshots/` — Historical short borrow files and comparisons used for alerts.

