# Short Stock Monitor

Small Python service that:
- Downloads IBKR public shortstock file (`usa.txt`) once a day
- Extracts borrow rate, rebate, and shares available for a set of tickers
- Compares to previous day and emails alerts when thresholds are breached

## Layout

- `shortstockmonitor.py` – main script
- `config/ym_tickers.csv` – list of tickers to watch
- `requirements.txt` – Python dependencies
- `Dockerfile` – container build
- `k8s/cronjob.yaml` – example Kubernetes CronJob

## Running locally

```bash
pip install -r requirements.txt
python shortstockmonitor.py
```

## Testing the daily pipeline locally

The GitHub Actions workflow runs the three core scripts in sequence: `etf_screener.py`, `ibkr_algo.py`, and `short_stock_monitor.py`. To mirror that locally:

1. Ensure the required environment variables are set (the same ones used in CI):
   - `IBKR_FTP_HOST`, `IBKR_FTP_USER`, `IBKR_FTP_PASS`, `IBKR_FTP_FILE` (defaults to `usa.txt`)
   - `BORROW_CAP`, `MIN_SHARES_AVAILABLE`
   - `IB_HOST`, `IB_PORT`, `IB_CLIENT_ID`, `DRY_RUN`
   - `SCREENED_CSV`, `SNAPSHOT_DIR`, `EMAIL_FROM`, `EMAIL_TO`, `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASS`
2. Run the helper script that uses the same defaults and ordering as the workflow:

```bash
./scripts/run_daily_pipeline.sh
```

The script installs dependencies, applies the same default paths as the workflow, and then executes the three scripts in order so you can validate the end-to-end run.
