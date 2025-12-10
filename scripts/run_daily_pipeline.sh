#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN=${PYTHON:-python}

# Default paths to mirror the GitHub Actions workflow.
export PYTHONUNBUFFERED=1
export GITHUB_WORKSPACE="${GITHUB_WORKSPACE:-$ROOT_DIR}"
export CAGR_CSV="${CAGR_CSV:-$ROOT_DIR/config/etf_cagr.csv}"
export OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/data}"
export OUTPUT_FILE="${OUTPUT_FILE:-$OUTPUT_DIR/etf_screened_today.csv}"
export SCREENED_CSV="${SCREENED_CSV:-$OUTPUT_FILE}"
export SNAPSHOT_DIR="${SNAPSHOT_DIR:-$ROOT_DIR/data/shortstock_snapshots}"
export IBKR_FTP_FILE="${IBKR_FTP_FILE:-usa.txt}"

cd "$ROOT_DIR"

if [ ! -d .venv ] && ! command -v $PYTHON_BIN >/dev/null; then
  echo "Python not found; set PYTHON to your interpreter path." >&2
  exit 1
fi

$PYTHON_BIN -m pip install --upgrade pip >/dev/null
$PYTHON_BIN -m pip install -r requirements.txt >/dev/null

echo "Running etf_screener.py..."
$PYTHON_BIN etf_screener.py

echo "Running ibkr_algo.py..."
$PYTHON_BIN ibkr_algo.py

echo "Running short_stock_monitor.py..."
$PYTHON_BIN short_stock_monitor.py
