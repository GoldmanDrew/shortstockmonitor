#!/usr/bin/env python3
"""
etf_screener.py (FTP version)

Workflow:

1) Load ETF list + CAGRs from config/etf_cagr.csv
   - Required columns: ETF, cagr_port_hist
   - Optional: Underlying, LevType, Leverage, etc.

2) Fetch IBKR shortstock file from FTP (default: usa.txt):
   - Columns include: SYM, AVAILABLE, FEERATE, REBATERATE, etc.
   - Compute net_borrow_annual = max(fee_annual - rebate_annual, 0)

3) For each ETF, map:
      borrow_current   = net_borrow_annual (decimal, e.g. 0.12 = 12%)
      shares_available = AVAILABLE
      borrow_spiking   = False (placeholder)

4) Apply screening rules (same as previous working version):
      - include if (borrow <= cap) OR (whitelisted)
      - exclude if shares_available < MIN_SHARES_AVAILABLE
      - exclude if borrow_spiking == True (placeholder currently always False)
      - NOTE: CAGR is informational only (NOT an exclusion)

5) Outputs (UPDATED: date-partitioning):
      - Always write dated output to:
            data/runs/YYYY-MM-DD/etf_screened_today.csv
      - Optionally also write the latest convenience copy to:
            data/etf_screened_today.csv
"""

from __future__ import annotations

import argparse
import io
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Iterable

import ftplib
import pandas as pd
import yaml

# --------------------------------------------------
# PATHS / CONFIG
# --------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(os.getenv("GITHUB_WORKSPACE", str(SCRIPT_DIR))).resolve()

# Strategy config (YAML)
STRATEGY_CONFIG = Path(os.getenv("STRATEGY_CONFIG", str(REPO_ROOT / "strategy_config.yml")))
if not STRATEGY_CONFIG.exists():
    alt = REPO_ROOT / "config" / "strategy_config.yml"
    if alt.exists():
        STRATEGY_CONFIG = alt


def load_strategy_config(path: Path = STRATEGY_CONFIG) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# Single input file: ETF + CAGRs (+ optional metadata)
CAGR_CSV = Path(os.getenv("CAGR_CSV", str(REPO_ROOT / "config" / "etf_cagr.csv")))

# Output (latest)
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(REPO_ROOT / "data")))
OUTPUT_FILE = Path(os.getenv("OUTPUT_FILE", str(OUTPUT_DIR / "etf_screened_today.csv")))

# IBKR FTP config
FTP_HOST = os.getenv("IBKR_FTP_HOST") or "ftp2.interactivebrokers.com"
FTP_USER = os.getenv("IBKR_FTP_USER") or "shortstock"
FTP_PASS = os.getenv("IBKR_FTP_PASS") or ""
FTP_FILE = os.getenv("IBKR_FTP_FILE") or "usa.txt"

# Screening thresholds (defaults)
BORROW_CAP_DEFAULT = 0.12
MIN_SHARES_AVAILABLE_DEFAULT = 1000


@dataclass
class ScreeningParams:
    borrow_cap: float = BORROW_CAP_DEFAULT
    min_shares_available: int = MIN_SHARES_AVAILABLE_DEFAULT
    borrow_whitelist_etfs: set | None = None


# --------------------------------------------------
# DATE-PARTITIONED OUTPUT HELPERS
# --------------------------------------------------

def today_str() -> str:
    return date.today().isoformat()


def run_dir(run_date: str) -> Path:
    return REPO_ROOT / "data" / "runs" / run_date


# --------------------------------------------------
# FTP SHORT FILE HELPERS (UNCHANGED WORKING BEHAVIOR)
# --------------------------------------------------

def fetch_ibkr_shortstock_file(filename: str = FTP_FILE) -> pd.DataFrame:
    """
    Download IBKR shortstock file from FTP and parse it into a DataFrame.

    The file is pipe-delimited with a header line starting with '#SYM|...'.
    We:
      - Find that header line
      - Build lowercase column names from it
      - Parse the remaining lines as data
    """
    if not FTP_HOST:
        raise RuntimeError("IBKR_FTP_HOST is empty/unset.")
    if not FTP_USER:
        raise RuntimeError("IBKR_FTP_USER is empty/unset.")

    print(f"Connecting to IBKR FTP: {FTP_HOST}, file: {filename}")

    ftp = ftplib.FTP(timeout=30)
    try:
        ftp.connect(FTP_HOST, 21)
        ftp.login(user=FTP_USER, passwd=FTP_PASS)
        ftp.set_pasv(True)

        buf = io.BytesIO()
        ftp.retrbinary(f"RETR {filename}", buf.write)
        print("FTP download complete, parsing...")
    finally:
        try:
            ftp.quit()
        except Exception:
            try:
                ftp.close()
            except Exception:
                pass

    text = buf.getvalue().decode("utf-8", errors="ignore")
    lines = [ln for ln in text.splitlines() if ln.strip()]

    header_idx = None
    for i, ln in enumerate(lines):
        if ln.startswith("#SYM|"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not find header line starting with '#SYM|'")

    header_line = lines[header_idx]
    data_lines = lines[header_idx + 1 :]

    header_cols = [c.strip().lstrip("#").lower() for c in header_line.split("|")]

    df = pd.read_csv(
        io.StringIO("\n".join(data_lines)),
        sep="|",
        header=None,
        engine="python",
    )

    n_cols = min(len(header_cols), df.shape[1])
    df = df.iloc[:, :n_cols]
    df.columns = header_cols[:n_cols]

    df = df.drop(columns=[c for c in df.columns if not c or str(c).startswith("unnamed")], errors="ignore")

    print(f"Parsed {df.shape[0]} rows and {df.shape[1]} columns from FTP file.")
    return df


def get_ibkr_borrow_snapshot_from_ftp(etf_list: Iterable[str]) -> pd.DataFrame:
    """
    Given a list of ETF tickers, return a DataFrame with:
        ETF, borrow_current, shares_available, borrow_spiking

    borrow_current is net borrow (fee - rebate, floored at 0) as decimal.
    shares_available comes from 'available' column.
    """
    etf_list = list(dict.fromkeys([str(x).strip().upper() for x in etf_list]))

    short_df = fetch_ibkr_shortstock_file(FTP_FILE)

    if "sym" not in short_df.columns:
        raise ValueError(f"Expected 'sym' column in IBKR FTP file; got: {list(short_df.columns)}")

    short_df["sym"] = short_df["sym"].astype(str).str.upper().str.strip()

    # Working expectation for IBKR usa.txt in your environment:
    if "rebaterate" not in short_df.columns or "feerate" not in short_df.columns:
        raise ValueError("Expected 'rebaterate' and 'feerate' columns in FTP file.")

    short_df["rebate_annual"] = pd.to_numeric(short_df["rebaterate"], errors="coerce") / 100.0
    short_df["fee_annual"] = pd.to_numeric(short_df["feerate"], errors="coerce") / 100.0
    short_df["available_int"] = pd.to_numeric(short_df.get("available", 0), errors="coerce")

    short_df["net_borrow_annual"] = (short_df["fee_annual"] - short_df["rebate_annual"]).clip(lower=0)

    sub = short_df[short_df["sym"].isin(etf_list)].copy()

    records: list[Dict] = []
    for _, row in sub.iterrows():
        sym = row["sym"]
        records.append(
            {
                "ETF": sym,
                "borrow_current": float(row["net_borrow_annual"]) if pd.notna(row["net_borrow_annual"]) else float("nan"),
                "shares_available": int(row["available_int"]) if pd.notna(row["available_int"]) else 0,
            }
        )

    missing = [sym for sym in etf_list if sym not in sub["sym"].values]
    for sym in missing:
        records.append({"ETF": sym, "borrow_current": float("nan"), "shares_available": 0})

    df_out = pd.DataFrame(records)
    df_out["borrow_spiking"] = False  # placeholder
    return df_out


# --------------------------------------------------
# SCREENING LOGIC (UNCHANGED WORKING BEHAVIOR)
# --------------------------------------------------

def screen_universe_for_algo(df: pd.DataFrame, params: ScreeningParams) -> pd.DataFrame:
    """
    df must contain at least:
        ETF, cagr_port_hist, borrow_current, shares_available, borrow_spiking (bool)
    Returns df with include_for_algo and diagnostic flags.
    """
    df = df.copy()

    df["cagr_positive"] = df["cagr_port_hist"] > 0  # informational only
    df["borrow_leq_cap"] = df["borrow_current"] <= params.borrow_cap
    df["borrow_gt_cap"] = ~df["borrow_leq_cap"]

    if "borrow_spiking" not in df.columns:
        df["borrow_spiking"] = False

    df["shares_available"] = df["shares_available"].fillna(0).astype(int)

    wl = params.borrow_whitelist_etfs or set()
    df["whitelisted"] = df["ETF"].isin(wl)

    # Keep diagnostics, but do NOT use shares_available / borrow_spiking to gate inclusion
    df["exclude_borrow_gt_cap"] = df["borrow_gt_cap"] & ~df["whitelisted"]

    # informational only (still written to CSV)
    df["exclude_no_shares"] = df["shares_available"] < params.min_shares_available
    df["exclude_borrow_spike"] = df["borrow_spiking"].fillna(False)

    # Simplified inclusion rule:
    df["include_for_algo"] = df["borrow_leq_cap"] | df["whitelisted"]
    return df


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run-date",
        default=os.getenv("RUN_DATE") or today_str(),
        help="YYYY-MM-DD; used for data/runs/<date>/ outputs (default: today).",
    )
    ap.add_argument(
        "--no-write-latest",
        action="store_true",
        help="If set, do not write/overwrite data/etf_screened_today.csv (latest).",
    )
    args = ap.parse_args()

    run_date = args.run_date
    dated_dir = run_dir(run_date)
    dated_dir.mkdir(parents=True, exist_ok=True)
    dated_output = dated_dir / "etf_screened_today.csv"

    print(f"Repo root: {REPO_ROOT}")
    print(f"CAGR CSV: {CAGR_CSV}")
    print(f"Run date: {run_date}")
    print(f"Dated output: {dated_output}")
    print(f"Latest output: {OUTPUT_FILE} (write_latest={not args.no_write_latest})")

    # Load strategy config (nested screener block)
    cfg = load_strategy_config()
    screener_cfg = (cfg or {}).get("screener", {}) or {}

    borrow_cap = float(screener_cfg.get("borrow_cap", BORROW_CAP_DEFAULT))
    min_shares_available = int(screener_cfg.get("min_shares_available", MIN_SHARES_AVAILABLE_DEFAULT))

    # Normalize whitelist in the same way we normalize ETFs from etf_cagr.csv
    wl_raw = screener_cfg.get("borrow_whitelist_etfs", []) or []
    whitelist = {str(x).strip().upper().replace(".", "-") for x in wl_raw if str(x).strip()}

    params = ScreeningParams(
        borrow_cap=borrow_cap,
        min_shares_available=min_shares_available,
        borrow_whitelist_etfs=whitelist,
    )

    print(f"Borrow cap: {params.borrow_cap:.2%}")
    print(f"Min shares available: {params.min_shares_available}")
    print(f"Whitelist size: {len(whitelist)}")
    if whitelist:
        print("Whitelist (sample):", sorted(list(whitelist))[:15])

    # 1) Load ETF + CAGR file
    if not CAGR_CSV.exists():
        raise FileNotFoundError(f"CAGR CSV not found: {CAGR_CSV}")

    cagr_df = pd.read_csv(CAGR_CSV)
    print("Loaded etf_cagr head:\n", cagr_df.head())

    if "ETF" not in cagr_df.columns or "cagr_port_hist" not in cagr_df.columns:
        raise ValueError(f"{CAGR_CSV} must contain columns: 'ETF', 'cagr_port_hist'")

    # Normalize ETF symbols
    cagr_df["ETF"] = cagr_df["ETF"].astype(str).str.strip().str.replace(".", "-", regex=False).str.upper()

    # 2) Borrow snapshot from IBKR FTP
    borrow_df = get_ibkr_borrow_snapshot_from_ftp(cagr_df["ETF"].unique())

    # 3) Merge CAGRs and borrow data
    metrics = cagr_df.merge(borrow_df, on="ETF", how="left")
    print("Merged metrics sample:\n", metrics.head())

    # 4) Screen
    screened = screen_universe_for_algo(metrics, params=params)

    # 5) Write outputs: dated always; latest optional
    dated_output.parent.mkdir(parents=True, exist_ok=True)
    screened.to_csv(dated_output, index=False)
    print(f"\n[SCREENER] Saved dated screened universe: {dated_output}")

    if not args.no_write_latest:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        screened.to_csv(OUTPUT_FILE, index=False)
        print(f"[SCREENER] Updated latest screened universe: {OUTPUT_FILE}")
    else:
        print("[SCREENER] Skipped writing latest output (per --no-write-latest).")

    included = screened[screened["include_for_algo"]]
    print(f"\nIncluded {len(included)} ETFs:")
    cols_to_show = [c for c in ["Underlying", "ETF", "LevType"] if c in included.columns]
    if cols_to_show:
        print(included[cols_to_show].sort_values(cols_to_show[0]))
    else:
        print(included[["ETF"]].sort_values("ETF"))


if __name__ == "__main__":
    main()
