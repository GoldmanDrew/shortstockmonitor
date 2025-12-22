#!/usr/bin/env python3
"""
ibkr_algo.py (FTP version)

Workflow:

1) Load ETF list + CAGRs from config/etf_cagr.csv
   - Required columns: ETF, cagr_port_hist
   - Optional: Underlying, LevType

2) Fetch IBKR shortstock file from FTP (default: usa.txt):
   - Columns include: SYM, AVAILABLE, FEE_RATE, REBATE_RATE, etc.
   - Compute net_borrow_annual = max(fee_annual - rebate_annual, 0)

3) For each ETF, map:
      borrow_current   = net_borrow_annual (decimal, e.g. 0.12 = 12%)
      shares_available = AVAILABLE

4) Apply screening rules:
      - cagr_positive = cagr_port_hist > 0
      - "cheap" borrow if borrow_current <= BORROW_CAP
      - allow expensive borrow (> BORROW_CAP) only if CAGR > 0
      - exclude if borrow > BORROW_CAP AND CAGR <= 0
      - exclude if shares_available < MIN_SHARES_AVAILABLE
      - placeholder borrow_spiking (False for now)

5) Save full table (including include_for_algo) to data/etf_screened_today.csv.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Dict

import ftplib
import pandas as pd

# --------------------------------------------------
# PATHS / CONFIG
# --------------------------------------------------

# Directory where this script lives
SCRIPT_DIR = Path(__file__).resolve().parent

# Treat this as repo root unless GITHUB_WORKSPACE overrides it
REPO_ROOT = Path(os.getenv("GITHUB_WORKSPACE", str(SCRIPT_DIR))).resolve()

# Single input file: ETF + CAGRs (+ optional metadata)
CAGR_CSV = Path(os.getenv("CAGR_CSV", str(REPO_ROOT / "config" / "etf_cagr.csv")))

# Output
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(REPO_ROOT / "data")))
OUTPUT_FILE = Path(os.getenv("OUTPUT_FILE", str(OUTPUT_DIR / "etf_screened_today.csv")))

# IBKR FTP config
# IMPORTANT: use `or` so empty strings injected by CI don't override defaults
FTP_HOST = os.getenv("IBKR_FTP_HOST") or "ftp2.interactivebrokers.com"
FTP_USER = os.getenv("IBKR_FTP_USER") or "shortstock"
FTP_PASS = os.getenv("IBKR_FTP_PASS") or ""
FTP_FILE = os.getenv("IBKR_FTP_FILE") or "usa.txt"

# Screening thresholds
BORROW_CAP = float(os.getenv("BORROW_CAP", "0.10"))                   # 10% "cheap" borrow cap
MIN_SHARES_AVAILABLE = int(os.getenv("MIN_SHARES_AVAILABLE", "1000")) # minimum shortable


@dataclass
class ScreeningParams:
    borrow_cap: float = BORROW_CAP
    min_shares_available: int = MIN_SHARES_AVAILABLE


# --------------------------------------------------
# FTP SHORT FILE HELPERS
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

    # Make connection explicit so we never hit ftp.sock == None in login()
    ftp = ftplib.FTP(timeout=30)
    try:
        ftp.connect(FTP_HOST, 21)
        ftp.login(user=FTP_USER, passwd=FTP_PASS)
        ftp.set_pasv(True)  # often helps on CI runners / firewalls

        buf = io.BytesIO()
        ftp.retrbinary(f"RETR {filename}", buf.write)
        print("FTP download complete, parsing...")
    finally:
        try:
            ftp.quit()
        except Exception:
            # if quit fails (connection already dropped), ensure close
            try:
                ftp.close()
            except Exception:
                pass

    buf.seek(0)
    text = buf.getvalue().decode("utf-8", errors="ignore")
    lines = [ln for ln in text.splitlines() if ln.strip()]

    # Find the header line that starts with #SYM|
    header_idx = None
    for i, ln in enumerate(lines):
        if ln.startswith("#SYM|"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not find header line starting with '#SYM|'")

    header_line = lines[header_idx]
    data_lines = lines[header_idx + 1:]

    # Build column names from header
    header_cols = [c.strip().lstrip("#").lower() for c in header_line.split("|")]

    # Join data lines back to a CSV-like string
    data_str = "\n".join(data_lines)
    data_buf = io.StringIO(data_str)

    # Read data; allow Python engine for safety
    df = pd.read_csv(
        data_buf,
        sep="|",
        header=None,
        engine="python"
    )

    # Trim / align columns
    n_cols = min(len(header_cols), df.shape[1])
    df = df.iloc[:, :n_cols]
    df.columns = header_cols[:n_cols]

    # Drop any empty / unnamed trailing columns
    df = df.drop(
        columns=[c for c in df.columns if not c or str(c).startswith("unnamed")],
        errors="ignore"
    )

    print(f"Parsed {df.shape[0]} rows and {df.shape[1]} columns from FTP file.")
    return df


def get_ibkr_borrow_snapshot_from_ftp(etf_list: Iterable[str]) -> pd.DataFrame:
    """
    Given a list of ETF tickers, return a DataFrame with:
        ETF, borrow_current, shares_available, borrow_spiking

    borrow_current is net borrow (fee - rebate, floored at 0) as decimal.
    shares_available comes from 'available' column.
    """
    etf_list = list(dict.fromkeys([str(x).strip().upper() for x in etf_list]))  # unique

    short_df = fetch_ibkr_shortstock_file(FTP_FILE)

    # Normalize symbol column from FTP
    if "sym" not in short_df.columns:
        raise ValueError("Expected 'sym' column in IBKR FTP file; got: "
                         f"{list(short_df.columns)}")

    short_df["sym"] = short_df["sym"].astype(str).str.upper().str.strip()

    # Some typical column names in the FTP file: 'rebaterate', 'feerate', 'available'
    # We convert them to numeric and build net_borrow_annual.
    if "rebaterate" not in short_df.columns or "feerate" not in short_df.columns:
        raise ValueError("Expected 'rebaterate' and 'feerate' columns in FTP file.")

    short_df["rebate_annual"] = pd.to_numeric(short_df["rebaterate"], errors="coerce") / 100.0
    short_df["fee_annual"] = pd.to_numeric(short_df["feerate"], errors="coerce") / 100.0
    short_df["available_int"] = pd.to_numeric(short_df.get("available", 0), errors="coerce")

    short_df["net_borrow_annual"] = short_df["fee_annual"] - short_df["rebate_annual"]
    short_df["net_borrow_annual"] = short_df["net_borrow_annual"].clip(lower=0)

    # Filter to requested ETFs
    sub = short_df[short_df["sym"].isin(etf_list)].copy()

    records: list[Dict] = []
    for _, row in sub.iterrows():
        sym = row["sym"]
        records.append({
            "ETF": sym,
            "borrow_current": float(row["net_borrow_annual"])
                if pd.notna(row["net_borrow_annual"]) else float("nan"),
            "shares_available": int(row["available_int"])
                if pd.notna(row["available_int"]) else 0,
        })

    # There may be ETFs in etf_cagr.csv that are not in usa.txt; fill with NaN/0
    # for completeness.
    missing = [sym for sym in etf_list if sym not in sub["sym"].values]
    for sym in missing:
        records.append({
            "ETF": sym,
            "borrow_current": float("nan"),
            "shares_available": 0,
        })

    df_out = pd.DataFrame(records)
    df_out["borrow_spiking"] = False  # placeholder
    return df_out


# --------------------------------------------------
# SCREENING LOGIC
# --------------------------------------------------

def screen_universe_for_algo(
    df: pd.DataFrame,
    params: ScreeningParams = ScreeningParams(),
) -> pd.DataFrame:
    """
    df must contain at least:
        ETF,
        cagr_port_hist,
        borrow_current,
        shares_available,
        borrow_spiking (bool)

    It can optionally contain:
        Underlying, LevType

    Returns df with extra boolean column: include_for_algo
    """

    df = df.copy()

    # Basic flags
    df["cagr_positive"] = df["cagr_port_hist"] > 0
    df["borrow_leq_cap"] = df["borrow_current"] <= params.borrow_cap
    df["borrow_gt_cap"] = ~df["borrow_leq_cap"]

    if "borrow_spiking" not in df.columns:
        df["borrow_spiking"] = False

    df["shares_available"] = df["shares_available"].fillna(0).astype(int)

    # Exclusions
    df["exclude_borrow_gt_cap_and_neg_cagr"] = (
        df["borrow_gt_cap"] & ~df["cagr_positive"]
    )
    df["exclude_no_shares"] = df["shares_available"] < params.min_shares_available
    df["exclude_borrow_spike"] = df["borrow_spiking"].fillna(False)

    # Inclusions
    cheap_borrow_mask = df["borrow_leq_cap"]
    expensive_but_ok_mask = (
        df["borrow_gt_cap"] & df["cagr_positive"] & ~df["borrow_spiking"]
    )
    base_inclusion = cheap_borrow_mask | expensive_but_ok_mask

    hard_excludes = (
        df["exclude_borrow_gt_cap_and_neg_cagr"]
        | df["exclude_no_shares"]
        | df["exclude_borrow_spike"]
    )

    df["include_for_algo"] = base_inclusion & ~hard_excludes

    return df


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

def main():
    print(f"Repo root: {REPO_ROOT}")
    print(f"CAGR CSV: {CAGR_CSV}")

    # 1) Load ETF + CAGR file
    if not CAGR_CSV.exists():
        raise FileNotFoundError(f"CAGR CSV not found: {CAGR_CSV}")

    cagr_df = pd.read_csv(CAGR_CSV)
    print("Loaded etf_cagr head:\n", cagr_df.head())

    if "ETF" not in cagr_df.columns or "cagr_port_hist" not in cagr_df.columns:
        raise ValueError(
            f"{CAGR_CSV} must contain columns: 'ETF', 'cagr_port_hist'"
        )

    # Clean ETF symbols
    cagr_df["ETF"] = (
        cagr_df["ETF"].astype(str).str.strip().str.replace(".", "-", regex=False).str.upper()
    )

    # 2) Get current borrow snapshot from IBKR FTP
    borrow_df = get_ibkr_borrow_snapshot_from_ftp(cagr_df["ETF"].unique())

    # 3) Merge CAGRs and borrow data
    metrics = cagr_df.merge(borrow_df, on="ETF", how="left")

    print("Merged metrics sample:\n", metrics.head())

    # 4) Run screen
    screened = screen_universe_for_algo(metrics)

    # 5) Save output for trading algo
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    screened.to_csv(OUTPUT_FILE, index=False)

    included = screened[screened["include_for_algo"]]
    print(f"\nScreened ETF universe saved to: {OUTPUT_FILE}")
    print(f"Included {len(included)} ETFs:")
    cols_to_show = [c for c in ["Underlying", "ETF", "LevType"] if c in included.columns]
    if cols_to_show:
        print(included[cols_to_show].sort_values(cols_to_show[0]))
    else:
        print(included[["ETF"]].sort_values("ETF"))


if __name__ == "__main__":
    main()
