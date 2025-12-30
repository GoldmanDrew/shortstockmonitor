#!/usr/bin/env python3
"""
generate_trade_plan.py

Quarterly rebalance-friendly trade plan generator.

Purpose:
- Build an ABSOLUTE target portfolio (in USD notionals per leg) from today's screened universe.
- This file does NOT compute deltas vs current holdings. That belongs in execute_trade_plan.py.
- Baseline is not used here; baseline is used in execution to compute strategy-only holdings and deltas.

Inputs:
- cfg["paths"]["screened_csv"]  (typically data/etf_screened_today.csv)
- cfg["strategy"]["blacklist"]  (inline list in config/strategy_config.yml)
- cfg["strategy"]["capital_usd"], cfg["strategy"]["gross_leverage"]
- include_for_algo from screened CSV

Outputs:
- data/runs/YYYY-MM-DD/proposed_trades.csv    (dated archive)
- cfg["paths"]["proposed_trades_csv"]         (latest convenience copy)
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Set

import pandas as pd
import yaml


CONFIG_PATH = Path("config/strategy_config.yml")


def today_str() -> str:
    return date.today().isoformat()


def run_dir(run_date: str) -> Path:
    return Path("data") / "runs" / run_date


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_blacklist(cfg: dict) -> Set[str]:
    raw = cfg.get("strategy", {}).get("blacklist", []) or []
    return {str(sym).upper().strip() for sym in raw if str(sym).strip()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-date", default=today_str(), help="YYYY-MM-DD for data/runs/<date>/ outputs")
    args = ap.parse_args()

    cfg = load_config()
    paths = cfg["paths"]
    strategy = cfg["strategy"]

    screened_csv = Path(paths["screened_csv"])
    proposed_latest_csv = Path(paths["proposed_trades_csv"])

    tag = str(strategy["tag"])
    blacklist = load_blacklist(cfg)

    print(f"[INFO] Loaded {len(blacklist)} blacklisted symbols: {sorted(blacklist)}")

    if not screened_csv.exists():
        raise FileNotFoundError(f"Screened CSV not found: {screened_csv}")

    # ------------------------
    # Load screened universe
    # ------------------------
    screened = pd.read_csv(screened_csv)

    if screened.empty:
        print("[WARN] Screened universe is empty.")
        proposed_latest_csv.parent.mkdir(parents=True, exist_ok=True)
        screened.to_csv(proposed_latest_csv, index=False)
        return

    # ------------------------
    # Normalize tickers & filter blacklist
    # ------------------------
    # (Keep consistent with your screener normalization)
    screened["ETF"] = screened["ETF"].astype(str).str.strip().str.upper().str.replace(".", "-", regex=False)
    screened["Underlying"] = screened["Underlying"].astype(str).str.strip().str.upper().str.replace(".", "-", regex=False)

    def is_allowed(row) -> bool:
        return (row["Underlying"] not in blacklist) and (row["ETF"] not in blacklist)

    screened = screened[screened.apply(is_allowed, axis=1)].copy()

    if screened.empty:
        print("[WARN] No eligible pairs after blacklist filtering.")
        proposed_latest_csv.parent.mkdir(parents=True, exist_ok=True)
        screened.to_csv(proposed_latest_csv, index=False)
        return

    # Only trade rows explicitly included by the screener
    eligible = screened[screened["include_for_algo"] == True].copy()

    if eligible.empty:
        print("[WARN] No eligible rows to size (include_for_algo is empty).")
        out = screened.copy()
        out["strategy_tag"] = tag
        out["long_usd"] = 0.0
        out["short_usd"] = 0.0
        # still write a dated + latest empty plan for traceability
        dated_path = run_dir(args.run_date) / "proposed_trades.csv"
        dated_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(dated_path, index=False)
        proposed_latest_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(proposed_latest_csv, index=False)
        return

    # ------------------------
    # Sizing logic (ABSOLUTE TARGETS)
    # ------------------------
    capital_usd = float(strategy["capital_usd"])
    gross_leverage = float(strategy["gross_leverage"])
    target_gross_usd = capital_usd * gross_leverage

    # Hedge ratio:
    # - CC: 1.0
    # - Levered: 1 / leverage_multiple (e.g., 2x => 0.5)
    eligible["hedge_ratio"] = 0.0
    is_cc = eligible["LevType"].astype(str).str.upper().eq("CC")
    eligible.loc[is_cc, "hedge_ratio"] = 1.0

    is_lev = ~is_cc
    lev_mult = pd.to_numeric(eligible.loc[is_lev, "Leverage"], errors="coerce")
    bad_mask = lev_mult.isna() | (lev_mult <= 0)
    if bad_mask.any():
        bad = eligible.loc[is_lev].loc[bad_mask, ["ETF", "Underlying", "LevType", "Leverage"]]
        raise ValueError(f"Invalid Leverage for leveraged rows. Examples:\n{bad.head(10)}")

    eligible.loc[is_lev, "hedge_ratio"] = 1.0 / lev_mult

    # Allocation scheme:
    # For each row i:
    #   gross_i = long_i + |short_i| = long_i * (1 + hedge_ratio_i)
    # We allocate equal "gross per row" and solve long_i from that.
    n = len(eligible)
    gross_per_row = target_gross_usd / n

    eligible["long_usd"] = gross_per_row / (1.0 + eligible["hedge_ratio"])
    # IMPORTANT: short_usd is negative notionally
    eligible["short_usd"] = -(eligible["hedge_ratio"] * eligible["long_usd"])

    # Write back into screened for formatting consistency
    screened["long_usd"] = 0.0
    screened["short_usd"] = 0.0
    screened.loc[eligible.index, "long_usd"] = eligible["long_usd"]
    screened.loc[eligible.index, "short_usd"] = eligible["short_usd"]

    screened["strategy_tag"] = tag

    # ------------------------
    # Formatting
    # ------------------------
    # Round CAGR columns to 4 decimals
    cagr_cols = [c for c in screened.columns if "cagr" in c.lower()]
    for c in cagr_cols:
        screened[c] = pd.to_numeric(screened[c], errors="coerce").round(4)

    # Round borrow columns to 4 decimals
    borrow_cols = [c for c in screened.columns if "borrow" in c.lower()]
    for c in borrow_cols:
        screened[c] = pd.to_numeric(screened[c], errors="coerce").round(4)

    # Round USD columns to 2 decimals
    usd_cols = [c for c in screened.columns if c.lower().endswith("_usd")]
    for c in usd_cols:
        screened[c] = pd.to_numeric(screened[c], errors="coerce").round(2)

    # ------------------------
    # Output (ONLY proposed trades)
    # ------------------------
    proposed = screened[screened["include_for_algo"] == True].copy()
    proposed = proposed[(proposed["long_usd"] != 0) | (proposed["short_usd"] != 0)]

    cols_to_drop = ["cagr_positive", "include_for_algo"]
    proposed = proposed.drop(columns=[c for c in cols_to_drop if c in proposed.columns])

    # Dated archive
    dated_path = run_dir(args.run_date) / "proposed_trades.csv"
    dated_path.parent.mkdir(parents=True, exist_ok=True)
    proposed.to_csv(dated_path, index=False)

    # Latest convenience copy
    proposed_latest_csv.parent.mkdir(parents=True, exist_ok=True)
    proposed.to_csv(proposed_latest_csv, index=False)

    print(f"[OK] Wrote proposed trades → {dated_path}  (n={len(proposed)})")
    print(f"[OK] Updated latest proposed trades → {proposed_latest_csv}  (n={len(proposed)})")


if __name__ == "__main__":
    main()
