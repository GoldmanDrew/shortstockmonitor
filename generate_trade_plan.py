#!/usr/bin/env python3
"""
generate_trade_plan.py

Build proposed ETF long/short trades using:
- screened universe CSV
- baseline snapshot
- inline strategy blacklist from config/strategy_config.yml

Outputs:
- proposed_trades.csv
"""

import yaml
import pandas as pd
from pathlib import Path
from typing import Set


# ------------------------
# Config
# ------------------------

CONFIG_PATH = Path("config/strategy_config.yml")


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def load_blacklist(cfg: dict) -> Set[str]:
    """
    Reads blacklist from:
      strategy:
        blacklist:
          - JPM
          - AAPL
    """
    raw = cfg.get("strategy", {}).get("blacklist", [])
    return {str(sym).upper().strip() for sym in raw if sym}


# ------------------------
# Main Logic
# ------------------------

def main():
    cfg = load_config()

    paths = cfg["paths"]
    strategy = cfg["strategy"]

    screened_csv = Path(paths["screened_csv"])
    baseline_csv = Path(paths["baseline_csv"])
    proposed_csv = Path(paths["proposed_trades_csv"])

    tag = strategy["tag"]

    blacklist = load_blacklist(cfg)

    print(f"[INFO] Loaded {len(blacklist)} blacklisted symbols: {sorted(blacklist)}")

    # ------------------------
    # Load data
    # ------------------------

    screened = pd.read_csv(screened_csv)
    baseline = pd.read_csv(baseline_csv)

    # ------------------------
    # Filter by blacklist
    # ------------------------

    def is_allowed(row) -> bool:
        return (
            row["Underlying"].upper() not in blacklist
            and row["ETF"].upper() not in blacklist
        )

    screened = screened[screened.apply(is_allowed, axis=1)]

    if screened.empty:
        print("[WARN] No eligible pairs after blacklist filtering.")
        proposed_csv.parent.mkdir(parents=True, exist_ok=True)
        screened.to_csv(proposed_csv, index=False)
        return

   # ------------------------
    # Sizing logic (use existing LevType/Leverage classification)
    # ------------------------
    capital_usd = float(strategy["capital_usd"])
    gross_leverage = float(strategy["gross_leverage"])
    target_gross_usd = capital_usd * gross_leverage

    # Only size the rows we actually plan to trade
    eligible = screened[screened["include_for_algo"] == True].copy()

    screened["long_usd"] = 0.0
    screened["short_usd"] = 0.0

    if eligible.empty:
        print("[WARN] No eligible rows to size (include_for_algo is empty).")
    else:
        # Hedge ratio:
        # - CC: 1.0
        # - Levered: 1 / leverage_multiple (e.g., 2x => 0.5)
        eligible["hedge_ratio"] = 0.0
        is_cc = eligible["LevType"].astype(str).str.upper().eq("CC")
        eligible.loc[is_cc, "hedge_ratio"] = 1.0

        is_lev = ~is_cc
        lev_mult = pd.to_numeric(eligible.loc[is_lev, "Leverage"], errors="coerce")
        if lev_mult.isna().any() or (lev_mult <= 0).any():
            bad = eligible.loc[is_lev & (lev_mult.isna() | (lev_mult <= 0)), ["ETF", "Underlying", "LevType", "Leverage"]]
            raise ValueError(f"Invalid Leverage for leveraged rows. Examples:\n{bad.head(10)}")

        eligible.loc[is_lev, "hedge_ratio"] = 1.0 / lev_mult

        denom = float((1.0 + eligible["hedge_ratio"]).sum())
        long_per_row = target_gross_usd / denom

        eligible["long_usd"] = long_per_row
        eligible["short_usd"] = eligible["hedge_ratio"] * eligible["long_usd"]

        # Write back into screened
        screened.loc[eligible.index, "long_usd"] = eligible["long_usd"]
        screened.loc[eligible.index, "short_usd"] = eligible["short_usd"]


    screened["strategy_tag"] = tag

    # ------------------------
    # Formatting + column cleanup
    # ------------------------

    # 1) Round CAGR-related columns to 4 decimals
    cagr_cols = [c for c in screened.columns if "cagr" in c.lower()]
    for c in cagr_cols:
        screened[c] = screened[c].round(4)

    # 2) Round borrow-related columns to 4 decimals
    borrow_cols = [c for c in screened.columns if "borrow" in c.lower()]
    for c in borrow_cols:
        screened[c] = screened[c].round(4)

    # 3) Round capital / USD columns to 2 decimals
    usd_cols = [c for c in screened.columns if c.lower().endswith("_usd")]
    for c in usd_cols:
        screened[c] = screened[c].round(2)


    # ------------------------
    # Output (ONLY proposed trades)
    # ------------------------
    proposed = screened[screened["include_for_algo"] == True].copy()

    # Optional safety: drop any zero-sized rows (in case include_for_algo is True but sizing failed)
    proposed = proposed[(proposed["long_usd"] != 0) | (proposed["short_usd"] != 0)]

    cols_to_drop = [
        "LevType",
        "cagr_positive",
        "include_for_algo",
    ]
    proposed = proposed.drop(columns=[c for c in cols_to_drop if c in proposed.columns])

    proposed_csv.parent.mkdir(parents=True, exist_ok=True)
    proposed.to_csv(proposed_csv, index=False)

    print(f"[OK] Wrote proposed trades → {proposed_csv}  (n={len(proposed)})")



if __name__ == "__main__":
    main()
