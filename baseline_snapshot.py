#!/usr/bin/env python3
"""
baseline_snapshot.py

Creates a baseline snapshot of IBKR positions to "segment" strategy holdings
inside a single IBKR account.

Strategy holdings are computed later as:
    strategy_qty(symbol) = current_ib_qty(symbol) - baseline_qty(symbol)

New behavior:
- Always writes an archival baseline under:
    data/baselines/YYYY-MM-DD/baseline_snapshot.csv
- By default also writes/overwrites the "active baseline" path defined in config:
    cfg["paths"]["baseline_csv"]  (often data/baseline_snapshot.csv)

This script never places orders.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
from ib_insync import IB

from strategy_config import load_config


# ---------------------------
# Symbol normalization helpers
# ---------------------------

IB_SYMBOL_MAP: Dict[str, Tuple[str, Optional[str]]] = {
    "BRK-B": ("BRK B", "NYSE"),
    "BRK-A": ("BRK A", "NYSE"),
}

REVERSE_IB_SYMBOL_MAP: Dict[str, str] = {ib_sym: uni for uni, (ib_sym, _) in IB_SYMBOL_MAP.items()}


def universal_symbol_from_ib(symbol: str) -> str:
    s = str(symbol).strip().upper()
    return REVERSE_IB_SYMBOL_MAP.get(s, s)


def today_str() -> str:
    return date.today().isoformat()


def archival_baseline_path(run_date: str) -> Path:
    return Path("data") / "baselines" / run_date / "baseline_snapshot.csv"


# ---------------------------
# IBKR connection
# ---------------------------

def connect_ib(host: str, port: int, client_id: int) -> IB:
    ib = IB()
    ib.connect(host, port, clientId=client_id)
    if not ib.isConnected():
        raise RuntimeError("Failed to connect to IBKR.")
    return ib


def fetch_positions_df(ib: IB) -> pd.DataFrame:
    rows = []
    for p in ib.positions():
        c = p.contract
        rows.append(
            {
                "account": p.account,
                "symbol_ib": str(c.symbol),
                "localSymbol": str(getattr(c, "localSymbol", "")),
                "conId": int(getattr(c, "conId", 0)),
                "secType": str(getattr(c, "secType", "")),
                "currency": str(getattr(c, "currency", "")),
                "exchange": str(getattr(c, "exchange", "")),
                "primaryExchange": str(getattr(c, "primaryExchange", "")),
                "qty": float(p.position),
                "avgCost": float(getattr(p, "avgCost", 0.0)),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "account",
                "symbol_ib",
                "localSymbol",
                "conId",
                "secType",
                "currency",
                "exchange",
                "primaryExchange",
                "qty",
                "avgCost",
            ]
        )

    df["symbol"] = df["symbol_ib"].map(universal_symbol_from_ib)
    df["captured_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Keep output schema stable for downstream tools
    return df[
        [
            "captured_at",
            "account",
            "symbol",
            "symbol_ib",
            "localSymbol",
            "conId",
            "secType",
            "currency",
            "exchange",
            "primaryExchange",
            "qty",
            "avgCost",
        ]
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run-date",
        default=today_str(),
        help="YYYY-MM-DD; used for archival folder naming (default: today).",
    )
    ap.add_argument(
        "--no-set-active",
        action="store_true",
        help="If set, do not overwrite the active baseline path from config; only write the dated archival baseline.",
    )
    args = ap.parse_args()

    cfg = load_config()  # reads config/strategy_config.yaml

    host = cfg["ibkr"]["host"]
    port = int(cfg["ibkr"]["port"])
    client_id = int(cfg["ibkr"]["client_id"])

    # Active baseline path (existing behavior)
    active_baseline_path = Path(cfg["paths"]["baseline_csv"])
    active_baseline_path.parent.mkdir(parents=True, exist_ok=True)

    # New: archival baseline path
    archive_path = archival_baseline_path(args.run_date)
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    ib = connect_ib(host, port, client_id)
    try:
        df = fetch_positions_df(ib)

        # Always write the archival baseline
        df.to_csv(archive_path, index=False)
        print(f"[BASELINE] Wrote archival baseline ({len(df)} rows): {archive_path}")

        # Optionally write/overwrite the active baseline
        if not args.no_set_active:
            df.to_csv(active_baseline_path, index=False)
            print(f"[BASELINE] Updated active baseline ({len(df)} rows): {active_baseline_path}")
        else:
            print("[BASELINE] Skipped updating active baseline (per --no-set-active).")

    finally:
        ib.disconnect()


if __name__ == "__main__":
    main()
