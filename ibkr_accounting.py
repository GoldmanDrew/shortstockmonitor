#!/usr/bin/env python3
"""
ibkr_accounting.py

Compute net PnL (realized + unrealized + cashflows) from IBKR Flex XML exports and
roll it up by:
- symbol
- underlying (ETF mapped to underlyingSymbol)

Inputs (defaults assume your flex puller wrote into data/runs/<RUN_DATE>/ibkr_flex/):
  flex_trades.xml
  flex_positions.xml
  flex_cash.xml

Outputs:
  data/runs/<RUN_DATE>/accounting/
    pnl_by_symbol.csv
    pnl_by_underlying.csv
    totals.json

Notes / Accounting model:
- Realized trading PnL: sum of Trades@fifoPnlRealized (IBKR FIFO realized PnL) by trade.
- Transaction fees: joined from UnbundledCommissionDetails@totalCommission and added to realized (commission is usually negative).
- Unrealized PnL: OpenPositions@fifoPnlUnrealized (IBKR FIFO unrealized PnL).
- Cashflows: CashTransactions + CorporateActions amounts (dividends received/paid, borrow fees if present, special dividends, CVR cash, etc.).
- InterestAccruals section in Flex is often summary-only; if you want symbol-level borrow you should ensure it appears
  as CashTransactions lines (common) or add a separate borrow/fees detail query if your account supports it.

This script is intentionally conservative: it only books cash items that IBKR posts as cash transactions / corporate actions.
"""

from __future__ import annotations
import os
import re
import json
from pathlib import Path
from datetime import date
from typing import Dict, Tuple, Optional, List
import xml.etree.ElementTree as ET

import pandas as pd
import numpy as np


def today_str() -> str:
    return date.today().isoformat()


def parse_float(x: Optional[str]) -> float:
    if x is None:
        return 0.0
    x = str(x).strip()
    if x == "" or x.lower() == "nan":
        return 0.0
    try:
        return float(x)
    except Exception:
        # Sometimes values contain commas
        try:
            return float(x.replace(",", ""))
        except Exception:
            return 0.0


def parse_int(x: Optional[str]) -> int:
    return int(round(parse_float(x)))


def _find_one(stmt: ET.Element, tag: str) -> Optional[ET.Element]:
    for child in stmt:
        if child.tag == tag:
            return child
    return None


def load_flex_statement(xml_path: Path) -> ET.Element:
    root = ET.fromstring(xml_path.read_text(encoding="utf-8", errors="ignore"))
    fs = root.find("FlexStatements")
    if fs is None:
        raise ValueError(f"No <FlexStatements> found in {xml_path}")
    stmt = fs.find("FlexStatement")
    if stmt is None:
        raise ValueError(f"No <FlexStatement> found in {xml_path}")
    return stmt


def parse_trades_and_commissions(stmt: ET.Element) -> pd.DataFrame:
    trades_el = _find_one(stmt, "Trades")
    comm_el = _find_one(stmt, "UnbundledCommissionDetails")

    trade_rows = []
    if trades_el is not None:
        for t in trades_el.findall("Trade"):
            a = t.attrib
            trade_rows.append({
                "trade_id": a.get("tradeID", ""),
                "dateTime": a.get("dateTime", ""),
                "trade_date": (a.get("dateTime", "")[:8] if a.get("dateTime") else ""),
                "symbol": (a.get("symbol") or "").strip().upper(),
                "underlying": ((a.get("underlyingSymbol") or a.get("symbol") or "").strip().upper()),
                "assetCategory": a.get("assetCategory", ""),
                "subCategory": a.get("subCategory", ""),
                "buySell": a.get("buySell", ""),
                "quantity": parse_float(a.get("quantity")),
                "tradePrice": parse_float(a.get("tradePrice")),
                "fifo_pnl_realized": parse_float(a.get("fifoPnlRealized")),
            })

    trades_df = pd.DataFrame(trade_rows)
    if trades_df.empty:
        trades_df = pd.DataFrame(columns=[
            "trade_id","dateTime","trade_date","symbol","underlying",
            "assetCategory","subCategory","buySell","quantity","tradePrice","fifo_pnl_realized"
        ])

    # Commissions: join by tradeID if possible, else by (symbol, dateTime) fallback
    comm_rows = []
    if comm_el is not None:
        for c in comm_el.findall("UnbundledCommissionDetail"):
            a = c.attrib
            comm_rows.append({
                "trade_id": a.get("tradeID", ""),
                "dateTime": a.get("dateTime", ""),
                "symbol": (a.get("symbol") or "").strip().upper(),
                "total_commission": parse_float(a.get("totalCommission")),
            })
    comm_df = pd.DataFrame(comm_rows)
    if comm_df.empty:
        comm_df = pd.DataFrame(columns=["trade_id","dateTime","symbol","total_commission"])

    if not trades_df.empty and not comm_df.empty:
        # Prefer trade_id join
        merged = trades_df.merge(
            comm_df.groupby("trade_id", as_index=False)["total_commission"].sum(),
            on="trade_id",
            how="left",
        )
        merged["total_commission"] = merged["total_commission"].fillna(0.0)
    else:
        merged = trades_df.copy()
        merged["total_commission"] = 0.0

    merged["realized_net"] = merged["fifo_pnl_realized"] + merged["total_commission"]
    return merged


def parse_open_positions(stmt: ET.Element) -> pd.DataFrame:
    op_el = _find_one(stmt, "OpenPositions")
    rows = []
    if op_el is not None:
        for p in op_el.findall("OpenPosition"):
            a = p.attrib
            sym = (a.get("symbol") or "").strip().upper()
            under = (a.get("underlyingSymbol") or sym).strip().upper()
            rows.append({
                "symbol": sym,
                "underlying": under if under else sym,
                "assetCategory": a.get("assetCategory", ""),
                "subCategory": a.get("subCategory", ""),
                "description": a.get("description",""),
                "position": parse_float(a.get("position")),
                "markPrice": parse_float(a.get("markPrice")),
                "costBasisPrice": parse_float(a.get("costBasisPrice")),
                "positionValue": parse_float(a.get("positionValue")),
                "fifo_pnl_unrealized": parse_float(a.get("fifoPnlUnrealized")),
                "side": a.get("side",""),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[df["position"].abs() > 0].copy()
    if df.empty:
        df = pd.DataFrame(columns=[
            "symbol","underlying","assetCategory","subCategory","description",
            "position","markPrice","costBasisPrice","positionValue","fifo_pnl_unrealized","side"
        ])
    return df


def _parse_cash_like(stmt: ET.Element, section_tag: str, row_tag: str) -> pd.DataFrame:
    sec = _find_one(stmt, section_tag)
    rows = []
    if sec is not None:
        for r in sec.findall(row_tag):
            a = r.attrib
            sym = (a.get("symbol") or "").strip().upper()
            under = (a.get("underlyingSymbol") or sym).strip().upper()
            # Most cash rows have 'amount' and 'dateTime'/'reportDate'
            amt = parse_float(a.get("amount") or a.get("proceeds") or a.get("netCash") or a.get("total") or a.get("cashAmount"))
            dt = a.get("dateTime") or a.get("reportDate") or a.get("date") or ""
            d = dt[:8] if dt else ""
            rows.append({
                "section": section_tag,
                "type": a.get("type",""),
                "description": a.get("description",""),
                "symbol": sym,
                "underlying": under if under else sym,
                "currency": a.get("currency",""),
                "date": d,
                "amount": amt,
            })
    return pd.DataFrame(rows)


def parse_cashflows(stmt: ET.Element) -> pd.DataFrame:
    # In your Q_CASH we included CashTransactions + CorporateActions (and maybe Dividends).
    # In Flex XML, those sections are <CashTransactions><CashTransaction  /></CashTransactions>
    # and <CorporateActions><CorporateAction  /></CorporateActions>
    cash_df = _parse_cash_like(stmt, "CashTransactions", "CashTransaction")
    ca_df = _parse_cash_like(stmt, "CorporateActions", "CorporateAction")

    # Some configs also emit <Dividends> section; parse if present
    div_df = _parse_cash_like(stmt, "Dividends", "Dividend")

    out = pd.concat([cash_df, ca_df, div_df], ignore_index=True)
    if out.empty:
        out = pd.DataFrame(columns=["section","type","description","symbol","underlying","currency","date","amount"])

    # Heuristic classification for later reporting
    desc = out["description"].astype(str).str.lower()
    typ = out["type"].astype(str).str.lower()
    out["cash_category"] = np.select(
        [
            desc.str.contains("dividend") | typ.str.contains("dividend") | desc.str.contains("payment in lieu"),
            desc.str.contains("borrow") | desc.str.contains("stock borrow") | desc.str.contains("short stock") | desc.str.contains("loan"),
            desc.str.contains("withholding") | desc.str.contains("tax"),
            desc.str.contains("merger") | desc.str.contains("tender") | desc.str.contains("spinoff") | desc.str.contains("corporate"),
        ],
        ["dividend", "borrow_fee", "tax", "corp_action"],
        default="other",
    )
    return out


def main() -> None:
    run_date = os.getenv("RUN_DATE") or today_str()
    base_dir = Path(os.getenv("BASE_DIR") or ".").resolve()

    # Default locations (your flex puller output)
    flex_dir = base_dir / "data" / "runs" / run_date / "ibkr_flex"
    trades_path = Path(os.getenv("FLEX_TRADES_PATH") or (flex_dir / "flex_trades.xml"))
    cash_path = Path(os.getenv("FLEX_CASH_PATH") or (flex_dir / "flex_cash.xml"))
    pos_path = Path(os.getenv("FLEX_POSITIONS_PATH") or (flex_dir / "flex_positions.xml"))

    if not trades_path.exists():
        raise FileNotFoundError(f"Missing {trades_path}")
    if not pos_path.exists():
        raise FileNotFoundError(f"Missing {pos_path}")
    if not cash_path.exists():
        # Allow cash to be missing; treat as empty
        cash_path = None

    stmt_trades = load_flex_statement(trades_path)
    stmt_pos = load_flex_statement(pos_path)
    stmt_cash = load_flex_statement(cash_path) if cash_path else None

    trades = parse_trades_and_commissions(stmt_trades)
    positions = parse_open_positions(stmt_pos)
    cashflows = parse_cashflows(stmt_cash) if stmt_cash is not None else pd.DataFrame(
        columns=["section","type","description","symbol","underlying","currency","date","amount","cash_category"]
    )
    # ---- Optional: override underlying mapping using etf_screened_today.csv (recommended) ----
    # IBKR Flex often leaves underlyingSymbol blank for many ETFs, which breaks rollups.
    # If etf_screened_today.csv is present, we map ETF -> Underlying from that file.
    map_path = Path(os.getenv("ETF_MAP_PATH") or (base_dir / "data" / "etf_screened_today.csv"))
    if map_path.exists():
        mdf = pd.read_csv(map_path)
        cols = {c.lower(): c for c in mdf.columns}
        etf_c = cols.get("etf")
        under_c = cols.get("underlying")
        if etf_c and under_c:
            mdf["_ETF"] = mdf[etf_c].astype(str).str.upper().str.strip()
            mdf["_UNDER"] = mdf[under_c].astype(str).str.upper().str.strip()
            etf_to_under = dict(zip(mdf["_ETF"], mdf["_UNDER"]))
            # Apply to trades/positions/cashflows where symbol matches an ETF
            trades["underlying"] = trades["symbol"].map(etf_to_under).fillna(trades["underlying"]).astype(str)
            positions["underlying"] = positions["symbol"].map(etf_to_under).fillna(positions["underlying"]).astype(str)
            cashflows["underlying"] = cashflows["symbol"].map(etf_to_under).fillna(cashflows["underlying"]).astype(str)
            print(f"[MAP] Loaded ETF->Underlying map from {map_path} ({len(etf_to_under)} ETFs)")
        else:
            print(f"[MAP] Found {map_path} but missing ETF/Underlying columns; skipping mapping.")
    else:
        print(f"[MAP] No ETF map found at {map_path}; using IBKR underlyingSymbol only.")


    # ---- Symbol-level aggregation ----
    realized_by_sym = trades.groupby(["symbol","underlying"], as_index=False).agg(
        realized_pnl=("realized_net","sum"),
        fifo_realized=("fifo_pnl_realized","sum"),
        commissions=("total_commission","sum"),
        n_trades=("trade_id","nunique"),
    )

    unreal_by_sym = positions.groupby(["symbol","underlying"], as_index=False).agg(
        unrealized_pnl=("fifo_pnl_unrealized","sum"),
        position=("position","sum"),
        markPrice=("markPrice","last"),
        positionValue=("positionValue","sum"),
        subCategory=("subCategory","last"),
    )

    cash_by_sym = cashflows.groupby(["symbol","underlying"], as_index=False).agg(
        cash_pnl=("amount","sum"),
        dividends=("amount", lambda s: float(s[cashflows.loc[s.index,"cash_category"].eq("dividend")].sum()) if len(s) else 0.0),
        borrow_fees=("amount", lambda s: float(s[cashflows.loc[s.index,"cash_category"].eq("borrow_fee")].sum()) if len(s) else 0.0),
        taxes=("amount", lambda s: float(s[cashflows.loc[s.index,"cash_category"].eq("tax")].sum()) if len(s) else 0.0),
        corp_actions=("amount", lambda s: float(s[cashflows.loc[s.index,"cash_category"].eq("corp_action")].sum()) if len(s) else 0.0),
        n_cash_rows=("amount","count"),
    )

    pnl_sym = realized_by_sym.merge(unreal_by_sym, on=["symbol","underlying"], how="outer").merge(
        cash_by_sym, on=["symbol","underlying"], how="outer"
    ).fillna(0.0).infer_objects(copy=False)

    pnl_sym["net_pnl"] = pnl_sym["realized_pnl"] + pnl_sym["unrealized_pnl"] + pnl_sym["cash_pnl"]

    # ---- Underlying-level aggregation ----
    pnl_under = pnl_sym.groupby("underlying", as_index=False).agg(
        realized_pnl=("realized_pnl","sum"),
        unrealized_pnl=("unrealized_pnl","sum"),
        cash_pnl=("cash_pnl","sum"),
        dividends=("dividends","sum"),
        borrow_fees=("borrow_fees","sum"),
        commissions=("commissions","sum"),
        net_pnl=("net_pnl","sum"),
        n_symbols=("symbol","nunique"),
    ).sort_values("net_pnl", ascending=False)

    # ---- Per-pair (Underlying, Symbol) breakdown ----
    # "Pair" here means: for each underlying, show PnL contribution of each associated symbol (ETFs and the underlying itself).
    pnl_pair = pnl_sym.copy()
    pnl_pair["pair_label"] = pnl_pair["underlying"] + " / " + pnl_pair["symbol"]
    pnl_pair = pnl_pair.sort_values(["underlying", "net_pnl"], ascending=[True, False]).reset_index(drop=True)

    # Classic underlying-vs-ETF pairs (exclude the underlying itself)
    pnl_pair_etf = pnl_pair[pnl_pair["symbol"] != pnl_pair["underlying"]].copy()

    totals = {
        "run_date": run_date,
        "net_pnl": float(pnl_sym["net_pnl"].sum()),
        "realized_pnl": float(pnl_sym["realized_pnl"].sum()),
        "unrealized_pnl": float(pnl_sym["unrealized_pnl"].sum()),
        "cash_pnl": float(pnl_sym["cash_pnl"].sum()),
        "dividends": float(pnl_sym["dividends"].sum()),
        "borrow_fees": float(pnl_sym["borrow_fees"].sum()),
        "commissions": float(pnl_sym["commissions"].sum()),
        "n_symbols": int(pnl_sym["symbol"].nunique()),
        "n_underlyings": int(pnl_under["underlying"].nunique()),
    }

    out_dir = base_dir / "data" / "runs" / run_date / "accounting"
    out_dir.mkdir(parents=True, exist_ok=True)

    pnl_sym.to_csv(out_dir / "pnl_by_symbol.csv", index=False)
    pnl_under.to_csv(out_dir / "pnl_by_underlying.csv", index=False)
    pnl_pair.to_csv(out_dir / "pnl_by_pair.csv", index=False)
    pnl_pair_etf.to_csv(out_dir / "pnl_by_pair_etf_only.csv", index=False)
    (out_dir / "totals.json").write_text(json.dumps(totals, indent=2), encoding="utf-8")

    print("[OK] Wrote:", out_dir / "pnl_by_symbol.csv")
    print("[OK] Wrote:", out_dir / "pnl_by_underlying.csv")
    print("[OK] Wrote:", out_dir / "totals.json")
    print("[TOTALS]", json.dumps(totals, indent=2))

    # Print per-underlying pair breakdown (top 10 legs per underlying by net_pnl)
    print("\n[PER-PAIR PnL BY UNDERLYING] (top 10 legs per underlying; includes underlying + ETFs)")
    # show only underlyings with any activity or non-zero pnl
    # Prefer underlyings that actually have non-zero contributions today.
    active_under = pnl_pair[
        (pnl_pair["net_pnl"].abs() > 1e-9)
        | (pnl_pair.get("n_trades", 0) > 0)
        | (pnl_pair.get("position", 0).abs() > 1e-9)
        | (pnl_pair.get("cash_pnl", 0).abs() > 1e-9)
    ].copy()

    if active_under.empty:
        active_list = pnl_under.sort_values("net_pnl", ascending=False)["underlying"].head(25).tolist()
    else:
        active_list = (
            active_under.groupby("underlying")["net_pnl"]
            .sum()
            .abs()
            .sort_values(ascending=False)
            .head(25)
            .index
            .tolist()
        )

    for u in active_list:
        sub = pnl_pair[pnl_pair["underlying"] == u].copy()
        if sub.empty:
            continue
        print(f"\n--- {u} ---")
        show = sub.sort_values("net_pnl", ascending=False).head(10)[
            ["symbol", "net_pnl", "realized_pnl", "unrealized_pnl", "cash_pnl", "dividends", "borrow_fees", "commissions"]
        ]
        with pd.option_context("display.width", 160):
            print(show.to_string(index=False))


if __name__ == "__main__":
    main()
