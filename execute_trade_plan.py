#!/usr/bin/env python3
"""
execute_trade_plan.py

Reads a proposed trade plan CSV and executes pair-by-pair with manual approval.

UPDATED for:
- Initial build + quarterly rebalancing
- Date-partitioned IO (reads/writes data/runs/YYYY-MM-DD/...)
- Delta execution vs strategy-only positions (current - baseline)

Core mechanics:
- Baseline snapshot protects pre-existing holdings:
      strategy_qty(symbol) = current_ib_qty(symbol) - baseline_qty(symbol)
- Plan expresses ABSOLUTE TARGET exposures (long_usd > 0, short_usd < 0).
- Executor places DELTA orders to move strategy-only holdings to the targets:
      delta_shares = target_shares - current_strategy_qty

Safety:
- DRY_RUN=1 environment variable to simulate without placing orders.

Usage examples:
  python execute_trade_plan.py --run-date 2025-12-28
  python execute_trade_plan.py --strategy-tag ETF_LS
  DRY_RUN=1 python execute_trade_plan.py
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Set

import pandas as pd
from ib_insync import IB, Stock, Order, Trade, MarketOrder, TagValue
import yaml


# ---------------------------
# Symbol normalization
# ---------------------------

IB_SYMBOL_MAP: Dict[str, Tuple[str, Optional[str]]] = {
    "BRK-B": ("BRK B", "NYSE"),
    "BRK-A": ("BRK A", "NYSE"),
}
REVERSE_IB_SYMBOL_MAP: Dict[str, str] = {ib_sym: uni for uni, (ib_sym, _) in IB_SYMBOL_MAP.items()}


def ib_symbol_from_universal(sym: str) -> Tuple[str, Optional[str]]:
    s = str(sym).strip().upper()
    if s in IB_SYMBOL_MAP:
        return IB_SYMBOL_MAP[s]
    return s, None


def universal_symbol_from_ib(sym: str) -> str:
    s = str(sym).strip().upper()
    return REVERSE_IB_SYMBOL_MAP.get(s, s)


def make_stock(symbol: str) -> Stock:
    ib_sym, primary = ib_symbol_from_universal(symbol)
    c = Stock(ib_sym, "SMART", "USD")
    if primary:
        c.primaryExchange = primary
    return c


# ---------------------------
# Paths / date helpers
# ---------------------------

def today_str() -> str:
    return date.today().isoformat()


def run_dir(run_date: str) -> Path:
    return Path("data") / "runs" / run_date


def exec_dir(run_date: str) -> Path:
    return run_dir(run_date) / "execution"


# ---------------------------
# IBKR connection & pricing
# ---------------------------

def connect_ib(host: str, port: int, client_id: int) -> IB:
    ib = IB()
    ib.connect(host, port, clientId=client_id)
    if not ib.isConnected():
        raise RuntimeError("Failed to connect to IBKR.")

    # Receive order updates for orders created in TWS / other clients too
    ib.reqAutoOpenOrders(True)

    # Pull THIS client's open orders once (lightweight)
    try:
        ib.reqOpenOrders()
        ib.sleep(0.5)
    except Exception:
        pass

    return ib


def safe_price(v) -> Optional[float]:
    try:
        x = float(v)
        return x if x > 0 else None
    except Exception:
        return None


def get_snapshot_price(ib: IB, symbol: str, prefer_delayed: bool = True) -> float:
    """
    Lightweight price fetch (snapshot):
    - If prefer_delayed, set marketDataType=3 (delayed) to avoid subscription errors.
    - Otherwise try live (1) and fall back to delayed (3).
    - Prefer mid (bid/ask), else last, else close, else marketPrice().
    - Final fallback: 1D historical close.
    """
    sym_u = symbol.upper()
    contract = make_stock(sym_u)
    ib.qualifyContracts(contract)

    def read_ticker(t) -> Optional[float]:
        bid = safe_price(getattr(t, "bid", None)) or safe_price(getattr(t, "delayedBid", None))
        ask = safe_price(getattr(t, "ask", None)) or safe_price(getattr(t, "delayedAsk", None))
        last = safe_price(getattr(t, "last", None)) or safe_price(getattr(t, "delayedLast", None))
        close = safe_price(getattr(t, "close", None)) or safe_price(getattr(t, "delayedClose", None))
        mkt = safe_price(t.marketPrice())
        if bid and ask:
            return (bid + ask) / 2.0
        return last or close or mkt

    def snapshot_with_type(data_type: int) -> Optional[float]:
        ib.reqMarketDataType(data_type)
        t = ib.reqMktData(contract, "", snapshot=True)
        try:
            for _ in range(12):
                ib.sleep(0.25)
                px = read_ticker(t)
                if px is not None:
                    return px
        finally:
            # Cancel snapshot subscription if we actually have a reqId.
            # ib_insync.cancelMktData(contract) can spam "No reqId found" if the snapshot failed (e.g. 10089),
            # so we cancel via the underlying client when possible.
            try:
                req_id = getattr(t, "reqId", None) or getattr(t, "tickerId", None)
                if isinstance(req_id, int) and req_id > 0:
                    ib.client.cancelMktData(req_id)
            except Exception:
                pass
        return None

    px: Optional[float] = None
    if prefer_delayed:
        # Try delayed-frozen first (more stable), then delayed.
        px = snapshot_with_type(4) or snapshot_with_type(3)
    else:
        px = snapshot_with_type(1) or snapshot_with_type(3)

    if px is None:
        # Fallback: last daily close via historical bars.
        # This is slower and subject to pacing / connectivity, so we retry a few times.
        for _attempt in range(3):
            # If connection dropped, give TWS a moment to recover.
            if not ib.isConnected():
                for _ in range(40):
                    ib.sleep(0.25)
                    if ib.isConnected():
                        break
            try:
                bars = ib.reqHistoricalData(
                    contract,
                    endDateTime="",
                    durationStr="2 D",
                    barSizeSetting="1 day",
                    whatToShow="TRADES",
                    useRTH=True,
                    formatDate=1,
                    keepUpToDate=False,
                )
            except Exception:
                bars = []
            if bars:
                px = safe_price(bars[-1].close)
                if px is not None:
                    break
            ib.sleep(0.5)

    if px is None:
        raise RuntimeError(f"No usable price for {sym_u}")
    return float(px)


# ---------------------------
# Orders
# ---------------------------

def build_limit_order(action: str, qty: int, ref_price: float, bps: float, order_ref: str) -> Order:
    """
    Simple limit order priced slightly through ref_price.
    """
    o = Order()
    o.action = action.upper()
    o.totalQuantity = int(qty)
    o.tif = "DAY"
    o.orderType = "LMT"
    offset = ref_price * (bps / 10_000.0)
    if o.action == "BUY":
        o.lmtPrice = round(ref_price + offset, 4)
    else:
        o.lmtPrice = round(ref_price - offset, 4)
    o.orderRef = order_ref
    return o


def build_market_order(action: str, qty: int, order_ref: str) -> Order:
    """Plain market order (no explicit price)."""
    o = MarketOrder(action.upper(), int(qty))
    o.tif = "DAY"
    o.transmit = True
    o.orderRef = order_ref
    return o


def build_adaptive_market_order(action: str, qty: int, order_ref: str, priority: str = "Normal") -> Order:
    """IBKR Adaptive algo + market order (no explicit price)."""
    o = Order()
    o.action = action.upper()
    o.totalQuantity = int(qty)
    o.orderType = "MKT"
    o.tif = "DAY"
    o.transmit = True
    o.orderRef = order_ref
    o.algoStrategy = "Adaptive"
    o.algoParams = [TagValue("adaptivePriority", str(priority))]
    return o


WARNING_ERROR_CODES = {10349}
ORDER_NEVER_LIVE_CANCEL_CODES = {10147}

# Ensure these exist somewhere in your file
ACCEPTED: Set[str] = {"presubmitted", "submitted", "filled", "pendingsubmit"}  # you can tweak
TERMINAL: Set[str] = {"filled", "cancelled", "inactive"}  # IB uses these commonly


def _iter_trade_log_entries(trade) -> list:
    """ib_insync Trade.log is a list of TradeLogEntry; be defensive."""
    try:
        return list(getattr(trade, "log", None) or [])
    except Exception:
        return []


def _trade_has_error_code(trade, code: int) -> bool:
    """Return True if Trade.log contains an entry mentioning 'Error <code>'."""
    for e in _iter_trade_log_entries(trade):
        msg = getattr(e, "message", "") or ""
        if f"Error {code}" in msg:
            return True
    return False


def _only_warnings(trade) -> bool:
    """
    Some IB messages show up as 'Cancelled' but are just warnings / transient.
    If you already have a warning-code list elsewhere, use it.
    """
    # If you track warning codes explicitly, wire it here.
    # Fallback: if there's *any* "Error " message, treat as not-only-warnings.
    for e in _iter_trade_log_entries(trade):
        msg = getattr(e, "message", "") or ""
        if "Error " in msg:
            # treat as a real issue, not only warnings
            return False
    return True


def _find_any_open_trade_by_order_ref(ib, order_ref: str):
    """
    Find a Trade in ib.trades() matching orderRef, where status is not terminal.
    (ib_insync keeps local trade objects; reqAllOpenOrders updates them.)
    """
    for t in list(getattr(ib, "trades", lambda: [])() or []):
        try:
            ref = getattr(getattr(t, "order", None), "orderRef", "") or ""
            st = (getattr(getattr(t, "orderStatus", None), "status", "") or "").lower()
            if ref == order_ref and st not in TERMINAL:
                return t
        except Exception:
            continue
    return None


def wait_for_trade_terminal(ib, trade, timeout: float = 90.0):
    """
    Wait until a Trade reaches a terminal status. Also periodically resync open orders
    to help TWS/Gateway update order state.
    """
    t0 = time.time()
    last_resync = 0.0
    while time.time() - t0 < timeout:
        st = (trade.orderStatus.status or "").lower()
        if st in TERMINAL:
            return trade

        # resync every ~1s to help resolve PendingSubmit / missing state updates
        now = time.time()
        if now - last_resync > 1.0:
            try:
                ib.reqAllOpenOrders()
            except Exception:
                pass
            last_resync = now

        ib.sleep(0.2)

    return trade


def cancel_and_wait_with_resync(ib, trade, timeout: float = 30.0):
    """
    Cancel an order and wait for terminal status.
    If IB says 10147 'not found', attempt to locate/cancel the live order by orderRef.
    """
    order_ref = getattr(getattr(trade, "order", None), "orderRef", "") or ""

    # Attempt cancel
    try:
        ib.cancelOrder(trade.order)
    except Exception:
        pass

    # Resync early
    try:
        ib.reqAllOpenOrders()
    except Exception:
        pass

    trade = wait_for_trade_terminal(ib, trade, timeout=timeout)

    # If "order not found", try to cancel by orderRef
    if order_ref and _trade_has_error_code(trade, 10147):
        live = _find_any_open_trade_by_order_ref(ib, order_ref)
        if live is not None:
            try:
                ib.cancelOrder(live.order)
            except Exception:
                pass
            trade = wait_for_trade_terminal(ib, live, timeout=timeout)

    return trade


def wait_for_trade_accepted(ib: IB, trade: Trade, timeout: float = 15.0) -> Tuple[bool, Trade]:
    """Wait until trade reaches an 'accepted' status.

    Important nuance: TWS/Gateway can leave orders in PendingSubmit while it is busy
    or the API client has not fully resynced. In that state we *resync and wait*,
    rather than cancelling immediately.
    """
    t0 = time.time()
    order_ref = getattr(getattr(trade, "order", None), "orderRef", "") or ""
    while time.time() - t0 < timeout:
        status = (trade.orderStatus.status or "").lower()

        if status in ACCEPTED:
            return True, trade

        if status == "pendingsubmit":
            # Give TWS time to register the order; resync open orders and
            # try to re-bind the Trade object via orderRef if needed.
            try:
                ib.reqAllOpenOrders()
            except Exception:
                pass
            ib.sleep(0.5)

            if order_ref:
                tr2 = _find_any_open_trade_by_order_ref(ib, order_ref)
                if tr2 is not None:
                    trade = tr2
            continue

        if status in TERMINAL:
            # If it "cancelled" but ONLY has warning codes (e.g. 10349),
            # don't treat it as a real rejection; give it a moment and resync.
            if status == "cancelled" and _only_warnings(trade):
                try:
                    ib.reqAllOpenOrders()
                except Exception:
                    pass
                ib.sleep(0.5)
                continue
            return False, trade

        ib.sleep(0.2)

    return False, trade


def wait_for_trade_done(ib: IB, trade: Trade, timeout: float = 90.0) -> Trade:
    t0 = time.time()
    while time.time() - t0 < timeout:
        status = (trade.orderStatus.status or "").lower()
        if status in TERMINAL:
            return trade
        ib.sleep(0.2)
    return trade


def execute_leg(
    ib: IB,
    symbol: str,
    action: str,
    qty: int,
    ref_price: float,
    bps: float,
    order_ref: str,
    exec_cfg: Dict,
    timeout: float = 90.0,
    max_retries: int = 3,
    dry_run: bool = False,
) -> Tuple[int, Optional[Trade]]:

    if qty <= 0:
        return 0, None

    contract = make_stock(symbol)
    ib.qualifyContracts(contract)

    filled_total = 0
    last_trade: Optional[Trade] = None

    aggressive_step_bps = float(exec_cfg.get("aggressive_bps_step", 25.0))
    market_fallback_third_try = bool(exec_cfg.get("market_fallback_third_try", True))

    accept_timeout = float(exec_cfg.get("accept_timeout_sec", 15.0))
    market_accept_timeout = float(exec_cfg.get("market_accept_timeout_sec", 45.0))
    cancel_timeout = float(exec_cfg.get("cancel_timeout_sec", 30.0))

    for attempt in range(1, max_retries + 1):
        remain = qty - filled_total
        if remain <= 0:
            break

        # Order routing that does NOT require a price:
        #   order_style = "ADAPTIVE_MKT"  -> IBKR Adaptive algo + Market
        #   order_style = "MKT"           -> plain Market
        # Optional (price-required):
        #   order_style = "LMT"           -> your limit ladder (uses snapshot price)
        order_style = str(exec_cfg.get("order_style", "ADAPTIVE_MKT")).strip().upper()

        use_market_fallback = (market_fallback_third_try and attempt == 3)

        ref_px_now: Optional[float] = None
        bps_now: Optional[float] = None

        if order_style == "ADAPTIVE_MKT":
            if "|UNDER_DELTA" in order_ref:
                priority = "Urgent"
            else:
                priority = "Normal" if attempt == 1 else "Urgent"

            o = build_adaptive_market_order(
                action=action,
                qty=remain,
                order_ref=f"{order_ref}|att{attempt}|ADAPT",
                priority=priority,
            )
            px_str = f"ADAPT_MKT({priority})"


        elif order_style == "MKT":
            o = build_market_order(
                action=action,
                qty=remain,
                order_ref=f"{order_ref}|att{attempt}|MKT",
            )
            px_str = "MKT"

        else:
            # Price-required path (original behavior):
            # Use limit ladder with optional market fallback on the last retry.
            ref_px_now = get_snapshot_price(ib, symbol, prefer_delayed=bool(exec_cfg.get("prefer_delayed", True)))
            bps_now = bps + (attempt - 1) * aggressive_step_bps

            if use_market_fallback:
                o = build_market_order(
                    action=action,
                    qty=remain,
                    order_ref=f"{order_ref}|att{attempt}|MKT",
                )
                px_str = "MKT"
            else:
                o = build_limit_order(
                    action=action,
                    qty=remain,
                    ref_price=ref_px_now,
                    bps=bps_now,
                    order_ref=f"{order_ref}|att{attempt}|LMT",
                )
                o.transmit = True
                px_str = f"{o.lmtPrice:.4f}"

        # Logging: only show ref/bps if we actually used a priced order
        if ref_px_now is not None and bps_now is not None:
            print(
                f"[LEG] {symbol} {action} qty={remain} ref_now={ref_px_now:.4f} "
                f"bps_now={bps_now:.1f} px={px_str} refTag={o.orderRef}"
            )
        else:
            print(f"[LEG] {symbol} {action} qty={remain} px={px_str} refTag={o.orderRef}")

        if dry_run:
            filled_total += remain
            continue

        trade = ib.placeOrder(contract, o)
        last_trade = trade

        # Immediate resync helps a lot with PendingSubmit weirdness
        try:
            ib.reqAllOpenOrders()
        except Exception:
            pass
        ib.sleep(0.5)

        is_market_like = (str(getattr(o, "orderType", "")) or "").upper() != "LMT"
        accepted, trade = wait_for_trade_accepted(
            ib, trade, timeout=(market_accept_timeout if is_market_like else accept_timeout)
        )

        if not accepted:
            st = (trade.orderStatus.status or "")
            print(f"[LEG] {symbol} not accepted within timeout (status={st}); attempting cancel/resync before retry.")
            cancel_and_wait_with_resync(ib, trade, timeout=cancel_timeout)
            continue

        done_timeout = float(
            exec_cfg.get(
                "market_done_timeout_sec" if is_market_like else "limit_done_timeout_sec",
                180.0 if is_market_like else timeout,
            )
        )
        trade = wait_for_trade_terminal(ib, trade, timeout=done_timeout)

        status = (trade.orderStatus.status or "").lower()

        filled = int(trade.orderStatus.filled or 0)
        remaining = trade.orderStatus.remaining
        remaining = None if remaining is None else int(remaining)

        print(f"[LEG] status={status} filled={filled} remaining={remaining}")
        filled_total = min(qty, filled_total + filled)

        if status not in TERMINAL and remaining and remaining > 0:
            print(f"[LEG] {symbol} still working (status={status}); cancelling before retry.")
            cancel_and_wait_with_resync(ib, trade, timeout=cancel_timeout)

    return int(filled_total), last_trade


# ---------------------------
# Baseline snapshot mechanics
# ---------------------------

def load_baseline_qty(path: Path) -> Dict[str, float]:
    if not path.exists():
        print(f"[BASELINE] No baseline file found at {path}. Treating baseline as empty.")
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    # baseline_snapshot.py writes qty in column "qty"
    if "qty" not in df.columns:
        raise ValueError(f"Baseline file {path} missing required column 'qty'. Columns={list(df.columns)}")
    return dict(df.groupby("symbol")["qty"].sum())


def current_ib_positions(ib: IB) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for p in ib.positions():
        sym = universal_symbol_from_ib(p.contract.symbol)
        out[sym] = out.get(sym, 0.0) + float(p.position)
    return out


def strategy_position_only(ib_pos: Dict[str, float], baseline: Dict[str, float]) -> Dict[str, float]:
    syms = set(ib_pos) | set(baseline)
    return {s: float(ib_pos.get(s, 0.0) - baseline.get(s, 0.0)) for s in syms}


# ---------------------------
# IO helpers
# ---------------------------

def append_fills(rows: List[dict], fills_path: Path) -> None:
    fills_path.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame(rows)
    if fills_path.exists():
        df_old = pd.read_csv(fills_path)
        df_out = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_out = df_new
    df_out.to_csv(fills_path, index=False)
    print(f"[FILLS] Appended {len(rows)} rows -> {fills_path}")


def resolve_plan_path(run_date: str, paths_cfg: dict) -> Path:
    """
    Prefer dated run folder plan if present:
      data/runs/<run_date>/proposed_trades.csv
    else:
      paths.proposed_trades_csv
    """
    dated = run_dir(run_date) / "proposed_trades.csv"
    if dated.exists():
        return dated
    return Path(paths_cfg.get("proposed_trades_csv", "data/proposed_trades.csv"))


def resolve_fills_path(run_date: str, paths_cfg: dict) -> Path:
    """
    Always write fills into dated execution folder, but you can also keep
    a global ledger if you want later.
    """
    return exec_dir(run_date) / "fills.csv"


def write_execution_snapshot(run_date: str, df: pd.DataFrame, name: str) -> None:
    p = exec_dir(run_date) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    print(f"[EXEC] Wrote {p}")


# ---------------------------
# Sizing helpers (targets -> shares)
# ---------------------------

def target_shares_from_usd(notional_usd: float, px: float) -> int:
    """
    Convert target USD notional to target shares using truncation toward 0.
    - Long (positive notional) -> positive shares
    - Short (negative notional) -> negative shares
    """
    if px <= 0:
        raise ValueError("Price must be > 0")
    return int(notional_usd / px)


# ---------------------------
# Main
# ---------------------------

def main() -> None:
    approve_y_required_first_n = 5
    pairs_approved_by_y = 0
    auto_after_first_n = False

    CONFIG_YML = Path("config/strategy_config.yml")

    ap = argparse.ArgumentParser()
    ap.add_argument("--run-date", default=os.getenv("RUN_DATE") or today_str(), help="YYYY-MM-DD")
    ap.add_argument("--strategy-tag", default=None, help="Override strategy.tag from config")
    args = ap.parse_args()

    if not CONFIG_YML.exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_YML}")

    cfg = yaml.safe_load(CONFIG_YML.read_text(encoding="utf-8")) or {}

    ibkr_cfg = cfg.get("ibkr", {}) or {}
    strat_cfg = cfg.get("strategy", {}) or {}
    paths_cfg = cfg.get("paths", {}) or {}
    exec_cfg = cfg.get("execution", {}) or {}

    run_date = args.run_date

    # --- Strategy tag ---
    strategy_tag = args.strategy_tag or str(strat_cfg.get("tag", "")).strip()
    if not strategy_tag:
        raise ValueError("Missing strategy.tag in config/strategy_config.yml (or pass --strategy-tag).")

    # --- IBKR connection params ---
    host = str(ibkr_cfg.get("host", "127.0.0.1"))
    port = int(ibkr_cfg.get("port", 7497))
    client_id = int(ibkr_cfg.get("client_id", 3))
    prefer_delayed = bool(ibkr_cfg.get("prefer_delayed", True))

    # Make prefer_delayed visible to execute_leg without threading args everywhere
    exec_cfg = dict(exec_cfg)
    exec_cfg["prefer_delayed"] = prefer_delayed

    # --- Execution params ---
    limit_bps = float(exec_cfg.get("limit_bps", 10.0))
    timeout = float(exec_cfg.get("timeout_sec", 90))
    short_first = bool(exec_cfg.get("short_first", True))
    max_retries = int(exec_cfg.get("max_retries", 3))

    if "DRY_RUN" in os.environ:
        dry_run = bool(int(os.getenv("DRY_RUN", "0")))
    else:
        dry_run = bool(exec_cfg.get("dry_run", False))

    if dry_run:
        print("[DRY_RUN] Enabled. No orders will be placed.")

    # --- Paths ---
    plan_path = resolve_plan_path(run_date, paths_cfg)
    baseline_csv = Path(paths_cfg.get("baseline_csv", "data/baseline_snapshot.csv"))
    fills_path = resolve_fills_path(run_date, paths_cfg)

    if not plan_path.exists():
        raise FileNotFoundError(f"Trade plan not found: {plan_path}")

    plan = pd.read_csv(plan_path)
    if "strategy_tag" in plan.columns:
        plan = plan[plan["strategy_tag"].astype(str) == strategy_tag].copy()

    if plan.empty:
        raise ValueError(f"No rows in {plan_path} for strategy_tag={strategy_tag}")

    # baseline protection
    baseline = load_baseline_qty(baseline_csv)

    # create execution folder
    exec_dir(run_date).mkdir(parents=True, exist_ok=True)

    ib = connect_ib(host, port, client_id)
    try:
        ib_pos = current_ib_positions(ib)
        strat_pos = strategy_position_only(ib_pos, baseline)

        print(
            f"[POS] current IB symbols={len(ib_pos)}; "
            f"baseline symbols={len(baseline)}; "
            f"strategy-only symbols={len(strat_pos)}"
        )
        print(f"[PLAN] Using: {plan_path}")
        print(f"[BASELINE] Using: {baseline_csv}")
        print(f"[EXEC] Writing to: {exec_dir(run_date)}")

        fills_to_append: List[dict] = []
        approve_all = False

        # Precompute prices once per unique symbol for this run (keeps execution consistent)
        symbols = sorted(set(plan["Underlying"].astype(str).str.upper()) | set(plan["ETF"].astype(str).str.upper()))
        prices: Dict[str, float] = {}
        for s in symbols:
            prices[s] = get_snapshot_price(ib, s, prefer_delayed=prefer_delayed)

        # Save execution pricing snapshot
        px_df = pd.DataFrame([{"symbol": k, "price": v} for k, v in prices.items()]).sort_values("symbol")
        write_execution_snapshot(run_date, px_df, "prices_snapshot.csv")

        # Manual approval loop (pair-by-pair)
        for _, row in plan.iterrows():
            u = str(row["Underlying"]).upper()
            e = str(row["ETF"]).upper()
            pair_id = str(row.get("pair_id", f"{u}__{e}"))

            tu = float(row["long_usd"])    # target underlying notional (positive)
            te = float(row["short_usd"])   # target ETF notional (negative for short)

            px_u = float(prices[u])
            px_e = float(prices[e])

            # Targets in shares (ABSOLUTE)
            target_sh_u = target_shares_from_usd(tu, px_u)
            target_sh_e = target_shares_from_usd(te, px_e)  # negative for short

            # Current strategy-only positions
            cur_strat_u = float(strat_pos.get(u, 0.0))
            cur_strat_e = float(strat_pos.get(e, 0.0))

            # Deltas we need to trade
            delta_u = int(target_sh_u - cur_strat_u)
            delta_e = int(target_sh_e - cur_strat_e)

            print("\n" + "-" * 100)
            print(f"[PAIR] {pair_id}")
            print(f"  Prices: u={px_u:.4f} e={px_e:.4f}")
            print(f"  Target shares: u={target_sh_u:+d} e={target_sh_e:+d}")
            print(f"  Strategy-only current: u={cur_strat_u:+.0f} e={cur_strat_e:+.0f}")
            print(f"  Delta to trade: u={delta_u:+d} e={delta_e:+d}")
            print(f"  Baseline qty: baseline[u]={baseline.get(u,0):+.0f} baseline[e]={baseline.get(e,0):+.0f}")

            if delta_u == 0 and delta_e == 0:
                print("  [SKIP] Already at target (no deltas).")
                continue

            if not auto_after_first_n:
                ans = input(f"Approve pair {pairs_approved_by_y+1}/{approve_y_required_first_n}? Type 'y' to run, anything else skips, 'q' quits: ").strip().lower()
                if ans == "q":
                    break
                if ans != "y":
                    continue

                pairs_approved_by_y += 1
                if pairs_approved_by_y >= approve_y_required_first_n:
                    auto_after_first_n = True


            order_base_ref = f"{strategy_tag}|{pair_id}"

            # Helper to execute a delta in shares:
            def exec_delta(symbol: str, delta: int, px: float, ref_suffix: str) -> Tuple[int, Optional[Trade]]:
                if delta == 0:
                    return 0, None
                action = "BUY" if delta > 0 else "SELL"
                qty = abs(delta)
                return execute_leg(
                    ib=ib,
                    symbol=symbol,
                    action=action,
                    qty=qty,
                    ref_price=px,
                    bps=limit_bps,
                    order_ref=f"{order_base_ref}|{ref_suffix}",
                    exec_cfg=exec_cfg,
                    timeout=timeout,
                    max_retries=max_retries,
                    dry_run=dry_run,
                )

            # Execute legs (short_first means: trade ETF delta first, then underlying delta)
            if short_first:
                # Execute ETF delta first. If this is a SHORT leg and we cannot place/fill it cleanly,
                # we skip the pair to avoid ending up long the underlying without the hedge.
                filled_e, trade_e = exec_delta(e, delta_e, px_e, "ETF_DELTA")

                short_intent = (delta_e < 0)
                short_qty = abs(int(delta_e))
                short_clean = (not short_intent) or (filled_e == short_qty)

                if short_intent and not short_clean:
                    st = (trade_e.orderStatus.status if trade_e else "NO_TRADE")
                    print(f"[PAIR] Skipping underlying leg for {pair_id}: ETF short not clean (filled={filled_e}/{short_qty}, status={st}).")
                    filled_u, trade_u = 0, None
                else:
                    filled_u, trade_u = exec_delta(u, delta_u, px_u, "UNDER_DELTA")
            else:
                filled_u, trade_u = exec_delta(u, delta_u, px_u, "UNDER_DELTA")
                filled_e, trade_e = exec_delta(e, delta_e, px_e, "ETF_DELTA")

            # Record fills
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            fills_to_append.append(
                {
                    "filled_at": now,
                    "run_date": run_date,
                    "strategy_tag": strategy_tag,
                    "pair_id": pair_id,
                    "underlying": u,
                    "etf": e,
                    "px_under": px_u,
                    "px_etf": px_e,
                    "target_sh_under": target_sh_u,
                    "target_sh_etf": target_sh_e,
                    "delta_sh_under": delta_u,
                    "delta_sh_etf": delta_e,
                    "filled_sh_under": filled_u,
                    "filled_sh_etf": filled_e,
                    "notes": "",
                }
            )

            # Refresh positions for next pair view (important for rebalance correctness if many pairs overlap)
            ib_pos = current_ib_positions(ib)
            strat_pos = strategy_position_only(ib_pos, baseline)

        if fills_to_append:
            append_fills(fills_to_append, fills_path)

        print("[DONE] Execution pass complete.")
    finally:
        ib.disconnect()


if __name__ == "__main__":
    main()
