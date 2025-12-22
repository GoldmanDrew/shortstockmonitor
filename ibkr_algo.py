#!/usr/bin/env python3
"""
ibkr_algo.py

High-level workflow:

1) Load screened ETF list from data/etf_screened_today.csv
   - Must contain: ETF, include_for_algo
   - Typically also has borrow_current, cagr_port_hist, etc.

2) Load static mapping from config/etf_cagr.csv
   - Must contain: ETF, Underlying, LevType ("CC" or "2x")

3) Merge to build a universe of pairs:
   - Pair(underlying, etf, lev_type)

4) Connect to IBKR TWS / Gateway via ib_insync.

5) On first run (no positions_state.csv and no IB positions):
   - Read NetLiquidation (account equity)
   - Allocate 50% (x) to CC leg, 50% (100-x) to LEV2 leg
   - For:
       CC leg: long 1x underlying, short 1x CC ETF, NEVER rebalance.
       LEV2 leg: long 1x underlying, short 0.5x 2x ETF, rebalance in future.
   - Use Adaptive limit orders (configurable priority: Patient/Normal/Urgent).
   - If one leg of a pair fills but the other fails, we flatten the filled leg.
     We retry the pair up to 3 times before giving up.

6) Save a simple positions_state.csv file with our intended positions.

Notes:
- This script assumes:
    - data/etf_screened_today.csv exists (built by your FTP screening script)
    - config/etf_cagr.csv exists with mapping info
    - TWS is running and API is enabled
    - You have either LIVE or DELAYED market data (set via MARKET_DATA_MODE)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import pandas as pd
from ib_insync import IB, Stock, Order, Trade, TagValue

# --------------------------------------------------
# PATHS / CONFIG
# --------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(os.getenv("GITHUB_WORKSPACE", str(SCRIPT_DIR))).resolve()

# Mapping file (contains ETF, Underlying, LevType)
CAGR_MAP_CSV = REPO_ROOT / "config" / "etf_cagr.csv"

# Screened daily universe (output of FTP screening job)
SCREENED_CSV = Path(
    os.getenv("SCREENED_CSV", str(REPO_ROOT / "data" / "etf_screened_today.csv"))
)

# Local state of positions
STATE_FILE = REPO_ROOT / "data" / "positions_state.csv"

# IBKR connection
IB_HOST = os.getenv("IB_HOST", "127.0.0.1")
IB_PORT = int(os.getenv("IB_PORT", "7497"))  # 7497 paper, 7496 live by default
IB_CLIENT_ID = int(os.getenv("IB_CLIENT_ID", "3"))

# Safety / sizing
MAX_SHARES_PER_ORDER = int(os.getenv("MAX_SHARES_PER_ORDER", "500"))

# 0 = send real orders; 1 = only log (no orders)
DRY_RUN = bool(int(os.getenv("DRY_RUN", "0")))

MARKET_DATA_MODE = os.getenv("MARKET_DATA_MODE", "DELAYED")

# Adaptive algo priority: must be one of "Patient", "Normal", "Urgent"
ADAPTIVE_PRIORITY = os.getenv("ADAPTIVE_PRIORITY", "Patient").strip().capitalize()
if ADAPTIVE_PRIORITY not in ("Patient", "Normal", "Urgent"):
    print(
        f"[CONFIG] Invalid ADAPTIVE_PRIORITY='{ADAPTIVE_PRIORITY}' supplied; "
        "falling back to 'Patient'."
    )
    ADAPTIVE_PRIORITY = "Patient"

# Execution mode:
#   LIMIT   - use plain limit orders around ref price (default, recommended)
#   ADAPTIVE - use IB Adaptive algo (requires robust market data)
EXECUTION_MODE = os.getenv("EXECUTION_MODE", "LIMIT").strip().upper()
if EXECUTION_MODE not in ("LIMIT", "ADAPTIVE"):
    print(f"[CONFIG] Invalid EXECUTION_MODE='{EXECUTION_MODE}', falling back to 'LIMIT'.")
    EXECUTION_MODE = "LIMIT"

# --------------------------------------------------
# BASIC DATA STRUCTURES
# --------------------------------------------------

@dataclass
class Pair:
    underlying: str   # e.g. AAPL, BRK-B
    etf: str          # e.g. AAPU
    lev_type: str     # "CC" or "2x"


# Map Yahoo-ish oddballs to IB symbols
IB_SYMBOL_MAP: Dict[str, Tuple[str, Optional[str]]] = {
    "BRK-B": ("BRK B", "NYSE"),
    "BRK-A": ("BRK A", "NYSE"),
    # Add more overrides here if needed
}

# Reverse map: IB symbol -> universal symbol
REVERSE_IB_SYMBOL_MAP: Dict[str, str] = {
    ib_sym: uni for uni, (ib_sym, _primary) in IB_SYMBOL_MAP.items()
}

# --------------------------------------------------
# STATE LOAD / SAVE
# --------------------------------------------------

def load_state() -> pd.DataFrame:
    """
    Load the current portfolio state from STATE_FILE.

    Error handling:
      - If file is missing → return empty template
      - If file is unreadable/corrupted → log + return empty template
      - If file is missing required columns → add them with default values
    """
    required_cols = [
        "symbol",
        "underlying",
        "etf",
        "lev_type",
        "sub",          # "CC" or "LEV2"
        "side",         # "LONG" or "SHORT"
        "shares",
        "opened_at",
    ]

    if not STATE_FILE.exists():
        print(f"[STATE] No existing state file found at {STATE_FILE}. Initializing empty state.")
        return pd.DataFrame(columns=required_cols)

    try:
        df = pd.read_csv(STATE_FILE)
    except Exception as e:
        print(f"[STATE] ERROR reading state file: {STATE_FILE}")
        print(f"[STATE] Exception: {e}")
        print("[STATE] Reinitializing to empty state.")
        return pd.DataFrame(columns=required_cols)

    if df.empty:
        print("[STATE] State file is empty. Using empty template.")
        return pd.DataFrame(columns=required_cols)

    for col in required_cols:
        if col not in df.columns:
            print(f"[STATE] Missing column '{col}' in saved state. Adding with defaults.")
            df[col] = "" if col != "shares" else 0

    # Normalize types
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0).astype(int)
    for col in ["symbol", "underlying", "etf", "lev_type", "sub", "side"]:
        df[col] = df[col].astype(str).str.strip().str.upper()

    print(f"[STATE] Loaded {len(df)} positions from state file.")
    return df[required_cols]


def save_state(df: pd.DataFrame) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(STATE_FILE, index=False)
    print(f"[STATE] State saved to {STATE_FILE}")

# --------------------------------------------------
# IBKR CONNECTION & SYMBOL HANDLING
# --------------------------------------------------

def ib_symbol_from_universal(sym: str) -> Tuple[str, Optional[str]]:
    sym_u = sym.upper().strip()
    if sym_u in IB_SYMBOL_MAP:
        return IB_SYMBOL_MAP[sym_u]
    return sym_u, None


def make_stock(symbol: str) -> Stock:
    ib_sym, primary = ib_symbol_from_universal(symbol)
    c = Stock(ib_sym, "SMART", "USD")
    if primary:
        c.primaryExchange = primary
    return c


def connect_ib() -> IB:
    ib = IB()
    print(f"Connecting to IB at {IB_HOST}:{IB_PORT}, clientId={IB_CLIENT_ID}")
    ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID)
    print("Connected:", ib.isConnected())
    return ib

# --------------------------------------------------
# MARKET DATA: PRICES (LIVE / DELAYED MODES)
# --------------------------------------------------
def get_last_prices(ib: IB, symbols: List[str]) -> Dict[str, float]:
    """
    Fetch a 'reasonable' price for each universal symbol.

    If MARKET_DATA_MODE == "LIVE":
        For each symbol:
          1) Try LIVE snapshot (marketDataType=1):
             - rt_mid (bid/ask)
             - rt_last
             - rt_close
             - marketPrice()
          2) If no live price, try DELAYED snapshot (marketDataType=3):
             - delayed_mid (delayedBid/delayedAsk)
             - delayed_last
             - delayed_close
             - fall back to any rt_* fields IB might populate
          3) If still nothing, use 1-day historical close.

    If MARKET_DATA_MODE == "DELAYED":
        For each symbol:
          1) Try DELAYED snapshot (marketDataType=3):
             - delayed_mid, delayed_last, delayed_close, or rt_* fields
          2) If nothing, use 1-day historical close.
    """

    def safe(v):
        return v if (v is not None and isinstance(v, (int, float)) and v > 0) else None

    prices: Dict[str, float] = {}

    mode = str(MARKET_DATA_MODE).upper().strip()
    if mode not in ("LIVE", "DELAYED"):
        print(f"[PRICE] Unknown MARKET_DATA_MODE='{MARKET_DATA_MODE}', defaulting to 'DELAYED'.")
        mode = "DELAYED"

    for sym in symbols:
        sym_u = sym.upper().strip()
        contract = make_stock(sym_u)
        ib.qualifyContracts(contract)

        price: Optional[float] = None
        source: Optional[str] = None

        # ------------------------------------------------
        # Helper: request a single snapshot with a given
        # market data type (1=live, 3=delayed)
        # ------------------------------------------------
        def snapshot_with_type(data_type: int, label: str) -> Tuple[Optional[float], Optional[str]]:
            """
            Returns (price, source_label) or (None, None) if no usable price.
            label is "live" or "delayed" purely for logging.
            """
            print(f"[PRICE] {sym_u}: requesting {label} snapshot (type={data_type})")
            try:
                ib.reqMarketDataType(data_type)
                ticker = ib.reqMktData(contract, "", snapshot=True)

                # Let data populate
                for _ in range(10):
                    ib.sleep(0.5)
                    rt_bid  = safe(getattr(ticker, "bid", None))
                    rt_ask  = safe(getattr(ticker, "ask", None))
                    rt_last = safe(getattr(ticker, "last", None))
                    rt_close = safe(getattr(ticker, "close", None))
                    rt_mkt  = safe(ticker.marketPrice())

                    d_bid   = safe(getattr(ticker, "delayedBid", None))
                    d_ask   = safe(getattr(ticker, "delayedAsk", None))
                    d_last  = safe(getattr(ticker, "delayedLast", None))
                    d_close = safe(getattr(ticker, "delayedClose", None))

                    if any([rt_bid, rt_ask, rt_last, rt_close, rt_mkt,
                            d_bid, d_ask, d_last, d_close]):
                        break

                # Re-read after waiting
                rt_bid  = safe(getattr(ticker, "bid", None))
                rt_ask  = safe(getattr(ticker, "ask", None))
                rt_last = safe(getattr(ticker, "last", None))
                rt_close = safe(getattr(ticker, "close", None))
                rt_mkt  = safe(ticker.marketPrice())

                d_bid   = safe(getattr(ticker, "delayedBid", None))
                d_ask   = safe(getattr(ticker, "delayedAsk", None))
                d_last  = safe(getattr(ticker, "delayedLast", None))
                d_close = safe(getattr(ticker, "delayedClose", None))

                if label == "live":
                    # Prefer live, then delayed
                    if rt_bid and rt_ask:
                        return (rt_bid + rt_ask) / 2.0, "rt_mid"
                    if rt_last:
                        return rt_last, "rt_last"
                    if rt_close:
                        return rt_close, "rt_close"
                    if rt_mkt:
                        return rt_mkt, "rt_marketPrice"
                    if d_bid and d_ask:
                        return (d_bid + d_ask) / 2.0, "delayed_mid"
                    if d_last:
                        return d_last, "delayed_last"
                    if d_close:
                        return d_close, "delayed_close"
                else:  # label == "delayed"
                    # Prefer delayed, then any live fields IB gives us anyway
                    if d_bid and d_ask:
                        return (d_bid + d_ask) / 2.0, "delayed_mid"
                    if d_last:
                        return d_last, "delayed_last"
                    if d_close:
                        return d_close, "delayed_close"
                    if rt_bid and rt_ask:
                        return (rt_bid + rt_ask) / 2.0, "rt_mid"
                    if rt_last:
                        return rt_last, "rt_last"
                    if rt_close:
                        return rt_close, "rt_close"
                    if rt_mkt:
                        return rt_mkt, "rt_marketPrice"

            except Exception as e:
                print(f"[PRICE] {sym_u}: error requesting {label} snapshot (type={data_type}): {e}")
            finally:
                try:
                    ib.cancelMktData(contract)
                except Exception:
                    pass

            print(f"[PRICE] {sym_u}: no usable {label} snapshot")
            return None, None

        # 1) If mode=LIVE, try live snapshot first
        if mode == "LIVE":
            price, source = snapshot_with_type(1, "live")

        # 2) If still None (or mode=DELAYED), try delayed snapshot
        if price is None:
            price, source = snapshot_with_type(3, "delayed")

        # 3) Historical close as final fallback
        if price is None:
            print(f"[PRICE] {sym_u}: falling back to 1D historical close")
            try:
                bars = ib.reqHistoricalData(
                    contract,
                    endDateTime="",
                    durationStr="1 D",
                    barSizeSetting="1 day",
                    whatToShow="TRADES",
                    useRTH=True,
                    formatDate=1,
                    keepUpToDate=False,
                )
                if bars:
                    price = safe(bars[-1].close)
                    source = "hist_close"
            except Exception as e:
                print(f"[PRICE] {sym_u}: error fetching historical data: {e}")

        if price is None:
            price = float("nan")
            source = "none"

        prices[sym_u] = price

        if source == "none":
            print(f"[PRICE] {sym_u}: no usable price (live+delayed+historical all failed, mode={mode})")
        else:
            print(f"[PRICE] {sym_u}: {price:.4f} ({source}, mode={mode})")

    return prices


# --------------------------------------------------
# ADAPTIVE ORDER HELPERS
# --------------------------------------------------

def adaptive_passive_order(action: str, quantity: int, ref_price: float) -> Order:
    """
    Build the order we use for each leg.

    - In EXECUTION_MODE == "LIMIT":
        Use a plain LIMIT order, priced slightly through the delayed mid.
        This is robust for paper accounts with delayed data only.

    - In EXECUTION_MODE == "ADAPTIVE":
        Use IB's Adaptive algo with 'adaptivePriority' (Patient/Normal/Urgent).
        Only recommended if you have proper live/delayed subscriptions.
    """
    action = action.upper()
    qty = int(quantity)
    px = float(ref_price)

    o = Order()
    o.action = action
    o.totalQuantity = qty
    o.tif = "DAY"

    if EXECUTION_MODE == "LIMIT":
        # How aggressive should we be? bps offset from ref price
        # e.g. 10 bps = 0.10% through the mid.
        bps = float(os.getenv("LIMIT_BPS_OFFSET", "10"))  # 10 bps default
        offset = px * (bps / 10_000.0)

        if action == "BUY":
            lmt = px + offset
        else:  # SELL
            lmt = px - offset

        o.orderType = "LMT"
        o.lmtPrice = round(lmt, 4)
        print(f"[ORDER] {action} {qty} @ LMT {o.lmtPrice:.4f} (ref={px:.4f}, {bps}bps, mode=LIMIT)")
        return o

    # EXECUTION_MODE == "ADAPTIVE"
    o.orderType = "LMT"
    o.lmtPrice = px
    o.algoStrategy = "Adaptive"
    o.algoParams = [TagValue("adaptivePriority", ADAPTIVE_PRIORITY)]
    print(f"[ORDER] {action} {qty} @ LMT {o.lmtPrice:.4f} (Adaptive {ADAPTIVE_PRIORITY})")
    return o



def wait_for_trade_completion(
    trade: Trade,
    timeout: float = 60.0,
    poll_interval: float = 0.5,
) -> Trade:
    """
    Block until the trade is done, cancelled, or timeout occurs.
    Returns the final Trade object (status updated).
    """
    start = time.time()
    while time.time() - start < timeout:
        if trade.isDone():
            break
        time.sleep(poll_interval)
    return trade


def execute_leg_adaptive(
    ib: IB,
    symbol: str,
    action: str,
    quantity: int,
    limit_price: float,
    max_retries: int = 3,
    timeout: float = 60.0,
) -> Tuple[bool, Optional[Trade]]:
    """
    Execute one leg (BUY or SELL) with retries.

    - If EXECUTION_MODE = LIMIT:
        Use a simple LMT order priced off `limit_price` with a small bps offset.
    - If EXECUTION_MODE = ADAPTIVE:
        Use IB Adaptive algo with priority ADAPTIVE_PRIORITY.

    Returns (success, Trade).
      - success=True means fully filled
      - success=False means could not fill after retries
    """
    if quantity <= 0:
        print(f"[LEG] Skipping {action} {symbol} with quantity {quantity} (<= 0).")
        return True, None

    # Enforce max shares per order (Precautionary Settings)
    if quantity > MAX_SHARES_PER_ORDER:
        print(f"[LEG] Clamping {symbol} {action} from {quantity} -> {MAX_SHARES_PER_ORDER} due to max size.")
        quantity = MAX_SHARES_PER_ORDER

    contract = make_stock(symbol)
    ib.qualifyContracts(contract)

    trade: Optional[Trade] = None

    for attempt in range(1, max_retries + 1):
        print(
            f"[LEG] Attempt {attempt}/{max_retries}: {action.upper()} {quantity} {symbol} "
            f"@ ~{limit_price:.4f} (mode={EXECUTION_MODE}, priority={ADAPTIVE_PRIORITY})"
        )

        order = adaptive_passive_order(action, quantity, limit_price)

        if DRY_RUN:
            print(f"[DRY_RUN] Would place order: {action.upper()} {quantity} {symbol}")
            return True, None

        trade = ib.placeOrder(contract, order)
        trade = wait_for_trade_completion(trade, timeout=timeout)

        status = trade.orderStatus.status or ""
        filled = trade.orderStatus.filled or 0
        remaining = trade.orderStatus.remaining or 0

        print(f"[LEG] {symbol} status={status}, filled={filled}, remaining={remaining}")

        # If fully filled, done
        if status.lower() in ("filled", "partiallyfilled") and remaining == 0:
            return True, trade

        # If order still working after timeout, cancel and retry
        if status.lower() in ("presubmitted", "submitted", "pendingsubmit"):
            print(f"[LEG] Cancelling working order for {symbol} after timeout (status={status}).")
            ib.cancelOrder(order)
            time.sleep(1.0)
        elif status.lower() in ("cancelled", "apicancelled", "inactive", "rejected"):
            print(f"[LEG] Order for {symbol} ended in terminal status={status}, will retry if attempts remain.")

    print(f"[LEG] FAILED: {action.upper()} {quantity} {symbol} after {max_retries} attempts.")
    return False, trade



def flatten_leg(
    ib: IB,
    symbol: str,
    side_filled: str,
    quantity: int,
    price_hint: float,
    timeout: float = 60.0,
):
    """
    Close out a filled leg if the other leg fails.

    side_filled: "BUY" if we are currently long and need to SELL to flatten
                 "SELL" if we are currently short and need to BUY to flatten
    """
    reverse_action = "SELL" if side_filled.upper() == "BUY" else "BUY"
    print(f"[SAFETY] Flattening {symbol}: {reverse_action} {quantity} (undo {side_filled}).")

    success, _ = execute_leg_adaptive(
        ib,
        symbol=symbol,
        action=reverse_action,
        quantity=quantity,
        limit_price=price_hint,
        max_retries=3,
        timeout=timeout,
    )
    if not success:
        print(f"[SAFETY] WARNING: Could not fully flatten {symbol}!")


def execute_pair(
    ib: IB,
    pair: Pair,
    qty_underlying: int,
    qty_etf_short: int,
    px_underlying: float,
    px_etf: float,
    pair_max_retries: int = 3,
    leg_max_retries: int = 3,
    timeout_per_leg: float = 60.0,
) -> bool:
    """
    Execute a pair:
        - LONG underlying (BUY qty_underlying)
        - SHORT ETF (SELL qty_etf_short)

    Uses Adaptive orders with retries.

    Logic:
      - Try to execute both legs up to pair_max_retries times.
      - Within each attempt, each leg is retried up to leg_max_retries times.
      - If in any attempt one leg fills but the other fails, we immediately
        flatten the filled leg, then (if we still have attempts left) we
        re-try the whole pair from a flat starting point.
      - If, after all attempts, we still cannot get both legs filled in
        the same attempt, we give up and ensure we are flat.

    Returns True if the pair ends up established, False otherwise.
    """
    u_sym = pair.underlying
    e_sym = pair.etf

    last_u_trade: Optional[Trade] = None
    last_e_trade: Optional[Trade] = None

    for pair_attempt in range(1, pair_max_retries + 1):
        print(f"\n[PAIR] Attempt {pair_attempt}/{pair_max_retries} for {u_sym} vs {e_sym}")

        # 1) Execute underlying BUY
        u_success, u_trade = execute_leg_adaptive(
            ib,
            symbol=u_sym,
            action="BUY",
            quantity=qty_underlying,
            limit_price=px_underlying,
            max_retries=leg_max_retries,
            timeout=timeout_per_leg,
        )
        last_u_trade = u_trade

        # 2) Execute ETF SELL
        e_success, e_trade = execute_leg_adaptive(
            ib,
            symbol=e_sym,
            action="SELL",
            quantity=qty_etf_short,
            limit_price=px_etf,
            max_retries=leg_max_retries,
            timeout=timeout_per_leg,
        )
        last_e_trade = e_trade

        if u_success and e_success:
            print(f"[PAIR] SUCCESS: Established pair LONG {u_sym} / SHORT {e_sym}")
            return True

        # Flatten any filled leg if the other failed
        if u_success and not e_success and u_trade is not None:
            filled_u = int(u_trade.orderStatus.filled)
            if filled_u > 0:
                flatten_leg(
                    ib,
                    symbol=u_sym,
                    side_filled="BUY",
                    quantity=filled_u,
                    price_hint=px_underlying,
                    timeout=timeout_per_leg,
                )

        if e_success and not u_success and e_trade is not None:
            filled_e = int(e_trade.orderStatus.filled)
            if filled_e > 0:
                flatten_leg(
                    ib,
                    symbol=e_sym,
                    side_filled="SELL",
                    quantity=filled_e,
                    price_hint=px_etf,
                    timeout=timeout_per_leg,
                )

        print(f"[PAIR] Pair attempt {pair_attempt} FAILED for {u_sym}/{e_sym}, will retry if attempts remain.")

    print(f"[PAIR] GAVE UP establishing pair {u_sym}/{e_sym} after {pair_max_retries} attempts.")
    return False

# --------------------------------------------------
# UNIVERSE LOADING & PAIR CONSTRUCTION
# --------------------------------------------------

def load_screened_universe() -> pd.DataFrame:
    """
    Load the daily screened universe that setup.py produced.

    Expected columns in data/etf_screened_today.csv:
      - ETF
      - cagr_port_hist
      - borrow_current
      - shares_available
      - include_for_algo
      - Underlying (required for pair building)
      - LevType   (required for pair building: e.g. 'CC', 'LEV2')

    Returns a DataFrame filtered to include_for_algo == True.
    """
    if not SCREENED_CSV.exists():
        raise FileNotFoundError(f"Screened CSV not found: {SCREENED_CSV}")

    df = pd.read_csv(SCREENED_CSV)

    required = {"ETF", "include_for_algo", "Underlying", "LevType"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{SCREENED_CSV} is missing required columns: {missing}. "
            "Make sure setup.py was run on a version of etf_cagr.csv that "
            "includes Underlying and LevType."
        )

    # Clean tickers
    df["ETF"] = (
        df["ETF"]
        .astype(str)
        .str.strip()
        .str.replace(".", "-", regex=False)
        .str.upper()
    )
    df["Underlying"] = (
        df["Underlying"]
        .astype(str)
        .str.strip()
        .str.replace(".", "-", regex=False)
        .str.upper()
    )

    included = df[df["include_for_algo"]].copy()

    print("Screened universe sample:")
    print(included.head())
    print(f"[UNIVERSE] {len(included)} ETFs included for algo.")

    return included


@dataclass
class Pair:
    underlying: str
    etf: str
    lev_type: str   # "CC" or "LEV2"


def build_pairs(screened_df: pd.DataFrame) -> list[Pair]:
    """
    Build pair list (underlying, etf, lev_type) from the *screened* universe.

    screened_df must already be filtered to include_for_algo == True and contain:
      - Underlying
      - ETF
      - LevType  (for now you can keep using 'CC' for covered-call, 'LEV2' for 2x)
    """
    required = {"Underlying", "ETF", "LevType"}
    missing = required - set(screened_df.columns)
    if missing:
        raise ValueError(
            f"screened_df must contain columns: {required}, but has: "
            f"{list(screened_df.columns)}"
        )

    pairs: list[Pair] = []
    for _, row in screened_df.iterrows():
        pairs.append(
            Pair(
                underlying=str(row["Underlying"]).upper(),
                etf=str(row["ETF"]).upper(),
                lev_type=str(row["LevType"]).upper(),
            )
        )

    print(f"[UNIVERSE] Built {len(pairs)} pairs.")
    return pairs


# --------------------------------------------------
# CAPITAL ALLOCATION & INITIAL PORTFOLIO
# --------------------------------------------------

def get_account_equity(ib: IB) -> float:
    """
    Get NetLiquidation value from accountSummary.
    """
    summary = ib.accountSummary()
    net = None
    for row in summary:
        if row.tag == "NetLiquidation":
            try:
                net = float(row.value)
            except ValueError:
                continue
    if net is None:
        raise RuntimeError("Could not find NetLiquidation in account summary.")
    print(f"Account equity (NetLiquidation): {net:.2f}")
    return net


def initialize_portfolio(ib: IB, pairs: List[Pair]) -> None:
    """
    First-time setup:
      - Split capital 50/50: CC vs LEV2
      - For CC pairs: long 1x underlying, short 1x CC ETF
      - For LEV2 pairs: long 1x underlying, short 0.5x 2x ETF

    This function sends orders but does not return a state DataFrame.
    After calling it, you should sync from IB using sync_state_from_ib().
    """
    equity = get_account_equity(ib)

    cc_pairs = [p for p in pairs if p.lev_type.upper() == "CC"]
    lev2_pairs = [p for p in pairs if p.lev_type.upper() == "2X"]

    if not cc_pairs and not lev2_pairs:
        print("[INIT] No pairs to trade.")
        return

    capital_cc = equity * 0.5 if cc_pairs else 0.0
    capital_lev2 = equity * 0.5 if lev2_pairs else 0.0

    print(f"Allocating {capital_cc:.2f} to CC leg, {capital_lev2:.2f} to LEV2 leg.")

    # CC leg
    if cc_pairs:
        per_pair_cc = capital_cc / len(cc_pairs)
        print(f"CC leg: {len(cc_pairs)} pairs, per-pair capital ~ {per_pair_cc:.2f}")

        syms_cc = sorted({p.underlying for p in cc_pairs} | {p.etf for p in cc_pairs})
        prices_cc = get_last_prices(ib, syms_cc)

        for p in cc_pairs:
            pu = prices_cc.get(p.underlying.upper(), float("nan"))
            pe = prices_cc.get(p.etf.upper(), float("nan"))
            if not (pu > 0 and pe > 0):
                print(f"[CC] Skipping {p.underlying}/{p.etf}: bad prices pu={pu}, pe={pe}")
                continue

            # Simple equal split of capital between long and short
            cap_long = per_pair_cc * 0.5
            cap_short = per_pair_cc * 0.5

            qty_u = int(cap_long // pu)
            qty_e = int(cap_short // pe)

            if qty_u <= 0 or qty_e <= 0:
                print(f"[CC] Skipping {p.underlying}/{p.etf}: too small qty_u={qty_u}, qty_e={qty_e}")
                continue

            print(f"[CC] Target {p.underlying}: {qty_u} shares @ ~{pu:.2f}; {p.etf}: short {qty_e} @ ~{pe:.2f}")
            execute_pair(
                ib,
                pair=p,
                qty_underlying=qty_u,
                qty_etf_short=qty_e,
                px_underlying=pu,
                px_etf=pe,
                pair_max_retries=3,
                leg_max_retries=3,
            )

    # LEV2 leg
    if lev2_pairs:
        per_pair_lev2 = capital_lev2 / len(lev2_pairs)
        print(f"LEV2 leg: {len(lev2_pairs)} pairs, per-pair capital ~ {per_pair_lev2:.2f}")

        syms_lev2 = sorted({p.underlying for p in lev2_pairs} | {p.etf for p in lev2_pairs})
        prices_lev2 = get_last_prices(ib, syms_lev2)

        for p in lev2_pairs:
            pu = prices_lev2.get(p.underlying.upper(), float("nan"))
            pe = prices_lev2.get(p.etf.upper(), float("nan"))
            if not (pu > 0 and pe > 0):
                print(f"[LEV2] Skipping {p.underlying}/{p.etf}: bad prices pu={pu}, pe={pe}")
                continue

            # Long 1x underlying, short 0.5x ETF (approx via capital split)
            cap_long = per_pair_lev2 * 0.67   # more weight to long
            cap_short = per_pair_lev2 * 0.33  # ~half exposure via 2x ETF

            qty_u = int(cap_long // pu)
            qty_e = int(cap_short // pe)

            if qty_u <= 0 or qty_e <= 0:
                print(f"[LEV2] Skipping {p.underlying}/{p.etf}: too small qty_u={qty_u}, qty_e={qty_e}")
                continue

            print(f"[LEV2] Target {p.underlying}: {qty_u} shares @ ~{pu:.2f}; {p.etf}: short {qty_e} @ ~{pe:.2f}")
            execute_pair(
                ib,
                pair=p,
                qty_underlying=qty_u,
                qty_etf_short=qty_e,
                px_underlying=pu,
                px_etf=pe,
                pair_max_retries=3,
                leg_max_retries=3,
            )

# --------------------------------------------------
# IB POSITIONS ↔ STATE SYNC
# --------------------------------------------------

def fetch_ib_positions(ib: IB, save_snapshots: bool = True) -> pd.DataFrame:
    """
    Pull current positions from IBKR and return as a DataFrame.

    Columns:
      account, conId, symbol, localSymbol, secType, currency,
      exchange, primaryExchange, position, avgCost
    """
    positions = ib.positions()
    rows = []

    for p in positions:
        c = p.contract
        rows.append({
            "account": p.account,
            "conId": c.conId,
            "symbol": str(c.symbol),
            "localSymbol": str(c.localSymbol),
            "secType": str(c.secType),
            "currency": str(c.currency),
            "exchange": str(c.exchange),
            "primaryExchange": str(getattr(c, "primaryExchange", "")),
            "position": float(p.position),
            "avgCost": float(p.avgCost),
        })

    df = pd.DataFrame(rows)

    if save_snapshots:
        out_dir = REPO_ROOT / "data"
        out_dir.mkdir(parents=True, exist_ok=True)

        latest_path = out_dir / "ib_positions_latest.csv"
        df.to_csv(latest_path, index=False)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        snap_path = out_dir / f"ib_positions_{ts}.csv"
        df.to_csv(snap_path, index=False)

        print(f"[IB_POS] Saved latest positions to {latest_path}")
        print(f"[IB_POS] Saved snapshot to {snap_path}")

    print(f"[IB_POS] Currently {len(df)} open positions in IB.")
    return df


from datetime import datetime
from typing import List, Dict
from ib_insync import IB

# ... make sure REVERSE_IB_SYMBOL_MAP and save_state(...) are already defined ...


def sync_state_from_ib(
    ib: IB,
    pairs: List[Pair],
) -> pd.DataFrame:
    """
    Pull positions directly from IB (ib.positions()), convert them into our
    positions_state-style DataFrame, and save to positions_state.csv.

    Columns in returned DataFrame:
      symbol       - universal symbol (e.g. BRK-B)
      underlying   - matched underlying symbol if in our pair universe, else ""
      etf          - matched ETF symbol if in our pair universe, else ""
      lev_type     - 'CC', 'LEV2', 'MULTI' or ''
      sub          - 'CC', 'LEV2', 'UNDERLYING' or 'RAW'
      side         - 'LONG' or 'SHORT'
      shares       - absolute share count
      opened_at    - timestamp when we synced (for now we just stamp "now")
    """

    positions = ib.positions()
    if not positions:
        print("[IB_POS] No open positions in IB.")
        state_df = pd.DataFrame(
            columns=[
                "symbol",
                "underlying",
                "etf",
                "lev_type",
                "sub",
                "side",
                "shares",
                "opened_at",
            ]
        )
        save_state(state_df)
        print("[SYNC] No non-zero positions in IB; state will be empty.")
        return state_df

    # Build a simple DataFrame from ib.positions()
    rows_pos = []
    for p in positions:
        c = p.contract
        rows_pos.append(
            {
                "account": p.account,
                "symbol": c.symbol,
                "localSymbol": c.localSymbol,
                "conId": c.conId,
                "exchange": c.exchange or "",
                "primaryExchange": getattr(c, "primaryExchange", "") or "",
                "position": float(p.position),
                "avgCost": float(p.avgCost),
            }
        )

    ib_positions_df = pd.DataFrame(rows_pos)
    ib_positions_df.to_csv(
        REPO_ROOT / "data" / "ib_positions_latest.csv", index=False
    )
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ib_positions_df.to_csv(
        REPO_ROOT / "data" / f"ib_positions_{ts}.csv", index=False
    )
    print(
        f"[IB_POS] Saved latest positions to data/ib_positions_latest.csv "
        f"({len(ib_positions_df)} rows)."
    )

    # ---------- map these into our internal state format ----------

    # Build lookup maps from pairs
    etf_map: Dict[str, Pair] = {}
    underlying_map: Dict[str, List[Pair]] = {}

    for p in pairs:
        e = p.etf.upper()
        u = p.underlying.upper()
        etf_map[e] = p
        underlying_map.setdefault(u, []).append(p)

    rows_state = []
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for _, row in ib_positions_df.iterrows():
        raw_sym = str(row["symbol"]).strip()
        pos = float(row["position"])

        if pos == 0:
            continue  # ignore flat lines

        # Convert IB symbol -> universal symbol (e.g. "BRK B" -> "BRK-B")
        uni_sym = REVERSE_IB_SYMBOL_MAP.get(raw_sym, raw_sym).upper()

        side = "LONG" if pos > 0 else "SHORT"
        shares = int(abs(pos))

        underlying = ""
        etf = ""
        lev_type = ""
        sub = "RAW"  # default label for unmatched stuff

        # First, see if it's one of our ETFs
        if uni_sym in etf_map:
            pair = etf_map[uni_sym]
            underlying = pair.underlying.upper()
            etf = pair.etf.upper()
            lev_type = pair.lev_type.upper()
            sub = pair.lev_type.upper()  # "CC" or "LEV2"

        # Otherwise, see if it's an underlying used in pairs
        elif uni_sym in underlying_map:
            underlying = uni_sym
            attached_pairs = underlying_map[uni_sym]
            lev_types = sorted({p.lev_type.upper() for p in attached_pairs})
            if len(lev_types) == 1:
                lev_type = lev_types[0]
            else:
                lev_type = "MULTI"
            sub = "UNDERLYING"

        rows_state.append(
            {
                "symbol": uni_sym,
                "underlying": underlying,
                "etf": etf,
                "lev_type": lev_type,
                "sub": sub,
                "side": side,
                "shares": shares,
                "opened_at": now_str,
            }
        )

    state_df = pd.DataFrame(rows_state)
    if state_df.empty:
        print("[SYNC] No non-zero positions in IB; state will be empty.")
        state_df = pd.DataFrame(
            columns=[
                "symbol",
                "underlying",
                "etf",
                "lev_type",
                "sub",
                "side",
                "shares",
                "opened_at",
            ]
        )

    save_state(state_df)
    print(f"[SYNC] Synced {len(state_df)} positions from IB into positions_state.csv")
    return state_df

# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    print(f"Repo root: {REPO_ROOT}")
    print(f"Screened CSV: {SCREENED_CSV}")

    ib = connect_ib()
    print(f"Connected: {ib.isConnected()}")

    # 1) Load screened universe from setup.py output
    screened_df = load_screened_universe()

    # 2) Build pairs from that universe
    pairs = build_pairs(screened_df)

    # 3) Sync state from IB positions into our positions_state.csv
    state_df = sync_state_from_ib(ib, pairs)

    # 4) If there are no positions at all, initialize portfolio; otherwise you would
    #    run your rebalance / maintenance logic here.
    if state_df.empty and not ib.positions():
        print("No existing IB positions and no state; initializing portfolio fresh.")
        initialize_portfolio(ib, pairs)
        # After sending initial orders, re-sync state from IB
        state_df = sync_state_from_ib(ib, pairs)
    else:
        print(
            "Existing positions/state found; "
            "here you would run rebalance / maintenance logic."
        )
        # TODO: add periodic rebalancing + close-on-borrow-spike

    ib.disconnect()


if __name__ == "__main__":
    main()


#Initially we should definitely have a trade by trade acceptance by the user