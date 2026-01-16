#!/usr/bin/env python3
"""
pull_ibkr_flex.py

Download multiple IBKR Flex Queries via Flex Web Service (v3).

How it works:
1) SendRequest -> returns a ReferenceCode
2) GetStatement -> poll until report is ready, then save content (CSV/XML/TXT)

Required env vars:
  IBKR_FLEX_TOKEN=...               # your Flex Web Service token
  IBKR_FLEX_Q_TRADES=123456         # query id for Trades+Commissions
  IBKR_FLEX_Q_CASH=234567           # query id for Cash Transactions (dividends, fees, borrow)
  IBKR_FLEX_Q_POSITIONS=345678      # query id for Positions / NAV / open positions snapshot

Optional:
  IBKR_FLEX_BASE_URL=https://ndcdyn.interactivebrokers.com/AccountManagement/FlexWebService
  RUN_DATE=YYYY-MM-DD
"""

from __future__ import annotations

import os
import sys
import time
import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional, Dict

import requests


DEFAULT_BASE_URL = "https://ndcdyn.interactivebrokers.com/AccountManagement/FlexWebService"


def today_str() -> str:
    return date.today().isoformat()


@dataclass
class FlexQuery:
    name: str
    query_id: str


class FlexError(RuntimeError):
    pass


def _env_required(key: str) -> str:
    v = os.getenv(key)
    if not v:
        raise FlexError(f"Missing required environment variable: {key}")
    return v.strip()


def send_request(base_url: str, token: str, query_id: str, version: int = 3) -> str:
    """
    Returns ReferenceCode string if Success, else raises.
    """
    url = f"{base_url}/SendRequest"
    params = {"t": token, "q": query_id, "v": str(version)}

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    text = r.text

    # Response is XML-ish; keep parsing simple/robust without extra deps
    if "<Status>Success</Status>" not in text:
        # Try to extract error message
        err_code = _extract_tag(text, "ErrorCode") or "UNKNOWN"
        err_msg = _extract_tag(text, "ErrorMessage") or text[:400]
        raise FlexError(f"SendRequest failed for q={query_id}: {err_code} {err_msg}")

    ref = _extract_tag(text, "ReferenceCode")
    if not ref:
        raise FlexError(f"SendRequest succeeded but no ReferenceCode found for q={query_id}")
    return ref

import random
import time

def send_request_with_backoff(base_url: str, token: str, query_id: str, version: int = 3,
                              max_attempts: int = 8, base_sleep: float = 5.0) -> str:
    """
    SendRequest with exponential backoff for IBKR rate limiting (ErrorCode 1018).
    """
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            return send_request(base_url, token, query_id, version=version)
        except FlexError as e:
            msg = str(e)
            last_err = e

            # Detect 1018 throttle
            if " 1018 " in msg or "ErrorCode>1018" in msg or "Too many requests" in msg:
                sleep = base_sleep * (2 ** (attempt - 1))
                sleep = min(sleep, 120.0)  # cap at 2 minutes
                sleep += random.uniform(0, 2.0)  # jitter
                print(f"[FLEX] Throttled (1018). Backing off {sleep:.1f}s (attempt {attempt}/{max_attempts})")
                time.sleep(sleep)
                continue

            # Other errors: raise immediately
            raise

    raise FlexError(f"SendRequest failed after {max_attempts} attempts for q={query_id}: {last_err}")


def get_statement(
    base_url: str,
    token: str,
    reference_code: str,
    version: int = 3,
) -> str:
    """
    Returns statement content as text if ready.
    If not ready, IBKR returns an XML response with 'Fail' and an error that indicates processing.
    """
    url = f"{base_url}/GetStatement"
    params = {"t": token, "q": reference_code, "v": str(version)}

    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.text


def _extract_tag(xml_text: str, tag: str) -> Optional[str]:
    """
    Minimal tag extractor for <Tag>value</Tag>.
    """
    open_t = f"<{tag}>"
    close_t = f"</{tag}>"
    a = xml_text.find(open_t)
    b = xml_text.find(close_t)
    if a == -1 or b == -1 or b <= a:
        return None
    return xml_text[a + len(open_t) : b].strip()


def _looks_like_processing_response(body: str) -> bool:
    """
    When not ready, IBKR often returns an XML response with Status=Fail and a message like
    "Statement is not ready" / "processing".
    """
    if "<FlexStatementResponse>" in body and "<Status>Fail</Status>" in body:
        msg = (_extract_tag(body, "ErrorMessage") or "").lower()
        return ("not ready" in msg) or ("processing" in msg) or ("please try again" in msg)
    return False


def _detect_extension(body: str) -> str:
    """
    Best-effort. If user configured Flex output as CSV/TXT/XML, body will reflect it.
    """
    b = body.lstrip()
    if b.startswith("<?xml") or b.startswith("<FlexQueryResponse") or b.startswith("<FlexStatementResponse"):
        return "xml"
    # CSV/TXT are usually plain text with commas or pipes
    if "," in b.splitlines()[0]:
        return "csv"
    if "|" in b.splitlines()[0]:
        return "txt"
    return "txt"


def fetch_and_save(
    *,
    base_url: str,
    token: str,
    q: FlexQuery,
    out_dir: Path,
    poll_every_sec: float = 5.0,
    max_wait_sec: float = 180.0,
) -> Path:
    """
    SendRequest -> poll GetStatement -> save to disk.
    """
    ref = send_request_with_backoff(base_url, token, q.query_id)
    t0 = time.time()

    last_body = None
    while True:
        body = get_statement(base_url, token, ref)
        last_body = body

        if _looks_like_processing_response(body):
            if time.time() - t0 > max_wait_sec:
                msg = _extract_tag(body, "ErrorMessage") or "Timed out waiting for report"
                raise FlexError(f"Timed out fetching {q.name} (q={q.query_id}, ref={ref}): {msg}")
            time.sleep(poll_every_sec)
            continue

        # If it isn't a "processing" XML response, treat it as report content
        ext = _detect_extension(body)
        out_path = out_dir / f"{q.name}.{ext}"
        out_path.write_text(body, encoding="utf-8", errors="ignore")
        return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-date", default=os.getenv("RUN_DATE") or today_str(), help="YYYY-MM-DD")
    ap.add_argument("--out-root", default="data/runs", help="output root folder")
    ap.add_argument("--poll", type=float, default=float(os.getenv("IBKR_FLEX_POLL_SEC", "5")), help="poll interval seconds")
    ap.add_argument("--timeout", type=float, default=float(os.getenv("IBKR_FLEX_TIMEOUT_SEC", "180")), help="max wait per report seconds")
    args = ap.parse_args()

    #token = _env_required("IBKR_FLEX_TOKEN")
    token ="195413602443563105417466"
    base_url = os.getenv("195413602443563105417466", DEFAULT_BASE_URL).strip()

    queries = [
        FlexQuery("flex_trades", "1374436"),#_env_required("IBKR_FLEX_Q_TRADES")),
        FlexQuery("flex_cash", "1374451"),#_env_required("IBKR_FLEX_Q_CASH")),
        FlexQuery("flex_positions", "1374454")#_env_required("IBKR_FLEX_Q_POSITIONS")),
    ]

    out_dir = Path(args.out_root) / args.run_date / "ibkr_flex"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[FLEX] base_url={base_url}")
    print(f"[FLEX] writing to {out_dir}")

    for q in queries:
        print(f"[FLEX] fetching {q.name} (query_id={q.query_id}) ...")
        p = fetch_and_save(
            base_url=base_url,
            token=token,
            q=q,
            out_dir=out_dir,
            poll_every_sec=args.poll,
            max_wait_sec=args.timeout,
        )
        print(f"[FLEX] wrote {p}")
        time.sleep(3)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FlexError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise SystemExit(2)