#!/usr/bin/env python3
import ftplib
import io
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import smtplib
from email.mime.text import MIMEText

FTP_HOST = "ftp2.interactivebrokers.com"
FTP_USER = "shortstock"
FTP_PASS = ""
FTP_FILE = os.getenv("IBKR_FTP_FILE", "usa.txt")


# Base workspace (GitHub Actions sets this, but fallback to ".")
WORKSPACE = Path(os.getenv("GITHUB_WORKSPACE", ".")).resolve()

# Allow override via env var, default to "state/shortstock_state.csv"
STATE_PATH = WORKSPACE / os.getenv("STATE_PATH", "state/shortstock_state.csv")

TICKER_FILE = os.getenv("TICKER_FILE", "/config/tickers.csv")

def load_ticker_list(path: str) -> List[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Ticker list file not found: {path}")

    if path.suffix == ".csv":
        df = pd.read_csv(path)
        return df[df.columns[0]].dropna().astype(str).str.upper().tolist()

    if path.suffix == ".json":
        with open(path, "r") as f:
            lst = json.load(f)
        return [str(x).upper() for x in lst]

    raise ValueError("Ticker file must be .csv or .json")


WATCH_TICKERS = load_ticker_list(TICKER_FILE)
print("Loaded tickers:", WATCH_TICKERS)

# Thresholds (can tune via env vars)
BORROW_ABS_THRESHOLD = float(os.getenv("BORROW_ABS_THRESHOLD", "0.30"))      # 30%+
BORROW_CHANGE_THRESHOLD = float(os.getenv("BORROW_CHANGE_THRESHOLD", "0.05")) # +/−5% change
AVAIL_ABS_THRESHOLD = int(os.getenv("AVAIL_ABS_THRESHOLD", "10000"))          # < 10k shares
AVAIL_CHANGE_THRESHOLD = int(os.getenv("AVAIL_CHANGE_THRESHOLD", "50000"))    # big change

# Email config (use env vars / k8s Secret)
EMAIL_FROM = os.getenv("EMAIL_FROM", "dag5wd@virginia.edu")
EMAIL_TO = os.getenv("EMAIL_TO", "dag5wd@virginia.edu")
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")

# ------------- IBKR FTP FETCH / PARSE ----------------

def fetch_ibkr_shortstock_file(filename: str = FTP_FILE) -> pd.DataFrame:
    ftp = ftplib.FTP(FTP_HOST)
    ftp.login(user=FTP_USER, passwd=FTP_PASS)

    buf = io.BytesIO()
    ftp.retrbinary(f"RETR {filename}", buf.write)
    ftp.quit()

    buf.seek(0)
    text = buf.getvalue().decode("utf-8", errors="ignore")
    lines = [ln for ln in text.splitlines() if ln.strip()]

    # Find header line
    header_idx = None
    for i, ln in enumerate(lines):
        if ln.startswith("#SYM|"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not find header line starting with '#SYM|'")

    header_line = lines[header_idx]
    data_lines = lines[header_idx + 1:]

    header_cols = [c.strip().lstrip("#").lower() for c in header_line.split("|")]

    data_str = "\n".join(data_lines)
    data_buf = io.StringIO(data_str)

    df = pd.read_csv(data_buf, sep="|", header=None, engine="python")
    n_cols = min(len(header_cols), df.shape[1])
    df = df.iloc[:, :n_cols]
    df.columns = header_cols[:n_cols]

    df = df.drop(
        columns=[c for c in df.columns if not c or str(c).startswith("unnamed")],
        errors="ignore",
    )

    return df


def build_short_maps(tickers: List[str], df: pd.DataFrame):
    tickers = [t.upper() for t in tickers]

    df["sym"] = df["sym"].astype(str).str.upper().str.strip()
    df["rebate_annual"] = pd.to_numeric(df["rebaterate"], errors="coerce") / 100.0
    df["fee_annual"] = pd.to_numeric(df["feerate"], errors="coerce") / 100.0
    df["available_int"] = pd.to_numeric(df["available"], errors="coerce")

    df["net_borrow_annual"] = df["fee_annual"] - df["rebate_annual"]
    df["net_borrow_annual"] = df["net_borrow_annual"].clip(lower=0)

    sub = df[df["sym"].isin(tickers)].copy()

    out = {}
    for _, row in sub.iterrows():
        sym = row["sym"]
        out[sym] = {
            "borrow": float(row["net_borrow_annual"]) if pd.notna(row["net_borrow_annual"]) else None,
            "rebate": float(row["rebate_annual"]) if pd.notna(row["rebate_annual"]) else None,
            "available": int(row["available_int"]) if pd.notna(row["available_int"]) else None,
        }
    return out


# ------------- STATE LOAD / SAVE ----------------

def load_previous_state() -> Dict:
    if not STATE_PATH.exists():
        return {}
    with STATE_PATH.open("r") as f:
        return json.load(f)


def save_state(current):
    """
    Save the latest borrow/availability snapshot to disk.

    Accepts either:
      - a pandas DataFrame, or
      - a dict of the form { "TICKER": { ...metrics... }, ... }
    """
    # Convert dict → DataFrame if needed
    if isinstance(current, dict):
        # Expect something like: {"ABNY": {"borrow": 0.446, "available": 6000}, ...}
        rows = []
        for symbol, metrics in current.items():
            row = {"symbol": symbol}
            if isinstance(metrics, dict):
                row.update(metrics)
            else:
                # If metrics is just a single value, store it under "value"
                row["value"] = metrics
            rows.append(row)
        df = pd.DataFrame(rows)
    else:
        # Assume it's already a DataFrame-like object
        df = current

    # Make sure directory exists
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving state to: {STATE_PATH}")
    df.to_csv(STATE_PATH, index=False)


# ------------- ALERT LOGIC ----------------

def generate_alerts(current: Dict, previous: Dict) -> str:
    lines = []

    for sym, cur in current.items():
        borrow = cur.get("borrow")
        avail = cur.get("available")

        prev = previous.get(sym, {})
        prev_borrow = prev.get("borrow")
        prev_avail = prev.get("available")

        msgs = []

        # Absolute levels
        if borrow is not None and borrow > BORROW_ABS_THRESHOLD:
            msgs.append(f"borrow {borrow:.1%} > {BORROW_ABS_THRESHOLD:.1%}")
        if avail is not None and avail < AVAIL_ABS_THRESHOLD:
            msgs.append(f"available {avail:,} < {AVAIL_ABS_THRESHOLD:,}")

        # Changes vs yesterday
        if prev_borrow is not None and borrow is not None:
            delta_b = borrow - prev_borrow
            if abs(delta_b) > BORROW_CHANGE_THRESHOLD:
                msgs.append(
                    f"borrow change {delta_b:+.1%} (prev {prev_borrow:.1%} → now {borrow:.1%})"
                )

        if prev_avail is not None and avail is not None:
            delta_a = avail - prev_avail
            if abs(delta_a) > AVAIL_CHANGE_THRESHOLD:
                msgs.append(
                    f"available change {delta_a:+,} (prev {prev_avail:,} → now {avail:,})"
                )

        if msgs:
            lines.append(f"{sym}: " + "; ".join(msgs))

    if not lines:
        return ""  # no alerts

    header = "IBKR Short Stock Monitor – Alerts\n\n"
    return header + "\n".join(lines)


# ------------- EMAIL SENDER ----------------

def send_email(subject: str, body: str):
    if not SMTP_USER or not SMTP_PASS:
        print("No SMTP creds set; skipping email. Message would have been:\n", body)
        return

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(EMAIL_FROM, [EMAIL_TO], msg.as_string())


# ------------- MAIN ----------------

def main():
    short_df = fetch_ibkr_shortstock_file()
    current = build_short_maps(WATCH_TICKERS, short_df)
    previous = load_previous_state()

    alert_text = generate_alerts(current, previous)

    if alert_text:
        send_email("IBKR Short Borrow / Availability Alert", alert_text)
        print(alert_text)
    else:
        print("No alerts today.")

    save_state(current)


if __name__ == "__main__":
    main()
