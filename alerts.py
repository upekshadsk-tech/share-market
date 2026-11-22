"""
Background alert checker, run by GitHub Actions (see .github/workflows/alerts.yml).

Every minute while a market is open, it checks the shares listed in alert_symbols.txt and
sends a phone notification when one turns BUY NOW or SELL NOW. Runs up to ~58 minutes,
then the next hourly run takes over.

Settings come from environment variables (GitHub repo secrets / variables):
  NTFY_TOPIC   your private ntfy topic name (required)
  APP_URL      link to your Streamlit app (optional, tapping the alert opens it)
  HORIZON      signal look-ahead in minutes (default 5)
  MIN_HIT      minimum "signal right %" (default 55)
"""

import os
import sys
import time
from pathlib import Path

import exchange_calendars as xc
import pandas as pd
import yfinance as yf

from notify import alert_text, send_push
from signals import live_signal

TOPIC = os.environ.get("NTFY_TOPIC", "").strip()
APP_URL = os.environ.get("APP_URL", "").strip() or None
HORIZON = int(os.environ.get("HORIZON") or 5)
MIN_HIT = float(os.environ.get("MIN_HIT") or 55)
RUN_MINUTES = float(os.environ.get("RUN_MINUTES") or 58)
CHECK_EVERY = 60  # seconds

CAL = {".AX": "XASX", ".NZ": "XNZE"}


def calendar_for(symbol: str):
    for suffix, code in CAL.items():
        if symbol.endswith(suffix):
            return xc.get_calendar(code)
    return xc.get_calendar("XNYS")


def market_open(symbol: str) -> bool:
    return bool(calendar_for(symbol).is_open_on_minute(pd.Timestamp.now(tz="UTC").floor("min")))


def load_symbols() -> list[str]:
    f = Path(__file__).with_name("alert_symbols.txt")
    out = []
    for line in f.read_text().splitlines():
        s = line.split("#")[0].strip().upper()
        if s:
            out.append(s)
    return list(dict.fromkeys(out))


def main():
    if not TOPIC:
        sys.exit("NTFY_TOPIC is not set. Add it under Settings > Secrets and variables > Actions.")
    symbols = load_symbols()
    if not any(market_open(s) for s in symbols):
        print("No market open for these symbols right now. Exiting.")
        return
    last_action: dict[str, str] = {}
    end = time.time() + RUN_MINUTES * 60
    while time.time() < end:
        for s in symbols:
            if not market_open(s):
                continue
            try:
                bars = yf.Ticker(s).history(period="5d", interval="1m")[["Open", "High", "Low", "Close", "Volume"]].dropna()
                if len(bars) < 30:
                    continue
                r = live_signal(s, bars, None, HORIZON, MIN_HIT)
            except Exception as e:
                print(f"{s}: error {e}")
                continue
            act = r["Action"]
            # Alert only when a share CHANGES into BUY NOW or SELL NOW (no repeats every minute)
            if act in ("BUY NOW", "SELL NOW") and last_action.get(s) != act:
                title, body = alert_text(r)
                ok = send_push(TOPIC, title, body, act, APP_URL)
                print(f"ALERT {title} sent={ok}")
            last_action[s] = act
        time.sleep(CHECK_EVERY)


if __name__ == "__main__":
    main()
