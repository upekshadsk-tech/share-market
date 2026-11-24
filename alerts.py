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
  TIGEROPEN_* optional Tiger keys (read-only US prices), see tiger_data.py
  LOG_PATH     where to save the signal log CSV (the workflow saves it to the signal-log branch)
"""

import csv
import os
import sys
import time
from pathlib import Path

import exchange_calendars as xc
import pandas as pd
import yfinance as yf
import logging
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

from notify import alert_text, send_push
from scanner import SCAN_MARKETS, is_open, market_candidates
from signals import (EVAL_HORIZONS, combine_bars, evaluate_signal, fetch_history_1m, fetch_today_1m,
                     fix_day_change, live_signal, official_prev_close)

TOPIC = os.environ.get("NTFY_TOPIC", "").strip()
APP_URL = os.environ.get("APP_URL", "").strip() or None
HORIZON = int(os.environ.get("HORIZON") or 5)
MIN_HIT = float(os.environ.get("MIN_HIT") or 55)
RUN_MINUTES = float(os.environ.get("RUN_MINUTES") or 58)
CHECK_EVERY = 60  # seconds
SCAN_ALL = (os.environ.get("SCAN_ALL") or "yes").lower() not in ("no", "false", "0")
SCAN_EVERY = 5 * 60   # whole-market scan every 5 minutes
SCAN_TOP = int(os.environ.get("SCAN_TOP") or 15)
COOLDOWN_MIN = 90     # don't repeat the same suggestion within 90 minutes
LOG_PATH = Path(os.environ.get("LOG_PATH") or "signal_log.csv")
LOG_FIELDS = ["Time (UTC)", "Symbol", "Action", "Price", "Stop-loss", "Target", "Score", "Right %",
              "Tests", "Horizon", "Source"] + [f"Price +{h}m" for h in EVAL_HORIZONS] + \
             [f"+{h}m %" for h in EVAL_HORIZONS] + ["Best %", "Worst %", "Result", "Minutes"]


def log_signal(r: dict, source: str = "github"):
    new = not LOG_PATH.exists()
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({"Time (UTC)": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S"),
                    "Symbol": r["Symbol"], "Action": r["Action"], "Price": round(r["Price"], 4),
                    "Stop-loss": r["Stop-loss"] and round(r["Stop-loss"], 4),
                    "Target": r["Target"] and round(r["Target"], 4), "Score": r["Score"],
                    "Right %": r["Right %"] and round(r["Right %"], 1), "Tests": r["Tests"],
                    "Horizon": HORIZON, "Source": source})


_HIST: dict[str, tuple[float, pd.DataFrame]] = {}


def get_bars(symbol: str) -> pd.DataFrame:
    """~2 weeks of 1-minute bars: history cached for 30 min, today's bars fresh every call."""
    t, hist = _HIST.get(symbol, (0.0, None))
    if hist is None or time.time() - t > 1800:
        hist = fetch_history_1m(symbol)
        _HIST[symbol] = (time.time(), hist)
    return combine_bars(hist, fetch_today_1m(symbol))


def recently_alerted(symbol: str, action: str) -> bool:
    if not LOG_PATH.exists():
        return False
    try:
        df = pd.read_csv(LOG_PATH, usecols=["Time (UTC)", "Symbol", "Action"])
    except Exception:
        return False
    t = pd.to_datetime(df["Time (UTC)"], utc=True)
    recent = df[(t > pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=COOLDOWN_MIN))]
    return bool(((recent["Symbol"] == symbol) & (recent["Action"] == action)).any())


def scan_whole_markets():
    """Stage 1: screen every share in each open market. Stage 2: full signal check on the best."""
    for market in SCAN_MARKETS:
        if not is_open(market):
            continue
        try:
            cands = market_candidates(market, "buy", SCAN_TOP)
        except Exception as e:
            print(f"{market} scan error: {e}")
            continue
        print(f"{market}: {len(cands)} candidates")
        picks = []
        for _, c in cands.iterrows():
            try:
                bars = get_bars(c['Symbol'])
                if len(bars) < 120:
                    continue
                r = fix_day_change(live_signal(c['Symbol'], bars, None, HORIZON, MIN_HIT),
                                   official_prev_close(c['Symbol']))
            except Exception as e:
                print(f"{c['Symbol']}: error {e}")
                continue
            if r["Action"] == "BUY NOW" and not recently_alerted(c['Symbol'], "BUY NOW"):
                r["_why"] = (f"{c['Company'] or ''} · {market}\nVolume {c['Volume ×']:.1f}× normal for this time of day")
                picks.append(r)
        picks.sort(key=lambda r: (-(r["Right %"] or 0), -r["Score"]))
        for r in picks[:3]:   # at most 3 suggestions per market per scan
            title, body = alert_text(r)
            title = "🚀 Scan pick: " + title.split(" ", 1)[1]
            body = r["_why"] + "\n" + body
            if r["Symbol"].endswith((".AX", ".NZ")):
                body += "\n(ASX/NZX prices ~20 min delayed)"
            ok = send_push(TOPIC, title, body, "BUY NOW", APP_URL)
            print(f"SCAN ALERT {title} sent={ok}")
            log_signal(r, "scan")


def fill_results():
    """Work out what happened after earlier signals (once at least an hour has passed)."""
    if not LOG_PATH.exists():
        return
    df = pd.read_csv(LOG_PATH)
    for col in LOG_FIELDS:
        if col not in df.columns:
            df[col] = None
    df["Result"] = df["Result"].astype(object)
    todo = df[(df["Result"].isna() | df["Result"].astype(str).str.startswith("⏳"))]
    now = pd.Timestamp.now(tz="UTC")
    for sym, grp in todo.groupby("Symbol"):
        try:
            bars = fetch_history_1m(sym)
        except Exception:
            continue
        for i, row in grp.iterrows():
            t = pd.Timestamp(row["Time (UTC)"], tz="UTC")
            if now - t < pd.Timedelta(minutes=65):
                continue
            ev = evaluate_signal(bars, t, row["Action"], row["Price"], row["Stop-loss"], row["Target"])
            for k, v in ev.items():
                df.at[i, k] = round(v, 4) if isinstance(v, float) else v
    df[LOG_FIELDS].to_csv(LOG_PATH, index=False)


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
    try:
        fill_results()
    except Exception as e:
        print(f"Could not update past results: {e}")
    if not any(market_open(s) for s in symbols) and not (SCAN_ALL and any(is_open(m) for m in SCAN_MARKETS)):
        print("No market open right now. Exiting.")
        return
    last_action: dict[str, str] = {}
    end = time.time() + RUN_MINUTES * 60
    next_scan = 0.0
    while time.time() < end:
        if SCAN_ALL and time.time() >= next_scan:
            scan_whole_markets()
            next_scan = time.time() + SCAN_EVERY
        for s in symbols:
            if not market_open(s):
                continue
            try:
                bars = get_bars(s)
                if len(bars) < 30:
                    continue
                r = fix_day_change(live_signal(s, bars, None, HORIZON, MIN_HIT), official_prev_close(s))
            except Exception as e:
                print(f"{s}: error {e}")
                continue
            act = r["Action"]
            # Alert only when a share CHANGES into BUY NOW or SELL NOW (no repeats every minute)
            if act in ("BUY NOW", "SELL NOW") and last_action.get(s) != act:
                title, body = alert_text(r)
                ok = send_push(TOPIC, title, body, act, APP_URL)
                print(f"ALERT {title} sent={ok}")
                log_signal(r)
            last_action[s] = act
        time.sleep(CHECK_EVERY)
    try:
        fill_results()
    except Exception as e:
        print(f"Could not update past results: {e}")


if __name__ == "__main__":
    main()
