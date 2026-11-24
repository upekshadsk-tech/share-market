"""
Day-trade tracker: record what you bought, then get told when to sell (same day).

Trades are saved as small messages on a private ntfy topic: "<your topic>-trades".
That lets the app (on any device) and the background checker (GitHub) see the same open
trades with no extra account. ntfy keeps messages for about 12 hours, which suits day trading.

Sell rules for a bought share (checked every minute):
  🛑 Stop-loss hit                  price <= stop (loss limit)
  🔒 Trailing stop hit              once up by the original risk, the stop follows the price up
  🎯 Target reached                 price >= target
  📉 Momentum turned down           signal score <= -2 while in profit
  ⏰ Market closing                 15 minutes before the close (day trading = no overnight holds)
Technical guide only, not financial advice.
"""

import json
import time
import uuid

import pandas as pd
import requests

NTFY = "https://ntfy.sh"
CLOSE_WARN_MIN = 15
TRAIL_ATR = 1.5


def trades_topic(topic: str) -> str:
    return f"{topic.strip()}-trades"


def _post(topic: str, event: dict) -> bool:
    try:
        r = requests.post(f"{NTFY}/{trades_topic(topic)}", data=json.dumps(event).encode(),
                          headers={"Priority": "min", "Tags": "floppy_disk"}, timeout=10)
        return r.ok
    except Exception:
        return False


def load_events(topic: str, since: str = "24h") -> list[dict]:
    """All trade events (open / close / alerted) from the last `since`, oldest first."""
    if not topic:
        return []
    try:
        r = requests.get(f"{NTFY}/{trades_topic(topic)}/json", params={"poll": "1", "since": since}, timeout=15)
        out = []
        for line in r.text.splitlines():
            try:
                m = json.loads(line)
                if m.get("event") == "message":
                    e = json.loads(m.get("message", "{}"))
                    if isinstance(e, dict) and e.get("type"):
                        out.append(e)
            except Exception:
                continue
        return out
    except Exception:
        return []


def open_trade(topic: str, symbol: str, entry: float, shares: float, stop: float, target: float,
               market: str, note: str = "") -> dict:
    t = {"type": "open", "id": uuid.uuid4().hex[:8], "symbol": symbol.upper(), "entry": float(entry),
         "shares": float(shares or 0), "stop": float(stop), "target": float(target), "market": market,
         "time": pd.Timestamp.now(tz="UTC").isoformat(), "note": note}
    t["ok"] = _post(topic, t)
    return t


def close_trade(topic: str, trade_id: str, price: float) -> bool:
    return _post(topic, {"type": "close", "id": trade_id, "price": float(price),
                         "time": pd.Timestamp.now(tz="UTC").isoformat()})


def mark_alerted(topic: str, trade_id: str, reason: str) -> bool:
    return _post(topic, {"type": "alerted", "id": trade_id, "reason": reason,
                         "time": pd.Timestamp.now(tz="UTC").isoformat()})


def record_signal_alert(topic: str, symbol: str, action: str) -> bool:
    return _post(topic, {"type": "signal", "symbol": symbol.upper(), "action": action,
                         "time": pd.Timestamp.now(tz="UTC").isoformat()})


def signal_alerted_recently(events: list[dict], symbol: str, action: str, minutes: int = 90) -> bool:
    """True if this BUY/SELL alert was already sent (by the app on any device, or the checker) recently."""
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=minutes)
    for e in events:
        if e.get("type") == "signal" and e.get("symbol") == symbol.upper() and e.get("action") == action:
            try:
                if pd.Timestamp(e["time"]) >= cutoff:
                    return True
            except Exception:
                pass
    return False


def build(events: list[dict]) -> tuple[list[dict], list[dict], dict]:
    """(open trades, closed trades, {trade id: set of reasons already alerted})."""
    opens, closes, alerted = {}, {}, {}
    for e in events:
        if e["type"] == "open":
            opens[e["id"]] = e
        elif e["type"] == "close":
            closes[e["id"]] = e
        elif e["type"] == "alerted":
            alerted.setdefault(e["id"], set()).add(e.get("reason"))
    still_open = [t for i, t in opens.items() if i not in closes]
    closed = []
    for i, c in closes.items():
        if i in opens:
            t = dict(opens[i])
            t["exit"], t["exit_time"] = c["price"], c["time"]
            t["pl_pct"] = (c["price"] / t["entry"] - 1) * 100
            t["pl"] = (c["price"] - t["entry"]) * t.get("shares", 0)
            closed.append(t)
    return still_open, closed, alerted


def default_levels(price: float, bars: pd.DataFrame | None, stop_atr: float = 1.5, reward: float = 2.0):
    """Stop-loss and target for a new trade from the share's normal 5-minute move (ATR)."""
    from signals import atr_5m
    atr = atr_5m(bars) if bars is not None and not bars.empty else None
    risk = stop_atr * atr if atr else price * 0.01
    return price - risk, price + reward * risk


def advice(trade: dict, bars: pd.DataFrame | None, price: float, minutes_to_close: float | None,
           score: int | None = None) -> dict:
    """What to do with an open trade right now: HOLD or SELL (with the reason)."""
    from signals import atr_5m
    entry, stop0, target = trade["entry"], trade["stop"], trade["target"]
    risk = max(entry - stop0, 1e-9)
    t0 = pd.Timestamp(trade["time"])
    high = price
    if bars is not None and not bars.empty:
        after = bars[bars.index >= t0.tz_convert(bars.index.tz)]
        if not after.empty:
            high = max(float(after["High"].max()), price)
    atr = atr_5m(bars) if bars is not None and not bars.empty else None
    stop = stop0
    if high >= entry + risk:                                   # up by 1x the risk: protect the trade
        trail = high - TRAIL_ATR * atr if atr else high - risk
        stop = max(stop0, entry, trail)
    pl_pct = (price / entry - 1) * 100
    res = {"price": price, "pl_pct": pl_pct, "pl": (price - entry) * trade.get("shares", 0),
           "stop_now": stop, "high": high, "action": "HOLD", "reason": None, "text": ""}
    if price <= stop:
        if stop > stop0:
            res.update(action="SELL", reason="trail", text=f"🔒 Trailing stop hit at {stop:,.3f}: sell to lock in profit")
        else:
            res.update(action="SELL", reason="stop", text=f"🛑 Stop-loss hit at {stop:,.3f}: sell now to limit the loss")
    elif price >= target:
        res.update(action="SELL", reason="target", text=f"🎯 Target {target:,.3f} reached: take your profit")
    elif minutes_to_close is not None and minutes_to_close <= CLOSE_WARN_MIN:
        res.update(action="SELL", reason="close",
                   text=f"⏰ Market closes in {max(minutes_to_close, 0):.0f} min: sell to finish the day")
    elif score is not None and score <= -2 and price > entry:
        res.update(action="SELL", reason="momentum", text="📉 Momentum turned down while you're in profit: consider selling")
    else:
        res["text"] = f"Hold · stop {stop:,.3f}" + (" (trailing)" if stop > stop0 else "") + f" · target {target:,.3f}"
    return res


def should_alert(already: set | None, reason: str | None) -> bool:
    """One SELL alert per trade, plus one 'market closing' reminder if it's still open near the close."""
    already = already or set()
    if not reason or reason in already:
        return False
    return not already or reason == "close"


def minutes_to_close(market_cal_code: str) -> float | None:
    try:
        import exchange_calendars as xc
        cal = xc.get_calendar(market_cal_code)
        now = pd.Timestamp.now(tz="UTC").floor("min")
        if not cal.is_open_on_minute(now):
            return None
        return (cal.next_close(now) - now).total_seconds() / 60
    except Exception:
        return None
