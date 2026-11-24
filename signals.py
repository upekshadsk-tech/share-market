"""
Short-term momentum signals built from 1-minute price bars.

These are NOT guarantees. They summarise what the price is doing right now
(trend, momentum, position vs VWAP, volume surge) into a score, and they
measure how often that score has called the next few minutes correctly
over the last ~10 trading days (never fewer than 5), so you can judge whether to trust it
for each stock.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

MIN_HISTORY_DAYS = 5      # never give a signal unless at least this many past trading days were tested
HISTORY_CALENDAR_DAYS = 14  # how far back to load 1-minute data (about 10 trading days)
OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def fetch_history_1m(symbol: str, days: int = HISTORY_CALENDAR_DAYS) -> pd.DataFrame:
    """About two weeks of 1-minute bars. Yahoo allows max ~7 days per request, so fetch in chunks."""
    import yfinance as yf
    now = datetime.now(timezone.utc)
    frames, end = [], now + timedelta(minutes=1)
    while end > now - timedelta(days=days):
        start = max(end - timedelta(days=7), now - timedelta(days=days))
        try:
            h = yf.Ticker(symbol).history(start=start, end=end, interval="1m", prepost=False)
            if not h.empty:
                frames.append(h[OHLCV])
        except Exception:
            pass
        end = start
    if not frames:
        return pd.DataFrame(columns=OHLCV)
    df = pd.concat(frames)
    return df[~df.index.duplicated(keep="last")].sort_index().dropna()


def fetch_today_1m(symbol: str) -> pd.DataFrame:
    """Today's 1-minute bars. US shares use Tiger (if connected), otherwise Yahoo Finance."""
    try:
        import tiger_data
        if tiger_data.configured() and tiger_data.is_us(symbol):
            t = tiger_data.today_bars_1m(symbol)
            if not t.empty and (pd.Timestamp.now(tz="UTC") - t.index[-1]) < pd.Timedelta(hours=20):
                return t
    except Exception:
        pass
    import yfinance as yf
    h = yf.Ticker(symbol).history(period="1d", interval="1m", prepost=False)
    return h[OHLCV].dropna() if not h.empty else pd.DataFrame(columns=OHLCV)


def combine_bars(history: pd.DataFrame, today: pd.DataFrame) -> pd.DataFrame:
    """Old history (cached) + the freshest bars of today."""
    if today is None or today.empty:
        return history
    if history is None or history.empty:
        return today
    df = pd.concat([history[history.index < today.index[0]], today])
    return df[~df.index.duplicated(keep="last")].sort_index()


def history_days(bars: pd.DataFrame) -> int:
    """Number of complete past trading days in the data (today's session not counted)."""
    if bars is None or bars.empty:
        return 0
    dates = pd.Index(bars.index.date).unique()
    return int(sum(d < bars.index[-1].date() for d in dates))


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """df has Open/High/Low/Close/Volume on 1-minute bars (several days)."""
    d = df.copy()
    c = d["Close"]
    d["EMA9"] = c.ewm(span=9, adjust=False).mean()
    d["EMA21"] = c.ewm(span=21, adjust=False).mean()

    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    d["RSI"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    # VWAP restarts every trading day
    day = d.index.date
    typical = (d["High"] + d["Low"] + d["Close"]) / 3
    pv = (typical * d["Volume"]).groupby(day).cumsum()
    vol = d["Volume"].groupby(day).cumsum()
    d["VWAP"] = pv / vol.replace(0, np.nan)

    d["Mom5"] = c.pct_change(5) * 100
    vol5 = d["Volume"].rolling(5).sum()
    d["VolSurge"] = vol5 / vol5.shift(5).rolling(60, min_periods=10).mean()

    rng = (d["High"] - d["Low"]) / c * 100
    d["RangeExp"] = rng.rolling(5).mean() / rng.rolling(60, min_periods=10).mean()

    # Score from -4 (strong down bias) to +4 (strong up bias)
    s = np.sign(d["EMA9"] - d["EMA21"])
    s = s + np.sign(d["Mom5"].fillna(0))
    s = s + np.sign(c - d["VWAP"]).fillna(0)
    s = s + np.where(d["RSI"] < 30, 1, np.where(d["RSI"] > 70, -1, 0))  # stretched → snap-back
    d["Score"] = s
    return d


def hit_rate(d: pd.DataFrame, horizon: int = 5, min_score: int = 2):
    """How often |Score| >= min_score correctly called the direction `horizon` bars later."""
    future = d["Close"].shift(-horizon) - d["Close"]
    mask = (d["Score"].abs() >= min_score) & future.notna() & (future != 0)
    n = int(mask.sum())
    if n == 0:
        return None, 0
    correct = (np.sign(d.loc[mask, "Score"]) == np.sign(future[mask])).mean()
    return float(correct * 100), n


def projection(d: pd.DataFrame, horizon: int = 5, lookback: int = 15):
    """Straight-line trend of the last `lookback` minutes, extended `horizon` minutes,
    with a +/- band from how noisy those minutes were."""
    y = d["Close"].dropna().tail(lookback).to_numpy()
    if len(y) < 5:
        return None
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    target = slope * (len(y) - 1 + horizon) + intercept
    band = resid.std() * np.sqrt(horizon) * 1.5
    return {"target": float(target), "low": float(target - band), "high": float(target + band),
            "slope": float(slope)}


def summarise(symbol: str, df: pd.DataFrame, horizon: int = 5) -> dict:
    d = add_indicators(df)
    last = d.iloc[-1]
    score = int(last["Score"]) if pd.notna(last["Score"]) else 0
    hr, n = hit_rate(d, horizon)
    proj = projection(d, horizon)
    days = history_days(df)
    if days < MIN_HISTORY_DAYS:
        call = f"– Not enough history ({days}d)"
    elif score >= 2:
        call = "▲ Up bias"
    elif score <= -2:
        call = "▼ Down bias"
    else:
        call = "– Unclear"
    # "About to move" = volume and price range both picking up vs normal
    activity = np.nanmean([last.get("VolSurge", np.nan), last.get("RangeExp", np.nan)])
    return {
        "Symbol": symbol,
        "Price": float(last["Close"]),
        "Signal": call,
        "Score": score,
        f"Next {horizon} min (trend)": proj["target"] if proj else None,
        "Range low": proj["low"] if proj else None,
        "Range high": proj["high"] if proj else None,
        "Last 5 min %": float(last["Mom5"]) if pd.notna(last["Mom5"]) else None,
        "RSI": float(last["RSI"]) if pd.notna(last["RSI"]) else None,
        "Volume surge ×": float(last["VolSurge"]) if pd.notna(last["VolSurge"]) else None,
        "Activity ×": float(activity) if pd.notna(activity) else None,
        "Signal right %": hr,
        "Tests": n,
        "History days": days,
        "Bar time": d.index[-1],
    }


def atr_5m(df: pd.DataFrame, periods: int = 14) -> float | None:
    """Average True Range on 5-minute bars: how far the price typically moves in 5 minutes."""
    b = df.resample("5min").agg({"High": "max", "Low": "min", "Close": "last"}).dropna()
    if len(b) < periods + 1:
        return None
    prev = b["Close"].shift()
    tr = pd.concat([b["High"] - b["Low"], (b["High"] - prev).abs(), (b["Low"] - prev).abs()], axis=1).max(axis=1)
    return float(tr.tail(periods).mean())


def trade_idea(symbol: str, df: pd.DataFrame, horizon: int, account: float, risk_pct: float,
               stop_atr: float = 1.5, reward_ratio: float = 2.0) -> dict | None:
    """Turn the current signal into a concrete plan: entry, stop-loss, target, size."""
    s = summarise(symbol, df, horizon)
    if s["Score"] == 0 or s["History days"] < MIN_HISTORY_DAYS:
        return None
    atr = atr_5m(df)
    if not atr:
        return None
    side = "BUY" if s["Score"] > 0 else "SELL"
    price = s["Price"]
    risk_per_share = stop_atr * atr
    if side == "BUY":
        stop, target = price - risk_per_share, price + reward_ratio * risk_per_share
    else:
        stop, target = price + risk_per_share, price - reward_ratio * risk_per_share
    shares = int((account * risk_pct / 100) // risk_per_share) if risk_per_share > 0 else 0
    shares = min(shares, int(account // price))  # never more than the account can pay for
    hr = s["Signal right %"] or 0
    dollar_vol = float((df["Close"] * df["Volume"]).tail(30).mean())
    # Quality: edge over a coin flip x signal strength x how active the stock is right now
    quality = max(hr - 50, 0) * abs(s["Score"]) * min(max(s["Activity ×"] or 1, 0.5), 3)
    return {
        "Symbol": symbol, "Action": side, "Price": price,
        "Stop-loss": stop, "Target": target,
        "Shares": shares, "Risk $": shares * risk_per_share, "Reward $": shares * reward_ratio * risk_per_share,
        "Score": s["Score"], "Signal right %": s["Signal right %"], "Tests": s["Tests"],
        "History days": s["History days"],
        "Activity ×": s["Activity ×"], "RSI": s["RSI"], "$ traded / min": dollar_vol,
        "Quality": quality, "Bar time": s["Bar time"],
    }


def live_signal(symbol: str, bars: pd.DataFrame, live_price: float | None, horizon: int = 5,
                min_hit: float = 55, stop_atr: float = 1.5, reward_ratio: float = 2.0) -> dict:
    """Re-score the stock using the newest live price on top of its 1-minute history."""
    d = add_indicators(bars)
    last = d.iloc[-1]
    price = float(live_price) if live_price else float(last["Close"])

    a9, a21 = 2 / 10, 2 / 22   # roll the EMAs forward to the live price
    ema9 = last["EMA9"] + a9 * (price - last["EMA9"])
    ema21 = last["EMA21"] + a21 * (price - last["EMA21"])
    ref5 = d["Close"].iloc[-5] if len(d) >= 5 else d["Close"].iloc[0]
    mom5 = (price / ref5 - 1) * 100
    vwap = last["VWAP"]
    rsi = last["RSI"]

    score = int(np.sign(ema9 - ema21) + np.sign(mom5)
                + (np.sign(price - vwap) if pd.notna(vwap) else 0)
                + (1 if pd.notna(rsi) and rsi < 30 else -1 if pd.notna(rsi) and rsi > 70 else 0))
    hr, n = hit_rate(d, horizon)
    days = history_days(bars)
    reliable = hr is not None and hr >= min_hit and n >= 50 and days >= MIN_HISTORY_DAYS

    if days < MIN_HISTORY_DAYS:
        action = "WAIT (not enough history)"
    elif score >= 2 and reliable:
        action = "BUY NOW"
    elif score <= -2 and reliable:
        action = "SELL NOW"
    elif abs(score) >= 2:
        action = "WAIT (unreliable)"
    else:
        action = "WAIT"

    atr = atr_5m(bars)
    stop = target = None
    if atr and action in ("BUY NOW", "SELL NOW"):
        r = stop_atr * atr
        stop, target = (price - r, price + reward_ratio * r) if action == "BUY NOW" else (price + r, price - reward_ratio * r)

    day = d[d.index.date == d.index[-1].date()]
    prev = d[d.index.date < d.index[-1].date()]["Close"]
    ref = prev.iloc[-1] if not prev.empty else day["Open"].iloc[0]
    return {
        "Symbol": symbol, "Action": action, "Price": price, "Day %": (price / ref - 1) * 100,
        "Score": score, "Right %": hr, "Tests": n, "History days": days, "Last 5 min %": mom5,
        "vs VWAP %": (price / vwap - 1) * 100 if pd.notna(vwap) and vwap else None,
        "RSI": float(rsi) if pd.notna(rsi) else None, "Stop-loss": stop, "Target": target,
    }


EVAL_HORIZONS = (5, 10, 15)


def evaluate_signal(bars: pd.DataFrame, t, action: str, price: float, stop=None, target=None,
                    max_minutes: int = 60) -> dict:
    """Check a past BUY/SELL signal against what the price actually did afterwards."""
    t = pd.Timestamp(t)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    direction = 1 if action == "BUY NOW" else -1
    after = bars[bars.index > t]
    after = after[after.index <= t + pd.Timedelta(minutes=max_minutes)]
    out = {}
    for h in EVAL_HORIZONS:
        at = after[after.index >= t + pd.Timedelta(minutes=h)]
        if at.empty:
            out[f"Price +{h}m"] = None
            out[f"+{h}m %"] = None
        else:
            p = float(at["Close"].iloc[0])
            out[f"Price +{h}m"] = p
            out[f"+{h}m %"] = (p / price - 1) * 100 * direction   # positive = signal was right
    result, minutes = "⏳ Waiting", None
    if not after.empty:
        for ts, row in after.iterrows():
            hit_stop = stop is not None and pd.notna(stop) and (
                row["Low"] <= stop if direction == 1 else row["High"] >= stop)
            hit_target = target is not None and pd.notna(target) and (
                row["High"] >= target if direction == 1 else row["Low"] <= target)
            if hit_stop:  # if both happen in the same minute, assume the worse case
                result, minutes = "❌ Stop hit", (ts - t).total_seconds() / 60
                break
            if hit_target:
                result, minutes = "✅ Target hit", (ts - t).total_seconds() / 60
                break
        else:
            span = (after.index[-1] - t).total_seconds() / 60
            if span >= max_minutes - 1 or after.index[-1].date() != t.tz_convert(after.index.tz).date():
                last = float(after["Close"].iloc[-1])
                pl = (last / price - 1) * 100 * direction
                result = f"➖ Neither ({pl:+.2f}%)"
        moves = (after["High"] / price - 1) * 100 * direction, (after["Low"] / price - 1) * 100 * direction
        best = max(moves[0].max(), moves[1].max())
        worst = min(moves[0].min(), moves[1].min())
        out["Best %"], out["Worst %"] = float(best), float(worst)
    else:
        out["Best %"] = out["Worst %"] = None
    out["Result"] = result
    out["Minutes"] = minutes
    return out


def buy_sell_pressure(bars: pd.DataFrame) -> tuple[float, float] | None:
    """Estimated buying vs selling pressure today: volume traded in minutes where the price rose
    vs minutes where it fell (flat minutes split evenly). Returns (buy %, sell %)."""
    if bars is None or bars.empty:
        return None
    today = bars[bars.index.date == bars.index[-1].date()]
    if len(today) < 5:
        return None
    move = today["Close"].diff().fillna(today["Close"] - today["Open"])
    vol = today["Volume"].astype(float)
    up, down, flat = vol[move > 0].sum(), vol[move < 0].sum(), vol[move == 0].sum()
    total = up + down + flat
    if total <= 0:
        return None
    buy = (up + flat / 2) / total * 100
    return float(buy), float(100 - buy)


def official_prev_close(symbol: str) -> float | None:
    """The exchange's official previous close (Tiger if connected, otherwise Yahoo's quote).
    Used for 'Day %' because the last 1-minute bar of the previous day can be wrong or missing."""
    try:
        import tiger_data
        if tiger_data.configured() and tiger_data.is_us(symbol):
            p = tiger_data.prices([symbol]).get(symbol, {}).get("prev_close")
            if p and p > 0:
                return float(p)
    except Exception:
        pass
    try:
        import yfinance as yf
        p = yf.Ticker(symbol).fast_info.previous_close
        if p and p > 0:
            return float(p)
    except Exception:
        pass
    return None


def fix_day_change(r: dict, prev_close: float | None) -> dict:
    """Replace 'Day %' with the change versus the official previous close."""
    if prev_close and r.get("Price"):
        r["Day %"] = (float(r["Price"]) / prev_close - 1) * 100
        r["Prev close"] = prev_close
    return r
