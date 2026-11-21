"""
Short-term momentum signals built from 1-minute price bars.

These are NOT guarantees. They summarise what the price is doing right now
(trend, momentum, position vs VWAP, volume surge) into a score, and they
measure how often that score has called the next few minutes correctly
over recent days, so you can judge whether to trust it for each stock.
"""

import numpy as np
import pandas as pd


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
    if score >= 2:
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
        "Signal right % (5d)": hr,
        "Tests": n,
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
    if s["Score"] == 0:
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
    hr = s["Signal right % (5d)"] or 0
    dollar_vol = float((df["Close"] * df["Volume"]).tail(30).mean())
    # Quality: edge over a coin flip x signal strength x how active the stock is right now
    quality = max(hr - 50, 0) * abs(s["Score"]) * min(max(s["Activity ×"] or 1, 0.5), 3)
    return {
        "Symbol": symbol, "Action": side, "Price": price,
        "Stop-loss": stop, "Target": target,
        "Shares": shares, "Risk $": shares * risk_per_share, "Reward $": shares * reward_ratio * risk_per_share,
        "Score": s["Score"], "Signal right %": s["Signal right % (5d)"], "Tests": s["Tests"],
        "Activity ×": s["Activity ×"], "RSI": s["RSI"], "$ traded / min": dollar_vol,
        "Quality": quality, "Bar time": s["Bar time"],
    }
