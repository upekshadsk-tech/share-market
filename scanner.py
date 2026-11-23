"""
Whole-market scanner for USA, ASX and NZX.

Stage 1 (cheap): one screener query per market looks at EVERY listed share and keeps the
liquid ones that are moving on unusual volume. Stage 2 (in alerts.py / app.py) runs the
full 1-minute signal + accuracy test (min. 5 past trading days) on only the best of those.
"""

import exchange_calendars as xc
import pandas as pd
import yfinance as yf
from yfinance import EquityQuery as Q

SCAN_MARKETS = {
    "USA": {"region": "us", "exchanges": ["NMS", "NYQ", "NGM", "NCM", "ASE"], "cal": "XNYS",
            "min_price": 2.0, "min_dollar_vol": 20_000_000, "pages": 2},
    "Australia (ASX)": {"region": "au", "exchanges": ["ASX"], "cal": "XASX",
                        "min_price": 0.20, "min_dollar_vol": 1_000_000, "pages": 1},
    "New Zealand (NZX)": {"region": "nz", "exchanges": ["NZE"], "cal": "XNZE",
                          "min_price": 0.20, "min_dollar_vol": 200_000, "pages": 1},
}


def is_open(market: str) -> bool:
    cal = xc.get_calendar(SCAN_MARKETS[market]["cal"])
    return bool(cal.is_open_on_minute(pd.Timestamp.now(tz="UTC").floor("min")))


def session_elapsed(market: str) -> float:
    """Fraction of today's session that has passed (0-1), used to compare volume fairly."""
    cal = xc.get_calendar(SCAN_MARKETS[market]["cal"])
    now = pd.Timestamp.now(tz="UTC")
    try:
        session = cal.minute_to_session(now.floor("min"), direction="previous")
        o, c = cal.session_open(session), cal.session_close(session)
        return float(min(max((now - o) / (c - o), 0.05), 1.0))
    except Exception:
        return 1.0


def _query(market: str, side: str):
    cfg = SCAN_MARKETS[market]
    parts = [Q("eq", ["region", cfg["region"]]), Q("is-in", ["exchange", *cfg["exchanges"]]),
             Q("gt", ["intradayprice", cfg["min_price"]])]
    parts.append(Q("gt", ["percentchange", 0.5]) if side == "buy" else Q("lt", ["percentchange", -0.5]))
    return Q("and", parts)


def market_candidates(market: str, side: str = "buy", top: int = 15) -> pd.DataFrame:
    """Best `top` shares in the whole market for a BUY (or SELL) check right now."""
    cfg = SCAN_MARKETS[market]
    rows = []
    for page in range(cfg["pages"]):
        res = yf.screen(_query(market, side), sortField="dayvolume", sortAsc=False, size=250, offset=page * 250)
        quotes = res.get("quotes", []) if isinstance(res, dict) else []
        rows += quotes
        if len(quotes) < 250:
            break
    if not rows:
        return pd.DataFrame()
    elapsed = session_elapsed(market)
    out = []
    for r in rows:
        price, vol = r.get("regularMarketPrice"), r.get("regularMarketVolume") or 0
        avg = r.get("averageDailyVolume3Month") or r.get("averageDailyVolume10Day") or 0
        hi, lo = r.get("regularMarketDayHigh"), r.get("regularMarketDayLow")
        chg = r.get("regularMarketChangePercent") or 0
        if not price or price * vol < cfg["min_dollar_vol"] * elapsed:
            continue
        rel_vol = vol / (avg * elapsed) if avg else 1.0          # volume vs normal for this time of day
        pos = (price - lo) / (hi - lo) if hi and lo and hi > lo else 0.5  # 1 = at day high, 0 = at day low
        if side == "buy":
            if rel_vol < 1.2 or pos < 0.6:
                continue
            rank = min(rel_vol, 5) * min(chg, 15) * pos
        else:
            if rel_vol < 1.2 or pos > 0.4:
                continue
            rank = min(rel_vol, 5) * min(-chg, 15) * (1 - pos)
        out.append({"Symbol": r.get("symbol"), "Company": r.get("shortName") or r.get("longName"),
                    "Market": market, "Price": price, "Day %": chg, "Volume ×": rel_vol,
                    "Near day high": pos, "Rank": rank})
    df = pd.DataFrame(out)
    if df.empty:
        return df
    return df.sort_values("Rank", ascending=False).head(top).reset_index(drop=True)


# ---------------------------------------------------------------- previous sessions (from daily history)
SESSION_UNIVERSE = {"USA": 300, "Australia (ASX)": 200, "New Zealand (NZX)": 150}


def sessions(market: str, n: int = 2) -> list:
    """The last `n` trading days, newest first. Today counts only if the market has opened."""
    cal = xc.get_calendar(SCAN_MARKETS[market]["cal"])
    now = pd.Timestamp.now(tz="UTC")
    out = []
    for s in reversed(cal.sessions_in_range(now.normalize().tz_localize(None) - pd.Timedelta(days=20),
                                            now.normalize().tz_localize(None))):
        if cal.session_open(s) <= now:
            out.append(pd.Timestamp(s).date())
        if len(out) == n:
            break
    return out


def largest_shares(market: str, size: int) -> pd.DataFrame:
    """The market's biggest companies by market value (works any time, open or closed)."""
    cfg = SCAN_MARKETS[market]
    q = Q("and", [Q("eq", ["region", cfg["region"]]), Q("is-in", ["exchange", *cfg["exchanges"]]),
                  Q("gt", ["intradayprice", cfg["min_price"]])])
    rows = []
    for page in range((size + 249) // 250):
        res = yf.screen(q, sortField="intradaymarketcap", sortAsc=False, size=min(250, size - page * 250),
                        offset=page * 250)
        quotes = res.get("quotes", []) if isinstance(res, dict) else []
        rows += quotes
        if len(quotes) < 250:
            break
    return pd.DataFrame([{"Symbol": r.get("symbol"), "Company": r.get("shortName") or r.get("longName")}
                         for r in rows if r.get("symbol")]).drop_duplicates("Symbol")


def session_movers(market: str, day) -> pd.DataFrame:
    """How each of the market's biggest shares moved on a given trading day (from daily prices)."""
    uni = largest_shares(market, SESSION_UNIVERSE[market])
    if uni.empty:
        return pd.DataFrame()
    syms_all = uni["Symbol"].tolist()
    frames = {}
    for i in range(0, len(syms_all), 100):          # smaller batches: one failure doesn't lose everything
        chunk = syms_all[i:i + 100]
        try:
            data = yf.download(chunk, period="2mo", interval="1d", group_by="ticker",
                               auto_adjust=False, threads=True, progress=False)
        except Exception:
            continue
        if data is None or data.empty:
            continue
        for sym in chunk:
            try:
                frames[sym] = data[sym] if isinstance(data.columns, pd.MultiIndex) else data
            except Exception:
                pass
    names = dict(zip(uni["Symbol"], uni["Company"]))
    out = []
    for sym in syms_all:
        try:
            if sym not in frames:
                continue
            d = frames[sym]
            d = d.dropna(subset=["Close"])
            dates = [pd.Timestamp(x).date() for x in d.index]
            if day not in dates:
                continue
            i = dates.index(day)
            if i == 0:
                continue
            row, prev = d.iloc[i], d.iloc[i - 1]
            vol_avg = d["Volume"].iloc[max(0, i - 20):i].mean()
            hi, lo, close = float(row["High"]), float(row["Low"]), float(row["Close"])
            out.append({"Symbol": sym, "Company": names.get(sym), "Market": market,
                        "Price": close, "Prev close": float(prev["Close"]),
                        "Change %": (close / float(prev["Close"]) - 1) * 100,
                        "Day range %": (hi - lo) / lo * 100 if lo else None,
                        "Near day high": (close - lo) / (hi - lo) if hi > lo else 0.5,
                        "Volume": float(row["Volume"]),
                        "Volume ×": float(row["Volume"]) / vol_avg if vol_avg else None,
                        "Date": day})
        except Exception:
            continue
    df = pd.DataFrame(out)
    df.attrs["universe"] = len(syms_all)
    df.attrs["loaded"] = len(frames)
    return df
