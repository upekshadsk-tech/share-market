"""
Stock Tracker: US, ASX (Australia) and NZX (New Zealand)

Screens:
  1. Big Movers  - the stocks with the largest price swings today (checked first)
  2. Watchlist   - live prices for only the companies you picked, auto-refreshing
  3. Pick Companies - search the full list of listed companies and add/remove them

Data: Yahoo Finance via the `yfinance` library (free, no API key).
US prices are near real-time; ASX and NZX are delayed ~20 minutes by the exchanges
unless you pay for a real-time data feed.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import exchange_calendars as xc
import pandas as pd
import streamlit as st
import yfinance as yf
import logging
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
from yfinance import EquityQuery as Q
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from signals import (EVAL_HORIZONS, MIN_HISTORY_DAYS, add_indicators, buy_sell_pressure, combine_bars, evaluate_signal,
                     fix_day_change, official_prev_close,
                     fetch_history_1m, fetch_today_1m, live_signal, summarise, trade_idea)
from live import LiveFeed
from notify import alert_text, send_push
import trades as tr
from scanner import fallback_live, market_candidates, session_movers, sessions, yahoo_call

st.set_page_config(page_title="Stock Tracker US / AU / NZ", page_icon="📈", layout="wide")

# Tiger keys live only in Streamlit Secrets; copy them to the environment for tiger_data.py
try:
    import os as _os
    for _k in ("TIGEROPEN_TIGER_ID", "TIGEROPEN_PRIVATE_KEY", "TIGEROPEN_ACCOUNT", "TIGEROPEN_LICENSE"):
        if _k in st.secrets and not _os.environ.get(_k):
            _os.environ[_k] = str(st.secrets[_k])
except Exception:
    pass
try:
    import tiger_data  # noqa: E402  (optional: read-only Tiger prices, never trades)
except Exception:      # tiger_data.py not uploaded: run on Yahoo prices only
    class _NoTiger:
        @staticmethod
        def configured():
            return False

        @staticmethod
        def prices(symbols):
            return {}

        @staticmethod
        def status():
            return {"ok": False, "message": "tiger_data.py not uploaded"}
    tiger_data = _NoTiger()

WATCHLIST_FILE = Path(__file__).with_name("watchlist.json")

MARKETS = {
    "USA": {"region": "us", "exchanges": ["NMS", "NYQ", "NGM", "NCM", "ASE"], "tz": "America/New_York",
            "cal": "XNYS", "flag": "🇺🇸"},
    "Australia (ASX)": {"region": "au", "exchanges": ["ASX"], "tz": "Australia/Sydney", "cal": "XASX", "flag": "🇦🇺"},
    "New Zealand (NZX)": {"region": "nz", "exchanges": ["NZE"], "tz": "Pacific/Auckland", "cal": "XNZE", "flag": "🇳🇿"},
}
SUFFIX_MARKET = {".AX": "Australia (ASX)", ".NZ": "New Zealand (NZX)"}


def market_of(symbol: str) -> str:
    for suffix, name in SUFFIX_MARKET.items():
        if symbol.upper().endswith(suffix):
            return name
    return "USA"


# ---------------------------------------------------------------- watchlist storage
def load_watchlist() -> list[str]:
    # A ?w=AAPL,BHP.AX link restores a watchlist on any device
    from_url = st.query_params.get("w")
    if from_url:
        return [s for s in from_url.split(",") if s]
    try:
        return json.loads(WATCHLIST_FILE.read_text())
    except Exception:
        return ["AAPL", "NVDA", "BHP.AX", "CBA.AX", "FPH.NZ", "AIR.NZ"]


def save_watchlist(symbols: list[str]) -> None:
    st.session_state.watchlist = symbols
    st.query_params["w"] = ",".join(symbols)
    try:
        WATCHLIST_FILE.write_text(json.dumps(symbols))
    except Exception:
        pass


if "watchlist" not in st.session_state:
    st.session_state.watchlist = load_watchlist()


# ---------------------------------------------------------------- data fetching
def _screen_rows(query, sort_field="percentchange", sort_asc=False, size=250, offset=0):
    res = yahoo_call(yf.screen, query, sortField=sort_field, sortAsc=sort_asc, size=size, offset=offset)
    return res.get("quotes", []) if isinstance(res, dict) else []


def _base_query(market: str):
    cfg = MARKETS[market]
    return Q("and", [
        Q("eq", ["region", cfg["region"]]),
        Q("is-in", ["exchange", *cfg["exchanges"]]),
    ])


@st.cache_data(ttl=24 * 3600, show_spinner="Loading full company list (once a day)...")
def company_list(market: str) -> pd.DataFrame:
    """Every listed equity in a market (Yahoo's screener), or the built-in main companies if it's blocked."""
    try:
        df = _company_list_screener(market)
        if not df.empty:
            return df
    except Exception:
        pass
    from universe import FALLBACK, name_of
    return pd.DataFrame({"Symbol": FALLBACK[market], "Company": [name_of(x) for x in FALLBACK[market]],
                         "Price": None, "Market cap": None})


def _company_list_screener(market: str) -> pd.DataFrame:
    rows, offset = [], 0
    while offset < 12000:
        batch = _screen_rows(Q("and", [_base_query(market), Q("gt", ["intradayprice", 0])]),
                             sort_field="intradaymarketcap", size=250, offset=offset)
        rows.extend(batch)
        if len(batch) < 250:
            break
        offset += 250
    df = pd.DataFrame([{
        "Symbol": r.get("symbol"),
        "Company": r.get("longName") or r.get("shortName") or r.get("symbol"),
        "Price": r.get("regularMarketPrice"),
        "Market cap": r.get("marketCap"),
    } for r in rows]).drop_duplicates("Symbol")
    return df


@st.cache_data(ttl=60, show_spinner=False)
def big_movers(market: str, min_move: float, min_price: float, min_volume: int, size: int) -> pd.DataFrame:
    """Stocks whose price has moved more than `min_move` % today, up or down."""
    base = [_base_query(market), Q("gt", ["intradayprice", min_price]), Q("gt", ["dayvolume", min_volume])]
    up = _screen_rows(Q("and", base + [Q("gt", ["percentchange", min_move])]), sort_asc=False, size=size)
    down = _screen_rows(Q("and", base + [Q("lt", ["percentchange", -min_move])]), sort_asc=True, size=size)
    rows = []
    for r in up + down:
        hi, lo, price = r.get("regularMarketDayHigh"), r.get("regularMarketDayLow"), r.get("regularMarketPrice")
        rows.append({
            "Symbol": r.get("symbol"),
            "Company": r.get("shortName") or r.get("longName"),
            "Price": price,
            "Change %": r.get("regularMarketChangePercent"),
            "Day range %": (hi - lo) / lo * 100 if hi and lo else None,
            "Volume": r.get("regularMarketVolume"),
            "Currency": r.get("currency"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates("Symbol").sort_values("Change %", key=abs, ascending=False)
    return df


def _quote(symbol: str) -> dict:
    try:
        fi = yf.Ticker(symbol).fast_info
        price, prev = fi.last_price, fi.previous_close
        hi, lo = fi.day_high, fi.day_low
        return {
            "Symbol": symbol,
            "Market": market_of(symbol),
            "Price": price,
            "Change": price - prev if price and prev else None,
            "Change %": (price - prev) / prev * 100 if price and prev else None,
            "Day low": lo,
            "Day high": hi,
            "Day range %": (hi - lo) / lo * 100 if hi and lo else None,
            "Volume": fi.last_volume,
            "Currency": fi.currency,
        }
    except Exception:
        return {"Symbol": symbol, "Market": market_of(symbol)}


@st.cache_data(ttl=15, show_spinner=False)
def quotes(symbols: tuple[str, ...]) -> pd.DataFrame:
    with ThreadPoolExecutor(max_workers=12) as pool:
        return pd.DataFrame(list(pool.map(_quote, symbols)))


@st.cache_data(ttl=60, show_spinner=False)
def intraday(symbol: str) -> pd.DataFrame:
    h = yf.Ticker(symbol).history(period="1d", interval="5m")
    if h.empty:  # market closed: show last 5 days instead
        h = yf.Ticker(symbol).history(period="5d", interval="30m")
    return h[["Close"]]


# ---------------------------------------------------------------- display helpers
def color_change(v):
    if v is None or pd.isna(v):
        return ""
    return "color: #16a34a" if v > 0 else "color: #dc2626" if v < 0 else ""


def show_table(df: pd.DataFrame, alert_pct: float | None = None):
    styler = df.style.format({
        "Price": "{:,.3f}", "Change": "{:+,.3f}", "Change %": "{:+.2f}%",
        "Day range %": "{:.2f}%", "Day low": "{:,.3f}", "Day high": "{:,.3f}",
        "Volume": "{:,.0f}", "Market cap": "{:,.0f}", "Volume ×": "{:.1f}×",
    }, na_rep="–")
    cols = [c for c in ("Change", "Change %") if c in df.columns]
    if cols:
        styler = styler.map(color_change, subset=cols)
    if alert_pct is not None and "Change %" in df.columns:
        styler = styler.apply(
            lambda row: ["background-color: rgba(250,204,21,.25)"] * len(row)
            if pd.notna(row["Change %"]) and abs(row["Change %"]) >= alert_pct else [""] * len(row),
            axis=1)
    st.dataframe(styler, hide_index=True, width="stretch")


def now_local():
    """Current time in the time zone chosen in Settings (the server itself runs on UTC)."""
    try:
        return pd.Timestamp.now(tz=st.session_state.get("my_tz_sel") or "Pacific/Auckland")
    except Exception:
        return pd.Timestamp.now(tz="Pacific/Auckland")


@st.cache_resource
def calendar(code: str):
    return xc.get_calendar(code)


def _until(delta: pd.Timedelta) -> str:
    mins = int(delta.total_seconds() // 60)
    d, h, m = mins // 1440, mins % 1440 // 60, mins % 60
    return f"{d}d {h}h" if d else f"{h}h {m:02d}m" if h else f"{m} min"


def market_status(name: str, my_tz: str) -> dict:
    """Open/closed right now, using the exchange's real calendar (holidays and early closes)."""
    cal = calendar(MARKETS[name]["cal"])
    now = pd.Timestamp.now(tz="UTC").floor("min")
    is_open = cal.is_open_on_minute(now)
    nxt = cal.next_close(now) if is_open else cal.next_open(now)
    local_next = nxt.tz_convert(my_tz)
    return {"open": bool(is_open), "next": local_next, "in": nxt - now,
            "exchange_time": now.tz_convert(MARKETS[name]["tz"])}


def open_markets(my_tz: str) -> list[str]:
    return [m for m in MARKETS if market_status(m, my_tz)["open"]]


def market_status_bar(my_tz: str):
    cols = st.columns(len(MARKETS))
    for col, name in zip(cols, MARKETS):
        try:
            ms = market_status(name, my_tz)
        except Exception:
            col.caption(f"{name}: status unavailable")
            continue
        if ms["open"]:
            badge, colr, line = "🟢 OPEN NOW", "#16a34a", f"closes in {_until(ms['in'])} ({ms['next']:%H:%M} your time)"
        else:
            badge, colr, line = "🔴 CLOSED", "#dc2626", f"opens in {_until(ms['in'])} ({ms['next']:%a %H:%M} your time)"
        col.markdown(
            f"<div style='border:1px solid {colr};border-radius:10px;padding:8px 10px'>"
            f"<div style='font-weight:600'>{MARKETS[name]['flag']} {name}</div>"
            f"<div style='color:{colr};font-weight:700'>{badge}</div>"
            f"<div style='font-size:.85em'>{line}</div>"
            f"<div style='font-size:.75em;opacity:.7'>Exchange time {ms['exchange_time']:%a %H:%M}</div></div>",
            unsafe_allow_html=True)


# ---------------------------------------------------------------- sidebar settings
with st.sidebar:
    st.header("Settings")
    my_tz = st.selectbox("My time zone", key="my_tz_sel", options=["Pacific/Auckland", "Australia/Sydney", "Australia/Perth",
                                          "America/New_York", "Asia/Colombo", "UTC"])
    only_open = st.toggle("Trade Ideas: scan only markets open now", value=True)
    chosen_markets = st.multiselect("Markets", list(MARKETS), default=list(MARKETS))
    refresh = st.select_slider("Auto-refresh every", [10, 15, 30, 60, 120], value=10,
                               format_func=lambda s: f"{s}s")
    st.subheader("Big-mover filter")
    min_move = st.slider("Price move of at least (%)", 1.0, 30.0, 5.0, 0.5)
    min_price = st.number_input("Ignore stocks priced under", 0.0, 100.0, 0.10, 0.05,
                                help="Hides penny stocks, which swing wildly on tiny trades")
    min_volume = st.number_input("Minimum volume today", 0, 10_000_000, 50_000, 10_000)
    max_rows = st.slider("Max stocks per direction", 10, 100, 25, 5)
    with st.expander("📲 Phone alerts", expanded=False):
        st.caption("Install the free **ntfy** app (Play Store), tap **+**, and subscribe to a private topic name "
                   "only you know, e.g. `upeksha-stocks-7391`. Enter the same name here. "
                   "These alerts work while this app is open; the GitHub checker covers when it is closed.")
        default_topic = st.query_params.get("ntfy", "")
        ntfy_topic = st.text_input("ntfy topic", value=default_topic, key="ntfy_topic").strip()
        alerts_on = st.toggle("Send BUY / SELL alerts to my phone", value=bool(ntfy_topic), key="alerts_on")
        if ntfy_topic:
            st.query_params["ntfy"] = ntfy_topic
        if ntfy_topic and st.button("Send test alert"):
            ok = send_push(ntfy_topic, "✅ Stock Tracker test", "Alerts are working.", "info")
            if ok:
                st.success(f"Sent to topic '{ntfy_topic}'. Check your phone.")
            else:
                st.error("Couldn't send. Check the topic name.")
    with st.expander("💼 My trade margin", expanded=False):
        st.caption("Used when you tap ✅ I bought. 0 = let the app choose from the share's normal move.")
        st.number_input("Sell when profit reaches (+%)", 0.0, 50.0, 0.0, 0.25, key="tp_pct")
        st.number_input("Stop-loss if it falls (−%)", 0.0, 50.0, 0.0, 0.25, key="sl_pct")
    st.subheader("Price source")
    if tiger_data.configured():
        ts = tiger_data.status()
        if ts["ok"]:
            st.success("🐯 Tiger connected (read-only): official live US prices")
        else:
            st.error(f"🐯 Tiger: {ts['message'][:160]}. Using Yahoo instead.")
    else:
        st.caption("Yahoo Finance (free). Add Tiger keys in Streamlit Secrets for official US prices.")
    st.caption("ASX and NZX are delayed about 20 minutes.")

st.title("📈 Stock Tracker: USA · Australia · New Zealand")

@st.fragment(run_every=30)
def status_view():
    market_status_bar(my_tz)
    live = open_markets(my_tz)
    st.caption("Live now: " + (", ".join(live) if live else "no market is open right now"))
status_view()

# Menu: only the selected screen runs (much faster than tabs, which all run on every refresh)
VIEWS = {"top": "🏆 Top Picks", "trades": "💼 My Trades", "now": "⚡ My Shares", "log": "📒 Signal Log", "ideas": "🎯 Trade Ideas",
         "movers": "🔥 Big Movers", "watch": "⭐ Watchlist", "chart": "📊 Live Chart", "signals": "🔮 Signals",
         "pick": "➕ Pick Companies"}
if st.session_state.get("_goto"):                # programmatic jump (e.g. after "I bought")
    st.session_state["view"] = st.session_state.pop("_goto")
_start = st.query_params.get("view", "top")
VIEW = st.segmented_control("Screen", list(VIEWS), default=_start if _start in VIEWS else "top", key="view",
                            format_func=lambda v: VIEWS[v], label_visibility="collapsed") or "top"
if st.query_params.get("view") != VIEW:
    st.query_params["view"] = VIEW

# ---------------------------------------------------------------- 1-minute bars
@st.cache_data(ttl=1800, show_spinner=False)
def bars_history(symbol: str) -> pd.DataFrame:
    """~2 weeks of 1-minute bars (about 10 trading days). Reloaded every 30 minutes."""
    return fetch_history_1m(symbol)


@st.cache_data(ttl=10, show_spinner=False)
def bars_today(symbol: str) -> pd.DataFrame:
    return fetch_today_1m(symbol)


def bars_1m(symbol: str) -> pd.DataFrame:
    """History + today's freshest bars. Signals need at least 5 full past trading days."""
    return combine_bars(bars_history(symbol), bars_today(symbol))


def bars_many(symbols: tuple[str, ...]) -> dict:
    def one(s):
        try:
            return s, bars_1m(s)
        except Exception:
            return s, pd.DataFrame()
    with ThreadPoolExecutor(max_workers=10) as pool:
        return dict(pool.map(one, symbols))


# ---------------------------------------------------------------- Buy / Sell now (my shares)
@st.cache_resource
def live_feed() -> LiveFeed:
    return LiveFeed()


@st.cache_data(ttl=2, show_spinner=False)
def tiger_prices(symbols: tuple[str, ...]) -> dict:
    return tiger_data.prices(symbols) if tiger_data.configured() else {}


def best_prices(symbols, feed) -> dict:
    """Newest price per share: Tiger (US, if connected) first, then the Yahoo live stream."""
    out = {}
    tg = tiger_prices(tuple(symbols))
    for s in symbols:
        if s in tg:
            out[s] = (tg[s]["price"], "tiger")
        else:
            m = feed.get(s)
            if m:
                out[s] = (m["price"], "yahoo")
    return out


@st.cache_data(ttl=900, show_spinner=False)
def prev_close(symbol: str):
    return official_prev_close(symbol)


def prev_closes(symbols) -> dict:
    with ThreadPoolExecutor(max_workers=10) as pool:
        return dict(zip(symbols, pool.map(prev_close, symbols)))


@st.cache_data(ttl=20, show_spinner=False)
def fallback_price(symbol: str):
    try:
        return float(yf.Ticker(symbol).fast_info.last_price)
    except Exception:
        return None


NOW_CSS = """
<style>
.sig-card{border-radius:12px;padding:10px 14px;margin:6px 0;border:2px solid}
.sig-buy{background:#dcfce7;border-color:#16a34a;color:#14532d}
.sig-sell{background:#e0f2fe;border-color:#0284c7;color:#0c4a6e}
.sig-wait{background:rgba(148,163,184,.12);border-color:rgba(148,163,184,.5)}
.sig-top{display:flex;justify-content:space-between;align-items:baseline;gap:8px;flex-wrap:wrap}
.sig-sym{font-size:1.25em;font-weight:800}
.sig-act{font-weight:800;letter-spacing:.03em}
.sig-price{font-size:1.2em;font-weight:700}
.sig-sub{font-size:.85em;opacity:.9}
.up{color:#15803d}.down{color:#b91c1c}
</style>
"""


def signal_card(r: dict, tick: str, live) -> str:
    cls = {"BUY NOW": "sig-buy", "SELL NOW": "sig-sell"}.get(r["Action"], "sig-wait")
    icon = {"BUY NOW": "🟢", "SELL NOW": "🔵"}.get(r["Action"], "⏸️")
    dcls = "up" if (r["Day %"] or 0) >= 0 else "down"
    plan = ""
    if r["Stop-loss"] is not None:
        plan = (f"<div class='sig-sub'><b>Stop-loss</b> {r['Stop-loss']:,.3f} · <b>Target</b> {r['Target']:,.3f}</div>")
    right = (f"{r['Right %']:.0f}% of {r['Tests']} ({r['History days']}d)"
             if r["Right %"] is not None else "–")
    rsi = f"{r['RSI']:.0f}" if r["RSI"] is not None else "–"
    return (f"<div class='sig-card {cls}'><div class='sig-top'>"
            f"<span class='sig-sym'>{r['Symbol']}</span>"
            f"<span class='sig-act'>{icon} {r['Action']}</span>"
            f"<span class='sig-price'>{tick} {r['Price']:,.3f} <span class='{dcls}' style='font-size:.75em'>"
            f"{r['Day %']:+.2f}%</span></span></div>"
            f"<div class='sig-sub'>Score {r['Score']:+d} · signal right {right} · last 5 min {r['Last 5 min %']:+.2f}% · "
            f"RSI {rsi} · {'⚡ live (Tiger)' if live == 'tiger' else '⚡ live' if live else '1-min data'}</div>{plan}</div>")


@st.cache_data(ttl=120, show_spinner=False)
def fallback_live_cached(market: str) -> pd.DataFrame:
    return fallback_live(market)


@st.cache_data(ttl=1800, show_spinner=False)
def _session_movers_ok(market: str, day) -> pd.DataFrame:
    df = session_movers(market, day)
    if df.empty:
        raise RuntimeError("no daily prices loaded")   # raising = not cached, so the next refresh retries
    return df


def session_movers_cached(market: str, day) -> pd.DataFrame:
    try:
        return _session_movers_ok(market, day)
    except Exception as e:
        out = pd.DataFrame()
        out.attrs["error"] = str(e)
        return out


@st.cache_data(ttl=600, show_spinner=False)
def sessions_cached(market: str, n: int = 2, _minute: str = "") -> list:
    return sessions(market, n)


def day_label(d) -> str:
    return pd.Timestamp(d).strftime("%a %d %b")


TRADES_HISTORY_URL = "https://raw.githubusercontent.com/upekshadsk-tech/share-market/signal-log/trade_events.jsonl"


@st.cache_data(ttl=120, show_spinner=False)
def trade_history() -> list:
    return tr.load_history_url(TRADES_HISTORY_URL)


@st.cache_data(ttl=8, show_spinner=False)
def _trade_events_fresh(topic: str) -> list:
    return tr.load_events(topic)


def trade_events(topic: str) -> list:
    """Every buy/sell/alert: permanent GitHub history + the latest from ntfy."""
    return tr.merge_events(trade_history(), _trade_events_fresh(topic))


trade_events.clear = _trade_events_fresh.clear   # same call as before when a new event is saved


def record_buy(sym: str, price: float, shares: float, stop=None, target=None, bars=None) -> bool:
    topic = st.session_state.get("ntfy_topic", "")
    if not topic:
        st.warning("Set your ntfy topic in the sidebar (📲 Phone alerts) first, so your trades can be tracked.")
        return False
    tp, sl = st.session_state.get("tp_pct", 0.0), st.session_state.get("sl_pct", 0.0)
    if tp or sl or not stop or not target:          # your own margin wins over the signal's levels
        m_stop, m_target = tr.levels_from_margin(price, tp, sl, bars)
        stop = m_stop if (sl or not stop) else stop
        target = m_target if (tp or not target) else target
    t = tr.open_trade(topic, sym, price, shares, stop, target, market_of(sym))
    trade_events.clear()
    if t.get("ok"):
        st.toast(f"✅ Tracking {sym}: bought at {price:,.3f}. I'll tell you when to sell.")
        return True
    st.error("Couldn't save the trade (ntfy not reachable). Try again.")
    return False


def log_and_alert(rows: list[dict], horizon: int, source: str = "app"):
    """Record new BUY/SELL signals in the Signal Log and send phone alerts (if switched on)."""
    sent = st.session_state.setdefault("alert_state", {})
    app_log = st.session_state.setdefault("app_log", [])
    push_on = st.session_state.get("alerts_on") and st.session_state.get("ntfy_topic")
    live = set(open_markets(my_tz))
    for r in rows:
        act = r["Action"]
        if market_of(r["Symbol"]) not in live:
            continue   # market closed: last-session setups are for practice only, never alert or log
        if act in ("BUY NOW", "SELL NOW") and sent.get(r["Symbol"]) != act:
            app_log.append({"Time (UTC)": pd.Timestamp.now(tz="UTC").floor("s"), "Symbol": r["Symbol"],
                            "Action": act, "Price": r["Price"], "Stop-loss": r["Stop-loss"],
                            "Target": r["Target"], "Score": r["Score"], "Right %": r["Right %"],
                            "Horizon": horizon, "Source": source})
            if push_on:
                topic_ = st.session_state["ntfy_topic"]
                if not tr.signal_alerted_recently(trade_events(topic_), r["Symbol"], act):
                    title, body = alert_text(r)
                    send_push(topic_, title, body, act)
                    tr.record_signal_alert(topic_, r["Symbol"], act)
                    trade_events.clear()
                    st.toast(title)
        sent[r["Symbol"]] = act



if VIEW == "now":
    st.markdown(NOW_CSS, unsafe_allow_html=True)
    st.markdown("**Buy now / sell now for my selected shares**")
    c1, c2, c3 = st.columns(3)
    now_h = c1.selectbox("Signal for next", [1, 2, 5, 10, 15], index=2, format_func=lambda m: f"{m} min",
                         key="now_h")
    now_hit = c2.slider("Min. signal right %", 50, 70, 55, key="now_hit")
    now_refresh = c3.selectbox("Refresh", [1, 2, 5], format_func=lambda s: f"every {s}s", key="now_ref")

    @st.fragment(run_every=now_refresh)
    def now_view():
        wl = [s for s in st.session_state.watchlist if market_of(s) in chosen_markets]
        if not wl:
            st.info("Add companies to your watchlist first (➕ Pick Companies).")
            return
        feed = live_feed()
        feed.watch(wl)
        bars = bars_many(tuple(wl))
        prev_px = st.session_state.setdefault("prev_px", {})
        rows, live_flags = [], {}
        px_now = best_prices(wl, feed)
        pcs = prev_closes(tuple(wl))
        for s in wl:
            b = bars.get(s)
            if b is None or b.empty or len(b) < 30:
                continue
            px, src = px_now.get(s, (None, None))
            live_flags[s] = src
            try:
                rows.append(fix_day_change(live_signal(s, b, px, now_h, now_hit), pcs.get(s)))
            except Exception:
                pass
        if not rows:
            st.info("No price data yet. The markets may be closed, or the data is still loading.")
            return

        def tick_for(r):
            old, new = prev_px.get(r["Symbol"]), r["Price"]
            prev_px[r["Symbol"]] = new
            if old is None or old == new:
                return "•"
            return "<span class='up'>▲</span>" if new > old else "<span class='down'>▼</span>"

        # Phone alerts + Signal Log (only for shares whose market is open right now)
        log_and_alert(rows, now_h, "app")

        buys = sorted([r for r in rows if r["Action"] == "BUY NOW"], key=lambda r: -r["Score"])
        sells = sorted([r for r in rows if r["Action"] == "SELL NOW"], key=lambda r: r["Score"])
        waits = [r for r in rows if r["Action"].startswith("WAIT")]

        cb, cs = st.columns(2)
        with cb:
            st.markdown(f"### 🟢 Good to BUY ({len(buys)})")
            st.markdown("".join(signal_card(r, tick_for(r), live_flags.get(r["Symbol"])) for r in buys)
                        or "<div class='sig-sub'>None right now</div>", unsafe_allow_html=True)
        with cs:
            st.markdown(f"### 🔵 Good to SELL ({len(sells)})")
            st.markdown("".join(signal_card(r, tick_for(r), live_flags.get(r["Symbol"])) for r in sells)
                        or "<div class='sig-sub'>None right now</div>", unsafe_allow_html=True)
        with st.expander(f"⏸️ Wait / no clear signal ({len(waits)})", expanded=False):
            st.markdown("".join(signal_card(r, tick_for(r), live_flags.get(r["Symbol"])) for r in waits),
                        unsafe_allow_html=True)
        n_live = sum(1 for v in live_flags.values() if v)
        n_tiger = sum(1 for v in live_flags.values() if v == "tiger")
        st.caption(f"{now_local():%H:%M:%S} · live stream: {'connected' if feed.connected else 'connecting…'} · "
                   f"{n_live}/{len(wl)} stocks ticking live" + (f" ({n_tiger} from Tiger)" if n_tiger else ""))

    now_view()
    st.caption("🟢 BUY NOW = score +2 or more and the signal has been right at least your minimum % on this stock "
               f"over at least {MIN_HISTORY_DAYS} past trading days (usually ~10). 🔵 SELL NOW = score -2 or less, same test. "
               "'WAIT (unreliable)' = the signal points one way but hasn't been accurate enough for this stock. "
               "Live ticks are near real-time for US stocks; ASX/NZX are ~20 min delayed. Not financial advice.")



# ---------------------------------------------------------------- Top picks (whole market, automatic)
@st.cache_data(ttl=180, show_spinner=False)
def top_candidates(market: str, per_side: int) -> pd.DataFrame:
    frames = []
    for side in ("buy", "sell"):
        try:
            c = market_candidates(market, side, per_side)
            if not c.empty:
                c["Side"] = side
                frames.append(c)
        except Exception:
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()




RANK_CSS = """
<style>
.tg-wrap{background:#16171d;border-radius:14px;padding:6px 14px 4px;color:#e8e8ee}
.tg-head{display:grid;grid-template-columns:34px 1fr 84px 118px;gap:8px;color:#8b8d98;font-size:.82em;
         padding:8px 0 6px}
.tg-head .r{text-align:right}
.tg-row{display:grid;grid-template-columns:34px 1fr 84px 118px;gap:8px;align-items:center;
        padding:12px 0;border-top:1px solid #262833}
.tg-rank{font-size:1.25em;font-weight:700;color:#8b8d98}
.tg-rank.r1{color:#ff5a4f}.tg-rank.r2{color:#ff9a3c}.tg-rank.r3{color:#f5c542}
.tg-co{font-size:1.08em;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.tg-sub{font-size:.74em;color:#9a9ca8;margin-top:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.tg-det{grid-column:2 / -1;font-size:.76em;color:#9a9ca8;margin-top:-2px;line-height:1.35}
.tg-det b{color:#d6d7de;font-weight:600}
.tg-badge{display:inline-block;font-weight:800;font-size:.95em;padding:1px 6px;border-radius:5px;margin-right:5px}
.tg-badge.buy{background:#123d27;color:#34d77c}.tg-badge.sell{background:#0d3550;color:#5cc8ff}
.tg-badge.wait{background:#2a2c36;color:#9a9ca8}
.tg-chg{text-align:right;font-size:1.08em;font-weight:600}
.tg-chg.up{color:#25c26e}.tg-chg.down{color:#ff5a4f}
.tg-pct{display:flex;justify-content:space-between;font-size:.82em;font-weight:600}
.tg-pct .g{color:#25c26e}.tg-pct .r{color:#ff5a4f}
.tg-track{display:flex;height:4px;border-radius:3px;overflow:hidden;margin-top:4px;gap:2px}
.tg-track .g{background:#25c26e}.tg-track .r{background:#ff5a4f}
.tg-title{font-size:1.35em;font-weight:700;text-align:center;margin-top:4px}
.tg-upd{color:#8b8d98;text-align:center;font-size:.85em;margin-bottom:6px}
.tg-row.sel{background:#1f2130;border-radius:10px}
[class*="st-key-tgwrap"]{background:#16171d;border-radius:14px;padding:6px 14px 8px;color:#e8e8ee}
[class*="st-key-tgwrap"] [data-testid="stVerticalBlock"]{gap:0 !important}
[class*="st-key-tgrow_"]{position:relative}
[class*="st-key-tgbtn_"]{position:absolute !important;inset:0;z-index:3;margin:0 !important}
[class*="st-key-tgbtn_"] *{width:100% !important;height:100% !important;max-width:none !important}
[class*="st-key-tgbtn_"] button{opacity:0;cursor:pointer;padding:0;border:0}
[class*="st-key-tgdetail"]{background:#1b1d27;border-radius:12px;padding:10px 12px;margin:2px 0 8px;color:#e8e8ee}
.tg-dgrid{display:grid;grid-template-columns:repeat(3,1fr);gap:6px 10px;font-size:.82em;color:#9a9ca8;margin:4px 0 6px;padding-bottom:10px;line-height:1.35}
.tg-dgrid b{display:block;color:#e8e8ee;font-size:1.1em}
.tg-pred{background:#16171d;border-radius:10px;padding:8px 10px;font-size:.85em;margin-top:4px}
[class*="st-key-tgclose_"] button{background:#2a2c36 !important;color:#e8e8ee !important;border:0 !important;
  min-height:0 !important;padding:2px 10px !important}
[class*="st-key-tgdetail"] [data-testid="stHorizontalBlock"]{flex-wrap:nowrap !important;align-items:center}
[class*="st-key-tgdetail"] [data-testid="stColumn"]{min-width:0 !important}
[class*="st-key-tgdetail"] [data-testid="stVerticalBlock"]{gap:.45rem !important}
</style>
"""


def rank_row(i: int, r: dict, live_src, selected: bool = False) -> str:
    rk = f"r{i}" if i <= 3 else ""
    badge = {"BUY NOW": ("buy", "BUY"), "SELL NOW": ("sell", "SELL")}.get(r["Action"], ("wait", "WAIT"))
    chg = r.get("Day %") or 0
    pr = r.get("Pressure")
    if pr:
        b, sl = pr
        bar = (f"<div class='tg-pct'><span class='g'>{b:.0f}%</span><span class='r'>{sl:.0f}%</span></div>"
               f"<div class='tg-track'><div class='g' style='width:{b:.0f}%'></div>"
               f"<div class='r' style='width:{sl:.0f}%'></div></div>")
    else:
        bar = "<div class='tg-pct'><span>–</span></div>"
    right = f" · right {r['Right %']:.0f}% ({r['History days']}d)" if r.get("Right %") is not None else ""
    plan = (f"<br>Stop <b>{r['Stop-loss']:,.3f}</b> · Target <b>{r['Target']:,.3f}</b>"
            if r.get("Stop-loss") is not None else "")
    live = " ⚡" if live_src else ""
    name = (r.get("Company") or r["Symbol"]).replace("<", "")
    return (f"<div class='tg-row{' sel' if selected else ''}'><div class='tg-rank {rk}'>{i}</div>"
            f"<div style='min-width:0'><div class='tg-co'>{name}</div>"
            f"<div class='tg-sub'><span class='tg-badge {badge[0]}'>{badge[1]}</span>{r['Symbol']}</div></div>"
            f"<div class='tg-chg {'up' if chg >= 0 else 'down'}'>{chg:+.2f}%</div>"
            f"<div>{bar}</div>"
            f"<div class='tg-det'>Price <b>{r['Price']:,.3f}</b>{live}{right}{plan}</div></div>")


def live_chart(symbol: str, d: pd.DataFrame, proj_minutes: int, dark: bool = False):
    today = d[d.index.date == d.index[-1].date()]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=today.index, open=today["Open"], high=today["High"], low=today["Low"],
                                 close=today["Close"], name="Price",
                                 increasing_line_color="#16a34a", decreasing_line_color="#dc2626"), 1, 1)
    for col, colr in (("EMA9", "#f59e0b"), ("EMA21", "#3b82f6"), ("VWAP", "#a855f7")):
        fig.add_trace(go.Scatter(x=today.index, y=today[col], name=col, line=dict(width=1.3, color=colr)), 1, 1)
    # Trend projection with uncertainty band
    from signals import projection
    p = projection(d, proj_minutes)
    if p:
        t0 = today.index[-1]
        t1 = t0 + pd.Timedelta(minutes=proj_minutes)
        last = float(today["Close"].iloc[-1])
        fig.add_trace(go.Scatter(x=[t0, t1, t1, t0], y=[last, p["high"], p["low"], last], fill="toself",
                                 fillcolor="rgba(148,163,184,.25)", line=dict(width=0), name="Likely range",
                                 hoverinfo="skip"), 1, 1)
        fig.add_trace(go.Scatter(x=[t0, t1], y=[last, p["target"]], name=f"Trend +{proj_minutes}m",
                                 line=dict(dash="dot", color="#64748b")), 1, 1)
    colors = ["#16a34a" if c >= o else "#dc2626" for o, c in zip(today["Open"], today["Close"])]
    fig.add_trace(go.Bar(x=today.index, y=today["Volume"], marker_color=colors, name="Volume", showlegend=False), 2, 1)
    # one giant block trade shouldn't squash every other volume bar: cap the scale, the spike goes off the top
    vcap = float(today["Volume"].quantile(0.99)) * 1.3 if len(today) > 20 else None
    if vcap and float(today["Volume"].max()) > vcap * 2:
        fig.update_yaxes(range=[0, vcap], row=2, col=1)
    if not dark:
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10), xaxis_rangeslider_visible=False,
                          legend=dict(orientation="h", y=1.08), title=f"{symbol} · today, 1-minute bars")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        return
    # Dark, phone-friendly version for the Top Buys & Sells panel
    bg, grid, txt = "#1b1d27", "#2a2c36", "#c9cad3"
    if p:
        t_end = today.index[-1] + pd.Timedelta(minutes=proj_minutes)
        span = today.index[-1] - today.index[0]
        fig.update_xaxes(range=[today.index[0], t_end + max(span * 0.04, pd.Timedelta(minutes=proj_minutes))])
        colr = "#25c26e" if p["target"] >= float(today["Close"].iloc[-1]) else "#ff5a4f"
        fig.add_annotation(x=t_end, y=p["target"], text=f"{p['target']:,.2f}", showarrow=True, arrowhead=0,
                           ax=0, ay=-26, font=dict(color=colr, size=12), bgcolor=bg, row=1, col=1)
    fig.update_layout(template="plotly_dark", paper_bgcolor=bg, plot_bgcolor=bg, height=430,
                      margin=dict(l=4, r=4, t=8, b=4), xaxis_rangeslider_visible=False, showlegend=True,
                      legend=dict(orientation="h", yanchor="top", y=-0.08, x=0, font=dict(size=10, color=txt)),
                      font=dict(color=txt, size=11), hovermode="x unified")
    fig.update_xaxes(gridcolor=grid, showline=False)
    fig.update_yaxes(gridcolor=grid, side="right")
    st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False, "scrollZoom": False})


def detail_panel(sym: str, r: dict, bars: pd.DataFrame, horizon: int):
    """Today's price movement for one share plus the short-term trend projection."""
    from signals import projection
    with st.container(key=f"tgdetail_{sym}"):
        top = st.columns([4, 1])
        top[0].markdown(f"<div style='font-weight:700;font-size:1.05em'>{(r.get('Company') or sym).replace('<', '')}"
                        f" <span style='color:#9a9ca8;font-weight:400'>· {sym}</span></div>", unsafe_allow_html=True)
        if top[1].button("✕", key=f"tgclose_{sym}", help="Close"):
            st.session_state["top_detail"] = None
            st.rerun()
        st.markdown("<div style='color:#9a9ca8;font-size:.8em'>🔮 Predict next</div>", unsafe_allow_html=True)
        mins = st.segmented_control("Predict next", [5, 10, 15, 30], default=horizon if horizon in (5, 10, 15, 30)
                                    else 10, key=f"tgpred_{sym}", format_func=lambda m: f"{m} min",
                                    label_visibility="collapsed") or 10
        d = add_indicators(bars)
        today = d[d.index.date == d.index[-1].date()]
        prev = d[d.index.date < d.index[-1].date()]["Close"]
        prev_close = float(r["Prev close"]) if r.get("Prev close") else (
            float(prev.iloc[-1]) if not prev.empty else float(today["Open"].iloc[0]))
        hi, lo, op = float(today["High"].max()), float(today["Low"].min()), float(today["Open"].iloc[0])
        price = float(r["Price"])
        rsi = f"{r['RSI']:.0f}" if r.get("RSI") is not None else "–"
        st.markdown(
            "<div class='tg-dgrid'>"
            f"<div>Price<b>{price:,.3f}</b></div>"
            f"<div>Change<b style='color:{'#25c26e' if price >= prev_close else '#ff5a4f'}'>{(price / prev_close - 1) * 100:+.2f}%</b></div>"
            f"<div>Prev close<b>{prev_close:,.3f}</b></div>"
            f"<div>Open<b>{op:,.3f}</b></div><div>Day high<b>{hi:,.3f}</b></div><div>Day low<b>{lo:,.3f}</b></div>"
            f"<div>Day range<b>{(hi - lo) / lo * 100:.2f}%</b></div><div>RSI<b>{rsi}</b></div>"
            f"<div>Signal<b>{r['Action'].replace(' NOW', '')}</b></div></div>", unsafe_allow_html=True)
        live_chart(sym, d, mins, dark=True)
        p = projection(d, mins)
        if p:
            direction = "up" if p["target"] > price else "down" if p["target"] < price else "flat"
            st.markdown(
                f"<div class='tg-pred'>🔮 <b>Next {mins} min (trend guide):</b> {direction} towards "
                f"<b>{p['target']:,.3f}</b> ({(p['target'] / price - 1) * 100:+.2f}%), likely range "
                f"<b>{p['low']:,.3f} – {p['high']:,.3f}</b>.<br><span style='color:#9a9ca8'>Straight-line trend of the "
                f"last 15 minutes with its normal noise. A guide, not a forecast. "
                + (f"This signal was right {r['Right %']:.0f}% of the time over {r['History days']} days."
                   if r.get("Right %") is not None else "") + "</span></div>", unsafe_allow_html=True)
        in_wl = sym in st.session_state.watchlist
        if st.button("⭐ In My Shares" if in_wl else "⭐ Add to My Shares", key=f"tgwl_{sym}", disabled=in_wl):
            save_watchlist(list(dict.fromkeys(st.session_state.watchlist + [sym])))
            st.toast(f"⭐ {sym} added to My Shares")
            st.rerun()
        if True:   # always available (you may buy pre-market, or on delayed ASX/NZX prices)
            if market_of(sym) not in open_markets(my_tz):
                st.caption("⚠️ This market is closed now. You can still record a buy you made.")
            bc = st.columns([1, 2])
            sh = bc[0].number_input("Shares", min_value=0.0, value=0.0, step=1.0, key=f"tgsh_{sym}",
                                    label_visibility="collapsed", help="How many shares you bought (optional)")
            tp_, sl_ = st.session_state.get("tp_pct", 0.0), st.session_state.get("sl_pct", 0.0)
            pv_stop, pv_tgt = tr.levels_from_margin(price, tp_, sl_, bars)
            if not tp_ and r.get("Target"):
                pv_tgt = r["Target"]
            if not sl_ and r.get("Stop-loss"):
                pv_stop = r["Stop-loss"]
            st.caption(f"If you buy now, I'll tell you to sell at **{pv_tgt:,.3f}** ({(pv_tgt / price - 1) * 100:+.2f}%) "
                       f"or if it falls to **{pv_stop:,.3f}** ({(pv_stop / price - 1) * 100:+.2f}%). "
                       "Change the margin in the sidebar: 💼 My trade margin.")
            if bc[1].button(f"✅ I bought at {price:,.3f}", key=f"tgbuy_{sym}", width="stretch"):
                if record_buy(sym, price, sh, r.get("Stop-loss"), r.get("Target"), bars):
                    st.session_state["_goto"] = "trades"
                    st.rerun()


TRADE_CSS = """
<style>
.tr-card{border-radius:12px;padding:10px 14px;margin:6px 0 2px;border:2px solid rgba(148,163,184,.5);
         background:rgba(148,163,184,.10)}
.tr-card.sell{background:#e0f2fe;border-color:#0284c7;color:#0c4a6e}
.tr-top{display:flex;justify-content:space-between;align-items:baseline;gap:8px;flex-wrap:wrap}
.tr-sym{font-size:1.25em;font-weight:800}
.tr-act{font-weight:800;letter-spacing:.03em}
.tr-pl{font-size:1.15em;font-weight:800}
.tr-sub{font-size:.85em;opacity:.9;margin-top:2px}
.tr-why{font-weight:700;margin-top:4px}
.up{color:#15803d}.down{color:#b91c1c}
</style>
"""

if VIEW == "trades":
    st.markdown(TRADE_CSS, unsafe_allow_html=True)
    st.markdown("**My day trades.** Tap **✅ I bought** on a company in 🏆 Top Picks (or add one below). "
                "The app watches it and tells you when to sell, before the market closes.")
    topic = st.session_state.get("ntfy_topic", "")
    if not topic:
        st.warning("First set your ntfy topic in the sidebar (**📲 Phone alerts**). Trades are saved there so the "
                   "app and the background checker can both track them and alert your phone.")
    else:
        add_sym = st.query_params.get("add", "")
        with st.expander("➕ Add a trade I made", expanded=bool(add_sym)):
            c1, c2, c3 = st.columns(3)
            picks = ["Type a ticker…"] + list(dict.fromkeys(st.session_state.watchlist))
            chosen = c1.selectbox("From my watchlist", picks, index=picks.index(add_sym) if add_sym in picks else 0)
            a_sym = (add_sym if chosen == picks[0] else chosen)
            a_sym = c1.text_input("Ticker", value=a_sym, placeholder="AAPL, BHP.AX, AIR.NZ").strip().upper()
            a_px = c2.number_input("Price I paid (0 = current price)", min_value=0.0, value=0.0, step=0.01, format="%.3f")
            a_sh = c3.number_input("Shares", min_value=0.0, value=0.0, step=1.0)
            if a_sym and st.button("Save trade", type="primary"):
                b = bars_1m(a_sym)
                px = a_px or (float(b["Close"].iloc[-1]) if b is not None and not b.empty else 0)
                if px <= 0:
                    st.error("Couldn't get a price for that ticker. Enter the price you paid.")
                elif record_buy(a_sym, px, a_sh, bars=b):
                    if "add" in st.query_params:
                        del st.query_params["add"]
                    st.rerun()

        @st.fragment(run_every=10)
        def trades_view():
            open_t, closed_t, alerted = tr.build(trade_events(topic))
            if not open_t:
                st.info("No open trades. When the app suggests a BUY, open it in 🏆 Top Picks and tap ✅ I bought.")
            feed = live_feed()
            syms = tuple(dict.fromkeys(t["symbol"] for t in open_t))
            if syms:
                feed.watch(syms)
            bars = bars_many(syms) if syms else {}
            px_now = best_prices(syms, feed) if syms else {}
            live_set = set(open_markets(my_tz))
            for t in open_t:
                sym = t["symbol"]
                b = bars.get(sym)
                price = px_now.get(sym, (None, None))[0] or (float(b["Close"].iloc[-1]) if b is not None and not b.empty
                                                              else t["entry"])
                score = None
                try:
                    if b is not None and len(b) > 120:
                        score = live_signal(sym, b, price, 5, 55)["Score"]
                except Exception:
                    pass
                mkt = t.get("market") or market_of(sym)
                mtc = tr.minutes_to_close(MARKETS[mkt]["cal"]) if mkt in MARKETS else None
                adv = tr.advice(t, b, price, mtc, score)
                sell = adv["action"] == "SELL"
                # phone alert once per reason (shared with the background checker, so no duplicates)
                if sell and mkt in live_set and tr.should_alert(alerted.get(t["id"]), adv["reason"]):
                    send_push(topic, f"🔵 SELL NOW: {sym} @ {price:,.3f}",
                              f"{adv['text']}\nP/L {adv['pl_pct']:+.2f}%" + (f" (${adv['pl']:+,.2f})" if t.get('shares') else "")
                              + f"\nBought at {t['entry']:,.3f}. Not financial advice.", "SELL NOW")
                    tr.mark_alerted(topic, t["id"], adv["reason"])
                    trade_events.clear()
                pl_cls = "up" if adv["pl_pct"] >= 0 else "down"
                since = pd.Timestamp(t["time"]).tz_convert(my_tz).strftime("%H:%M")
                money = f" (${adv['pl']:+,.2f})" if t.get("shares") else ""
                st.markdown(
                    f"<div class='tr-card {'sell' if sell else ''}'><div class='tr-top'>"
                    f"<span class='tr-sym'>{sym}</span>"
                    f"<span class='tr-act'>{'🔵 SELL NOW' if sell else '🟢 HOLD'}</span>"
                    f"<span class='tr-pl {pl_cls}'>{adv['pl_pct']:+.2f}%{money}</span></div>"
                    f"<div class='tr-sub'>Now <b>{price:,.3f}</b> · bought <b>{t['entry']:,.3f}</b> at {since}"
                    + (f" · {t['shares']:g} shares" if t.get("shares") else "")
                    + (f" · closes in {mtc:.0f} min" if mtc is not None else " · market closed") + "</div>"
                    f"<div class='tr-why'>{adv['text']}</div>"
                    f"<div class='tr-sub'>Margin: sell at <b>{(t['target'] / t['entry'] - 1) * 100:+.2f}%</b> "
                    f"({t['target']:,.3f}) · stop <b>{(t['stop'] / t['entry'] - 1) * 100:+.2f}%</b> ({t['stop']:,.3f})</div>"
                    f"</div>", unsafe_allow_html=True)
                with st.expander(f"✏️ Change margin for {sym}"):
                    e1, e2, e3 = st.columns(3)
                    cur_tp = min(max((t["target"] / t["entry"] - 1) * 100, 0.05), 50.0)
                    cur_sl = min(max((1 - t["stop"] / t["entry"]) * 100, 0.05), 50.0)
                    n_tp = e1.number_input("Sell at +%", 0.05, 50.0, float(round(cur_tp, 2)), 0.25, key=f"etp_{t['id']}")
                    n_sl = e2.number_input("Stop at −%", 0.05, 50.0, float(round(cur_sl, 2)), 0.25, key=f"esl_{t['id']}")
                    e3.caption(f"Target **{t['entry'] * (1 + n_tp / 100):,.3f}**  \nStop **{t['entry'] * (1 - n_sl / 100):,.3f}**")
                    if st.button("Save margin", key=f"esave_{t['id']}"):
                        tr.update_trade(topic, t, t["entry"] * (1 - n_sl / 100), t["entry"] * (1 + n_tp / 100))
                        trade_events.clear()
                        st.toast(f"{sym}: sell at +{n_tp:g}%, stop at −{n_sl:g}%")
                        st.rerun()
                sc = st.columns([1, 2])
                sold_px = sc[0].number_input("Sold at", min_value=0.0, value=float(round(price, 3)), step=0.01,
                                             format="%.3f", key=f"soldpx_{t['id']}", label_visibility="collapsed",
                                             help="The price your broker actually sold at")
                if sc[1].button(f"💰 I sold {sym}", key=f"sold_{t['id']}", width="stretch"):
                    px_ = sold_px or price
                    tr.close_trade(topic, t["id"], px_)
                    trade_events.clear()
                    st.toast(f"Recorded: sold {sym} at {px_:,.3f} ({(px_ / t['entry'] - 1) * 100:+.2f}%)")
                    st.rerun()

            st.markdown("#### 📒 Buy & sell log")
            period = st.segmented_control("Period", ["today", "week", "all"], default="today", key="log_period",
                                          format_func=lambda k: {"today": "Today", "week": "This week",
                                                                 "all": "All"}[k],
                                          label_visibility="collapsed") or "today"
            start = {"today": pd.Timestamp.now(tz=my_tz).normalize(),
                     "week": pd.Timestamp.now(tz=my_tz).normalize() - pd.Timedelta(days=7),
                     "all": pd.Timestamp("2000-01-01", tz=my_tz)}[period]
            loc = lambda x: pd.Timestamp(x).tz_convert(my_tz)
            log_rows = []
            for t in closed_t:
                if loc(t["exit_time"]) >= start or loc(t["time"]) >= start:
                    log_rows.append({"Ticker": t["symbol"], "Status": "✅ Sold", "Bought": t["entry"],
                                     "Buy time": loc(t["time"]), "Sold": t["exit"], "Sell time": loc(t["exit_time"]),
                                     "P/L %": t["pl_pct"], "P/L $": t["pl"] if t.get("shares") else None,
                                     "Shares": t.get("shares") or None})
            for t in open_t:
                if loc(t["time"]) >= start or period != "today":
                    log_rows.append({"Ticker": t["symbol"], "Status": "🟢 Holding", "Bought": t["entry"],
                                     "Buy time": loc(t["time"]), "Sold": None, "Sell time": None,
                                     "P/L %": None, "P/L $": None, "Shares": t.get("shares") or None})
            if not log_rows:
                st.caption("No buys or sells in this period yet.")
            else:
                df = pd.DataFrame(log_rows).sort_values("Buy time", ascending=False)
                done = df[df["Status"] == "✅ Sold"]
                if not done.empty:
                    wins = int((done["P/L %"] > 0).sum())
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Finished trades", len(done))
                    m2.metric("Win rate", f"{wins / len(done) * 100:.0f}%", f"{wins} of {len(done)}")
                    tot = done["P/L $"].dropna().sum()
                    m3.metric("Total P/L", f"${tot:+,.2f}" if done["P/L $"].notna().any()
                              else f"{done['P/L %'].mean():+.2f}% avg")
                for c in ("Buy time", "Sell time"):
                    df[c] = df[c].apply(lambda v: v.strftime("%a %d %b %H:%M") if pd.notna(v) and v is not None else "")
                st.dataframe(df.style.format({"Bought": "{:,.3f}", "Sold": "{:,.3f}", "P/L %": "{:+.2f}%",
                                              "P/L $": "{:+,.2f}", "Shares": "{:g}"}, na_rep="–")
                             .map(lambda v: "color:#15803d;font-weight:600" if isinstance(v, (int, float)) and v > 0
                                  else "color:#b91c1c;font-weight:600" if isinstance(v, (int, float)) and v < 0 else "",
                                  subset=["P/L %", "P/L $"]), hide_index=True, width="stretch")
                st.download_button("⬇️ Download log (CSV)", df.to_csv(index=False).encode(), "my_trades.csv", "text/csv")
            st.caption(f"Updated {now_local():%H:%M:%S} · Sell rules: stop-loss, trailing stop once you're up, target, "
                       f"momentum turning down while in profit, and {tr.CLOSE_WARN_MIN} min before the close. "
                       "Brokerage fees not included. Not financial advice.")
        trades_view()


if VIEW == "top":
    st.markdown(NOW_CSS + RANK_CSS, unsafe_allow_html=True)
    short = {"USA": "US", "Australia (ASX)": "ASX", "New Zealand (NZX)": "NZX"}
    live_now = open_markets(my_tz)
    default = next((m for m in MARKETS if m in live_now), "USA")
    top_mkt = st.segmented_control("Market", list(MARKETS), default=default, key="top_mkt",
                                   format_func=lambda m: f"{MARKETS[m]['flag']} {short[m]}"
                                   + (" ●" if m in live_now else ""), label_visibility="collapsed") or default
    sess = sessions_cached(top_mkt, 2, pd.Timestamp.now().strftime("%Y%m%d%H"))
    top_live = top_mkt in live_now
    day_opts = ["latest", "before"] if len(sess) > 1 else ["latest"]
    day_lbl = {"latest": ("Today (live)" if top_live else f"Last session · {day_label(sess[0])}") if sess else "Latest",
               "before": f"Day before · {day_label(sess[1])}" if len(sess) > 1 else ""}
    top_day = st.segmented_control("Day", day_opts, default="latest", key="top_day",
                                   format_func=lambda k: day_lbl[k], label_visibility="collapsed") or "latest"
    top_rising = st.checkbox("📈 Only rising shares that are good to buy", key="top_rising",
                             help="Shows shares that are up today with an upward signal. BUY signals first.")
    with st.expander("⚙️ Options", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        top_h = c1.selectbox("Signal for next", [1, 2, 5, 10, 15], index=2, format_func=lambda m: f"{m} min",
                             key="top_h")
        top_hit = c2.slider("Min. signal right %", 50, 70, 55, key="top_hit")
        top_show = c3.selectbox("Show", ["BUY & SELL first, then others", "Only BUY & SELL signals"], key="top_show")
        top_ref = c4.selectbox("Refresh", [2, 5, 10, 30], index=1, format_func=lambda s: f"every {s}s", key="top_ref")

    def session_rows(day):
        """Top movers of a finished trading day, with the signal as it stood at that day's close."""
        mv = session_movers_cached(top_mkt, day)
        if mv.empty:
            return [], {}
        ups = mv[(mv["Change %"] > 0.5) & (mv["Near day high"] >= 0.5)].nlargest(10, "Change %")
        downs = mv[(mv["Change %"] < -0.5) & (mv["Near day high"] <= 0.5)].nsmallest(10, "Change %")
        if ups.empty and downs.empty:
            ups, downs = mv.nlargest(10, "Change %"), mv.nsmallest(10, "Change %")
        pick = pd.concat([ups, downs]).drop_duplicates("Symbol")
        syms = tuple(pick["Symbol"])
        with st.spinner(f"Loading {len(syms)} shares for {day_label(day)}…"):
            bars = bars_many(syms)
        rows, used = [], {}
        for _, m in pick.iterrows():
            s = m["Symbol"]
            b = bars.get(s)
            r = None
            if b is not None and not b.empty:
                bt = b[b.index.date <= day]
                if len(bt) > 120 and bt.index[-1].date() == day:
                    try:
                        r = live_signal(s, bt, None, top_h, top_hit)
                        r["Pressure"] = buy_sell_pressure(bt)
                        used[s] = bt
                    except Exception:
                        r = None
            if r is None:   # no minute data (some small NZX/ASX shares): show the day's move only
                r = {"Symbol": s, "Action": "WAIT", "Score": 0, "Right %": None, "History days": 0,
                     "Stop-loss": None, "Target": None, "RSI": None, "Pressure": None}
            r["Price"], r["Day %"], r["Company"] = m["Price"], m["Change %"], m["Company"]
            rows.append(r)
        return rows, used

    @st.fragment(run_every=top_ref if (top_live and top_day == "latest") else None)
    def top_view():
        is_live = top_live and top_day == "latest"
        session_day = None
        if not is_live:
            if not sess:
                st.info("No trading days found.")
                return
            session_day = sess[0] if top_day == "latest" else sess[1]
            rows, bars = session_rows(session_day)
            if not rows:
                st.warning("Couldn't load that day's prices yet (Yahoo may be busy). It retries automatically, "
                           "or refresh the page in a minute.")
                return
            live_flags = {}
            cands = None
        else:
            cands = top_candidates(top_mkt, 10)
        if is_live and cands.empty:
            st.info("The market scan returned nothing yet. Try again in a minute.")
            return
        if is_live:
            syms = tuple(dict.fromkeys(cands["Symbol"]))
            feed = live_feed()
            feed.watch(syms)
            first = f"top_loaded_{top_mkt}" not in st.session_state
            with st.spinner(f"Checking {len(syms)} shares (at least 5 trading days of history each)…"
                            + (" The first load takes about a minute." if first else "")):
                bars = bars_many(syms)
            st.session_state[f"top_loaded_{top_mkt}"] = True

            info = cands.drop_duplicates("Symbol").set_index("Symbol")
            px_now = best_prices(syms, feed)
            pcs = prev_closes(syms)
            rows, live_flags = [], {}
            for s in syms:
                b = bars.get(s)
                if b is None or b.empty or len(b) < 120:
                    continue
                px, src = px_now.get(s, (None, None))
                live_flags[s] = src
                try:
                    r = fix_day_change(live_signal(s, b, px, top_h, top_hit), pcs.get(s))
                except Exception:
                    continue
                r["Company"] = info.at[s, "Company"]
                r["Pressure"] = buy_sell_pressure(b)
                rows.append(r)
            if not rows:
                st.info("No price data yet. Try again shortly.")
                return
            log_and_alert(rows, top_h, "app")

        def order(r):
            act = {"BUY NOW": 0, "SELL NOW": 0}.get(r["Action"], 1)
            return (act, -(r["Right %"] or 0), -abs(r["Score"]))
        rows.sort(key=order)
        if top_rising:
            rows = [r for r in rows if (r.get("Day %") or 0) > 0 and (r["Action"] == "BUY NOW" or (r.get("Score") or 0) >= 2)]
        elif top_show.startswith("Only"):
            rows = [r for r in rows if r["Action"] in ("BUY NOW", "SELL NOW")]

        tz = MARKETS[top_mkt]["tz"]
        upd = pd.Timestamp.now(tz=tz).strftime("%m/%d %H:%M:%S %Z") if is_live else "close of day"
        note = "" if is_live else f" · {day_label(session_day)} (closed day, no alerts)"
        n_buy = sum(r["Action"] == "BUY NOW" for r in rows)
        n_sell = sum(r["Action"] == "SELL NOW" for r in rows)
        sel = st.session_state.get("top_detail")
        with st.container(key=f"tgwrap_{top_mkt.split()[0]}"):
            st.markdown(
                f"<div class='tg-title'>Top Buys &amp; Sells</div>"
                f"<div class='tg-upd'>Updated at {upd}{note}<br>{n_buy} BUY · {n_sell} SELL · tap a company for its chart</div>"
                f"<div class='tg-head'><span>Rank</span><span>Company</span><span class='r'>Chg%</span>"
                f"<span class='r'>Buy / Sell</span></div>", unsafe_allow_html=True)
            if not rows:
                st.markdown("<div class='tg-row' style='grid-template-columns:1fr'><div class='tg-sub'>"
                            "No share passes all the checks right now. Not trading is a valid choice.</div></div>",
                            unsafe_allow_html=True)
            for i, r in enumerate(rows):
                sym = r["Symbol"]
                with st.container(key=f"tgrow_{sym}"):
                    st.markdown(rank_row(i + 1, r, live_flags.get(sym), sel == sym), unsafe_allow_html=True)
                    if st.button("Open", key=f"tgbtn_{sym}", width="stretch"):
                        st.session_state["top_detail"] = None if sel == sym else sym
                        st.rerun()
                if sel == sym:
                    try:
                        detail_panel(sym, r, bars[sym], top_h)
                    except Exception as e:
                        st.caption(f"Chart not available ({e}).")

    top_view()
    with st.expander("ℹ️ How this list works"):
        st.markdown(f"""
- **Which shares:** every listed share in the market is screened. The list keeps liquid shares moving on volume at least
  1.2× normal, near the day's high (buy side) or low (sell side), then checks the best 10 each way closely.
- **BUY / SELL badge:** trend, momentum, VWAP and RSI agree, **and** the same rule was right at least your minimum %
  on that share over at least **{MIN_HISTORY_DAYS} past trading days** (shown as e.g. "right 58% (9d)").
  **WAIT** = no reliable signal right now. BUY & SELL are listed first, best first.
- **Chg%:** today's price change.
- **Buy / Sell bar:** *estimated* buying vs selling pressure today, from volume traded in minutes where the price
  rose vs fell. (Tiger's own bar uses its customers' orders, which only Tiger can see.)
- **⚡** = live price. ● next to a market = open now. ASX/NZX are ~20 min delayed. Not financial advice.
""")

# ---------------------------------------------------------------- Signal log
DEFAULT_LOG_URL = "https://raw.githubusercontent.com/upekshadsk-tech/share-market/signal-log/signal_log.csv"


@st.cache_data(ttl=120, show_spinner=False)
def load_github_log(url: str) -> pd.DataFrame:
    import io
    import requests
    r = requests.get(url, timeout=15)
    if r.status_code != 200:
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(r.text))


def bars_7d(symbol: str) -> pd.DataFrame:
    return bars_1m(symbol)


if VIEW == "log":
    st.markdown("**Were the signals right?** Every BUY/SELL signal with its time and price, "
                "checked against what the price did next.")
    c1, c2 = st.columns([3, 1])
    log_url = c1.text_input("GitHub signal log (from the alert checker)", DEFAULT_LOG_URL)
    days = c2.selectbox("Show last", [1, 3, 7, 30, 365], index=2, format_func=lambda d: f"{d} days")
    sources = st.multiselect("Signals from", ["github", "scan", "app"], default=["github", "scan", "app"],
                             help="github = your alert shares; scan = whole-market buy suggestions (both saved permanently); "
                                  "app = signals seen while this app was open (cleared when the app restarts)")

    frames = []
    if "github" in sources or "scan" in sources:
        try:
            g = load_github_log(log_url)
            if not g.empty:
                frames.append(g[g["Source"].isin([x for x in sources if x != "app"])] if "Source" in g else g)
        except Exception as e:
            st.caption(f"Couldn't load the GitHub log yet ({e}).")
    if "app" in sources and st.session_state.get("app_log"):
        frames.append(pd.DataFrame(st.session_state["app_log"]))

    if not frames:
        st.info("No signals logged yet. They appear here as soon as a share turns BUY NOW or SELL NOW "
                "(GitHub checker during market hours, or this app while it's open).")
    else:
        log = pd.concat(frames, ignore_index=True)
        log["Time (UTC)"] = pd.to_datetime(log["Time (UTC)"], utc=True)
        log = log[log["Time (UTC)"] >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)]
        log = log.sort_values("Time (UTC)", ascending=False).reset_index(drop=True)

        # Fill in results that are not saved yet (fresh signals, or app signals)
        need = log["Result"].isna() | log["Result"].astype(str).str.startswith("⏳") if "Result" in log else \
            pd.Series(True, index=log.index)
        for i in log[need].index:
            row = log.loc[i]
            try:
                ev = evaluate_signal(bars_7d(row["Symbol"]), row["Time (UTC)"], row["Action"], float(row["Price"]),
                                     row.get("Stop-loss"), row.get("Target"))
                for k, v in ev.items():
                    log.at[i, k] = v
            except Exception:
                pass

        h = st.radio("Judge signals after", list(EVAL_HORIZONS), index=0, horizontal=True,
                     format_func=lambda m: f"{m} min")
        col = f"+{h}m %"
        done = log[pd.to_numeric(log.get(col), errors="coerce").notna()]
        n = len(done)
        right = int((pd.to_numeric(done[col]) > 0).sum()) if n else 0
        res = log["Result"].astype(str) if "Result" in log else pd.Series(dtype=str)
        tgt, stp = int(res.str.startswith("✅").sum()), int(res.str.startswith("❌").sum())
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Signals", len(log))
        m2.metric(f"Right after {h} min", f"{right / n * 100:.0f}%" if n else "–", f"{right} of {n}")
        m3.metric("Target hit first", f"{tgt / (tgt + stp) * 100:.0f}%" if tgt + stp else "–", f"{tgt} vs {stp} stops")
        m4.metric(f"Avg move after {h} min", f"{pd.to_numeric(done[col]).mean():+.2f}%" if n else "–",
                  help="In the signal's direction, before fees. Positive = the signal made money on average.")

        show = log.copy()
        show.insert(0, "Time", show["Time (UTC)"].dt.tz_convert(my_tz).dt.strftime("%a %d %b %H:%M:%S"))
        cols = ["Time", "Symbol", "Action", "Price", "Stop-loss", "Target"] + \
               [c for h_ in EVAL_HORIZONS for c in (f"Price +{h_}m", f"+{h_}m %")] + \
               ["Result", "Minutes", "Best %", "Worst %", "Right %", "Source"]
        show = show[[c for c in cols if c in show.columns]]
        num = {c: "{:,.3f}" for c in ["Price", "Stop-loss", "Target"] + [f"Price +{x}m" for x in EVAL_HORIZONS]}
        num.update({c: "{:+.2f}%" for c in [f"+{x}m %" for x in EVAL_HORIZONS] + ["Best %", "Worst %"]})
        num.update({"Minutes": "{:.0f}", "Right %": "{:.0f}%"})

        def row_color(r):
            c = "background-color:#dcfce7" if r["Action"] == "BUY NOW" else "background-color:#e0f2fe"
            return [c if k in ("Time", "Symbol", "Action") else "" for k in r.index]

        def pct_color(v):
            try:
                return "color:#15803d;font-weight:600" if float(v) > 0 else "color:#b91c1c;font-weight:600" if float(v) < 0 else ""
            except (TypeError, ValueError):
                return ""

        pct_cols = [c for c in show.columns if c.endswith("m %")]
        st.dataframe(show.style.format(num, na_rep="–").apply(row_color, axis=1).map(pct_color, subset=pct_cols),
                     hide_index=True, width="stretch")
        st.download_button("⬇️ Download log (CSV)", show.to_csv(index=False).encode(), "signal_log.csv", "text/csv")
        st.caption(f"Times in {my_tz}. '+5m %' is the move in the signal's direction (green = the signal was right). "
                   "'Result' checks the first hour: did the price reach the target or the stop-loss first? "
                   "Fees are not included. Results appear once enough minutes have passed.")


# ---------------------------------------------------------------- 0. Trade ideas
@st.cache_data(ttl=120, show_spinner=False)
def scan_universe(markets: tuple[str, ...], watchlist: tuple[str, ...]) -> list[str]:
    """Watchlist + the best BUY and SELL candidates from a scan of EVERY share in each market."""
    syms = list(watchlist)
    for m in markets:
        for side in ("buy", "sell"):
            try:
                syms += market_candidates(m, side, 15)["Symbol"].tolist()
            except Exception:
                pass
    return list(dict.fromkeys(syms))[:100]


if VIEW == "ideas":
    st.markdown("**Best buy / sell setups right now**, ranked by how reliable the signal has been "
                "for each stock and how active it is.")
    c1, c2, c3 = st.columns(3)
    account = c1.number_input("My trading money ($)", 100.0, 10_000_000.0, 5000.0, 500.0)
    risk_pct = c2.number_input("Risk per trade (%)", 0.25, 5.0, 1.0, 0.25,
                               help="Max you lose if the stop-loss is hit. 1% is a common rule.")
    idea_h = c3.selectbox("Hold for about", [5, 10, 15, 30], index=1, format_func=lambda m: f"{m} min")
    c4, c5 = st.columns(2)
    min_hit = c4.slider("Only show if signal was right at least (%)", 50, 70, 55)
    min_tests = c5.slider("...over at least this many past tests", 20, 500, 100, 20)

    @st.fragment(run_every=max(refresh, 30))
    def ideas_view():
        scan_mkts = chosen_markets
        if only_open:
            live = [m for m in chosen_markets if m in open_markets(my_tz)]
            if live:
                scan_mkts = live
                st.caption("Scanning markets open now: " + ", ".join(live))
            else:
                st.caption("No chosen market is open, so showing the last session's setups for practice.")
        universe = scan_universe(tuple(scan_mkts),
                                 tuple(s for s in st.session_state.watchlist if market_of(s) in scan_mkts))
        if not universe:
            st.info("Pick at least one market in Settings.")
            return
        with st.spinner(f"Scanning {len(universe)} stocks..."):
            ideas = []
            for s, d in bars_many(tuple(universe)).items():
                if d is None or len(d) < 120:
                    continue
                try:
                    i = trade_idea(s, d, idea_h, account, risk_pct)
                except Exception:
                    i = None
                if i:
                    i["Market"] = market_of(s)
                    ideas.append(i)
        if not ideas:
            st.info("No intraday data right now: the markets may be closed.")
            return
        df = pd.DataFrame(ideas)
        now = pd.Timestamp.now(tz="UTC")
        df["Live"] = df["Bar time"].apply(lambda t: (now - t.tz_convert("UTC")) < pd.Timedelta(minutes=30))
        good = df[(df["Signal right %"].fillna(0) >= min_hit) & (df["Tests"] >= min_tests)
                  & (df["Score"].abs() >= 2) & (df["Shares"] > 0)]
        good = good.sort_values("Quality", ascending=False)
        if not df["Live"].any():
            st.warning("All markets look closed. These are setups from the last session, for practice only.")

        fmt = {"Price": "{:,.3f}", "Stop-loss": "{:,.3f}", "Target": "{:,.3f}", "Shares": "{:,.0f}",
               "Risk $": "{:,.0f}", "Reward $": "{:,.0f}", "Signal right %": "{:.0f}%",
               "Activity ×": "{:.1f}×", "RSI": "{:.0f}", "$ traded / min": "{:,.0f}", "Quality": "{:.0f}",
               "History days": "{:.0f}",
               "Bar time": lambda t: f"{t:%H:%M}"}
        cols = ["Symbol", "Market", "Price", "Stop-loss", "Target", "Shares", "Risk $", "Reward $",
                "Signal right %", "Tests", "History days", "Activity ×", "RSI", "$ traded / min", "Quality", "Bar time"]
        for side, title, colr in (("BUY", "🟢 Top BUY setups", "#16a34a"), ("SELL", "🔴 Top SELL setups", "#dc2626")):
            st.subheader(title)
            part = good[good["Action"] == side].head(5)
            if part.empty:
                st.caption("Nothing passes your reliability filter right now. That's normal: no trade is a valid choice.")
                continue
            top = part.iloc[0]
            st.markdown(f"<div style='border-left:4px solid {colr};padding:6px 12px;margin-bottom:8px'>"
                        f"<b>#1 {top.Symbol}</b> · {side} about <b>{top.Shares:,}</b> at ~{top.Price:,.3f} · "
                        f"stop {top['Stop-loss']:,.3f} · target {top.Target:,.3f} · "
                        f"risk ${top['Risk $']:,.0f} to make ${top['Reward $']:,.0f} · "
                        f"right {top['Signal right %']:.0f}% of {top.Tests} past tests over {top['History days']} days</div>",
                        unsafe_allow_html=True)
            st.dataframe(part[cols].style.format(fmt, na_rep="–"), hide_index=True, width="stretch")
        st.caption(f"Scanned {len(df)} stocks, {len(good)} passed your filters · updated {now_local():%H:%M:%S}")

    ideas_view()
    with st.expander("How these are picked (read before trading)"):
        st.markdown("""
- **Scanned:** every listed share in each open market is screened first. The app keeps liquid shares
  moving on unusual volume (at least 1.2× normal for the time of day), near the day's high (for buys)
  or low (for sells), then runs the full signal check on the best 15 each way, plus your watchlist.
- **BUY** = score of +2 or more (trend, momentum, VWAP, RSI agree upward). **SELL** = -2 or less.
  SELL means sell shares you own, or short-sell if your broker allows it (short selling is riskier).
- **Signal right %** = how often this exact rule called the direction correctly on this stock over the last
  ~10 trading days (**History days**, minimum 5; shares with less history are skipped).
  The table hides anything below your minimum.
- **Stop-loss** = 1.5 × the stock's normal 5-minute move away from the price. **Target** = twice that distance
  (risk $1 to make $2). With a 2:1 target you can be right only 40% of the time and still roughly break even before fees.
- **Shares** is sized so hitting the stop loses only your chosen risk %. Brokerage and currency fees are not included.
- ASX/NZX prices are ~20 min delayed, so use these only for US stocks during live trading.
- Practise on a paper (demo) account first. This is a technical screen, not financial advice.
""")

# ---------------------------------------------------------------- 1. Big movers
if VIEW == "movers":
    mv_day = st.segmented_control("Day", ["latest", "before"], default="latest", key="mv_day",
                                  format_func=lambda k: "Today / last session" if k == "latest" else "Day before",
                                  label_visibility="collapsed") or "latest"
    live_mk = open_markets(my_tz)

    @st.fragment(run_every=refresh if (mv_day == "latest" and live_mk) else None)
    def movers_view():
        for market in chosen_markets:
            sess_m = sessions_cached(market, 2, pd.Timestamp.now().strftime("%Y%m%d%H"))
            use_live = mv_day == "latest" and market in live_mk
            day = None if use_live else (sess_m[0] if mv_day == "latest" else (sess_m[1] if len(sess_m) > 1 else None))
            st.subheader(market + ("  ·  🟢 live" if use_live else f"  ·  {day_label(day)}" if day else ""))
            try:
                if use_live:
                    try:
                        df = big_movers(market, min_move, min_price, int(min_volume), max_rows)
                    except Exception:   # screener blocked (e.g. HTTP 401): use the main companies list instead
                        fb = fallback_live_cached(market)
                        df = fb if fb.empty else fb[(fb["Day %"].abs() >= min_move) & (fb["Price"] >= min_price)
                                                     ].rename(columns={"Day %": "Change %"}).sort_values(
                            "Change %", key=abs, ascending=False).head(max_rows * 2)[
                            ["Symbol", "Price", "Change %", "Day range %", "Volume", "Volume ×"]]
                        st.caption("Yahoo's market scanner is busy, so this uses each market's main companies.")
                elif day is None:
                    df = pd.DataFrame()
                else:
                    mv = session_movers_cached(market, day)
                    if mv.empty:
                        st.warning(f"Couldn't load {market} daily prices right now "
                                   f"({mv.attrs.get('error', 'no data')}). It will retry on the next refresh.")
                        continue
                    cols = ["Symbol", "Company", "Price", "Change %", "Day range %", "Volume", "Volume ×"]
                    ok = mv[(mv["Price"] >= min_price) & (mv["Volume"] >= min_volume)]
                    df = ok[ok["Change %"].abs() >= min_move]
                    st.caption(f"Checked {len(mv)} of {mv.attrs.get('universe', len(mv))} biggest companies "
                               f"for {day_label(day)}.")
                    if df.empty and not ok.empty:
                        st.info(f"No share moved {min_move:g}% or more. Biggest moves that day:")
                        df = pd.concat([ok.nlargest(5, "Change %"), ok.nsmallest(5, "Change %")]
                                       ).drop_duplicates("Symbol")
                    df = df.sort_values("Change %", key=abs, ascending=False).head(max_rows * 2)[cols] \
                        if not df.empty else df
            except Exception as e:
                st.warning(f"Couldn't load {market} movers right now ({e}).")
                continue
            if df.empty:
                st.info(f"No shares moved {min_move:g}% or more" + ("" if use_live else
                        " that day (checked the market's biggest companies)") + ". Try a smaller % in Settings.")
                continue
            up, down = df[df["Change %"] > 0], df[df["Change %"] < 0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Big risers", len(up))
            c2.metric("Big fallers", len(down))
            c3.metric("Biggest move", f"{df.iloc[0]['Symbol']}", f"{df.iloc[0]['Change %']:+.1f}%")
            show_table(df)
            pick = st.multiselect("Add to my watchlist", df["Symbol"].tolist(), key=f"add_{market}_{mv_day}")
            if pick and st.button("Add selected", key=f"btn_{market}_{mv_day}"):
                save_watchlist(list(dict.fromkeys(st.session_state.watchlist + pick)))
                st.toast(f"Added {', '.join(pick)}")
        st.caption(f"Up or down at least {min_move:g}%. Live markets use today's scan of every share; closed days use "
                   f"daily prices of each market's biggest companies · {now_local():%H:%M:%S}")
    movers_view()

# ---------------------------------------------------------------- 2. Watchlist
if VIEW == "watch":
    alert_pct = st.number_input("Highlight my stocks moving more than (%)", 0.5, 50.0, 3.0, 0.5)

    @st.fragment(run_every=refresh)
    def watch_view():
        wl = [s for s in st.session_state.watchlist if market_of(s) in chosen_markets]
        if not wl:
            st.info("Your watchlist is empty. Add companies from **Pick Companies** or **Big Movers**.")
            return
        df = quotes(tuple(wl))
        st.caption(f"{len(wl)} companies · updated {now_local():%H:%M:%S}")
        movers = df[df["Change %"].abs() >= alert_pct] if "Change %" in df else pd.DataFrame()
        if not movers.empty:
            st.warning("⚠️ Big moves: " + ", ".join(
                f"{r['Symbol']} {r['Change %']:+.1f}%" for _, r in movers.iterrows()))
        for market in chosen_markets:
            part = df[df["Market"] == market]
            if not part.empty:
                st.markdown(f"**{market}**")
                show_table(part.drop(columns="Market"), alert_pct)
    watch_view()

    st.divider()
    col1, col2 = st.columns([2, 1])
    with col1:
        if st.session_state.watchlist:
            chart_sym = st.selectbox("Price chart", st.session_state.watchlist)
            try:
                st.line_chart(intraday(chart_sym), y="Close", height=260)
            except Exception as e:
                st.warning(f"No chart data ({e}).")
    with col2:
        remove = st.multiselect("Remove from watchlist", st.session_state.watchlist)
        if remove and st.button("Remove"):
            save_watchlist([s for s in st.session_state.watchlist if s not in remove])
            st.rerun()


# ---------------------------------------------------------------- 3. Live chart
if VIEW == "chart":
    if not st.session_state.watchlist:
        st.info("Add companies to your watchlist first.")
    else:
        c1, c2 = st.columns([3, 1])
        sym = c1.selectbox("Company", st.session_state.watchlist, key="chart_sym")
        proj_min = c2.selectbox("Project ahead", [1, 2, 5, 10, 15], index=2, format_func=lambda m: f"{m} min")

        @st.fragment(run_every=refresh)
        def chart_view():
            try:
                raw = bars_1m(sym)
            except Exception as e:
                st.warning(f"No data for {sym} ({e}).")
                return
            if raw.empty:
                st.info("No intraday data yet (market closed or ticker not found).")
                return
            d = add_indicators(raw)
            last, prev_close = d.iloc[-1], d[d.index.date < d.index[-1].date()]["Close"]
            ref = prev_close.iloc[-1] if not prev_close.empty else d["Open"].iloc[0]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Price", f"{last['Close']:,.3f}", f"{(last['Close'] / ref - 1) * 100:+.2f}%")
            m2.metric("Last 5 min", f"{last['Mom5']:+.2f}%" if pd.notna(last["Mom5"]) else "–")
            m3.metric("RSI", f"{last['RSI']:.0f}" if pd.notna(last["RSI"]) else "–",
                      help="Above 70 = stretched up, below 30 = stretched down")
            m4.metric("Volume vs normal", f"{last['VolSurge']:.1f}×" if pd.notna(last["VolSurge"]) else "–")
            live_chart(sym, d, proj_min)
            st.caption(f"Last bar {d.index[-1]:%H:%M} exchange time · refreshed {now_local():%H:%M:%S}. "
                       "Grey cone = where a straight-line trend of the last 15 minutes would land, "
                       "with its normal noise. It is a guide, not a forecast.")
        chart_view()

# ---------------------------------------------------------------- 4. Signals
if VIEW == "signals":
    st.markdown("**Which of my stocks look ready to move in the next few minutes?**")
    horizon = st.radio("Look ahead", [1, 2, 5, 10, 15], index=2, horizontal=True,
                       format_func=lambda m: f"{m} min",
                       help="1-2 min signals are faster but closer to a coin flip; check Signal right %")

    @st.fragment(run_every=refresh)
    def signals_view():
        wl = tuple(s for s in st.session_state.watchlist if market_of(s) in chosen_markets)
        if not wl:
            st.info("Add companies to your watchlist first.")
            return
        rows = []
        for s, d in bars_many(wl).items():
            if d is not None and len(d) > 30:
                try:
                    rows.append(summarise(s, d, horizon))
                except Exception:
                    pass
        if not rows:
            st.info("No intraday data available right now.")
            return
        df = pd.DataFrame(rows)
        df["_rank"] = df["Score"].abs() * 10 + df["Activity ×"].fillna(0)
        df = df.sort_values("_rank", ascending=False).drop(columns="_rank")

        hot = df[(df["Activity ×"] >= 1.5) | (df["Score"].abs() >= 3)]
        if not hot.empty:
            st.warning("⚡ Watch now: " + ", ".join(f"{r.Symbol} ({r.Signal})" for r in hot.itertuples()))

        fmt = {"Price": "{:,.3f}", f"Next {horizon} min (trend)": "{:,.3f}", "Range low": "{:,.3f}",
               "Range high": "{:,.3f}", "Last 5 min %": "{:+.2f}%", "RSI": "{:.0f}",
               "Volume surge ×": "{:.1f}×", "Activity ×": "{:.1f}×", "Signal right %": "{:.0f}%", "History days": "{:.0f}",
               "Bar time": lambda t: f"{t:%H:%M}"}
        sig_color = lambda v: ("color:#16a34a;font-weight:600" if "Up" in str(v)
                               else "color:#dc2626;font-weight:600" if "Down" in str(v) else "")
        trust = lambda v: ("" if pd.isna(v) else "background-color:rgba(22,163,74,.18)" if v >= 55
                           else "background-color:rgba(220,38,38,.15)" if v < 50 else "")
        st.dataframe(df.style.format(fmt, na_rep="–").map(sig_color, subset=["Signal"])
                     .map(trust, subset=["Signal right %"]), hide_index=True, width="stretch")
        st.caption(f"Updated {now_local():%H:%M:%S}")
    signals_view()

    with st.expander("How to read this (please read)"):
        st.markdown("""
- **Signal / Score (-4 to +4)** adds up four things: short trend (EMA9 vs EMA21), the last 5 minutes' move,
  price above/below VWAP (today's average traded price), and RSI being stretched (which often snaps back).
  +2 or more = up bias, -2 or less = down bias.
- **Activity ×** compares the last 5 minutes' volume and price range with normal. Above 1.5× means the
  stock is waking up, and a bigger move is more likely, in either direction.
- **Signal right %** tests the same rule on every minute of the last ~10 trading days (never fewer than 5,
  shown in **History days**; with less history there is no signal at all): when it said up/down,
  how often was the price actually up/down N minutes later? **50% is a coin flip.** Only trust stocks
  where this stays clearly above 55% with plenty of tests, and it can change any day.
- ASX/NZX data is ~20 min delayed, so their "next 5 minutes" has already happened on the exchange.
  Signals are most useful for US stocks.
- This is a technical guide, not financial advice. Nobody can reliably predict the next move.
""")

# ---------------------------------------------------------------- 5. Pick companies
if VIEW == "pick":
    st.markdown("**Tick ⭐ the shares you want to follow, then tap Save.** No need to search: the list below "
                "shows each market's companies, largest first. These become **⚡ My Shares**.")
    if st.session_state.watchlist:
        st.caption("My shares now: " + ", ".join(st.session_state.watchlist))
    market = st.radio("Market", list(MARKETS), horizontal=True)
    try:
        all_cos = company_list(market)
    except Exception as e:
        all_cos = pd.DataFrame(columns=["Symbol", "Company", "Price", "Market cap"])
        st.warning(f"Couldn't load the company list ({e}). You can still add a ticker below.")

    search = st.text_input("Search (optional)", placeholder="e.g. Apple, BHP, Fisher & Paykel")
    view = all_cos
    if search:
        s = search.lower()
        view = all_cos[all_cos["Symbol"].str.lower().str.contains(s, regex=False)
                       | all_cos["Company"].str.lower().str.contains(s, regex=False)]
    st.caption(f"{len(view):,} of {len(all_cos):,} companies (largest first)")

    view = view.head(300).copy()
    view.insert(0, "Watch", view["Symbol"].isin(st.session_state.watchlist))
    edited = st.data_editor(
        view, hide_index=True, width="stretch", disabled=["Symbol", "Company", "Price", "Market cap"],
        column_config={"Watch": st.column_config.CheckboxColumn("⭐ Watch"),
                       "Market cap": st.column_config.NumberColumn(format="compact")},
        key=f"editor_{market}_{search}")
    n_ticked = int(edited["Watch"].sum()) if not edited.empty else 0
    b1, b2 = st.columns(2)
    if b2.button("Untick all in this market"):
        save_watchlist([x for x in st.session_state.watchlist if market_of(x) != market])
        st.rerun()
    if b1.button(f"💾 Save my selection ({n_ticked} ticked here)", type="primary"):
        ticked = set(edited.loc[edited["Watch"], "Symbol"])
        shown = set(edited["Symbol"])
        keep = [s for s in st.session_state.watchlist if s not in shown or s in ticked]
        save_watchlist(list(dict.fromkeys(keep + sorted(ticked))))
        st.success(f"Watchlist saved: {len(st.session_state.watchlist)} companies")

    with st.expander("Add a ticker manually"):
        st.caption("Use Yahoo symbols: AAPL (US), BHP.AX (Australia), AIR.NZ (New Zealand)")
        manual = st.text_input("Ticker").strip().upper()
        if manual and st.button("Add ticker"):
            save_watchlist(list(dict.fromkeys(st.session_state.watchlist + [manual])))
            st.success(f"Added {manual}")
