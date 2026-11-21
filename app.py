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
from yfinance import EquityQuery as Q
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from signals import add_indicators, summarise, trade_idea

st.set_page_config(page_title="Stock Tracker US / AU / NZ", page_icon="📈", layout="wide")

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
    res = yf.screen(query, sortField=sort_field, sortAsc=sort_asc, size=size, offset=offset)
    return res.get("quotes", []) if isinstance(res, dict) else []


def _base_query(market: str):
    cfg = MARKETS[market]
    return Q("and", [
        Q("eq", ["region", cfg["region"]]),
        Q("is-in", ["exchange", *cfg["exchanges"]]),
    ])


@st.cache_data(ttl=24 * 3600, show_spinner="Loading full company list (once a day)...")
def company_list(market: str) -> pd.DataFrame:
    """Every listed equity in a market, paged from Yahoo's screener."""
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
        "Volume": "{:,.0f}", "Market cap": "{:,.0f}",
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
    my_tz = st.selectbox("My time zone", ["Pacific/Auckland", "Australia/Sydney", "Australia/Perth",
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
    st.caption("US prices are near real-time. ASX and NZX are delayed about 20 minutes.")

st.title("📈 Stock Tracker: USA · Australia · New Zealand")

@st.fragment(run_every=30)
def status_view():
    market_status_bar(my_tz)
    live = open_markets(my_tz)
    st.caption("Live now: " + (", ".join(live) if live else "no market is open right now"))
status_view()

tab_ideas, tab_movers, tab_watch, tab_chart, tab_signals, tab_pick = st.tabs(
    ["🎯 Trade Ideas", "🔥 Big Movers", "⭐ My Watchlist", "📊 Live Chart", "🔮 Signals", "➕ Pick Companies"])

# ---------------------------------------------------------------- 1-minute bars
@st.cache_data(ttl=10, show_spinner=False)
def bars_1m(symbol: str) -> pd.DataFrame:
    """Last 5 trading days of 1-minute bars (Yahoo's limit for 1m data is ~7 days)."""
    h = yf.Ticker(symbol).history(period="5d", interval="1m", prepost=False)
    return h[["Open", "High", "Low", "Close", "Volume"]].dropna() if not h.empty else h


def bars_many(symbols: tuple[str, ...]) -> dict:
    def one(s):
        try:
            return s, bars_1m(s)
        except Exception:
            return s, pd.DataFrame()
    with ThreadPoolExecutor(max_workers=10) as pool:
        return dict(pool.map(one, symbols))


# ---------------------------------------------------------------- 0. Trade ideas
@st.cache_data(ttl=120, show_spinner=False)
def scan_universe(markets: tuple[str, ...], watchlist: tuple[str, ...]) -> list[str]:
    """Watchlist + today's most active and biggest-moving stocks in each chosen market."""
    syms = list(watchlist)
    for m in markets:
        try:
            base = [_base_query(m), Q("gt", ["intradayprice", 1]), Q("gt", ["dayvolume", 200_000])]
            for sort_field, asc in (("dayvolume", False), ("percentchange", False), ("percentchange", True)):
                syms += [r["symbol"] for r in _screen_rows(Q("and", base), sort_field, asc, size=10)]
        except Exception:
            pass
    return list(dict.fromkeys(syms))[:45]


with tab_ideas:
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
               "Bar time": lambda t: f"{t:%H:%M}"}
        cols = ["Symbol", "Market", "Price", "Stop-loss", "Target", "Shares", "Risk $", "Reward $",
                "Signal right %", "Tests", "Activity ×", "RSI", "$ traded / min", "Quality", "Bar time"]
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
                        f"right {top['Signal right %']:.0f}% of {top.Tests} past tests</div>",
                        unsafe_allow_html=True)
            st.dataframe(part[cols].style.format(fmt, na_rep="–"), hide_index=True, width="stretch")
        st.caption(f"Scanned {len(df)} stocks, {len(good)} passed your filters · updated {datetime.now():%H:%M:%S}")

    ideas_view()
    with st.expander("How these are picked (read before trading)"):
        st.markdown("""
- **Scanned:** your watchlist plus the 10 most traded, 10 biggest risers and 10 biggest fallers in each market
  (priced over $1, over 200k shares traded today, so you can get in and out easily).
- **BUY** = score of +2 or more (trend, momentum, VWAP, RSI agree upward). **SELL** = -2 or less.
  SELL means sell shares you own, or short-sell if your broker allows it (short selling is riskier).
- **Signal right %** = how often this exact rule called the direction correctly on this stock over the last 5 days.
  The table hides anything below your minimum.
- **Stop-loss** = 1.5 × the stock's normal 5-minute move away from the price. **Target** = twice that distance
  (risk $1 to make $2). With a 2:1 target you can be right only 40% of the time and still roughly break even before fees.
- **Shares** is sized so hitting the stop loses only your chosen risk %. Brokerage and currency fees are not included.
- ASX/NZX prices are ~20 min delayed, so use these only for US stocks during live trading.
- Practise on a paper (demo) account first. This is a technical screen, not financial advice.
""")

# ---------------------------------------------------------------- 1. Big movers
with tab_movers:
    @st.fragment(run_every=refresh)
    def movers_view():
        st.caption(f"Stocks up or down at least {min_move:g}% today · updated {datetime.now():%H:%M:%S}")
        for market in chosen_markets:
            st.subheader(market)
            try:
                df = big_movers(market, min_move, min_price, int(min_volume), max_rows)
            except Exception as e:
                st.warning(f"Couldn't load {market} movers right now ({e}).")
                continue
            if df.empty:
                st.info("No stocks past the threshold (the market may be closed or quiet).")
                continue
            up, down = df[df["Change %"] > 0], df[df["Change %"] < 0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Big risers", len(up))
            c2.metric("Big fallers", len(down))
            c3.metric("Biggest move", f"{df.iloc[0]['Symbol']}", f"{df.iloc[0]['Change %']:+.1f}%")
            show_table(df)
            pick = st.multiselect("Add to my watchlist", df["Symbol"].tolist(), key=f"add_{market}")
            if pick and st.button("Add selected", key=f"btn_{market}"):
                save_watchlist(list(dict.fromkeys(st.session_state.watchlist + pick)))
                st.toast(f"Added {', '.join(pick)}")
    movers_view()

# ---------------------------------------------------------------- 2. Watchlist
with tab_watch:
    alert_pct = st.number_input("Highlight my stocks moving more than (%)", 0.5, 50.0, 3.0, 0.5)

    @st.fragment(run_every=refresh)
    def watch_view():
        wl = [s for s in st.session_state.watchlist if market_of(s) in chosen_markets]
        if not wl:
            st.info("Your watchlist is empty. Add companies from **Pick Companies** or **Big Movers**.")
            return
        df = quotes(tuple(wl))
        st.caption(f"{len(wl)} companies · updated {datetime.now():%H:%M:%S}")
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


def live_chart(symbol: str, d: pd.DataFrame, proj_minutes: int):
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
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10), xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", y=1.08), title=f"{symbol} · today, 1-minute bars")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


# ---------------------------------------------------------------- 3. Live chart
with tab_chart:
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
            st.caption(f"Last bar {d.index[-1]:%H:%M} exchange time · refreshed {datetime.now():%H:%M:%S}. "
                       "Grey cone = where a straight-line trend of the last 15 minutes would land, "
                       "with its normal noise. It is a guide, not a forecast.")
        chart_view()

# ---------------------------------------------------------------- 4. Signals
with tab_signals:
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
               "Volume surge ×": "{:.1f}×", "Activity ×": "{:.1f}×", "Signal right % (5d)": "{:.0f}%",
               "Bar time": lambda t: f"{t:%H:%M}"}
        sig_color = lambda v: ("color:#16a34a;font-weight:600" if "Up" in str(v)
                               else "color:#dc2626;font-weight:600" if "Down" in str(v) else "")
        trust = lambda v: ("" if pd.isna(v) else "background-color:rgba(22,163,74,.18)" if v >= 55
                           else "background-color:rgba(220,38,38,.15)" if v < 50 else "")
        st.dataframe(df.style.format(fmt, na_rep="–").map(sig_color, subset=["Signal"])
                     .map(trust, subset=["Signal right % (5d)"]), hide_index=True, width="stretch")
        st.caption(f"Updated {datetime.now():%H:%M:%S}")
    signals_view()

    with st.expander("How to read this (please read)"):
        st.markdown("""
- **Signal / Score (-4 to +4)** adds up four things: short trend (EMA9 vs EMA21), the last 5 minutes' move,
  price above/below VWAP (today's average traded price), and RSI being stretched (which often snaps back).
  +2 or more = up bias, -2 or less = down bias.
- **Activity ×** compares the last 5 minutes' volume and price range with normal. Above 1.5× means the
  stock is waking up, and a bigger move is more likely, in either direction.
- **Signal right % (5d)** tests the same rule on every minute of the last 5 days: when it said up/down,
  how often was the price actually up/down N minutes later? **50% is a coin flip.** Only trust stocks
  where this stays clearly above 55% with plenty of tests, and it can change any day.
- ASX/NZX data is ~20 min delayed, so their "next 5 minutes" has already happened on the exchange.
  Signals are most useful for US stocks.
- This is a technical guide, not financial advice. Nobody can reliably predict the next move.
""")

# ---------------------------------------------------------------- 5. Pick companies
with tab_pick:
    market = st.radio("Market", list(MARKETS), horizontal=True)
    try:
        all_cos = company_list(market)
    except Exception as e:
        all_cos = pd.DataFrame(columns=["Symbol", "Company", "Price", "Market cap"])
        st.warning(f"Couldn't load the company list ({e}). You can still add a ticker below.")

    search = st.text_input("Search by name or ticker", placeholder="e.g. Apple, BHP, Fisher & Paykel")
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
    if st.button("Save my selection", type="primary"):
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
