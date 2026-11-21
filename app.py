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

import pandas as pd
import streamlit as st
import yfinance as yf
from yfinance import EquityQuery as Q

st.set_page_config(page_title="Stock Tracker US / AU / NZ", page_icon="📈", layout="wide")

WATCHLIST_FILE = Path(__file__).with_name("watchlist.json")

MARKETS = {
    "USA": {"region": "us", "exchanges": ["NMS", "NYQ", "NGM", "NCM", "ASE"], "tz": "America/New_York"},
    "Australia (ASX)": {"region": "au", "exchanges": ["ASX"], "tz": "Australia/Sydney"},
    "New Zealand (NZX)": {"region": "nz", "exchanges": ["NZE"], "tz": "Pacific/Auckland"},
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


def market_clocks():
    cols = st.columns(len(MARKETS))
    for col, (name, cfg) in zip(cols, MARKETS.items()):
        col.caption(f"**{name}** · {datetime.now(ZoneInfo(cfg['tz'])):%a %H:%M}")


# ---------------------------------------------------------------- sidebar settings
with st.sidebar:
    st.header("Settings")
    chosen_markets = st.multiselect("Markets", list(MARKETS), default=list(MARKETS))
    refresh = st.select_slider("Auto-refresh every", [10, 15, 30, 60, 120], value=30,
                               format_func=lambda s: f"{s}s")
    st.subheader("Big-mover filter")
    min_move = st.slider("Price move of at least (%)", 1.0, 30.0, 5.0, 0.5)
    min_price = st.number_input("Ignore stocks priced under", 0.0, 100.0, 0.10, 0.05,
                                help="Hides penny stocks, which swing wildly on tiny trades")
    min_volume = st.number_input("Minimum volume today", 0, 10_000_000, 50_000, 10_000)
    max_rows = st.slider("Max stocks per direction", 10, 100, 25, 5)
    st.caption("US prices are near real-time. ASX and NZX are delayed about 20 minutes.")

st.title("📈 Stock Tracker: USA · Australia · New Zealand")
market_clocks()

tab_movers, tab_watch, tab_pick = st.tabs(["🔥 Big Movers", "⭐ My Watchlist", "➕ Pick Companies"])

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

# ---------------------------------------------------------------- 3. Pick companies
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
