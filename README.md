# Stock Tracker: USA · Australia (ASX) · New Zealand (NZX)

## What it does
1. **🔥 Big Movers** (first tab): stocks that moved more than X% today in each market, risers and fallers. Penny stocks and thinly traded stocks are filtered out (you can change this in Settings). Tick any of them to add them to your watchlist.
2. **⭐ My Watchlist**: live prices for only the companies you picked. It refreshes on its own (every 10 to 120 s), highlights any stock moving more than your alert %, and shows an intraday chart.
3. **➕ Pick Companies**: search the full list of listed companies in each market by name or ticker, tick ⭐ Watch, then tap **Save my selection**.

Ticker format: `AAPL` (US), `BHP.AX` (ASX), `AIR.NZ` (NZX).

## Run it on your computer
```bash
pip install -r requirements.txt
streamlit run app.py
```
Then open http://localhost:8501.

## Use it on your phone (free hosting)
1. Create a free GitHub account and a new repository. Upload `app.py` and `requirements.txt`.
2. Go to https://share.streamlit.io, sign in with GitHub, choose **Create app**, then pick the repository and `app.py`.
3. You'll get a link like `https://yourname-stocks.streamlit.app`. Open it on your phone and use **Add to Home Screen**.

Your watchlist is saved in the page link (`?w=AAPL,BHP.AX,...`), so bookmark the link after you pick your companies.

## About the data
- The data comes from Yahoo Finance through the `yfinance` library. It's free and needs no API key, but it's unofficial, so Yahoo can rate-limit it or change it.
- **US** prices are near real-time. **ASX and NZX are delayed about 20 minutes.** True real-time ASX/NZX prices need a paid exchange data feed.
- The full company lists are loaded once a day (the first load takes about 30 seconds for the US).
- Prices are for information only. This app is not financial advice.
