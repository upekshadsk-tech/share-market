# Stock Tracker: USA · Australia (ASX) · New Zealand (NZX)

## What it does
- **⚡ Buy / Sell Now**: green BUY and light-blue SELL cards for your watchlist, refreshing every second, with phone alerts (ntfy).
- **📒 Signal Log**: every signal with its time and price, checked against the price 5/10/15 min later and whether the target or stop-loss was hit first.
- **🎯 Trade Ideas**: scans every share in the open markets for the best BUY/SELL setups, with stop-loss, target and position size.
- **🔥 Big Movers**, **⭐ Watchlist**, **📊 Live Chart**, **🔮 Signals**, **➕ Pick Companies**.
- **Background checker** (GitHub Actions, `.github/workflows/alerts.yml`): during market hours it watches `alert_symbols.txt` and scans all of USA, ASX and NZX every 5 minutes, sends 🚀 buy suggestions to your phone, and saves every signal to the `signal-log` branch.

## Signal rules
- A signal needs **at least 5 complete past trading days** of 1-minute data (the app loads about 10). With less, it shows "not enough history" and never BUY/SELL.
- BUY/SELL only shows if the same rule was right at least your minimum % (default 55%) on that share over that history.

## Setup checklist
1. Streamlit Community Cloud: deploy `app.py` from this repo.
2. ntfy app on your phone, subscribed to your private topic.
3. Repo Settings > Secrets and variables > Actions: secret `NTFY_TOPIC` = your topic. Optional variable `APP_URL`.
4. Repo Settings > Actions > General > Workflow permissions: **Read and write**.

## About the data
- Yahoo Finance via `yfinance` (free, unofficial, may rate-limit). US prices near real-time; **ASX/NZX delayed ~20 min**.
- Technical signals only. Not financial advice.
