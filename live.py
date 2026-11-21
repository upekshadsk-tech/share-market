"""
Live price stream from Yahoo Finance (push updates, usually within a second for US stocks).

One background connection is shared by the whole app. If it drops, it reconnects and
re-subscribes automatically. Symbols that do not stream (or while a market is closed)
simply have no live tick, and the app falls back to the latest 1-minute bar.
"""

import threading
import time

import yfinance as yf


class LiveFeed:
    def __init__(self):
        self.latest: dict[str, dict] = {}     # symbol -> last message (+ "_rx" receive time)
        self.symbols: set[str] = set()
        self._lock = threading.Lock()
        self._ws = None
        self._thread = None
        self.connected = False
        self.last_error = None

    # ------------------------------------------------------------ public
    def watch(self, symbols):
        symbols = {s.upper() for s in symbols}
        with self._lock:
            new = symbols - self.symbols
            self.symbols |= symbols
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        elif new and self._ws is not None:
            try:
                self._ws.subscribe(sorted(new))
            except Exception as e:
                self.last_error = str(e)

    def get(self, symbol: str, max_age: float = 120.0) -> dict | None:
        m = self.latest.get(symbol.upper())
        if m and time.time() - m["_rx"] <= max_age:
            return m
        return None

    # ------------------------------------------------------------ internals
    def _on_message(self, msg: dict):
        sym = msg.get("id")
        if sym and "price" in msg:
            msg["_rx"] = time.time()
            self.latest[sym.upper()] = msg

    def _run(self):
        while True:
            try:
                self._ws = yf.WebSocket(verbose=False)
                with self._lock:
                    syms = sorted(self.symbols)
                if syms:
                    self._ws.subscribe(syms)
                self.connected = True
                self._ws.listen(self._on_message)   # blocks until the connection drops
            except Exception as e:
                self.last_error = str(e)
            finally:
                self.connected = False
                try:
                    self._ws.close()
                except Exception:
                    pass
            time.sleep(5)  # wait, then reconnect
