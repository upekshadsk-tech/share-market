"""Send phone push notifications through ntfy.sh (free Android/iPhone app, no account needed)."""

import requests


def send_push(topic: str, title: str, message: str, kind: str = "info", click_url: str | None = None) -> bool:
    if not topic:
        return False
    tags = {"BUY NOW": "green_circle,chart_with_upwards_trend",
            "SELL NOW": "large_blue_circle,chart_with_downwards_trend"}.get(kind, "bell")
    headers = {"Title": title.encode("utf-8"), "Tags": tags, "Priority": "high"}
    if click_url:
        headers["Click"] = click_url
    try:
        r = requests.post(f"https://ntfy.sh/{topic.strip()}", data=message.encode("utf-8"),
                          headers=headers, timeout=10)
        return r.ok
    except Exception:
        return False


def alert_text(r: dict) -> tuple[str, str]:
    """Title and body for a BUY NOW / SELL NOW signal row from signals.live_signal()."""
    icon = "🟢" if r["Action"] == "BUY NOW" else "🔵"
    title = f"{icon} {r['Action']}: {r['Symbol']} @ {r['Price']:,.3f}"
    lines = [f"Day {r['Day %']:+.2f}% · score {r['Score']:+d}"]
    if r.get("Right %") is not None:
        lines.append(f"Signal right {r['Right %']:.0f}% of {r['Tests']} past tests")
    if r.get("Stop-loss") is not None:
        lines.append(f"Stop {r['Stop-loss']:,.3f} · Target {r['Target']:,.3f}")
    lines.append("Not financial advice. Check the chart first.")
    return title, "\n".join(lines)
