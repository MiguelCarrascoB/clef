"""Capture Clef Console screenshots (each screen, light + dark) against a LIVE server.

    pip install playwright            # uses an installed Edge or Chrome; no browser download needed
    python bench/http_bench.py --concurrency 2,4 --requests 400 --mixed &   # real traffic for the Ops charts
    python scripts/screenshots.py --out docs/screenshots

The script drives the UI like a user: runs the SystemOne example twice (so the previous-run overlay shows),
classifies a ticket twice, runs a small batch, then captures Playground, Classify, Batch, History and Ops.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

TICKETS = """id,text
1,"Checkout is down, orders blocked since 03:00"
2,"How do I change the billing address on my invoice?"
3,"I was charged twice for the same order"
4,"The app crashes when I upload a photo"
5,"Can I get a refund for last month?"
6,"Password reset email never arrives"
7,"Is there a discount for annual plans?"
8,"API returns 500 on every request"
"""

STATES = (
    "Customer asks whether the invoice for September can be sent to a different email address.",
    "The login page shows a blank screen for some users since this morning; others can sign in.",
    "Payments fail for every customer at checkout with error 502; orders are blocked.",
)


def capture(page, url: str, out: Path, theme: str, run_timeout_ms: int) -> None:
    def shot(name: str) -> None:
        path = out / f"{name}-{theme}.png"
        page.screenshot(path=str(path), full_page=True)
        print("wrote", path)

    def wait_idle() -> None:
        # Buttons read "Running…" while a request is in flight
        page.wait_for_function(
            "() => ![...document.querySelectorAll('button')].some(b => b.textContent.includes('Running'))",
            timeout=run_timeout_ms,
        )

    page.goto(f"{url}/#playground")
    page.wait_for_function("() => document.body.innerText.includes('Ready')", timeout=run_timeout_ms)

    # Playground, SystemOne mode: a few different tickets (History trend), the example last so the
    # previous-run overlay compares two real runs
    page.get_by_role("radio", name="SystemOne").click()
    state = page.get_by_label("State", exact=True)
    example = state.input_value()
    for text in (*STATES, example):
        state.fill(text)
        page.get_by_role("button", name="Run", exact=False).filter(has_text="Run").first.click()
        page.wait_for_timeout(300)
        wait_idle()
    page.wait_for_timeout(600)  # bar animations
    shot("playground")

    # Classify mode: run twice with a different input for the overlay
    page.get_by_role("radio", name="Classify").click()
    box = page.get_by_label("Input to classify")
    for text in (
        "The checkout page shows an error and orders are blocked.",
        "Checkout is down, orders blocked",
    ):
        box.fill(text)
        page.get_by_role("button", name="Classify", exact=False).filter(has_text="Classify").last.click()
        page.wait_for_timeout(300)
        wait_idle()
    page.wait_for_timeout(600)
    shot("classify")
    page.get_by_role("radio", name="SystemOne").click()

    # Batch: paste a small CSV and run it with the current schema
    page.goto(f"{url}/#batch")
    page.get_by_label("Batch input").fill(TICKETS)
    page.get_by_role("button", name="Run batch").click()
    page.wait_for_function(
        "() => [...document.querySelectorAll('button')].some(b => b.textContent.trim() === 'Run batch')",
        timeout=run_timeout_ms,
    )
    page.wait_for_timeout(800)
    shot("batch")

    page.goto(f"{url}/#history")
    page.wait_for_timeout(800)
    shot("history")

    page.goto(f"{url}/#ops")
    page.wait_for_timeout(6000)  # SSE connect + a couple of live points
    shot("ops")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("CLEF_URL", "http://127.0.0.1:8910").rstrip("/"))
    ap.add_argument("--out", type=Path, default=Path("docs/screenshots"))
    ap.add_argument("--themes", default="light,dark")
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--height", type=int, default=1000)
    ap.add_argument("--channel", default="msedge", help="installed browser: msedge, chrome, ...")
    ap.add_argument("--timeout", type=int, default=120, help="seconds per model run")
    args = ap.parse_args()
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("pip install playwright", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    key = os.environ.get("CLEF_API_KEY")
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel=args.channel)
        for theme in args.themes.split(","):
            ctx = browser.new_context(
                viewport={"width": args.width, "height": args.height},
                color_scheme=theme,
                device_scale_factor=1,
            )
            if key:  # the console keeps the key in localStorage
                ctx.add_init_script(f"localStorage.setItem('clef.apiKey', {key!r})")
            page = ctx.new_page()
            errors: list[str] = []
            page.on("pageerror", lambda exc, errors=errors: errors.append(str(exc)))
            capture(page, args.url, args.out, theme, args.timeout * 1000)
            for err in errors:
                print(f"[{theme}] page error: {err}", file=sys.stderr)
            ctx.close()
        browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
