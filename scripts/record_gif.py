"""Record docs/screenshots/console.gif: Playground classify, Evaluate (sample), Ops live charts (dark).

    pip install playwright pillow       # uses an installed Edge or Chrome; no browser download needed
    python scripts/record_gif.py --out docs/screenshots/console.gif

Run against a LIVE server with the real model. The script drives the console like a user and grabs frames
with page.screenshot at a fixed rate (Playwright's own video is WebM, so frames are taken directly), then
writes one GIF with a single shared palette (no flicker between frames, small file). A background thread sends
a little classify traffic so the Ops charts move.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path

TICKET = "Checkout is down for every customer, orders have been blocked since 03:00."
LABELS = "billing, technical, account"
VIEW = (1280, 720)  # captured size; the GIF is scaled down to --width


class Recorder:
    def __init__(self, page, fps: float) -> None:
        self.page, self.dt, self.frames = page, 1.0 / fps, []
        self.t0 = time.perf_counter()

    def grab(self) -> None:
        self.frames.append((time.perf_counter() - self.t0, self.page.screenshot(type="png")))

    def hold(self, seconds: float) -> None:
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            start = time.perf_counter()
            self.grab()
            time.sleep(max(0.0, self.dt - (time.perf_counter() - start)))

    def until(self, cond_js: str, timeout_s: float = 120) -> None:
        end = time.perf_counter() + timeout_s
        while time.perf_counter() < end and not self.page.evaluate(cond_js):
            start = time.perf_counter()
            self.grab()
            time.sleep(max(0.0, self.dt - (time.perf_counter() - start)))

    def type(self, locator, text: str, per_char_s: float = 0.025) -> None:
        locator.fill("")
        for i in range(0, len(text), 2):
            locator.fill(text[: i + 2])
            self.hold(per_char_s)


def traffic(url: str, stop: threading.Event) -> None:
    body = json.dumps({"input": TICKET, "labels": ["billing", "technical", "account"]}).encode()
    while not stop.is_set():
        try:
            req = urllib.request.Request(url + "/v1/classify", body, {"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=30).read()
        except Exception:  # noqa: BLE001 - background noise only
            time.sleep(0.5)


def to_gif(frames, out: Path, fps: float, width: int, max_mb: float) -> None:
    from PIL import Image

    # resample to a fixed rate: the frame nearest to each tick
    total = frames[-1][0]
    ticks = int(total * fps)
    picked, j = [], 0
    for k in range(ticks):
        t = k / fps
        while j + 1 < len(frames) and abs(frames[j + 1][0] - t) <= abs(frames[j][0] - t):
            j += 1
        picked.append(frames[j][1])
    images = []
    for raw in picked:
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        h = round(im.height * width / im.width)
        images.append(im.resize((width, h), Image.LANCZOS))
    # one palette from a spread of frames
    step = max(1, len(images) // 12)
    sheet = Image.new("RGB", (width, images[0].height * len(images[::step])))
    for n, im in enumerate(images[::step]):
        sheet.paste(im, (0, n * im.height))
    pal = sheet.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    quant = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in images]
    out.parent.mkdir(parents=True, exist_ok=True)
    quant[0].save(
        out,
        save_all=True,
        append_images=quant[1:],
        duration=round(1000 / fps),
        loop=0,
        optimize=True,
        disposal=1,
    )
    size = out.stat().st_size / 1e6
    print(
        f"wrote {out} ({len(quant)} frames, {len(quant) / fps:.1f}s, "
        f"{width}x{images[0].height}, {size:.2f} MB)"
    )
    if size > max_mb:
        print(f"warning: larger than {max_mb} MB; lower --fps / --width", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("CLEF_URL", "http://127.0.0.1:8910").rstrip("/"))
    ap.add_argument("--out", type=Path, default=Path("docs/screenshots/console.gif"))
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--width", type=int, default=1000)
    ap.add_argument("--max-mb", type=float, default=6.0)
    ap.add_argument("--channel", default="msedge")
    ap.add_argument("--theme", default="dark")
    args = ap.parse_args()
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("pip install playwright pillow", file=sys.stderr)
        return 2
    stop = threading.Event()
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel=args.channel)
        ctx = browser.new_context(
            viewport={"width": VIEW[0], "height": VIEW[1]}, color_scheme=args.theme, device_scale_factor=1
        )
        key = os.environ.get("CLEF_API_KEY")
        if key:
            ctx.add_init_script(f"localStorage.setItem('clef.apiKey', {key!r})")
        page = ctx.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        rec = Recorder(page, args.fps)
        busy = "() => [...document.querySelectorAll('button')].some(b => b.textContent.includes('Running'))"

        # 1. Playground, Classify mode
        page.goto(f"{args.url}/#playground")
        page.wait_for_function("() => document.body.innerText.includes('Ready')", timeout=120_000)
        page.get_by_role("radio", name="Classify").click()
        page.wait_for_timeout(300)
        rec.hold(0.8)
        box = page.get_by_label("Input to classify")
        rec.type(box, TICKET)
        rec.hold(0.4)
        page.get_by_role("button", name="Classify", exact=False).filter(has_text="Classify").last.click()
        rec.until(f"() => !({busy[6:]})", 60)
        rec.hold(2.2)

        # 2. Evaluate: load the sample, run it, scroll through the charts
        page.get_by_role("button", name="Evaluate").click()
        page.evaluate("window.scrollTo(0, 0)")
        rec.hold(0.8)
        page.get_by_role("button", name="Load sample").click()
        rec.hold(1.0)
        page.get_by_role("button", name="Run evaluation").click()
        page.evaluate("window.scrollTo(0, 0)")  # the Run button sits below the fold: undo the auto-scroll
        rec.until("() => document.body.innerText.includes('Confusion matrix')", 120)
        rec.hold(1.6)
        for _ in range(14):  # smooth scroll of the results column
            page.evaluate("document.querySelector('.pane.right').scrollBy(0, 38)")
            rec.hold(0.1)
        rec.hold(1.2)

        # 3. Ops: live charts
        threading.Thread(target=traffic, args=(args.url, stop), daemon=True).start()
        page.get_by_role("button", name="Ops").click()
        page.evaluate("window.scrollTo(0, 0)")
        rec.hold(4.5)
        stop.set()
        for err in errors:
            print("page error:", err, file=sys.stderr)
        ctx.close()
        browser.close()
    to_gif(rec.frames, args.out, args.fps, args.width, args.max_mb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
