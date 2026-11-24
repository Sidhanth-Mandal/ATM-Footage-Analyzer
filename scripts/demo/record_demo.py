"""Drive the web UI through the demo flow and capture it frame by frame.

Needs the API (8000), worker and Vite dev server (5173) running, an empty case list,
and ATMCCTV_DEMO_USER / ATMCCTV_DEMO_PASSWORD in the environment (see run_with_env.ps1).

    powershell -File scripts\\demo\\run_with_env.ps1 .venv\\Scripts\\python.exe scripts\\demo\\record_demo.py

Output: <out>/frames/*.jpg, <out>/timeline.json (frame times, speed-up ranges, captions).
Then run build_demo.py to cut the MP4 and the voice-over script.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import os
import re
import shutil
import time
from pathlib import Path

import pymupdf as fitz  # renders the downloaded PDF for the on-screen preview
from playwright.async_api import Locator, Page, async_playwright

BASE = "http://127.0.0.1:5173"
ROOT = Path(__file__).resolve().parents[2]
SRC_VIDEOS = sorted((ROOT / "samples" / "Video Demo").glob("*.mp4"))  # 1st = transaction, 2nd = gallery
# Friendly names shown in the UI (the originals are copied, not renamed).
NAMES = ["ATM01_CAM02_2026-11-24_211915.mp4", "ATM02_CAM01_2026-11-24_213000.mp4"]
CLIP1_START = "2026-11-24T21:19:14"   # DVR clock burned into the first frame of video 1
TXN_TIME = "2026-11-24T21:19:20"      # bank journal time of the withdrawal
VIEW_W, VIEW_H, DPR = 1536, 864, 1.25  # -> 1920x1080 frames

OVERLAY_JS = r"""
(() => {
  if (window.__demo) return;
  const css = `
  #__cur{position:fixed;left:0;top:0;width:26px;height:26px;z-index:2147483647;pointer-events:none;
         transform:translate(-3px,-2px);filter:drop-shadow(0 1px 2px rgba(0,0,0,.5));transition:opacity .2s}
  .__rip{position:fixed;width:34px;height:34px;margin:-17px 0 0 -17px;border-radius:50%;z-index:2147483646;
         pointer-events:none;border:3px solid rgba(56,189,248,.95);animation:__r .5s ease-out forwards}
  @keyframes __r{from{transform:scale(.3);opacity:1}to{transform:scale(1.6);opacity:0}}
  #__cap{position:fixed;left:50%;bottom:26px;transform:translateX(-50%);max-width:56%;z-index:2147483645;
         pointer-events:none;background:rgba(10,14,24,.9);color:#fff;font:600 19px/1.35 "Segoe UI",system-ui,sans-serif;
         padding:11px 20px;border-radius:12px;box-shadow:0 8px 30px rgba(0,0,0,.35);border:1px solid rgba(255,255,255,.12);
         text-align:center;transition:opacity .35s;opacity:0}
  #__cap .k{display:block;font:700 11px/1.2 "Segoe UI",sans-serif;letter-spacing:.14em;text-transform:uppercase;
            color:#38bdf8;margin-bottom:4px}
  #__card{position:fixed;inset:0;z-index:2147483644;display:flex;flex-direction:column;align-items:center;
          justify-content:center;gap:18px;background:radial-gradient(circle at 30% 20%,#13233d,#05070c 70%);
          color:#fff;font-family:"Segoe UI",system-ui,sans-serif;transition:opacity .6s;opacity:0;pointer-events:none}
  #__card h1{font-size:54px;font-weight:700;margin:0;letter-spacing:-.01em}
  #__card p{font-size:22px;color:#9fb3c8;margin:0;max-width:900px;text-align:center;line-height:1.45}
  #__card .tag{font-size:13px;letter-spacing:.3em;text-transform:uppercase;color:#38bdf8;font-weight:700}`;
  const add = () => {
    if (document.getElementById('__cur')) return;
    const st = document.createElement('style'); st.textContent = css; document.head.appendChild(st);
    const c = document.createElement('div'); c.id = '__cur';
    c.innerHTML = '<svg viewBox="0 0 24 24" width="26" height="26"><path d="M3 2l7.5 19 2.6-7.6L21 10.8z" fill="#fff" stroke="#111" stroke-width="1.6" stroke-linejoin="round"/></svg>';
    const p = JSON.parse(sessionStorage.getItem('__cur') || '[760,430]');
    c.style.left = p[0] + 'px'; c.style.top = p[1] + 'px';
    document.body.appendChild(c);
    const cap = document.createElement('div'); cap.id = '__cap'; document.body.appendChild(cap);
    const card = document.createElement('div'); card.id = '__card'; document.body.appendChild(card);
    const saved = sessionStorage.getItem('__capv');
    if (saved) window.__demo.caption(...JSON.parse(saved), true);
  };
  window.__demo = {
    caption(kicker, text, instant) {
      sessionStorage.setItem('__capv', JSON.stringify([kicker, text]));
      const cap = document.getElementById('__cap'); if (!cap) return;
      if (!text) { cap.style.opacity = 0; return; }
      cap.innerHTML = (kicker ? '<span class="k">' + kicker + '</span>' : '') + text;
      cap.style.transition = instant ? 'none' : 'opacity .35s'; cap.style.opacity = 1;
    },
    card(tag, title, sub) {
      const el = document.getElementById('__card');
      if (!title) { el.style.opacity = 0; return; }
      el.innerHTML = '<div class="tag">' + tag + '</div><h1>' + title + '</h1><p>' + sub + '</p>';
      el.style.opacity = 1;
    },
  };
  document.addEventListener('mousemove', e => {
    const c = document.getElementById('__cur'); if (!c) return;
    c.style.left = e.clientX + 'px'; c.style.top = e.clientY + 'px';
    sessionStorage.setItem('__cur', JSON.stringify([e.clientX, e.clientY]));
  }, true);
  document.addEventListener('mousedown', e => {
    const r = document.createElement('div'); r.className = '__rip';
    r.style.left = e.clientX + 'px'; r.style.top = e.clientY + 'px';
    document.body.appendChild(r); setTimeout(() => r.remove(), 600);
  }, true);
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', add); else add();
})();
"""


class Demo:
    def __init__(self, page: Page, out: Path):
        self.page = page
        self.out = out
        self.frames: list[tuple[float, str]] = []
        self.fast: list[dict] = []
        self.captions: list[dict] = []
        self.marks: dict[str, float] = {}
        self.mx, self.my = 760.0, 430.0
        self._n = 0

    # ---------- capture ----------
    async def start_capture(self) -> None:
        self.cdp = await self.page.context.new_cdp_session(self.page)
        (self.out / "frames").mkdir(parents=True, exist_ok=True)

        def on_frame(ev):
            self._n += 1
            name = f"{self._n:06d}.jpg"
            (self.out / "frames" / name).write_bytes(base64.b64decode(ev["data"]))
            self.frames.append((ev["metadata"]["timestamp"], name))
            asyncio.ensure_future(self.cdp.send("Page.screencastFrameAck", {"sessionId": ev["sessionId"]}))

        self.cdp.on("Page.screencastFrame", on_frame)
        await self.cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 92,
                                                     "maxWidth": int(VIEW_W * DPR), "maxHeight": int(VIEW_H * DPR)})

    async def stop_capture(self) -> None:
        await self.cdp.send("Page.stopScreencast")
        (self.out / "timeline.json").write_text(json.dumps(
            {"frames": self.frames, "fast": self.fast, "captions": self.captions, "marks": self.marks}, indent=1))

    def mark(self, name: str) -> None:
        self.marks[name] = time.time()

    async def keepalive(self) -> None:
        """Headless Chrome only emits frames on change; nudge the cursor so time keeps flowing."""
        await self.page.mouse.move(self.mx + 0.5, self.my)
        await self.page.mouse.move(self.mx, self.my)

    # ---------- overlay ----------
    async def say(self, kicker: str, text: str, vo: str = "") -> None:
        self.captions.append({"t": time.time(), "kicker": kicker, "text": text, "vo": vo})
        await self.page.evaluate("([k, t]) => window.__demo.caption(k, t)", [kicker, text])

    async def card(self, tag: str, title: str, sub: str, vo: str, hold: float) -> None:
        self.captions.append({"t": time.time(), "kicker": tag, "text": title, "vo": vo})
        await self.page.evaluate("([a, b, c]) => window.__demo.card(a, b, c)", [tag, title, sub])
        await self.wait(hold)
        await self.page.evaluate("() => window.__demo.card('', '', '')")
        await self.wait(0.7)

    async def wait(self, s: float) -> None:
        end = time.time() + s
        while time.time() < end:
            await self.page.wait_for_timeout(min(250, max(10, (end - time.time()) * 1000)))
            await self.keepalive()

    # ---------- pointer ----------
    async def show(self, loc: Locator) -> None:
        await loc.wait_for(state="visible", timeout=60_000)
        inview = await loc.evaluate(
            "e => { const r = e.getBoundingClientRect(); return r.top >= 70 && r.bottom <= innerHeight - 90; }")
        if not inview:
            await loc.evaluate("e => e.scrollIntoView({behavior: 'smooth', block: 'center'})")
            await self.wait(0.8)

    async def move(self, x: float, y: float) -> None:
        dist = math.hypot(x - self.mx, y - self.my)
        steps = max(6, min(40, int(dist / 22)))
        x0, y0 = self.mx, self.my
        for i in range(1, steps + 1):
            k = i / steps
            e = k * k * (3 - 2 * k)  # smoothstep
            await self.page.mouse.move(x0 + (x - x0) * e, y0 + (y - y0) * e)
            await self.page.wait_for_timeout(12)
        self.mx, self.my = x, y

    async def point(self, loc: Locator, dx: float = 0.5, dy: float = 0.5) -> tuple[float, float]:
        await self.show(loc)
        b = await loc.bounding_box()
        x, y = b["x"] + b["width"] * dx, b["y"] + b["height"] * dy
        await self.move(x, y)
        return x, y

    async def click(self, loc: Locator, pause: float = 0.35) -> None:
        x, y = await self.point(loc)
        await self.page.wait_for_timeout(120)
        await self.page.mouse.click(x, y)
        await self.wait(pause)

    async def type(self, loc: Locator, text: str, delay: int = 45) -> None:
        await self.click(loc, 0.15)
        await loc.press_sequentially(text, delay=delay)
        await self.wait(0.3)

    async def fill(self, loc: Locator, value: str) -> None:
        await self.click(loc, 0.2)
        await loc.fill(value)
        await self.wait(0.5)

    async def select(self, loc: Locator, **kw) -> None:
        await self.click(loc, 0.2)
        await loc.select_option(**kw)
        await self.wait(0.5)

    # ---------- waiting (shown sped-up in the final cut) ----------
    async def until(self, loc: Locator, label: str, timeout: float = 600, out_s: float = 2.5) -> None:
        t0 = time.time()
        deadline = t0 + timeout
        while True:
            try:
                if await loc.first.is_visible():
                    break
            except Exception:
                pass
            if time.time() > deadline:
                raise TimeoutError(f"timed out waiting for: {label}")
            await self.wait(0.5)
        self._record_wait(t0, label, out_s)

    async def until_js(self, js: str, arg, label: str, timeout: float = 600, out_s: float = 2.5) -> None:
        t0 = time.time()
        while not await self.page.evaluate(js, arg):
            if time.time() > t0 + timeout:
                raise TimeoutError(f"timed out waiting for: {label}")
            await self.wait(0.5)
        self._record_wait(t0, label, out_s)

    def _record_wait(self, t0: float, label: str, out_s: float) -> None:
        t1 = time.time()
        if t1 - t0 > out_s + 1:
            self.fast.append({"start": t0 + 0.5, "end": t1 - 0.3, "out": out_s, "label": label})
        print(f"  waited {t1 - t0:5.1f}s for {label}")


async def render_pdf_viewer(pdf: Path, out_dir: Path) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = []
    with fitz.open(pdf) as doc:
        for i, pg in enumerate(doc):
            pix = pg.get_pixmap(dpi=130)
            p = out_dir / f"page_{i + 1}.png"
            pix.save(p)
            imgs.append(base64.b64encode(p.read_bytes()).decode())
    pages = "".join(f'<img src="data:image/png;base64,{b}">' for b in imgs)
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{pdf.name}</title><style>
      body{{margin:0;background:#2b2f36;font-family:"Segoe UI",sans-serif}}
      .bar{{position:sticky;top:0;z-index:5;height:44px;display:flex;align-items:center;gap:14px;padding:0 18px;
            background:#1d2026;color:#e6e6e6;font-size:14px;box-shadow:0 2px 6px rgba(0,0,0,.4)}}
      .bar b{{font-weight:600}} .bar span{{color:#9aa3ad}}
      .pages{{display:flex;flex-direction:column;align-items:center;gap:18px;padding:22px 0 140px}}
      img{{width:820px;box-shadow:0 4px 18px rgba(0,0,0,.5);background:#fff}}
    </style></head><body><div class="bar"><b>{pdf.name}</b><span>{len(imgs)} page{'s' if len(imgs) != 1 else ''}
    · Downloads</span></div><div class="pages">{pages}</div></body></html>"""


async def run(out: Path, headed: bool) -> None:
    user, pw = os.environ["ATMCCTV_DEMO_USER"], os.environ["ATMCCTV_DEMO_PASSWORD"]
    tmp = out / "videos"
    tmp.mkdir(parents=True, exist_ok=True)
    vids = []
    for src, name in zip(SRC_VIDEOS[:2], NAMES):
        dst = tmp / name
        shutil.copyfile(src, dst)
        vids.append(dst)

    async with async_playwright() as p:
        browser = await p.chromium.launch(channel="chrome", headless=not headed,
                                          args=["--autoplay-policy=no-user-gesture-required", "--hide-scrollbars"])
        ctx = await browser.new_context(viewport={"width": VIEW_W, "height": VIEW_H}, device_scale_factor=DPR,
                                        locale="en-GB", accept_downloads=True)
        await ctx.add_init_script(OVERLAY_JS)
        page = await ctx.new_page()
        d = Demo(page, out)
        pdf_html: dict[str, str] = {}
        await page.route(f"{BASE}/__evidence_preview", lambda r: r.fulfill(content_type="text/html",
                                                                          body=pdf_html.get("html", "")))

        await page.goto(f"{BASE}/login")
        await page.wait_for_load_state("networkidle")
        await d.start_capture()
        d.mark("start")

        # ---- title ----
        await d.card("Investigator demo", "ATM CCTV Detection",
                     "From a disputed withdrawal to the customer's best face images, a court-ready evidence sheet, "
                     "and the same person found in other footage.",
                     "ATM CCTV Detection finds the person behind a disputed ATM withdrawal.",
                     4.5)

        # ---- login ----
        await d.say("Sign in", "Investigators sign in with their own account. Every action is audit-logged.",
                    "An investigator signs in. Every action is audit-logged.")
        await d.type(page.locator('input[autocomplete="username"]'), user)
        await d.type(page.locator('input[type="password"]'), pw, delay=25)
        await d.click(page.get_by_role("button", name="Sign in"), 1.2)

        # ---- create case ----
        await d.say("1 · New case", "Open a case for the disputed withdrawal.",
                    "First, we open a case for the disputed withdrawal.")
        await d.click(page.get_by_role("button", name="New case"), 0.5)
        await d.type(page.get_by_placeholder("e.g. FIR-2026-0412"), "FIR-2026-1124")
        await d.type(page.get_by_placeholder("Disputed withdrawal, ATM 01"), "Disputed withdrawal, ATM 01")
        await d.click(page.get_by_role("dialog").get_by_role("button", name="Create", exact=True), 1.0)
        await page.wait_for_url("**/cases/*")
        case_url = page.url

        # ---- upload video 1 ----
        await d.say("2 · Upload footage", "Upload the ATM camera clip as a <b>transaction clip</b>. "
                    "The original is hashed and stored read-only.",
                    "We upload the footage. The original is hashed and kept read-only.")
        selects = page.locator("select")
        await d.select(selects.nth(0), value="transaction")
        await d.select(selects.nth(1), label="KAGGLE-ATM-01 · fascia")
        drop = page.get_by_text("Drop video files here or click to choose")
        await d.point(drop)
        async with page.expect_file_chooser() as fc:
            await page.mouse.click(d.mx, d.my)
        await (await fc.value).set_files(str(vids[0]))
        await d.wait(1.5)
        await d.say("GPU processing", "The worker prepares the video and tracks every person on screen.",
                    "The GPU worker prepares the video and tracks every person in it.")
        row1 = page.locator("li.py-3", has_text=NAMES[0])
        await d.until(row1, "clip row", out_s=1)
        await d.point(row1, 0.6, 0.3)
        await d.until(row1.get_by_text("People found"), "tracking video 1", out_s=3)
        await d.wait(1.2)

        # ---- set the time ----
        await d.say("3 · Set the time", "Line up the DVR clock with the bank journal. "
                    "The clip's first frame reads <b>21:19:14</b>.",
                    "Next we set the time. The DVR clock in the first frame reads twenty-one nineteen fourteen.")
        await d.click(row1.get_by_role("link", name="Set up"), 1.2)
        frame = page.locator('img[src*="/frame"]').first
        await d.until(frame, "clip frame", out_s=1)
        await d.point(frame, 0.86, 0.08)
        await d.wait(2.2)
        start_in = page.locator('input[type="datetime-local"]').nth(0)
        await d.fill(start_in, CLIP1_START)
        await d.click(page.get_by_role("button", name="Set", exact=True), 0.6)
        await d.until(page.get_by_text("Set manually"), "time mapping saved", out_s=1)
        await d.point(page.get_by_text("Set manually"))
        await d.wait(1.5)

        # ---- transaction ----
        await d.say("Bank journal", "Enter the withdrawal from the bank journal: <b>21:19:20</b>, amount 20,000.",
                    "Then we enter the withdrawal from the bank journal: the time and the amount.")
        await d.type(page.get_by_placeholder("journal ref"), "TXN-884120")
        await d.fill(page.locator('input[type="datetime-local"]').nth(1), TXN_TIME)
        await d.type(page.locator('input[type="number"][step="0.01"]'), "20000")
        await d.click(page.get_by_role("button", name="Add transaction and rank candidates"), 0.8)
        await d.until(page.get_by_text("Candidates ready"), "ranking", out_s=1.5)
        await d.wait(0.8)

        # ---- select the person ----
        await d.click(page.get_by_role("link", name="Pick the customer").first, 1.0)
        await d.say("4 · Pick the customer", "People near the machine at that moment are ranked. "
                    "The investigator watches and decides.",
                    "The system ranks everyone who was at the machine around that time. The investigator watches and decides.")
        cand = page.locator('div[role="button"][aria-pressed]').first
        await d.until(cand, "candidates", out_s=1)
        await d.wait(0.8)
        await d.click(cand, 0.6)
        await page.evaluate("() => { const v = document.querySelector('video'); if (v) { v.muted = true; v.play(); } }")
        await d.point(page.locator("video").first, 0.5, 0.45)
        await d.wait(4.0)
        await page.evaluate("() => document.querySelector('video')?.pause()")
        await d.point(page.get_by_text("Who was on screen"))
        await d.wait(1.5)
        await d.click(page.get_by_role("button", name="Confirm person").first, 0.6)
        await d.type(page.get_by_placeholder("e.g. only person at the keypad during the dispense"),
                     "Only person at the machine during the dispense", delay=28)
        await d.click(page.get_by_role("button", name="Confirm and extract faces"), 0.5)
        await d.say("GPU processing", "Every frame of this person is scanned for faces and quality-scored.",
                    "Every frame of that person is scanned and each face is scored.")
        await d.wait(1.0)
        await d.until_js("""() => location.pathname.endsWith('/faces')
                              && !!document.querySelector('button[aria-label^="Open face"] img')""",
                         None, "dense face extraction", out_s=2.5)
        faces_url = page.url
        await d.wait(0.8)

        # ---- best images ----
        await d.say("5 · Best images", "The clearest frames are picked automatically and graded A, B or C.",
                    "The best face images are picked and graded automatically.")
        await d.move(VIEW_W * 0.5, VIEW_H * 0.55)
        await d.wait(2.0)
        await d.click(page.get_by_role("button", name="Details", exact=True).first, 1.0)
        await d.say("Quality details", "Each image keeps its full frame, timestamp and quality scores.",
                    "Each image keeps the full frame, the exact time, and why it scored the way it did.")
        await d.move(VIEW_W * 0.7, VIEW_H * 0.45)
        await d.wait(3.5)
        await d.click(page.locator('[aria-label="close"]').last, 0.6)

        # ---- enhancement ----
        await d.say("6 · Enhancement", "Make an enhanced copy for viewing. It is labelled and never used for matching.",
                    "We can make an enhanced copy. It's labelled and never used for matching.")
        await d.click(page.get_by_role("button", name="Enhance").first, 0.8)
        dlg = page.get_by_role("dialog")
        await d.select(dlg.locator("select"), value="clahe")
        await d.click(dlg.get_by_role("button", name="Create enhanced copy"), 0.4)
        await d.until(dlg.get_by_text("ENHANCED · NOT ORIGINAL"), "CLAHE enhance", out_s=1)
        first_src = await dlg.locator('img[src*="/media/enhanced/"]').first.get_attribute("src")
        await d.wait(2.0)
        await d.say("6 · Enhancement", "GFPGAN face restoration runs on the GPU, flagged as generative.",
                    "GFPGAN face restoration is also available, flagged as generative.")
        await d.select(dlg.locator("select"), value="gfpgan_v1.4_onnx")
        await d.click(dlg.get_by_role("button", name="Create enhanced copy"), 0.4)
        await d.until_js("""(old) => [...document.querySelectorAll('[role=dialog] img[src*="/media/enhanced/"]')]
                              .some(i => i.getAttribute('src') !== old && i.complete && i.naturalWidth > 0)""",
                         first_src, "GFPGAN enhance", out_s=1.5)
        await d.wait(3.0)
        await d.click(dlg.locator('[aria-label="close"]'), 0.6)

        # ---- export evidence ----
        await d.say("7 · Evidence export", "Build a PDF evidence sheet with hashes for every image.",
                    "Now we export the evidence: a PDF sheet, with a hash for every image.")
        step7 = page.locator('a[title="Step 7: Export"]')
        if await step7.count():
            await d.click(step7, 1.0)
        else:
            await page.goto(f"{case_url}/export")
        await d.until(page.get_by_role("button", name="Generate PDF"), "export page", out_s=1)
        await d.click(page.get_by_label("Include enhanced images (separate, labelled section)"), 0.5)
        await d.click(page.get_by_role("button", name="Generate PDF"), 0.5)
        dl_link = page.get_by_role("link", name="Download PDF").first
        await d.until(dl_link, "PDF export", out_s=1.5)
        await d.wait(0.6)
        href = await dl_link.get_attribute("href")
        m = re.search(r"/exports/(\d+)/download", href or "")
        pdf_name = f"evidence_{m.group(1)}.pdf" if m else "evidence.pdf"
        downloads: list = []
        ctx.on("page", lambda pg: pg.on("download", lambda dl: downloads.append(dl)))
        await d.click(dl_link, 2.0)
        pdf_path = out / "evidence.pdf"
        if downloads:
            await downloads[0].save_as(pdf_path)
            pdf_name = downloads[0].suggested_filename or pdf_name
        else:  # fall back to fetching the same URL
            print("  no browser download event; fetching the PDF directly")
            resp = await page.request.get(href if href.startswith("http") else BASE + href)
            pdf_path.write_bytes(await resp.body())
        for pg in ctx.pages[1:]:
            await pg.close()
        await page.bring_to_front()
        pdf_html["html"] = (await render_pdf_viewer(pdf_path, out / "evidence_pages")).replace(
            "<title>evidence.pdf", f"<title>{pdf_name}").replace("<b>evidence.pdf", f"<b>{pdf_name}")
        await d.say("Downloaded evidence", f"<b>{pdf_name}</b>: case details, best frames, quality grades and "
                    "SHA-256 hashes. Enhanced images sit in their own labelled section.",
                    "Here's the downloaded sheet: the case, the best frames with grades and times, and the hashes.")
        await page.goto(f"{BASE}/__evidence_preview")
        await d.wait(3.0)
        n_pages = len(list((out / "evidence_pages").glob("page_*.png")))
        for _ in range(min(6, max(1, 2 * (n_pages - 1)))):
            await page.evaluate("() => window.scrollBy({top: 540, behavior: 'smooth'})")
            await d.wait(1.7)

        # ---- video 2: face matching ----
        await page.goto(case_url)
        await d.say("8 · Face matching", "New footage: a different ATM, eleven minutes later. Upload it as a "
                    "<b>gallery clip</b> to search.",
                    "Next, face matching. We upload footage from another ATM as a gallery clip.")
        selects = page.locator("select")
        await d.until(selects.nth(0), "case page", out_s=1)
        await d.select(selects.nth(0), value="gallery")
        await d.select(selects.nth(1), label="KAGGLE-ATM-01 · fascia")
        drop = page.get_by_text("Drop video files here or click to choose")
        await d.point(drop)
        async with page.expect_file_chooser() as fc:
            await page.mouse.click(d.mx, d.my)
        await (await fc.value).set_files(str(vids[1]))
        await d.wait(1.5)
        await d.say("GPU processing", "Every face in the new clip is indexed for search.",
                    "Every face in the new clip is indexed for search.")
        row2 = page.locator("li.py-3", has_text=NAMES[1])
        await d.until(row2.get_by_text("Searchable"), "gallery indexing", out_s=3)
        await d.wait(1.2)

        await page.goto(faces_url)
        await d.say("8 · Face matching", "Select the customer's best searchable faces and search the other clips.",
                    "Back on the customer's faces, we select the best ones and search the other clips.")
        await d.until(page.locator('button[aria-label^="Open face"]'), "faces page", out_s=1)
        await d.wait(1.0)
        labels = await page.evaluate("""() => [...document.querySelectorAll('input[aria-label^="select face"]')]
            .filter(cb => [...cb.closest('div.rounded-card').querySelectorAll('span')]
                          .some(s => s.textContent.trim() === 'Searchable'))
            .map(cb => cb.getAttribute('aria-label')).slice(0, 3)""")
        if not labels:
            raise RuntimeError("no searchable faces on the customer's track")
        for lab in labels:
            await d.click(page.locator(f'input[aria-label="{lab}"]'), 0.3)
        await d.click(page.get_by_role("button", name="Search other clips with"), 1.0)
        await page.wait_for_url("**/search/*")
        await d.until(page.get_by_text("Search results"), "search results", out_s=1)
        await d.say("Match found", "The same person is found in the second clip. "
                    "A score is a lead for a human to verify, never an identification.",
                    "The same person turns up in the second clip. A score is a lead to check, never an identification.")
        hit = page.get_by_role("button", name="Play at moment").first
        await d.point(page.get_by_text("Searched with"))
        await d.wait(1.5)
        await d.point(page.get_by_text("Strong matches"))
        await d.wait(1.5)
        if await hit.count():
            await d.click(hit, 0.5)
            await d.wait(3.5)
            await d.click(page.get_by_role("dialog").locator('[aria-label="close"]'), 0.6)
            await d.say("Human review", "The investigator confirms the match, with a note, for the export.",
                        "The investigator confirms the match, with a note, for the export.")
            await d.type(page.get_by_placeholder("review note").first, "Same face and jacket, ATM-02 at 21:30", delay=28)
            await d.click(page.get_by_role("button", name="Possible match: confirm").first, 1.0)
            await d.wait(2.0)
        else:
            print("  WARNING: no match hits to show")
            await d.wait(3.0)

        await page.evaluate("() => window.__demo.caption('', '')")
        await d.card("ATM CCTV Detection", "Customer found. Evidence ready.",
                     "Case · Upload · Time sync · Pick the customer · Best faces · Enhance · Export · Face search",
                     "From one disputed withdrawal to verified evidence, in a few minutes.", 4.5)
        d.mark("end")
        await d.stop_capture()
        await browser.close()
    print(f"frames: {len(d.frames)}  fast ranges: {len(d.fast)}  captions: {len(d.captions)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(os.environ.get("TEMP", "/tmp")) / "atmdemo" / "rec"))
    ap.add_argument("--headed", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    asyncio.run(run(out, a.headed))
