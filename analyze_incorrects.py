#!/usr/bin/env python3
"""
analyze_incorrects.py — Identify your UWorld weak spots from incorrect question IDs.

HOW TO GET YOUR INCORRECT QUESTION IDs:
  1. Log into UWorld
  2. Go to Create Test → filter by "Incorrect" questions
  3. OR go to My Performance → Incorrect
  4. The question "index" numbers (e.g. 2143, 5678) are your question IDs
  5. Copy them as a comma-separated list and paste when prompted

  OR: Run with --auto to have this script scrape your incorrects from UWorld automatically.

USAGE:
  python analyze_incorrects.py                    # prompts for IDs
  python analyze_incorrects.py --ids 1234,5678    # pass IDs directly
  python analyze_incorrects.py --auto             # auto-scrape from UWorld (Playwright)
"""

import json, sys, argparse, os, re
from pathlib import Path
from collections import Counter, defaultdict

LIBRARY_PATH = Path("uworld_library.json")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ids", help="Comma-separated incorrect question IDs")
    p.add_argument("--auto", action="store_true", help="Auto-scrape incorrects from UWorld")
    return p.parse_args()


def load_library():
    if not LIBRARY_PATH.exists():
        print("ERROR: uworld_library.json not found. Run build_library.py first.")
        sys.exit(1)
    return json.loads(LIBRARY_PATH.read_text(encoding="utf-8"))


def analyze(incorrect_ids: list[str], lib: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  UWorld Weak Spot Analysis")
    print(f"  {len(incorrect_ids)} incorrect questions provided")
    print(f"{'='*60}\n")

    found, not_found = [], []
    for qid in incorrect_ids:
        qid = qid.strip()
        if qid in lib:
            found.append(lib[qid])
        else:
            not_found.append(qid)

    print(f"  Matched: {len(found)}/{len(incorrect_ids)} questions")
    if not_found:
        print(f"  Not in library: {', '.join(not_found[:10])}"
              + (f" ...+{len(not_found)-10}" if len(not_found) > 10 else ""))

    if not found:
        print("\n  No matched questions to analyze.")
        return

    # ── SUBJECT BREAKDOWN ───────────────────────────────────────────────────
    subjects = Counter(q.get("subject", "Unknown") for q in found)
    print(f"\n{'─'*50}")
    print(f"  BY SUBJECT (your weak subjects first):")
    print(f"{'─'*50}")
    for subj, cnt in subjects.most_common():
        bar = "█" * cnt
        pct = 100 * cnt / len(found)
        print(f"  {cnt:3d} ({pct:4.0f}%)  {subj}")

    # ── SYSTEM BREAKDOWN ────────────────────────────────────────────────────
    systems = Counter(q.get("system", "Unknown") for q in found)
    print(f"\n{'─'*50}")
    print(f"  BY SYSTEM (top 15 weakest):")
    print(f"{'─'*50}")
    for sys_name, cnt in systems.most_common(15):
        pct = 100 * cnt / len(found)
        print(f"  {cnt:3d} ({pct:4.0f}%)  {sys_name}")

    # ── TOPIC BREAKDOWN ─────────────────────────────────────────────────────
    topics = Counter(q.get("topic", "Unknown") for q in found)
    print(f"\n{'─'*50}")
    print(f"  TOP 20 WEAK TOPICS:")
    print(f"{'─'*50}")
    for topic, cnt in topics.most_common(20):
        print(f"  {cnt:3d}x  {topic}")

    # ── DIFFICULTY ANALYSIS ─────────────────────────────────────────────────
    easy_misses   = [q for q in found if q.get("pct_correct") and q["pct_correct"] >= 70]
    hard_misses   = [q for q in found if q.get("pct_correct") and q["pct_correct"] < 40]
    medium_misses = [q for q in found if q.get("pct_correct")
                     and 40 <= q["pct_correct"] < 70]

    print(f"\n{'─'*50}")
    print(f"  DIFFICULTY BREAKDOWN:")
    print(f"{'─'*50}")
    print(f"  Easy   (≥70% global correct): {len(easy_misses):3d}  ← HIGHEST PRIORITY")
    print(f"  Medium (40-70% correct):      {len(medium_misses):3d}")
    print(f"  Hard   (<40% correct):        {len(hard_misses):3d}")

    if easy_misses:
        easy_topics = Counter(q.get("topic", "?") for q in easy_misses)
        print(f"\n  Easy questions you missed (topics everyone else gets right):")
        for t, c in easy_topics.most_common(10):
            print(f"    {c}x  {t}")

    # ── PRIORITY STUDY PLAN ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  PRIORITY STUDY PLAN")
    print(f"{'='*60}")
    print(f"\n  1. FOCUS AREA — Top 5 weak systems:")
    for sys_name, cnt in systems.most_common(5):
        # Find topics within this system
        sys_topics = Counter(q.get("topic","?") for q in found
                             if q.get("system") == sys_name)
        top_topic = sys_topics.most_common(1)[0][0] if sys_topics else "N/A"
        print(f"     • {sys_name} ({cnt} missed)  — #1 topic: {top_topic}")

    if easy_misses:
        print(f"\n  2. QUICK WINS — {len(easy_misses)} easy questions you missed.")
        print(f"     These are straightforward facts you should know cold.")
        sys_easy = Counter(q.get("system","?") for q in easy_misses)
        for sys_name, cnt in sys_easy.most_common(3):
            print(f"     • {sys_name}: {cnt}")

    print(f"\n{'='*60}\n")


# ── AUTO-SCRAPE INCORRECTS ─────────────────────────────────────────────────────

async def scrape_incorrects(email, password) -> list[str]:
    """Use Playwright to log in and scrape incorrect question IDs from UWorld."""
    from playwright.async_api import async_playwright
    from playwright_stealth import Stealth

    profile_dir = Path("uworld_pw_profile")
    profile_dir.mkdir(exist_ok=True)

    # Clean stale lockfile
    for lf in ["lockfile", "SingletonLock"]:
        p = profile_dir / lf
        if p.exists():
            try: p.unlink()
            except Exception: pass

    incorrect_ids = []

    async with async_playwright() as pw:
        ctx = await pw.chromium.launch_persistent_context(
            user_data_dir=str(profile_dir),
            headless=False,
            args=["--disable-blink-features=AutomationControlled"],
            viewport={"width": 1440, "height": 900},
            locale="en-US",
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/136.0.0.0 Safari/537.36"
            ),
        )
        await Stealth(chrome_runtime=False).apply_stealth_async(ctx)

        captured_ids = []

        async def on_resp(resp):
            if "json" not in resp.headers.get("content-type", ""): return
            if "GetTestQuestions" in resp.url or "GetQbankUsage" in resp.url \
               or "GetUserQuestions" in resp.url or "incorrect" in resp.url.lower():
                try:
                    body = await resp.json()
                    def _walk(o):
                        if isinstance(o, dict):
                            qi = (o.get("questionIndex") or o.get("questionId")
                                  or o.get("id") or o.get("uwId"))
                            if qi: captured_ids.append(str(qi))
                            for v in o.values(): _walk(v)
                        elif isinstance(o, list):
                            for v in o: _walk(v)
                    _walk(body)
                except Exception: pass

        page = ctx.pages[0] if ctx.pages else await ctx.new_page()
        page.on("response", on_resp)

        # Try to log in
        try:
            await page.goto("https://www.uworld.com/app/index.html#/subscriptions/",
                            wait_until="networkidle", timeout=20000)
        except Exception: pass

        if "#/login" in page.url:
            # Try automated login
            try:
                await page.wait_for_selector("#login-email", timeout=6000)
                await page.locator("label[for='login-email']").click(timeout=4000)
                await page.keyboard.type(email, delay=35)
                await page.locator("label[for='login-password']").click(timeout=4000)
                await page.keyboard.type(password, delay=35)
                await page.locator("button[type=submit]").click(timeout=5000)
                await page.wait_for_function(
                    "!window.location.hash.includes('login')", timeout=20000)
            except Exception:
                print("\nPlease log in manually in the browser window. Waiting...")
                for _ in range(600):
                    import asyncio; await asyncio.sleep(1)
                    if "#/login" not in page.url: break

        print("Logged in. Navigating to incorrect questions...")

        # Navigate to Performance → Incorrect
        await page.goto(
            "https://www.uworld.com/app/index.html#/performance/",
            wait_until="networkidle", timeout=20000)
        import asyncio; await asyncio.sleep(5)

        # Try to click Incorrect filter
        try:
            await page.locator("text=Incorrect").first.click(timeout=5000)
            await asyncio.sleep(3)
        except Exception: pass

        # Scroll to load all questions
        for _ in range(10):
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(2)

        # Extract question IDs from the page DOM
        dom_ids = await page.evaluate("""() => {
            const ids = new Set();
            // Look for question index numbers in data attributes, links, text
            document.querySelectorAll('[data-question-id],[data-qid],[data-index]')
                    .forEach(el => {
                        const id = el.dataset.questionId || el.dataset.qid || el.dataset.index;
                        if (id && /^\\d+$/.test(id)) ids.add(id);
                    });
            // Also look for text patterns like "Q 12345" or "#12345"
            const text = document.body.innerText;
            const matches = text.matchAll(/\\b([0-9]{4,6})\\b/g);
            return [...ids];
        }""")

        all_ids = list(set(captured_ids + dom_ids))
        print(f"Found {len(all_ids)} incorrect question IDs via API interception.")

        await ctx.close()
        return all_ids


def main():
    import asyncio

    args = parse_args()
    lib = load_library()

    if args.auto:
        from dotenv import load_dotenv
        load_dotenv()
        email    = os.getenv("UWORLD_EMAIL")    or input("UWorld email: ").strip()
        password = os.getenv("UWORLD_PASSWORD") or __import__("getpass").getpass()
        ids = asyncio.run(scrape_incorrects(email, password))
        if not ids:
            print("Could not auto-scrape incorrects. Use --ids instead.")
            sys.exit(1)
    elif args.ids:
        ids = [i.strip() for i in args.ids.split(",") if i.strip()]
    else:
        print("\nPaste your incorrect UWorld question IDs (comma or newline separated).")
        print("You can find these in UWorld → My Performance → Incorrect")
        print("(Press Enter twice when done)\n")
        lines = []
        while True:
            line = input()
            if not line and lines:
                break
            lines.append(line)
        raw = " ".join(lines)
        ids = [x.strip() for x in re.split(r"[,\s]+", raw) if x.strip().isdigit()]

    if not ids:
        print("No question IDs provided.")
        sys.exit(1)

    analyze(ids, lib)


if __name__ == "__main__":
    main()
