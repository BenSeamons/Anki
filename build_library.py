#!/usr/bin/env python3
"""
build_library.py — Build a local database mapping every UWorld question ID
                   to its Subject / System / Topic / %Correct.

HOW IT WORKS:
  1. Logs into UWorld, clicks Launch to enter the qbank.
  2. Uses page.evaluate(fetch(...)) to call the search API from INSIDE
     the browser — this uses the browser's localStorage JWT automatically.
  3. Tries multiple search terms (system names, common words) to cover
     all questions. Saves incrementally after every term.

RUN:
    python build_library.py              # full run
    python build_library.py --headless   # no browser window
    python build_library.py --fresh      # wipe and start over
    python build_library.py --show       # print library stats and exit
    python build_library.py --debug      # verbose output + screenshots
"""

import asyncio
import json
import os
import re
import sys
import argparse
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

LIBRARY_PATH  = Path("uworld_library.json")
PROGRESS_PATH = Path("uworld_library_progress.json")
APPS_BASE     = "https://apps.uworld.com/courseapp/usmle/v53/en-US"

# ─── SEARCH TERMS ─────────────────────────────────────────────────────────────
# UWorld returns max ~1000 results per search.
# System names give clean non-overlapping subsets.
# Common words mop up cross-system questions.

SEARCH_TERMS = [
    # By system (core coverage)
    "cardiovascular", "renal", "pulmonary", "gastrointestinal",
    "endocrine", "neurology", "hematology", "infectious",
    "obstetrics", "gynecology", "pediatrics", "psychiatry",
    "dermatology", "musculoskeletal", "oncology", "surgery",
    "emergency", "biostatistics", "pharmacology", "immunology",
    # Common vignette words (mop up)
    "fever", "chest", "abdominal", "headache", "fatigue",
    "shortness", "hypertension", "diabetes", "pregnancy",
    "trauma", "blood", "fracture", "weight", "nausea",
    "seizure", "rash", "vision", "urinary", "alcohol",
    "screening", "ethics", "vaccine", "antibiotic",
]


# ─── ARG PARSING ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--headless", action="store_true")
    p.add_argument("--fresh",    action="store_true", help="Wipe and rebuild")
    p.add_argument("--show",     action="store_true", help="Print stats and exit")
    p.add_argument("--debug",    action="store_true")
    return p.parse_args()


# ─── LIBRARY I/O ─────────────────────────────────────────────────────────────

def load_library():
    if LIBRARY_PATH.exists():
        with open(LIBRARY_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_library(lib):
    with open(LIBRARY_PATH, "w", encoding="utf-8") as f:
        json.dump(lib, f, indent=2)


def load_progress():
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"completed_terms": [], "sub_id": None}


def save_progress(prog):
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(prog, f, indent=2)


def show_stats(lib):
    if not lib:
        print("Library is empty.")
        return
    subjects, systems, topics = {}, {}, {}
    for q in lib.values():
        s = q.get("subject","?"); subjects[s] = subjects.get(s, 0) + 1
        s = q.get("system", "?"); systems [s] = systems .get(s, 0) + 1
        t = q.get("topic",  "?"); topics  [t] = topics  .get(t, 0) + 1

    print(f"\n{'='*55}")
    print(f"  UWorld Library: {len(lib)} questions")
    print(f"{'='*55}")
    print("\nBy Subject:")
    for s, n in sorted(subjects.items(), key=lambda x: -x[1]):
        print(f"  {n:4d}  {s}")
    print("\nBy System (top 20):")
    for s, n in sorted(systems.items(), key=lambda x: -x[1])[:20]:
        print(f"  {n:4d}  {s}")
    missing = sum(1 for q in lib.values() if not q.get("topic"))
    print(f"\n  Total unique topics: {len(topics)}")
    print(f"  Questions missing topic: {missing}")


# ─── BROWSER SETUP ───────────────────────────────────────────────────────────

async def launch_browser(headless: bool):
    from playwright.async_api import async_playwright
    pw      = await async_playwright().start()
    browser = await pw.chromium.launch(
        headless=headless,
        args=["--disable-blink-features=AutomationControlled"]
    )
    ctx = await browser.new_context(
        viewport={"width": 1280, "height": 800},
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
        )
    )
    return pw, browser, ctx


# ─── LOGIN ───────────────────────────────────────────────────────────────────

async def login(ctx, debug: bool):
    """Login and return the page (still on account page)."""
    email    = os.getenv("UWORLD_EMAIL")    or input("UWorld email: ").strip()
    password = os.getenv("UWORLD_PASSWORD") or __import__("getpass").getpass("UWorld password: ")

    page = await ctx.new_page()
    print("  🔐 Logging in...")
    await page.goto("https://www.uworld.com/app/index.html#/login/",
                    wait_until="networkidle", timeout=30000)
    await asyncio.sleep(2)

    try:
        await page.wait_for_selector("#login-email", timeout=15000)
    except Exception:
        await page.wait_for_selector('input[type="email"]', timeout=5000)

    await page.evaluate("""
        ([em, pw]) => {
            function fill(el, val) {
                if (!el) return;
                el.value = val;
                ['input','change','blur'].forEach(ev =>
                    el.dispatchEvent(new Event(ev, {bubbles:true})));
            }
            fill(document.getElementById('login-email') ||
                 document.querySelector('input[type="email"]'), em);
            fill(document.querySelector('input[type="password"]'), pw);
        }
    """, [email, password])
    await asyncio.sleep(0.4)

    await page.evaluate("""
        () => {
            const btn = document.querySelector('button[type="submit"]') ||
                [...document.querySelectorAll('button')]
                    .find(b => /sign in|log in|login/i.test(b.textContent));
            if (btn) btn.click();
        }
    """)

    try:
        await page.wait_for_function(
            "!window.location.hash.includes('login')", timeout=25000)
    except Exception:
        pass
    await asyncio.sleep(3)

    if "login" in (await page.evaluate("window.location.hash")).lower():
        raise RuntimeError("Login failed. Check UWORLD_EMAIL / UWORLD_PASSWORD.")
    print(f"  ✅ Logged in!")
    if debug:
        await page.screenshot(path="debug_01_loggedin.png")
    return page


# ─── FIND SUBSCRIPTION ID ────────────────────────────────────────────────────

async def get_sub_id(ctx, debug: bool):
    """Intercept GetAllSubscriptions to find the real QBank subscription ID."""
    captured = {}

    async def on_resp(resp):
        if "GetAllSubscriptions" in resp.url or "GetPayments" in resp.url:
            try:
                captured[resp.url] = await resp.json()
            except Exception:
                pass

    # Attach to all existing pages
    for p in ctx.pages:
        p.on("response", on_resp)

    # Wait briefly for any in-flight calls
    await asyncio.sleep(2)

    sub_ids = []
    for url in captured:
        m = re.search(r'GetPaymentsForSubscription/(\d+)', url)
        if m and m.group(1) not in sub_ids:
            sub_ids.append(m.group(1))

    all_subs = next((v for k, v in captured.items() if "GetAllSubscriptions" in k), None)

    sub_id = None
    if all_subs and isinstance(all_subs, list):
        for item in all_subs:
            if not isinstance(item, dict): continue
            if item.get("IsSim") or item.get("isSim"): continue
            if item.get("FormId") or item.get("formId"): continue
            name = str(item.get("CourseName") or item.get("courseName") or "").lower()
            if any(x in name for x in ["self-assessment","free trial"]): continue
            sid = item.get("SubscriptionId") or item.get("subscriptionId")
            if sid:
                sub_id = str(sid)
                if debug:
                    print(f"    Found QBank sub: {name} → {sid}")
                break

    if not sub_id and sub_ids:
        sub_ids_sorted = sorted(sub_ids, key=int)
        sub_id = sub_ids_sorted[1] if len(sub_ids_sorted) > 1 else sub_ids_sorted[0]

    return sub_id


# ─── ENTER THE QBANK ─────────────────────────────────────────────────────────

async def enter_qbank(ctx, sub_id: str, debug: bool):
    """
    Navigate to the qbank. Tries:
      1. Extract href from Launch button and navigate directly
      2. Click Launch button and capture new tab
      3. Navigate directly to dashboard URL
    Returns the page inside the qbank.
    """
    # Get the account page (should be open)
    account_page = ctx.pages[0] if ctx.pages else None
    if not account_page:
        raise RuntimeError("No account page found")

    # Try to get the launch URL from the button's href
    launch_url = await account_page.evaluate("""
        () => {
            const all = [...document.querySelectorAll('a, button')];
            for (const el of all) {
                if (!/launch/i.test(el.textContent)) continue;
                // Check if it or its parent has an href
                const href = el.href || el.closest('a')?.href;
                if (href && href.includes('uworld.com')) return href;
            }
            return null;
        }
    """)

    if launch_url and "apps.uworld.com" in launch_url:
        print(f"  🚀 Found launch URL: {launch_url[:70]}")
        qb_page = await ctx.new_page()
        await qb_page.goto(launch_url, wait_until="domcontentloaded", timeout=25000)
        await asyncio.sleep(4)
        if debug:
            await qb_page.screenshot(path="debug_02_qbank.png")
            print(f"     Qbank URL: {qb_page.url}")
        return qb_page

    # Try clicking Launch and catching new tab
    print("  🚀 Clicking Launch button...")
    try:
        async with ctx.expect_page(timeout=8000) as pg_info:
            await account_page.evaluate("""
                () => {
                    const all = [...document.querySelectorAll('a, button')];
                    const medical = all.find(el => {
                        if (!/launch/i.test(el.textContent)) return false;
                        const row = el.closest('tr,li,div');
                        return !row || /step|ck|qbank|medical/i.test(row.textContent);
                    }) || all.find(el => /launch/i.test(el.textContent));
                    if (medical) medical.click();
                }
            """)
        qb_page = await pg_info.value
        await qb_page.wait_for_load_state("domcontentloaded", timeout=20000)
        await asyncio.sleep(4)
        if "apps.uworld.com" in qb_page.url:
            print(f"  ✅ In qbank: {qb_page.url[:70]}")
            return qb_page
    except Exception as e:
        if debug:
            print(f"     Launch click: {e}")

    # Direct navigation fallback
    print(f"  🚀 Navigating directly to qbank...")
    qb_page = await ctx.new_page()
    direct_url = f"{APPS_BASE}/dashboard/{sub_id}"
    await qb_page.goto(direct_url, wait_until="domcontentloaded", timeout=25000)
    await asyncio.sleep(4)
    if debug:
        await qb_page.screenshot(path="debug_02_qbank_direct.png")
        print(f"     URL: {qb_page.url}")

    if "login" in qb_page.url.lower():
        raise RuntimeError(
            "Could not enter the qbank — redirected to login.\n"
            "The Launch button needs to be clicked to authenticate on apps.uworld.com."
        )

    print(f"  ✅ In qbank: {qb_page.url[:70]}")
    return qb_page


# ─── SEARCH FROM INSIDE THE BROWSER ─────────────────────────────────────────

async def search_in_browser(qb_page, sub_id: str, term: str, debug: bool):
    """
    Call the search API using fetch() from inside the browser page.
    The browser has the JWT in localStorage / cookies so this works
    even when direct httpx calls return empty responses.

    Returns list of {id, subject, system, topic, pct_correct} dicts.
    """
    import urllib.parse
    q = urllib.parse.quote(term)

    # Candidate search API paths to try (run inside browser via fetch)
    candidates = [
        f"{APPS_BASE}/api/search?subscriptionId={sub_id}&query={q}&pageSize=1000",
        f"{APPS_BASE}/api/search?subscriptionId={sub_id}&q={q}&pageSize=1000",
        f"{APPS_BASE}/api/questions/search?subscriptionId={sub_id}&query={q}&pageSize=1000",
        f"https://gateway-api.uworld.com/api/search/questions?subscriptionId={sub_id}&query={q}&pageSize=1000",
        f"https://gateway-api.uworld.com/api/questions/search?subscriptionId={sub_id}&query={q}&pageSize=1000",
        f"https://gateway-api.uworld.com/api/qbank/search?subscriptionId={sub_id}&query={q}&pageSize=1000",
        f"https://gateway-api.uworld.com/api/search?subscriptionId={sub_id}&query={q}&pageSize=1000",
    ]

    for url in candidates:
        try:
            result = await qb_page.evaluate(f"""
                async () => {{
                    try {{
                        const resp = await fetch({json.dumps(url)}, {{
                            method: 'GET',
                            credentials: 'include',
                            headers: {{
                                'Accept': 'application/json',
                                'Content-Type': 'application/json',
                            }}
                        }});
                        const text = await resp.text();
                        return {{status: resp.status, body: text, url: resp.url}};
                    }} catch(e) {{
                        return {{error: e.toString()}};
                    }}
                }}
            """)

            if result.get("error"):
                if debug:
                    print(f"    fetch error {url[:70]}: {result['error']}")
                continue

            status = result.get("status", 0)
            body   = result.get("body", "")

            if debug:
                print(f"    {status}  {url[:80]}")
                if body:
                    print(f"      body[:100]: {body[:100]}")

            if status != 200 or not body:
                continue

            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                if debug:
                    print(f"      non-JSON body: {body[:80]}")
                continue

            items = _parse_search_results(data)
            if items:
                print(f"    ✅ {len(items)} results from {url[:70]}")
                return items

        except Exception as e:
            if debug:
                print(f"    eval error: {e}")
            continue

    # If API calls failed, try scraping the search page DOM directly
    return await scrape_search_page(qb_page, sub_id, term, debug)


async def scrape_search_page(qb_page, sub_id: str, term: str, debug: bool):
    """
    Navigate to the search page, fill the input via JS, wait for results,
    then extract the table rows from the DOM.
    """
    search_url = f"{APPS_BASE}/search/{sub_id}/false"

    if qb_page.url != search_url:
        await qb_page.goto(search_url, wait_until="networkidle", timeout=20000)
        await asyncio.sleep(2)

    # Fill search input via JS (more reliable than wait_for_selector + click)
    filled = await qb_page.evaluate(f"""
        () => {{
            const inputs = [...document.querySelectorAll('input')];
            const inp = inputs.find(el =>
                el.offsetParent !== null &&  // visible
                (el.type === 'text' || el.type === 'search' || !el.type)
            );
            if (!inp) return false;
            inp.value = {json.dumps(term)};
            ['input','change'].forEach(ev =>
                inp.dispatchEvent(new Event(ev, {{bubbles:true}})));
            return true;
        }}
    """)

    if not filled:
        if debug:
            print(f"    Could not find search input on {qb_page.url}")
            await qb_page.screenshot(path=f"debug_search_{term[:8]}.png")
        return []

    await qb_page.keyboard.press("Enter")
    await asyncio.sleep(5)  # wait for results to render

    if debug:
        await qb_page.screenshot(path=f"debug_results_{term[:8]}.png")

    # Extract rows from the results table
    rows = await qb_page.evaluate("""
        () => {
            // Try table rows first
            const tableRows = document.querySelectorAll('table tbody tr, [class*="row"]:not(:first-child)');
            if (tableRows.length > 0) {
                return [...tableRows].map(row => {
                    const cells = row.querySelectorAll('td, [class*="cell"]');
                    const idText = cells[0]?.textContent?.trim() || '';
                    // ID format is "1 - 2134" → extract the number after the dash
                    const idMatch = idText.match(/\d+\s*-\s*(\d+)/) || idText.match(/(\d+)/);
                    return {
                        id:          idMatch ? idMatch[1] : null,
                        subject:     cells[1]?.textContent?.trim() || null,
                        system:      cells[2]?.textContent?.trim() || null,
                        topic:       cells[3]?.textContent?.trim() || null,
                        pct_correct: cells[4]?.textContent?.replace('%','').trim() || null,
                    };
                }).filter(r => r.id);
            }
            return [];
        }
    """)

    if debug:
        print(f"    DOM scrape: {len(rows)} rows")

    # Clean up
    results = []
    for r in rows:
        if not r.get("id"):
            continue
        pct = None
        try:
            pct = float(r["pct_correct"]) if r.get("pct_correct") else None
        except (ValueError, TypeError):
            pass
        results.append({
            "id":          str(r["id"]),
            "subject":     r.get("subject") or None,
            "system":      r.get("system")  or None,
            "topic":       r.get("topic")   or None,
            "pct_correct": pct,
        })
    return results


# ─── PARSE API RESPONSE ───────────────────────────────────────────────────────

def _parse_search_results(data):
    """Extract question records from any JSON shape."""
    results = []

    def _extract(obj):
        if not isinstance(obj, dict):
            return
        qid = (obj.get("id") or obj.get("qid") or obj.get("questionId")
               or obj.get("itemId") or obj.get("uwId"))
        if not qid:
            return
        subject = (obj.get("subject") or obj.get("subjectName")
                   or obj.get("discipline") or obj.get("disciplineName"))
        system  = (obj.get("system")  or obj.get("systemName")
                   or obj.get("bodySystem") or obj.get("category"))
        topic   = (obj.get("topic")   or obj.get("topicName")
                   or obj.get("concept") or obj.get("topicTitle"))
        pct     = (obj.get("percentCorrect") or obj.get("pctCorrect")
                   or obj.get("correctPercent") or obj.get("percentCorrectGlobal"))
        results.append({
            "id":          str(qid),
            "subject":     str(subject) if subject else None,
            "system":      str(system)  if system  else None,
            "topic":       str(topic)   if topic   else None,
            "pct_correct": round(float(pct), 1) if pct is not None else None,
        })

    def _walk(obj):
        if isinstance(obj, dict):
            _extract(obj)
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for v in obj:
                _walk(v)

    _walk(data)
    return results


# ─── MAIN ─────────────────────────────────────────────────────────────────────

async def main():
    args = parse_args()

    if args.show:
        show_stats(load_library())
        return

    if args.fresh:
        LIBRARY_PATH.unlink(missing_ok=True)
        PROGRESS_PATH.unlink(missing_ok=True)
        print("  🗑️  Wiped existing library.")

    lib      = load_library()
    progress = load_progress()

    print(f"\n{'='*55}")
    print(f"  UWorld Library Builder")
    print(f"{'='*55}")
    print(f"  Library so far:    {len(lib)} questions")
    print(f"  Terms completed:   {len(progress['completed_terms'])}/{len(SEARCH_TERMS)}")

    # ── Launch browser and login ──────────────────────────────────────────
    print(f"\n  🌐 Opening browser{'(headless)' if args.headless else ''}...")
    pw, browser, ctx = await launch_browser(args.headless)

    try:
        # Attach subscription ID interceptor before login
        sub_id_box = {}
        async def on_any_resp(resp):
            if "GetAllSubscriptions" in resp.url or "GetPayments" in resp.url:
                try:
                    sub_id_box[resp.url] = await resp.json()
                except Exception:
                    pass

        # Login
        await login(ctx, args.debug)

        # Wait for account API calls
        for p in ctx.pages:
            p.on("response", on_any_resp)
        await asyncio.sleep(3)

        # Find sub ID
        sub_id = progress.get("sub_id")
        if not sub_id:
            # Parse from intercepted responses
            all_subs = next((v for k, v in sub_id_box.items()
                             if "GetAllSubscriptions" in k), None)
            if all_subs and isinstance(all_subs, list):
                for item in all_subs:
                    if not isinstance(item, dict): continue
                    if item.get("IsSim") or item.get("isSim"): continue
                    if item.get("FormId") or item.get("formId"): continue
                    name = str(item.get("CourseName") or "").lower()
                    if any(x in name for x in ["self-assessment","free trial"]): continue
                    sid = item.get("SubscriptionId") or item.get("subscriptionId")
                    if sid:
                        sub_id = str(sid)
                        if args.debug:
                            print(f"    QBank: {item.get('CourseName')} → {sid}")
                        break

            if not sub_id:
                # Parse sub IDs from URL patterns
                ids = []
                for url in sub_id_box:
                    m = re.search(r'GetPaymentsForSubscription/(\d+)', url)
                    if m: ids.append(m.group(1))
                ids = sorted(set(ids), key=int)
                sub_id = ids[1] if len(ids) > 1 else (ids[0] if ids else None)

            if not sub_id:
                raise RuntimeError("Could not find subscription ID.")

            progress["sub_id"] = sub_id
            save_progress(progress)

        print(f"\n  📋 Subscription ID: {sub_id}")

        # ── Enter the qbank ───────────────────────────────────────────────
        qb_page = await enter_qbank(ctx, sub_id, args.debug)

        # ── Search loop ───────────────────────────────────────────────────
        terms_to_run = [t for t in SEARCH_TERMS
                        if t not in progress["completed_terms"]]
        print(f"\n  Running {len(terms_to_run)} search terms...\n")

        for i, term in enumerate(terms_to_run):
            print(f"  [{i+1}/{len(terms_to_run)}] '{term}'")
            items = await search_in_browser(qb_page, sub_id, term, args.debug)

            new_this = 0
            for q in items:
                qid = q["id"]
                if qid not in lib:
                    lib[qid] = q
                    new_this += 1
                else:
                    for f in ["subject","system","topic","pct_correct"]:
                        if not lib[qid].get(f) and q.get(f):
                            lib[qid][f] = q[f]

            print(f"    {len(items)} results, {new_this} new  (total: {len(lib)})")
            save_library(lib)
            progress["completed_terms"].append(term)
            save_progress(progress)
            await asyncio.sleep(0.3)

    finally:
        await browser.close()
        await pw.stop()

    # ── Final stats ───────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Done! {len(lib)} questions in library.")
    print(f"  Saved to: {LIBRARY_PATH}")
    show_stats(lib)


if __name__ == "__main__":
    asyncio.run(main())
