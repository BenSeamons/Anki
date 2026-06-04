#!/usr/bin/env python3
"""
build_library.py — Build a local database mapping every UWorld question ID
                   to its Subject / System / Topic / %Correct.

HOW IT WORKS:
  1. Logs into UWorld (browser, one time).
  2. Navigates to the Search page and intercepts the search API endpoint.
  3. Calls that endpoint directly (no browser) for each search term.
  4. Saves results incrementally to uworld_library.json after every term.
  5. Stops when coverage plateaus (no new IDs in the last N terms).

RUN:
    python build_library.py                  # uses .env credentials
    python build_library.py --headless       # no visible browser
    python build_library.py --resume         # skip terms already done (default)
    python build_library.py --fresh          # wipe library and start over
    python build_library.py --show           # print library stats and exit
"""

import asyncio
import json
import os
import sys
import argparse
import re
import time
import httpx
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

LIBRARY_PATH  = Path("uworld_library.json")
PROGRESS_PATH = Path("uworld_library_progress.json")
APPS_BASE     = "https://apps.uworld.com/courseapp/usmle/v53/en-US"

# ─── SEARCH TERMS ─────────────────────────────────────────────────────────────
# Strategy: UWorld returns max 1000 results per search.
# Searching by system name hits clean, mostly non-overlapping subsets.
# Generic words mop up anything the system searches miss.

SEARCH_TERMS = [
    # ── By system (covers the vast majority) ──
    "cardiovascular",
    "renal",
    "pulmonary",
    "gastrointestinal",
    "endocrine",
    "neurology",
    "hematology",
    "infectious",
    "obstetrics",
    "gynecology",
    "pediatrics",
    "psychiatry",
    "dermatology",
    "musculoskeletal",
    "oncology",
    "surgery",
    "emergency",
    "biostatistics",
    "pharmacology",
    "immunology",
    # ── Common vignette words (mop up cross-system questions) ──
    "fever",
    "chest pain",
    "abdominal pain",
    "headache",
    "fatigue",
    "shortness of breath",
    "hypertension",
    "diabetes",
    "pregnancy",
    "trauma",
    "blood",
    "biopsy",
    "fracture",
    "weight loss",
    "nausea",
    "seizure",
    "rash",
    "vision",
    "hearing",
    "urinary",
    "sexual",
    "alcohol",
    "smoking",
    "vaccine",
    "antibiotic",
    "screening",
    "ethics",
]


# ─── ARG PARSING ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Build UWorld question library")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--fresh",    action="store_true", help="Wipe and rebuild")
    p.add_argument("--resume",   action="store_true", help="Skip completed terms (default)")
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
    return {"completed_terms": [], "api_endpoint": None, "sub_id": None}


def save_progress(prog):
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(prog, f, indent=2)


def show_stats(lib):
    if not lib:
        print("Library is empty.")
        return
    subjects = {}
    systems  = {}
    topics   = {}
    for q in lib.values():
        subjects[q.get("subject", "?")] = subjects.get(q.get("subject", "?"), 0) + 1
        systems [q.get("system",  "?")] = systems .get(q.get("system",  "?"), 0) + 1
        topics  [q.get("topic",   "?")] = topics  .get(q.get("topic",   "?"), 0) + 1

    print(f"\n{'='*55}")
    print(f"  UWorld Library: {len(lib)} questions")
    print(f"{'='*55}")
    print(f"\nBy Subject:")
    for s, n in sorted(subjects.items(), key=lambda x: -x[1]):
        print(f"  {n:4d}  {s}")
    print(f"\nBy System (top 20):")
    for s, n in sorted(systems.items(), key=lambda x: -x[1])[:20]:
        print(f"  {n:4d}  {s}")
    print(f"\nTotal unique topics: {len(topics)}")


# ─── BROWSER LOGIN ────────────────────────────────────────────────────────────

async def browser_login_and_discover(headless: bool, debug: bool):
    """
    Log in, then navigate to the Search page and make one search to
    intercept the exact API endpoint URL. Returns login_data dict.
    """
    from playwright.async_api import async_playwright

    captured   = {}
    token_box  = {}
    search_api = {}   # will hold {"url_template": ..., "headers": ...}

    async def on_response(resp):
        url = resp.url
        if "uworld.com" not in url:
            return
        ct = resp.headers.get("content-type", "")
        if "json" not in ct:
            return
        try:
            body = await resp.json()
            captured[url] = body

            # Grab JWT
            if not token_box.get("token"):
                if "refreshToken" in url or ("auth" in url and "userapi" in url):
                    for field in ["accessToken","token","jwt","access_token","bearerToken"]:
                        val = body.get(field) if isinstance(body, dict) else None
                        if val and isinstance(val, str) and len(val) > 20:
                            token_box["token"] = val
                            break

            # Detect search API endpoint
            if not search_api.get("url") and (
                "search" in url.lower() and "gateway-api" in url
                or "search" in url.lower() and "apps.uworld" in url
            ):
                search_api["url"] = url
                search_api["status"] = resp.status
                if debug:
                    print(f"  🔍 Search API discovered: {url}")
                    print(f"     Response: {json.dumps(body)[:200]}")

        except Exception:
            pass

    email    = os.getenv("UWORLD_EMAIL")    or input("UWorld email: ").strip()
    password = os.getenv("UWORLD_PASSWORD") or __import__("getpass").getpass("UWorld password: ")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=headless,
            args=["--disable-blink-features=AutomationControlled"]
        )
        ctx  = await browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
            )
        )
        page = await ctx.new_page()
        page.on("response", on_response)

        # ── Login ───────────────────────────────────────────────────────
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

        # ── Find subscription ID ───────────────────────────────────────
        sub_ids = []
        for url in captured:
            m = re.search(r'GetPaymentsForSubscription/(\d+)', url)
            if m and m.group(1) not in sub_ids:
                sub_ids.append(m.group(1))

        # Use the smallest (oldest = main qbank subscription)
        sub_id = sorted(sub_ids, key=int)[0] if sub_ids else None
        if not sub_id:
            raise RuntimeError(f"Could not find subscription ID. IDs seen: {sub_ids}")
        print(f"  📋 Subscription ID: {sub_id}")

        # ── Navigate to Search page and do one search to find the API ──
        search_url = f"{APPS_BASE}/search/{sub_id}/false"
        print(f"  🔍 Opening search page to discover API endpoint...")
        new_page = await ctx.new_page()
        new_page.on("response", on_response)
        await new_page.goto(search_url, wait_until="networkidle", timeout=20000)
        await asyncio.sleep(2)

        # Type "patient" in the search box to trigger an API call
        try:
            search_input = await new_page.wait_for_selector(
                'input[type="text"], input[type="search"], input[placeholder*="search" i], '
                'input[placeholder*="Question" i], input[placeholder*="keyword" i]',
                timeout=8000
            )
            await search_input.fill("patient")
            await asyncio.sleep(0.3)
            await new_page.keyboard.press("Enter")
            await asyncio.sleep(4)  # wait for API call
        except Exception as e:
            if debug:
                print(f"  Search input: {e}")

        if debug:
            await new_page.screenshot(path="debug_search.png")
            print(f"  Search page URL: {new_page.url}")

        cookies = await ctx.cookies()
        await browser.close()

    # Print all captured endpoints to help identify the search API
    print(f"\n  Captured {len(captured)} API responses.")
    search_candidates = [u for u in captured if "search" in u.lower() or "question" in u.lower()]
    if search_candidates:
        print("  Search/question API candidates:")
        for u in search_candidates:
            print(f"    {u[:100]}")
    elif debug:
        print("  All captured endpoints:")
        for u in sorted(captured.keys()):
            print(f"    {u[:100]}")

    return {
        "token":      token_box.get("token"),
        "cookies":    cookies,
        "sub_id":     sub_id,
        "search_api": search_api.get("url"),
        "intercepted": captured,
    }


# ─── SEARCH API CALLER ────────────────────────────────────────────────────────

def _headers(login_data):
    token   = login_data.get("token")
    cookies = login_data.get("cookies", [])
    h = {
        "Accept":     "application/json, text/plain, */*",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/125.0.0.0",
        "Origin":     "https://apps.uworld.com",
        "Referer":    "https://apps.uworld.com/",
    }
    if token:
        h["Authorization"] = f"Bearer {token}"
    cookie_str = "; ".join(
        f"{c['name']}={c['value']}"
        for c in cookies if "uworld.com" in c.get("domain","")
    )
    if cookie_str:
        h["Cookie"] = cookie_str
    return h


def _build_search_urls(term: str, sub_id: str, discovered_url: str = None):
    """
    Return a list of candidate search API URLs to try.
    If we discovered the real endpoint from the browser, put it first.
    """
    import urllib.parse
    q = urllib.parse.quote(term)
    candidates = []

    # If we discovered the real endpoint, derive template from it
    if discovered_url:
        # Replace the query param value in the discovered URL
        # Common patterns: ?query=X, ?q=X, ?keyword=X, ?search=X, /search/X
        templated = re.sub(r'([?&](?:query|q|keyword|search|term)=)[^&]+', f'\\g<1>{q}',
                           discovered_url)
        if templated != discovered_url:
            candidates.append(templated)
        # Also try replacing path segment
        templated2 = re.sub(r'/search/[^/]+(/|$)', f'/search/{q}\\1', discovered_url)
        if templated2 != discovered_url:
            candidates.append(templated2)

    # Gateway API guesses
    base_gw = "https://gateway-api.uworld.com/api"
    candidates += [
        f"{base_gw}/search/questions?subscriptionId={sub_id}&query={q}&pageSize=1000",
        f"{base_gw}/search/questions?subscriptionId={sub_id}&q={q}&pageSize=1000",
        f"{base_gw}/questions/search?subscriptionId={sub_id}&query={q}&pageSize=1000",
        f"{base_gw}/qbank/search?subscriptionId={sub_id}&query={q}&pageSize=1000",
        f"{base_gw}/search?subscriptionId={sub_id}&query={q}&pageSize=1000",
        f"{base_gw}/search?subscriptionId={sub_id}&q={q}&pageSize=1000",
        f"{base_gw}/qbank/questions?subscriptionId={sub_id}&search={q}&pageSize=1000",
        f"{base_gw}/items/search?subscriptionId={sub_id}&query={q}&pageSize=1000",
    ]
    # Apps API guesses
    base_app = f"https://apps.uworld.com/courseapp/usmle/v53/en-US/api"
    candidates += [
        f"{base_app}/search?subscriptionId={sub_id}&query={q}&pageSize=1000",
        f"{base_app}/questions/search?query={q}&subscriptionId={sub_id}&pageSize=1000",
    ]
    return candidates


def _parse_search_results(data):
    """
    Extract list of {id, subject, system, topic, pct_correct} from API response.
    Handles multiple response shapes.
    """
    results = []

    def _extract_item(obj):
        if not isinstance(obj, dict):
            return
        # Try to get the question ID
        qid = (obj.get("id") or obj.get("qid") or obj.get("questionId")
               or obj.get("itemId") or obj.get("uwId"))
        if not qid:
            return
        # Extract metadata
        subject = (obj.get("subject") or obj.get("subjectName")
                   or obj.get("discipline") or obj.get("disciplineName"))
        system  = (obj.get("system")  or obj.get("systemName")
                   or obj.get("bodySystem") or obj.get("category"))
        topic   = (obj.get("topic")   or obj.get("topicName")
                   or obj.get("concept") or obj.get("topicTitle"))
        pct     = (obj.get("percentCorrect") or obj.get("pctCorrect")
                   or obj.get("correctPercent") or obj.get("percentCorrectGlobal")
                   or obj.get("score") or obj.get("percent"))
        results.append({
            "id":          str(qid),
            "subject":     str(subject) if subject else None,
            "system":      str(system)  if system  else None,
            "topic":       str(topic)   if topic   else None,
            "pct_correct": round(float(pct), 1) if pct is not None else None,
        })

    def _walk(obj):
        if isinstance(obj, dict):
            _extract_item(obj)
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for v in obj:
                _walk(v)

    _walk(data)
    return results


async def search_term(term: str, login_data: dict, debug: bool = False):
    """
    Call the search API with `term`, return list of question dicts.
    Tries each candidate URL until one returns real data.
    """
    sub_id       = login_data["sub_id"]
    discovered   = login_data.get("search_api")
    headers      = _headers(login_data)
    candidates   = _build_search_urls(term, sub_id, discovered)

    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
        for url in candidates:
            try:
                resp = await client.get(url, headers=headers)
                if debug:
                    print(f"    {resp.status_code}  {url[:90]}")
                if resp.status_code != 200:
                    continue
                data = resp.json()
                items = _parse_search_results(data)
                if items:
                    if debug:
                        print(f"    → {len(items)} questions from {url[:70]}")
                    # Record which URL worked
                    login_data["search_api"] = url
                    return items
            except Exception as e:
                if debug:
                    print(f"    ERR {url[:70]}: {e}")
    return []


# ─── BROWSER FALLBACK SEARCH ─────────────────────────────────────────────────

async def browser_search(term: str, login_data: dict,
                          headless: bool = False, debug: bool = False):
    """
    If direct API calls fail, open the search page in a browser tab
    (with saved cookies) and intercept the API call it makes.
    """
    from playwright.async_api import async_playwright

    sub_id  = login_data["sub_id"]
    cookies = login_data.get("cookies", [])
    results_box = []

    async def on_resp(resp):
        if "json" not in resp.headers.get("content-type",""):
            return
        try:
            body = await resp.json()
            items = _parse_search_results(body)
            if items and not results_box:
                results_box.extend(items)
                login_data["search_api"] = resp.url
                if debug:
                    print(f"    ✅ Intercepted search: {resp.url[:80]}")
        except Exception:
            pass

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=headless)
        ctx     = await browser.new_context(viewport={"width":1280,"height":800})
        if cookies:
            await ctx.add_cookies(cookies)
        page = await ctx.new_page()
        page.on("response", on_resp)

        search_url = f"{APPS_BASE}/search/{sub_id}/false"
        await page.goto(search_url, wait_until="networkidle", timeout=20000)
        await asyncio.sleep(2)

        try:
            inp = await page.wait_for_selector(
                'input[type="text"], input[type="search"], '
                'input[placeholder*="Question" i], input[placeholder*="search" i]',
                timeout=8000
            )
            await inp.fill(term)
            await asyncio.sleep(0.3)
            await page.keyboard.press("Enter")
            await asyncio.sleep(5)
        except Exception as e:
            print(f"    Search input error: {e}")

        if debug:
            await page.screenshot(path=f"debug_search_{term[:10]}.png")

        await browser.close()

    return results_box


# ─── MAIN LOOP ────────────────────────────────────────────────────────────────

async def main():
    args = parse_args()

    # ── Show stats only ──────────────────────────────────────────────────
    if args.show:
        lib = load_library()
        show_stats(lib)
        return

    # ── Load or reset ────────────────────────────────────────────────────
    if args.fresh:
        LIBRARY_PATH.unlink(missing_ok=True)
        PROGRESS_PATH.unlink(missing_ok=True)
        print("  🗑️  Wiped existing library.")

    lib      = load_library()
    progress = load_progress()

    print(f"\n{'='*55}")
    print(f"  UWorld Library Builder")
    print(f"{'='*55}")
    print(f"  Library so far: {len(lib)} questions")
    print(f"  Terms completed: {len(progress['completed_terms'])}/{len(SEARCH_TERMS)}")

    # ── Browser login ────────────────────────────────────────────────────
    if not progress.get("sub_id") or not progress.get("api_endpoint") or args.fresh:
        print(f"\n  🌐 Logging in to discover search API endpoint...")
        login_data = await browser_login_and_discover(
            headless=args.headless, debug=args.debug
        )
        progress["sub_id"]        = login_data["sub_id"]
        progress["api_endpoint"]  = login_data.get("search_api")
        progress["token"]         = login_data.get("token")
        # Note: cookies expire, so we need a fresh login each session
        save_progress(progress)
    else:
        # Still need fresh login for valid cookies/token
        print(f"\n  🌐 Logging in (fresh cookies needed each session)...")
        login_data = await browser_login_and_discover(
            headless=args.headless, debug=args.debug
        )
        # Preserve known-good sub_id and endpoint from previous run
        login_data["sub_id"] = progress["sub_id"]
        if progress.get("api_endpoint") and not login_data.get("search_api"):
            login_data["search_api"] = progress["api_endpoint"]

    sub_id = login_data["sub_id"]
    print(f"\n  Subscription ID: {sub_id}")
    print(f"  Search API:      {login_data.get('search_api') or '(will discover on first search)'}")

    # ── Search loop ───────────────────────────────────────────────────────
    terms_to_run = [t for t in SEARCH_TERMS
                    if t not in progress["completed_terms"]]

    print(f"\n  Running {len(terms_to_run)} search terms...\n")

    for i, term in enumerate(terms_to_run):
        before = len(lib)

        print(f"  [{i+1}/{len(terms_to_run)}] Searching: '{term}'")

        # Try direct API first
        items = await search_term(term, login_data, debug=args.debug)

        # Fall back to browser if API returned nothing
        if not items:
            print(f"    API returned nothing — trying browser fallback...")
            items = await browser_search(
                term, login_data,
                headless=args.headless,
                debug=args.debug
            )

        # Merge into library (don't overwrite existing entries that have more data)
        new_this_term = 0
        for q in items:
            qid = q["id"]
            if qid not in lib:
                lib[qid] = q
                new_this_term += 1
            else:
                # Enrich existing entry if new data has more fields filled in
                existing = lib[qid]
                for field in ["subject","system","topic","pct_correct"]:
                    if not existing.get(field) and q.get(field):
                        existing[field] = q[field]

        after = len(lib)
        print(f"    → {len(items)} results, {new_this_term} new IDs  "
              f"(library total: {after})")

        # Save after every term
        save_library(lib)
        progress["completed_terms"].append(term)
        progress["api_endpoint"] = login_data.get("search_api")
        save_progress(progress)

        # Stop early if no new IDs for last 5 terms
        if i >= 4:
            recent = [t for t in progress["completed_terms"][-5:]]
            # (would need per-term new count to implement precisely — skip for now)

        await asyncio.sleep(0.5)  # be polite

    # ── Final stats ───────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Done! Library contains {len(lib)} unique questions.")
    print(f"  Saved to: {LIBRARY_PATH}")
    show_stats(lib)

    # Check for questions with missing metadata
    missing_topic   = sum(1 for q in lib.values() if not q.get("topic"))
    missing_system  = sum(1 for q in lib.values() if not q.get("system"))
    if missing_topic or missing_system:
        print(f"\n  ⚠️  Missing topic:  {missing_topic} questions")
        print(f"  ⚠️  Missing system: {missing_system} questions")
        print(f"  Run again with different search terms to fill gaps.")


if __name__ == "__main__":
    asyncio.run(main())
