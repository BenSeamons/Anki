#!/usr/bin/env python3
"""
uworld_auto.py — Automated UWorld incorrect-question analyzer
──────────────────────────────────────────────────────────────
Phase 1  (Browser)  : Log into UWorld, capture the JWT auth token and
                      subscription ID from the account-page API calls.
Phase 2  (API)      : Use that token to call UWorld's internal gateway API
                      directly — no more UI navigation.
Phase 3  (Claude)   : Feed the structured data to Claude for analysis.

SETUP (one time):
    pip install playwright
    playwright install chromium

Add to your .env file:
    UWORLD_EMAIL=you@email.com
    UWORLD_PASSWORD=yourpassword

RUN:
    python uworld_auto.py
    python uworld_auto.py --headless        # no browser window
    python uworld_auto.py --debug           # verbose output + screenshots
    python uworld_auto.py --anki-only       # skip UWorld, use Anki data only
"""

import asyncio
import json
import os
import sys
import argparse
import getpass
import re
import time
import httpx
from datetime import datetime
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv
import anthropic

load_dotenv()

# ─── CONFIG ──────────────────────────────────────────────────────────────────

LOGIN_URL    = "https://www.uworld.com/app/index.html#/login/"
GATEWAY_API  = "https://gateway-api.uworld.com/api"
APPS_BASE    = "https://apps.uworld.com/courseapp/usmle/v53/en-US"
ANKI_URL     = "http://localhost:8765"
OUTPUT_DIR   = Path("uworld_reports")
CLAUDE_MODEL = "claude-sonnet-4-6"

ANALYSIS_PROMPT = """You are an expert USMLE tutor. A medical student has shared their UWorld performance data.

Your job: identify their true weak spots and give an actionable study plan.

PERFORMANCE DATA:
{performance_data}

INCORRECT QUESTIONS ({incorrect_count} total):
{incorrect_summary}

Produce exactly these sections (be specific and dense, no filler):

## 🔴 Top Weak Systems/Subjects
List the 3-5 areas with the lowest % correct. For each, give the exact % and the most likely conceptual gaps based on the question topics.

## 🧠 Recurring Concept Gaps
Identify the specific mechanisms, pathways, or facts the incorrect topics point to. Be granular — not just "cardiology" but "distinguish systolic vs diastolic HF management."

## ⚡ Priority Study Plan (Next 2 Weeks)
Ordered list. What to hit first, why, and how (specific Anki tags, First Aid sections, or Sketchy videos).

## 📊 Quick Stats
- Total questions done: X
- Overall %: X%
- Strongest area: X (X%)
- Weakest area: X (X%)
- Incorrect questions to review: X

Be direct. This student needs to pass boards."""


# ─── ARG PARSING ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Automated UWorld weakness analyzer")
    p.add_argument("--email",    help="UWorld email (or set UWORLD_EMAIL in .env)")
    p.add_argument("--password", help="UWorld password (or set UWORLD_PASSWORD in .env)")
    p.add_argument("--headless", action="store_true", help="Run browser invisibly")
    p.add_argument("--debug",    action="store_true", help="Verbose output + screenshots")
    p.add_argument("--anki-only",action="store_true", help="Skip UWorld login")
    p.add_argument("--output",   default=None, help="Output file path")
    return p.parse_args()


def get_credentials(args):
    email    = args.email    or os.getenv("UWORLD_EMAIL")    or ""
    password = args.password or os.getenv("UWORLD_PASSWORD") or ""
    if not email:
        email    = input("UWorld email: ").strip()
    if not password:
        password = getpass.getpass("UWorld password: ")
    return email, password


# ─── ANKI CONNECT ────────────────────────────────────────────────────────────

async def anki_request(action, **params):
    import urllib.request
    payload = json.dumps({"action": action, "version": 6, "params": params}).encode()
    try:
        req = urllib.request.Request(ANKI_URL, data=payload)
        with urllib.request.urlopen(req, timeout=10) as r:
            result = json.loads(r.read())
            if result.get("error"):
                raise RuntimeError(result["error"])
            return result["result"]
    except Exception as e:
        raise RuntimeError(f"AnkiConnect: {e}")


async def get_anki_weak_cards():
    print("  📚 Querying AnkiConnect for weak AnKing/UWorld cards...")
    try:
        note_ids = await anki_request("findNotes", query="tag:AK_Step*UWorld*QID*")
        if not note_ids:
            print("  ⚠️  No AnKing UWorld cards found.")
            return [], []

        BATCH = 500
        all_cards = []
        for i in range(0, len(note_ids), BATCH):
            cards = await anki_request("notesInfo", notes=note_ids[i:i+BATCH])
            all_cards.extend(cards)

        qid_pat = re.compile(r'QID[_::](\d+)', re.I)
        all_qids = []
        for card in all_cards:
            for tag in card.get("tags", []):
                m = qid_pat.search(tag)
                if m:
                    all_qids.append(m.group(1))
                    break

        weak_ids = await anki_request("findNotes",
            query="tag:AK_Step*UWorld*QID* prop:lapses>0")
        struggling = []
        if weak_ids:
            wc = await anki_request("notesInfo", notes=weak_ids[:500])
            for card in wc:
                for tag in card.get("tags", []):
                    m = qid_pat.search(tag)
                    if m:
                        struggling.append(m.group(1))
                        break

        print(f"  Found {len(all_qids)} UWorld-tagged cards, {len(struggling)} with lapses")
        return all_qids, struggling
    except RuntimeError as e:
        print(f"  ⚠️  AnkiConnect not available: {e}")
        return [], []


# ─── PHASE 1: BROWSER LOGIN ───────────────────────────────────────────────────

async def browser_login(email: str, password: str, headless: bool = False, debug: bool = False):
    """
    Open a browser, log into UWorld, and capture:
      - The JWT access token (from the refreshToken or auth API response)
      - All subscription IDs (from GetPaymentsForSubscription URLs)
      - The raw GetAllSubscriptions response body
      - Any cookies set on uworld.com / apps.uworld.com

    Returns a dict: {
        "token":        str,           # Bearer token for gateway-api calls
        "cookies":      list,          # All browser cookies (list of dicts)
        "sub_ids":      list[str],     # Subscription IDs seen in API URLs
        "all_subs_raw": any,           # Raw GetAllSubscriptions response
        "intercepted":  dict,          # url → response body (for debugging)
    }
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        print("  Installing Playwright...")
        os.system(f"{sys.executable} -m pip install playwright -q")
        os.system(f"{sys.executable} -m playwright install chromium --quiet")
        from playwright.async_api import async_playwright

    captured = {}   # url → JSON body
    token_box = {}  # filled by on_response when we see the auth token

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=headless,
            args=["--disable-blink-features=AutomationControlled"]
        )
        ctx = await browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            )
        )
        page = await ctx.new_page()

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

                # Grab JWT from auth endpoints
                if not token_box.get("token"):
                    if "refreshToken" in url or ("auth" in url and "userapi" in url):
                        for field in ["accessToken", "token", "jwt", "access_token",
                                      "bearerToken", "AuthToken", "idToken"]:
                            val = body.get(field) if isinstance(body, dict) else None
                            if val and isinstance(val, str) and len(val) > 20:
                                token_box["token"] = val
                                if debug:
                                    print(f"    🔑 Captured token from {url} [{field}]")
                                break
                        # Also check nested
                        if not token_box.get("token") and isinstance(body, dict):
                            for v in body.values():
                                if isinstance(v, dict):
                                    for field in ["accessToken", "token", "jwt"]:
                                        val = v.get(field)
                                        if val and isinstance(val, str) and len(val) > 20:
                                            token_box["token"] = val
                                            break
            except Exception:
                pass

        page.on("response", on_response)

        # ── Login ────────────────────────────────────────────────────────
        print("  🔐 Logging into UWorld...")
        await page.goto(LOGIN_URL, wait_until="networkidle", timeout=30000)
        await asyncio.sleep(2)

        if debug:
            await page.screenshot(path="debug_01_login.png")

        # Wait for login form (id="login-email" confirmed from error logs)
        try:
            await page.wait_for_selector("#login-email", timeout=15000)
        except Exception:
            await page.wait_for_selector('input[type="email"]', timeout=5000)

        # Fill via JS to bypass the navbar overlay that blocks physical clicks
        await page.evaluate("""
            ([em, pw]) => {
                function fill(el, val) {
                    if (!el) return;
                    el.value = val;
                    ['input','change','blur'].forEach(ev =>
                        el.dispatchEvent(new Event(ev, {bubbles:true}))
                    );
                }
                fill(
                    document.getElementById('login-email') ||
                    document.querySelector('input[type="email"]'), em
                );
                fill(document.querySelector('input[type="password"]'), pw);
            }
        """, [email, password])

        await asyncio.sleep(0.4)

        if debug:
            await page.screenshot(path="debug_02_filled.png")

        # Submit via JS
        submitted = await page.evaluate("""
            () => {
                const btn = document.querySelector('button[type="submit"]') ||
                    document.querySelector('input[type="submit"]') ||
                    [...document.querySelectorAll('button')]
                        .find(b => /sign in|log in|login/i.test(b.textContent));
                if (btn) { btn.click(); return true; }
                return false;
            }
        """)
        if not submitted:
            await page.keyboard.press("Enter")

        # Wait for the hash to leave #/login/
        print("  ⏳ Waiting for login to complete...")
        try:
            await page.wait_for_function(
                "!window.location.hash.includes('login')", timeout=25000)
        except Exception:
            pass

        await asyncio.sleep(3)

        current_hash = await page.evaluate("window.location.hash")
        if "login" in current_hash.lower():
            raise RuntimeError(
                "Login failed — still on login page.\n"
                "Check UWORLD_EMAIL and UWORLD_PASSWORD in your .env file."
            )

        print(f"  ✅ Logged in! (landed at: {page.url})")

        if debug:
            await page.screenshot(path="debug_03_post_login.png")

        # Wait for account-page API calls to finish
        await asyncio.sleep(3)

        # Find subscription ID now so we can visit apps.uworld.com
        sub_ids_early = []
        for url in captured:
            m = re.search(r'GetPaymentsForSubscription/(\d+)', url)
            if m and m.group(1) not in sub_ids_early:
                sub_ids_early.append(m.group(1))

        # Pick the real QBank sub ID early (exclude IsSim forms)
        all_subs_early = next(
            (v for k, v in captured.items() if "GetAllSubscriptions" in k), None)
        best_sub_early = None
        if all_subs_early and isinstance(all_subs_early, list):
            for item in all_subs_early:
                if not isinstance(item, dict): continue
                if item.get("IsSim") or item.get("isSim"): continue
                if item.get("FormId") or item.get("formId"): continue
                sid = item.get("SubscriptionId") or item.get("subscriptionId")
                if sid:
                    best_sub_early = str(sid)
                    break
        if not best_sub_early and sub_ids_early:
            best_sub_early = sorted(sub_ids_early, key=int)[1 if len(sub_ids_early) > 1 else 0]

        # Visit apps.uworld.com so its auth cookies get set in the browser context
        if best_sub_early:
            apps_url = f"{APPS_BASE}/dashboard/{best_sub_early}"
            print(f"  🔗 Visiting apps.uworld.com to capture auth cookies...")
            try:
                apps_page = await ctx.new_page()
                apps_page.on("response", on_response)
                await apps_page.goto(apps_url, wait_until="domcontentloaded", timeout=20000)
                await asyncio.sleep(3)
                if debug:
                    print(f"     apps URL: {apps_page.url}")
                    await apps_page.screenshot(path="debug_04_apps.png")
            except Exception as e:
                if debug:
                    print(f"     apps visit failed: {e}")

        # Collect ALL cookies (now includes apps.uworld.com cookies too)
        cookies = await ctx.cookies()
        if debug:
            apps_cookies = [c for c in cookies if "apps.uworld" in c.get("domain","")]
            print(f"     apps.uworld.com cookies: {len(apps_cookies)}")

        await browser.close()

    # ── Extract subscription IDs from intercepted URL patterns ───────────
    sub_ids = []
    for url in captured:
        m = re.search(r'GetPaymentsForSubscription/(\d+)', url)
        if m and m.group(1) not in sub_ids:
            sub_ids.append(m.group(1))

    all_subs_raw = next(
        (v for k, v in captured.items() if "GetAllSubscriptions" in k), None)

    if debug:
        print(f"  Token captured: {'yes' if token_box.get('token') else 'NO'}")
        print(f"  Subscription IDs seen: {sub_ids}")
        print(f"  GetAllSubscriptions: {json.dumps(all_subs_raw)[:300]}")

    return {
        "token":        token_box.get("token"),
        "cookies":      cookies,
        "sub_ids":      sub_ids,
        "all_subs_raw": all_subs_raw,
        "intercepted":  captured,
    }


# ─── PHASE 2: FIND SUBSCRIPTION ID ───────────────────────────────────────────

def find_medical_sub_id(login_data):
    """
    Find the Step 2 CK QBank subscription ID.
    Key rule: exclude IsSim=true entries (self-assessment forms) and
    anything with 'form', 'self-assessment', or 'free trial' in the name.
    The real QBank has IsSim=false and 'QBank' in CourseName.
    """
    raw = login_data.get("all_subs_raw")

    if raw and isinstance(raw, list):
        # First pass: look for explicit QBank, non-sim entry
        for item in raw:
            if not isinstance(item, dict):
                continue
            is_sim  = item.get("IsSim") or item.get("isSim") or False
            form_id = item.get("FormId") or item.get("formId")
            name    = str(item.get("CourseName") or item.get("courseName") or "").lower()
            if is_sim or form_id:
                continue
            if any(x in name for x in ["self-assessment","free trial","free_trial"]):
                continue
            sid = item.get("SubscriptionId") or item.get("subscriptionId")
            if sid and any(x in name for x in ["qbank","step 2","step2","ck","usmle","medical"]):
                return str(sid)

        # Second pass: any non-sim entry
        for item in raw:
            if not isinstance(item, dict):
                continue
            is_sim  = item.get("IsSim") or item.get("isSim") or False
            form_id = item.get("FormId") or item.get("formId")
            if not is_sim and not form_id:
                sid = item.get("SubscriptionId") or item.get("subscriptionId")
                if sid:
                    return str(sid)

    # Fallback: from GetPaymentsForSubscription URLs, pick middle ID
    # (smallest is often the first self-assessment form, largest may be shelf)
    sub_ids = sorted(login_data.get("sub_ids", []), key=int)
    if sub_ids:
        # Pick the second one — first tends to be self-assessment form 1
        return sub_ids[1] if len(sub_ids) > 1 else sub_ids[0]

    return None


# ─── PHASE 2: DIRECT API CALLS ────────────────────────────────────────────────

def _auth_headers(token, cookies):
    """Build HTTP headers for gateway-api calls."""
    headers = {
        "Accept":       "application/json, text/plain, */*",
        "User-Agent":   "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/125.0.0.0",
        "Origin":       "https://apps.uworld.com",
        "Referer":      "https://apps.uworld.com/",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # Convert Playwright cookies → Cookie header string
    cookie_str = "; ".join(
        f"{c['name']}={c['value']}"
        for c in cookies
        if "uworld.com" in c.get("domain", "")
    )
    if cookie_str:
        headers["Cookie"] = cookie_str

    return headers


async def fetch_performance_data(sub_id, token, cookies, debug=False):
    """
    Call UWorld's gateway API to get:
    - Performance breakdown by subject, system, topic
    - List of incorrect questions with metadata

    We try several endpoint patterns UWorld commonly uses.
    Returns (performance_dict, questions_list).
    """
    headers = _auth_headers(token, cookies)
    performance = {"overall": None, "by_subject": [], "by_system": [], "by_topic": []}
    questions = []

    # Endpoint candidates — we try all of them and parse whatever responds
    perf_endpoints = [
        f"{GATEWAY_API}/performance/subject?subscriptionId={sub_id}",
        f"{GATEWAY_API}/performance/system?subscriptionId={sub_id}",
        f"{GATEWAY_API}/performance/topic?subscriptionId={sub_id}",
        f"{GATEWAY_API}/performance/report?subscriptionId={sub_id}",
        f"{GATEWAY_API}/performance/overall?subscriptionId={sub_id}",
        f"{GATEWAY_API}/reporting/performance/{sub_id}",
        f"{GATEWAY_API}/reporting/subject/{sub_id}",
        f"{GATEWAY_API}/reporting/system/{sub_id}",
        f"{GATEWAY_API}/qbank/performance/{sub_id}",
        f"{GATEWAY_API}/qbank/report/{sub_id}",
    ]
    qbank_endpoints = [
        f"{GATEWAY_API}/qbank/questions?subscriptionId={sub_id}&filter=incorrect&pageSize=500",
        f"{GATEWAY_API}/qbank/questions?subscriptionId={sub_id}&status=incorrect&pageSize=500",
        f"{GATEWAY_API}/qbank/incorrect?subscriptionId={sub_id}&pageSize=500",
        f"{GATEWAY_API}/qbank/items?subscriptionId={sub_id}&result=incorrect&pageSize=500",
        f"{GATEWAY_API}/qbank?subscriptionId={sub_id}&filter=incorrect",
    ]

    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
        print("\n  📊 Fetching performance data from API...")
        for url in perf_endpoints:
            try:
                resp = await client.get(url, headers=headers)
                if debug:
                    print(f"    {resp.status_code}  {url[:80]}")
                if resp.status_code == 200:
                    try:
                        data = resp.json()
                        _parse_performance_blob(data, performance, url)
                        if debug:
                            print(f"      → {json.dumps(data)[:200]}")
                    except Exception:
                        pass
            except Exception as e:
                if debug:
                    print(f"    ERR {url[:80]}: {e}")

        print("\n  ❌ Fetching incorrect questions from API...")
        for url in qbank_endpoints:
            try:
                resp = await client.get(url, headers=headers)
                if debug:
                    print(f"    {resp.status_code}  {url[:80]}")
                if resp.status_code == 200:
                    try:
                        data = resp.json()
                        _parse_question_blob(data, questions)
                        if debug:
                            print(f"      → found {len(questions)} questions so far")
                    except Exception:
                        pass
            except Exception as e:
                if debug:
                    print(f"    ERR {url[:80]}: {e}")

    # Deduplicate questions by ID
    seen, unique = set(), []
    for q in questions:
        if q["id"] not in seen:
            seen.add(q["id"])
            unique.append(q)

    return performance, unique


# ─── PHASE 2B: BROWSER FALLBACK (if API auth fails) ──────────────────────────

async def fetch_via_browser(login_data: dict, sub_id: str,
                             headless: bool = False, debug: bool = False):
    """
    If direct API calls fail (auth/token issues), fall back to opening the
    known performance and previous-tests pages in a new browser tab (reusing
    the captured cookies) and intercepting the API calls those pages make.
    """
    from playwright.async_api import async_playwright

    intercepted_extra = {}

    async def on_resp(resp):
        if "uworld.com" not in resp.url:
            return
        ct = resp.headers.get("content-type", "")
        if "json" not in ct:
            return
        try:
            intercepted_extra[resp.url] = await resp.json()
        except Exception:
            pass

    cookies = login_data.get("cookies", [])

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=headless)
        ctx = await browser.new_context(viewport={"width": 1280, "height": 800})

        # Restore cookies from login session
        if cookies:
            await ctx.add_cookies(cookies)

        page = await ctx.new_page()
        page.on("response", on_resp)

        # Navigate to the performance reports page
        perf_url = f"{APPS_BASE}/performance/reports/{sub_id}"
        print(f"  📊 Opening performance page: {perf_url}")
        try:
            await page.goto(perf_url, wait_until="networkidle", timeout=25000)
            await asyncio.sleep(4)

            if debug:
                await page.screenshot(path="debug_perf.png")

            # Click Systems and Topics tabs to load all data
            for tab_name in ["Systems", "Topics"]:
                try:
                    tab = page.get_by_role("tab", name=re.compile(tab_name, re.I))
                    if await tab.count() == 0:
                        tab = page.locator(
                            f'button:has-text("{tab_name}"), '
                            f'[role="tab"]:has-text("{tab_name}")'
                        )
                    if await tab.count() > 0:
                        await tab.first.click()
                        await asyncio.sleep(3)
                        print(f"    ✅ Clicked {tab_name} tab")
                except Exception as e:
                    if debug:
                        print(f"    {tab_name} tab: {e}")
        except Exception as e:
            print(f"  ⚠️  Performance page failed: {e}")

        # Navigate to previous tests page
        prev_url = f"{APPS_BASE}/previoustests/{sub_id}"
        print(f"  ❌ Opening previous tests: {prev_url}")
        try:
            await page.goto(prev_url, wait_until="networkidle", timeout=25000)
            await asyncio.sleep(5)
            if debug:
                await page.screenshot(path="debug_previoustests.png")
        except Exception as e:
            print(f"  ⚠️  Previous tests page failed: {e}")

        await asyncio.sleep(3)
        await browser.close()

    return intercepted_extra


# ─── DATA PARSERS ─────────────────────────────────────────────────────────────

def _parse_performance_blob(data, performance, source_url=""):
    """Recursively extract performance stats from any JSON shape."""
    if isinstance(data, dict):
        # Overall stats
        for key in ["overall", "total", "summary", "stats"]:
            if key in data and isinstance(data[key], dict):
                sub = data[key]
                pct = (sub.get("percent") or sub.get("percentage")
                       or sub.get("score") or sub.get("correctPercent"))
                if pct is not None and performance["overall"] is None:
                    performance["overall"] = {
                        "percent":  round(float(pct), 1),
                        "correct":  sub.get("correct") or sub.get("correctCount"),
                        "total":    sub.get("total") or sub.get("totalCount"),
                    }

        # Category arrays
        for key in ["subjects", "systems", "topics", "disciplines", "categories",
                    "bySubject", "bySystem", "byTopic",
                    "subjectBreakdown", "systemBreakdown", "topicBreakdown"]:
            if key in data and isinstance(data[key], list):
                parsed = _parse_category_list(data[key])
                if parsed:
                    url_key = source_url.lower()
                    if "system" in key.lower() or "system" in url_key:
                        performance["by_system"].extend(parsed)
                    elif "topic" in key.lower() or "topic" in url_key:
                        performance["by_topic"].extend(parsed)
                    else:
                        performance["by_subject"].extend(parsed)

        # If data itself looks like a category list (common for subject/system endpoints)
        if "data" in data and isinstance(data["data"], list):
            parsed = _parse_category_list(data["data"])
            if parsed:
                url_key = source_url.lower()
                if "system" in url_key:
                    performance["by_system"].extend(parsed)
                elif "topic" in url_key:
                    performance["by_topic"].extend(parsed)
                else:
                    performance["by_subject"].extend(parsed)

        for v in data.values():
            if isinstance(v, (dict, list)):
                _parse_performance_blob(v, performance, source_url)

    elif isinstance(data, list):
        # Top-level list — could be the category list itself
        parsed = _parse_category_list(data)
        if parsed:
            url_key = source_url.lower()
            if "system" in url_key:
                performance["by_system"].extend(parsed)
            elif "topic" in url_key:
                performance["by_topic"].extend(parsed)
            elif "subject" in url_key:
                performance["by_subject"].extend(parsed)
        for item in data:
            if isinstance(item, (dict, list)):
                _parse_performance_blob(item, performance, source_url)


def _parse_category_list(items):
    results = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = (item.get("name") or item.get("system") or item.get("subject")
                or item.get("topic") or item.get("category") or item.get("label")
                or item.get("title") or item.get("disciplineName")
                or item.get("systemName") or item.get("topicName"))
        pct = (item.get("percent") or item.get("percentage") or item.get("score")
               or item.get("correctPercent") or item.get("percentCorrect")
               or item.get("pctCorrect"))
        if pct is None:
            # Try to compute from correct/total
            correct = item.get("correct") or item.get("correctCount") or 0
            total   = item.get("total")   or item.get("totalCount")   or 0
            if total and int(total) > 0:
                pct = round(100 * int(correct) / int(total), 1)
        correct  = item.get("correct")  or item.get("correctCount")
        total    = item.get("total")    or item.get("totalCount") or item.get("count")
        incorrect= item.get("incorrect") or item.get("incorrectCount")
        if name and pct is not None:
            results.append({
                "name":      str(name),
                "percent":   round(float(pct), 1),
                "correct":   correct,
                "total":     total,
                "incorrect": incorrect,
            })
    return results


def _parse_question_blob(data, questions):
    """Recursively find incorrect question objects in any JSON shape."""
    if isinstance(data, dict):
        qid = (data.get("id") or data.get("qid") or data.get("questionId")
               or data.get("question_id") or data.get("itemId")
               or data.get("uwId") or data.get("questionNumber"))
        status = str(data.get("status") or data.get("result") or
                     data.get("answerResult") or "").lower()
        is_incorrect = (
            "incorrect" in status or "wrong" in status
            or data.get("incorrect") is True
            or data.get("isIncorrect") is True
        )
        if qid and is_incorrect:
            questions.append({
                "id":       str(qid),
                "system":   data.get("system")  or data.get("bodySystem")  or data.get("systemName"),
                "subject":  data.get("subject") or data.get("discipline")  or data.get("subjectName"),
                "topic":    data.get("topic")   or data.get("concept")     or data.get("topicName"),
                "subtopic": data.get("subtopic") or data.get("subTopic"),
            })
        for v in data.values():
            if isinstance(v, (dict, list)):
                _parse_question_blob(v, questions)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                _parse_question_blob(item, questions)


# ─── SUMMARIZE FOR CLAUDE ─────────────────────────────────────────────────────

def build_performance_summary(performance):
    lines = []
    if performance["overall"]:
        o = performance["overall"]
        lines.append(f"Overall: {o['percent']}% correct "
                     f"({o.get('correct','?')}/{o.get('total','?')} questions)")
    else:
        lines.append("Overall: Not available")

    def _section(label, items):
        if not items:
            return
        sorted_items = sorted(items, key=lambda x: x["percent"])
        lines.append(f"\nPerformance by {label} ({len(sorted_items)} total):")
        for s in sorted_items:
            c = s.get("correct","?")
            t = s.get("total","?")
            inc = s.get("incorrect")
            inc_str = f", {inc} incorrect" if inc else ""
            lines.append(f"  {s['percent']:5.1f}%  {s['name']}  ({c}/{t}{inc_str})")

    _section("Subject",  performance["by_subject"])
    _section("System",   performance["by_system"])
    _section("Topic",    performance["by_topic"][:30])
    return "\n".join(lines)


def build_incorrect_summary(questions):
    if not questions:
        return "No incorrect question data available."

    by_system  = defaultdict(list)
    by_subject = defaultdict(list)
    by_topic   = defaultdict(int)

    for q in questions:
        by_system [q.get("system")  or "Unknown System" ].append(q)
        by_subject[q.get("subject") or "Unknown Subject"].append(q)
        by_topic  [q.get("topic")   or "Unknown Topic"  ] += 1

    lines = [f"Total incorrect: {len(questions)}"]
    lines.append("\nBy System:")
    for sys, qs in sorted(by_system.items(), key=lambda x: -len(x[1])):
        lines.append(f"  {len(qs):3d}x  {sys}")
    lines.append("\nBy Subject:")
    for sub, qs in sorted(by_subject.items(), key=lambda x: -len(x[1])):
        lines.append(f"  {len(qs):3d}x  {sub}")
    lines.append("\nTop Topics in Incorrects:")
    for topic, count in sorted(by_topic.items(), key=lambda x: -x[1])[:30]:
        lines.append(f"  {count:3d}x  {topic}")

    qids = [q["id"] for q in questions[:50] if q.get("id")]
    if qids:
        lines.append(f"\nSample QIDs: {', '.join(qids[:20])}")

    return "\n".join(lines)


# ─── CLAUDE ANALYSIS ─────────────────────────────────────────────────────────

def analyze_with_claude(perf_summary: str, incorrect_summary: str,
                         incorrect_count: int) -> str:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    prompt = ANALYSIS_PROMPT.format(
        performance_data=perf_summary,
        incorrect_count=incorrect_count,
        incorrect_summary=incorrect_summary,
    )
    print("\n  🤖 Asking Claude to analyze your weak spots...")
    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


# ─── OUTPUT ──────────────────────────────────────────────────────────────────

def save_report(analysis: str, questions: list, performance: dict,
                output_path: Path):
    output_path.parent.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    md = f"# UWorld Weakness Analysis\nGenerated: {ts}\n\n---\n\n{analysis}\n\n---\n\n"
    md += "## Incorrect Questions\n\n"
    md += "| QID | Subject | System | Topic |\n|-----|---------|--------|-------|\n"
    for q in questions[:200]:
        md += f"| {q.get('id','?')} | {q.get('subject','?')} | {q.get('system','?')} | {q.get('topic','?')} |\n"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)

    json_path = output_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated":          ts,
            "performance":        performance,
            "incorrect_questions": questions,
            "question_ids":       [q["id"] for q in questions if q.get("id")],
        }, f, indent=2)

    print(f"\n  💾 Report: {output_path}")
    print(f"  💾 JSON:   {json_path}")
    print(f"     (Paste question IDs into MedTools UWorld Review for drill questions)")


# ─── ANKI-ONLY FALLBACK ───────────────────────────────────────────────────────

async def run_anki_only(output_path: Path):
    print("\n📚 Anki-only mode...")
    all_qids, struggling = await get_anki_weak_cards()
    if not struggling:
        print("❌ No struggling AnKing UWorld cards found.")
        return
    performance = {"overall": None, "by_system": [], "by_subject": [], "by_topic": []}
    questions   = [{"id": q, "system": None, "subject": None, "topic": None}
                   for q in struggling]
    perf_sum  = (f"Source: AnkiConnect\n"
                 f"Total UWorld-tagged cards: {len(all_qids)}\n"
                 f"Cards with lapses: {len(struggling)}")
    inc_sum   = build_incorrect_summary(questions)
    analysis  = analyze_with_claude(perf_sum, inc_sum, len(struggling))
    save_report(analysis, questions, performance, output_path)
    print("\n" + "="*60 + "\n" + analysis + "\n" + "="*60)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

async def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(exist_ok=True)
    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output) if args.output else OUTPUT_DIR / f"report_{ts}.md"

    print("\n" + "="*60)
    print("  UWorld Auto-Analyzer")
    print("="*60)

    if args.anki_only:
        await run_anki_only(output_path)
        return

    email, password = get_credentials(args)
    if not email or not password:
        print("❌ Email and password required. Set UWORLD_EMAIL / UWORLD_PASSWORD in .env")
        sys.exit(1)

    # Kick off AnkiConnect query in parallel (enriches report if available)
    anki_task = asyncio.create_task(get_anki_weak_cards())

    # ── Phase 1: Browser login ────────────────────────────────────────────
    print(f"\n  🌐 Opening UWorld {'(headless)' if args.headless else '(browser window will open)'}...")
    if not args.headless:
        print("  ℹ️  Watch the browser — complete any 2FA prompt manually.")

    try:
        login_data = await browser_login(
            email, password,
            headless=args.headless,
            debug=args.debug,
        )
    except Exception as e:
        print(f"\n  ❌ Login failed: {e}")
        print("  Falling back to Anki-only mode...")
        await run_anki_only(output_path)
        return

    # ── Find subscription ID ─────────────────────────────────────────────
    sub_id = find_medical_sub_id(login_data)
    if not sub_id:
        print("  ❌ Could not identify Step 2 CK subscription ID.")
        print(f"     Sub IDs seen: {login_data['sub_ids']}")
        print(f"     GetAllSubscriptions: {json.dumps(login_data['all_subs_raw'])[:400]}")
        sys.exit(1)

    print(f"\n  📋 Step 2 CK subscription ID: {sub_id}")
    token = login_data.get("token")
    print(f"  🔑 Auth token: {'captured ✅' if token else 'not found — will use cookie auth'}")

    # ── Phase 2: Direct API calls ─────────────────────────────────────────
    performance, questions = await fetch_performance_data(
        sub_id, token, login_data["cookies"], debug=args.debug
    )

    has_perf = bool(performance["by_subject"] or performance["by_system"])
    has_qs   = bool(questions)

    print(f"\n  📊 API results:")
    print(f"      Subjects: {len(performance['by_subject'])}")
    print(f"      Systems:  {len(performance['by_system'])}")
    print(f"      Topics:   {len(performance['by_topic'])}")
    print(f"      Incorrect Qs: {len(questions)}")

    # ── Phase 2B: Browser fallback if API calls didn't get data ──────────
    if not has_perf or not has_qs:
        print("\n  🔄 Direct API calls didn't return data.")
        print("  Opening qbank pages in browser to intercept their API calls...")
        extra = await fetch_via_browser(
            login_data, sub_id,
            headless=args.headless,
            debug=args.debug,
        )
        print(f"  Captured {len(extra)} additional API responses")

        if args.debug:
            print("  Extra endpoints:")
            for url in sorted(extra.keys()):
                print(f"    {url[:100]}")

        for url, data in extra.items():
            _parse_performance_blob(data, performance, url)
            _parse_question_blob(data, questions)

        # Deduplicate again after merge
        seen, unique = set(), []
        for q in questions:
            if q["id"] not in seen:
                seen.add(q["id"])
                unique.append(q)
        questions = unique

        print(f"\n  📊 After browser fallback:")
        print(f"      Subjects: {len(performance['by_subject'])}")
        print(f"      Systems:  {len(performance['by_system'])}")
        print(f"      Incorrect Qs: {len(questions)}")

    # ── Enrich with Anki data ─────────────────────────────────────────────
    try:
        anki_all, anki_weak = await anki_task
        if anki_weak and not questions:
            print(f"\n  ℹ️  Using {len(anki_weak)} weak Anki cards as question list")
            questions = [{"id": q, "system": None, "subject": None, "topic": None}
                         for q in anki_weak]
    except Exception:
        pass

    if not questions and not performance["by_subject"] and not performance["by_system"]:
        print("\n  ❌ No data retrieved from UWorld or Anki.")
        print("  Run with --debug for more detail.")
        sys.exit(1)

    # ── Phase 3: Claude analysis ──────────────────────────────────────────
    perf_summary = build_performance_summary(performance)
    inc_summary  = build_incorrect_summary(questions)

    print("\n" + "─"*60)
    print("PERFORMANCE SUMMARY:")
    print(perf_summary)

    analysis = analyze_with_claude(perf_summary, inc_summary, len(questions))
    save_report(analysis, questions, performance, output_path)

    print("\n" + "="*60)
    print(analysis)
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
