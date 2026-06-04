#!/usr/bin/env python3
"""
uworld_auto.py — Automated UWorld incorrect-question analyzer
──────────────────────────────────────────────────────────────
Logs into UWorld, intercepts its internal API calls, grabs your
performance breakdown + incorrect question list, then asks Claude
to pinpoint your weak spots.

SETUP (one time):
    pip install playwright playwright-stealth
    playwright install chromium

Add to your .env file:
    UWORLD_EMAIL=you@email.com
    UWORLD_PASSWORD=yourpassword

RUN:
    python uworld_auto.py
    python uworld_auto.py --headless        # no browser window
    python uworld_auto.py --debug           # saves screenshots + raw API dumps
    python uworld_auto.py --anki-only       # skip UWorld login, use Anki data only
"""

import asyncio
import json
import os
import sys
import argparse
import getpass
import re
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv
import anthropic

load_dotenv()

# ─── CONFIG ──────────────────────────────────────────────────────────────────

UWORLD_URL = "https://www.uworld.com/app/index.html#/login/"
ANKI_CONNECT_URL = "http://localhost:8765"
OUTPUT_DIR = Path("uworld_reports")

CLAUDE_MODEL = "claude-sonnet-4-6"

ANALYSIS_PROMPT = """You are an expert USMLE tutor. A medical student has shared their UWorld performance data.

Your job: identify their true weak spots and give an actionable study plan.

PERFORMANCE DATA:
{performance_data}

INCORRECT QUESTIONS ({incorrect_count} total):
{incorrect_summary}

Produce exactly these sections (be specific and dense, no filler):

## 🔴 Top Weak Systems
List the 3-5 systems with the lowest % correct. For each, give the exact % and the most likely conceptual gaps based on the question topics listed.

## 🧠 Recurring Concept Gaps
Identify the specific mechanisms, pathways, or facts that the incorrect question topics point to. Be granular — not just "cardiology" but "distinguish systolic vs diastolic HF management" or "beta-blocker contraindications."

## ⚡ Priority Study Plan (Next 2 Weeks)
Ordered list. What to hit first, why, and how (specific Anki tags, First Aid sections, or Sketchy videos if applicable).

## 📊 Quick Stats
- Total questions done: X
- Overall %: X%
- Strongest system: X (X%)
- Weakest system: X (X%)
- Incorrect questions to review: X

Be direct. This student needs to pass boards, not hear filler."""


# ─── ARG PARSING ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Automated UWorld weakness analyzer")
    p.add_argument("--email", help="UWorld email (or set UWORLD_EMAIL in .env)")
    p.add_argument("--password", help="UWorld password (or set UWORLD_PASSWORD in .env)")
    p.add_argument("--headless", action="store_true", help="Run browser invisibly")
    p.add_argument("--debug", action="store_true", help="Save screenshots and raw API data")
    p.add_argument("--anki-only", action="store_true", help="Skip UWorld login, pull from AnkiConnect only")
    p.add_argument("--output", default=None, help="Output file path (default: uworld_reports/report_DATE.md)")
    return p.parse_args()


def get_credentials(args):
    email = args.email or os.getenv("UWORLD_EMAIL") or ""
    password = args.password or os.getenv("UWORLD_PASSWORD") or ""
    if not email:
        email = input("UWorld email: ").strip()
    if not password:
        password = getpass.getpass("UWorld password: ")
    return email, password


# ─── ANKI CONNECT HELPER ─────────────────────────────────────────────────────

async def anki_request(action, **params):
    """Call local AnkiConnect API."""
    import urllib.request
    payload = json.dumps({"action": action, "version": 6, "params": params}).encode()
    try:
        req = urllib.request.Request(ANKI_CONNECT_URL, data=payload)
        with urllib.request.urlopen(req, timeout=10) as r:
            result = json.loads(r.read())
            if result.get("error"):
                raise RuntimeError(result["error"])
            return result["result"]
    except Exception as e:
        raise RuntimeError(f"AnkiConnect error: {e}")


async def get_anki_weak_cards():
    """
    Pull AnKing cards that map to UWorld questions where ease is low
    (i.e., cards you keep failing in Anki).
    AnKing tags look like: AK_Step1_v12::UWorld::QID::12345
    """
    print("  📚 Querying AnkiConnect for weak AnKing/UWorld cards...")

    # Cards in relearning or with ease < 2000 (struggling)
    # Also cards due that have lapse count > 1
    try:
        # Find all AnKing UWorld-tagged cards that are in 'relearn' or have high lapses
        note_ids = await anki_request("findNotes", query="tag:AK_Step*UWorld*QID*")
        if not note_ids:
            print("  ⚠️  No AnKing UWorld cards found. Is the AnKing deck installed?")
            return [], []

        print(f"  Found {len(note_ids)} AnKing UWorld-tagged cards total")

        # Get card info for all of them (batch to avoid timeout)
        BATCH = 500
        all_cards = []
        for i in range(0, len(note_ids), BATCH):
            batch = note_ids[i:i+BATCH]
            cards_info = await anki_request("notesInfo", notes=batch)
            all_cards.extend(cards_info)

        # Extract UWorld QIDs from tags and find struggling cards
        qid_pattern = re.compile(r'QID[_::](\d+)', re.IGNORECASE)
        weak_qids = []
        all_qids = []

        for card in all_cards:
            tags = card.get("tags", [])
            fields = card.get("fields", {})
            # Try to determine if this card is "weak" — has lapses in any field or tag hints
            # We'll pull all QIDs; the weak detection will come from card stats
            for tag in tags:
                m = qid_pattern.search(tag)
                if m:
                    qid = m.group(1)
                    all_qids.append(qid)
                    break

        # Now get card stats to find weak ones
        # Find cards with lapses > 0 (cards you've failed in Anki)
        weak_note_ids = await anki_request(
            "findNotes",
            query="tag:AK_Step*UWorld*QID* prop:lapses>0"
        )

        struggling_qids = []
        if weak_note_ids:
            weak_cards_info = await anki_request("notesInfo", notes=weak_note_ids[:500])
            for card in weak_cards_info:
                for tag in card.get("tags", []):
                    m = qid_pattern.search(tag)
                    if m:
                        struggling_qids.append(m.group(1))
                        break

        print(f"  Found {len(all_qids)} total UWorld-tagged cards, {len(struggling_qids)} with lapses")
        return all_qids, struggling_qids

    except RuntimeError as e:
        print(f"  ⚠️  AnkiConnect not available: {e}")
        return [], []


# ─── UWORLD SCRAPER ──────────────────────────────────────────────────────────

async def scrape_uworld(email, password, headless=False, debug=False):
    """
    Log into UWorld and intercept its internal API calls to get:
    1. Performance breakdown by system/subject
    2. Full list of incorrect questions with topics

    Returns (performance_data, incorrect_questions) or raises on failure.
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        print("\n  Installing Playwright (one-time setup)...")
        os.system(f"{sys.executable} -m pip install playwright -q")
        os.system(f"{sys.executable} -m playwright install chromium --quiet")
        from playwright.async_api import async_playwright

    # Try to import stealth (optional but helps)
    stealth_available = False
    try:
        from playwright_stealth import stealth_async
        stealth_available = True
        print("  🥷 Stealth mode active")
    except ImportError:
        print("  ℹ️  For better reliability: pip install playwright-stealth")

    intercepted = {}  # url -> parsed JSON response

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=headless,
            args=["--disable-blink-features=AutomationControlled"] if not headless else []
        )
        context = await browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            )
        )
        page = await context.new_page()

        if stealth_available:
            await stealth_async(page)

        # ── Intercept API responses ──────────────────────────────────────
        async def on_response(response):
            url = response.url
            # Capture JSON responses from UWorld's API
            if ("uworld.com" in url or "amboss" in url) and response.status == 200:
                ct = response.headers.get("content-type", "")
                if "json" in ct:
                    try:
                        data = await response.json()
                        intercepted[url] = data
                        if debug:
                            print(f"    🔗 Captured: {url[:80]}")
                    except Exception:
                        pass

        page.on("response", on_response)

        # ── LOGIN ────────────────────────────────────────────────────────
        # UWorld is a hash-based SPA: base URL never changes, only #/route/ does
        print("\n  🔐 Logging into UWorld...")
        await page.goto(UWORLD_URL, wait_until="networkidle", timeout=30000)
        await asyncio.sleep(2)

        if debug:
            await page.screenshot(path="debug_01_landing.png")
            print(f"    URL: {page.url}")
            print(f"    Hash: {await page.evaluate('window.location.hash')}")

        # Wait for the AngularJS login form to render
        # The email field has id="login-email" per the error log
        print("  ⏳ Waiting for login form...")
        try:
            await page.wait_for_selector('#login-email', timeout=15000)
        except Exception:
            # Fallback to generic email input
            try:
                await page.wait_for_selector('input[type="email"]', timeout=5000)
            except Exception:
                if debug:
                    await page.screenshot(path="debug_02_no_form.png")
                    html = await page.content()
                    with open("debug_page_source.html", "w", encoding="utf-8") as f:
                        f.write(html)
                    print("    Page source saved to debug_page_source.html")
                raise RuntimeError(
                    "Could not find the login form on UWorld.\n"
                    "Run with --debug to save a screenshot."
                )

        if debug:
            await page.screenshot(path="debug_02_form_found.png")

        # Fill fields via JS — bypasses the navbar overlay that blocks physical clicks.
        # Also triggers AngularJS input/change events so the model updates.
        await page.evaluate("""
            ([em, pw]) => {
                function fill(el, val) {
                    if (!el) return;
                    el.value = val;
                    ['input', 'change', 'blur'].forEach(ev =>
                        el.dispatchEvent(new Event(ev, {bubbles: true}))
                    );
                }
                fill(
                    document.getElementById('login-email') ||
                    document.querySelector('input[type="email"]'),
                    em
                );
                fill(
                    document.querySelector('input[type="password"]'),
                    pw
                );
            }
        """, [email, password])

        await asyncio.sleep(0.5)

        if debug:
            await page.screenshot(path="debug_03_filled.png")

        # Click submit via JS to avoid any overlay issues
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

        # ── WAIT FOR LOGIN TO COMPLETE ───────────────────────────────────
        # UWorld SPA: after login the hash changes from #/login/ to something else
        # (e.g. #/dashboard/ or #/qbank/). We can't use wait_for_url because
        # the base URL stays as /app/index.html the whole time.
        print("  ⏳ Waiting for login to complete...")
        try:
            await page.wait_for_function(
                "!window.location.hash.includes('login')",
                timeout=25000
            )
        except Exception:
            pass

        await asyncio.sleep(2)

        current_hash = await page.evaluate("window.location.hash")
        if debug:
            await page.screenshot(path="debug_03_post_login.png")
            print(f"    URL: {page.url}")
            print(f"    Hash: {current_hash}")

        if "login" in current_hash.lower():
            if debug:
                await page.screenshot(path="debug_04_login_failed.png")
            raise RuntimeError(
                "Login failed — still on the login page after submitting.\n"
                "Check your UWORLD_EMAIL and UWORLD_PASSWORD in .env.\n"
                "If you have 2FA enabled, run without --headless and complete it manually."
            )

        print(f"  ✅ Logged in! (landed at: {page.url})")

        # ── FIND SUBSCRIPTION ID AND NAVIGATE TO QBANK ──────────────────
        # Now we know the exact URL structure:
        #   https://apps.uworld.com/courseapp/usmle/v53/en-US/performance/reports/{sub_id}
        #   https://apps.uworld.com/courseapp/usmle/v53/en-US/previoustests/{sub_id}
        # We just need to find the right sub_id for the Step 2 CK QBank.

        print("\n  🚀 Identifying QBank subscription ID...")
        await asyncio.sleep(2)
        if debug:
            await page.screenshot(path="debug_04_subscriptions.png")

        # Collect all subscription IDs from the payment API URLs
        sub_ids_seen = []
        for url in intercepted:
            m2 = re.search(r'GetPaymentsForSubscription/(\d+)', url)
            if m2:
                sub_ids_seen.append(m2.group(1))

        print(f"    Subscription IDs from account page: {sub_ids_seen}")

        # Parse GetAllSubscriptions to find the medical/Step 2 one
        print(f"    GetAllSubscriptions → ", end="")
        medical_sub_id = None
        for url, data in intercepted.items():
            if "GetAllSubscriptions" in url:
                print(json.dumps(data)[:1000])  # full print for debugging
                def _find_sub(obj, depth=0):
                    if depth > 6: return None
                    if isinstance(obj, dict):
                        # Check if this dict looks like a subscription with a medical product
                        name = " ".join(str(v) for v in obj.values() if isinstance(v, str)).lower()
                        is_medical = any(k in name for k in ["step 2","step2","usmle","ck qbank","step2ck"])
                        if is_medical:
                            for idf in ["subscriptionId","id","courseSubscriptionId","subId","courseId"]:
                                if obj.get(idf):
                                    return str(obj[idf])
                        for v in obj.values():
                            r = _find_sub(v, depth+1)
                            if r: return r
                    elif isinstance(obj, list):
                        for v in obj:
                            r = _find_sub(v, depth+1)
                            if r: return r
                    return None
                medical_sub_id = _find_sub(data)
                break
        else:
            print("(not captured)")

        if medical_sub_id:
            print(f"    Found medical sub ID from GetAllSubscriptions: {medical_sub_id}")
        else:
            # Try each sub ID — open a throwaway page and see which one loads the qbank
            print(f"    Probing each subscription ID to find the qbank...")
            APP = "https://apps.uworld.com/courseapp/usmle/v53/en-US"
            for sid in sub_ids_seen:
                probe_url = f"{APP}/performance/reports/{sid}"
                try:
                    probe = await context.new_page()
                    probe.on("response", on_response)
                    await probe.goto(probe_url, wait_until="domcontentloaded", timeout=12000)
                    landed = probe.url
                    await probe.close()
                    if "apps.uworld.com" in landed and "login" not in landed.lower():
                        medical_sub_id = sid
                        print(f"    ✅ Subscription {sid} loads the qbank")
                        break
                    print(f"    ✗  {sid} → {landed[:60]}")
                except Exception as e:
                    try: await probe.close()
                    except: pass
                    print(f"    ✗  {sid} → error: {e}")

        if not medical_sub_id:
            raise RuntimeError(
                "Could not identify the Step 2 QBank subscription ID.\n"
                "Paste the full output above to Ben/Claude for analysis."
            )

        APP_BASE = "https://apps.uworld.com/courseapp/usmle/v53/en-US"
        print(f"  ✅ Using subscription ID: {medical_sub_id}")

        # Open a dedicated page for the qbank (cookies carry over in the same context)
        qb_page = await context.new_page()
        qb_page.on("response", on_response)

        # ── PERFORMANCE REPORTS: Subjects → Systems → Topics ─────────────
        print("\n  📊 Loading performance reports (Subjects / Systems / Topics)...")

        perf_url = f"{APP_BASE}/performance/reports/{medical_sub_id}"
        await qb_page.goto(perf_url, wait_until="networkidle", timeout=25000)
        await asyncio.sleep(3)

        if debug:
            await qb_page.screenshot(path="debug_05_perf_subjects.png")
        print(f"    Loaded: {qb_page.url}")

        # Click Systems tab
        try:
            systems_tab = qb_page.get_by_role("tab", name=re.compile("systems", re.I))
            if await systems_tab.count() == 0:
                systems_tab = qb_page.locator('button:has-text("Systems"), [role="tab"]:has-text("Systems")')
            if await systems_tab.count() > 0:
                await systems_tab.first.click()
                await asyncio.sleep(3)
                print("    ✅ Clicked Systems tab")
                if debug:
                    await qb_page.screenshot(path="debug_06_perf_systems.png")
        except Exception as e:
            print(f"    Systems tab: {e}")

        # Click Topics tab
        try:
            topics_tab = qb_page.get_by_role("tab", name=re.compile("topics", re.I))
            if await topics_tab.count() == 0:
                topics_tab = qb_page.locator('button:has-text("Topics"), [role="tab"]:has-text("Topics")')
            if await topics_tab.count() > 0:
                await topics_tab.first.click()
                await asyncio.sleep(3)
                print("    ✅ Clicked Topics tab")
                if debug:
                    await qb_page.screenshot(path="debug_07_perf_topics.png")
        except Exception as e:
            print(f"    Topics tab: {e}")

        # ── PREVIOUS TESTS (for incorrect question block info) ────────────
        print("\n  ❌ Loading previous tests / incorrect questions...")

        prev_url = f"{APP_BASE}/previoustests/{medical_sub_id}"
        await qb_page.goto(prev_url, wait_until="networkidle", timeout=25000)
        await asyncio.sleep(4)
        print(f"    Loaded: {qb_page.url}")

        if debug:
            await qb_page.screenshot(path="debug_08_previoustests.png")

        # Also try the Step 2 Review tab (as opposed to Shelf Review)
        try:
            step2_tab = qb_page.get_by_role("tab", name=re.compile("step 2", re.I))
            if await step2_tab.count() == 0:
                step2_tab = qb_page.locator('[role="tab"]:has-text("Step 2"), button:has-text("Step 2")')
            if await step2_tab.count() > 0:
                await step2_tab.first.click()
                await asyncio.sleep(3)
                print("    ✅ Clicked Step 2 Review tab")
        except Exception:
            pass

        # Try navigating to QBank create-test page with incorrect filter
        qbank_url_try = f"{APP_BASE}/qbank/{medical_sub_id}"
        try:
            await qb_page.goto(qbank_url_try, wait_until="networkidle", timeout=12000)
            await asyncio.sleep(3)
            print(f"    QBank page: {qb_page.url}")
        except Exception:
            pass

        # Give everything time to settle and fire remaining API calls
        await asyncio.sleep(5)
        page = qb_page  # update reference for final screenshot

        if debug:
            await page.screenshot(path="debug_final.png")
            with open("debug_api_responses.json", "w") as f:
                json.dump({k: v for k, v in list(intercepted.items())[:30]}, f, indent=2, default=str)
            print(f"    Saved {len(intercepted)} API responses to debug_api_responses.json")

        await browser.close()

    return intercepted


# ─── DATA EXTRACTION ─────────────────────────────────────────────────────────

def extract_performance_data(intercepted: dict):
    """
    Parse the raw intercepted API responses into structured performance data.
    UWorld's internal API structure varies, so we look for recognizable patterns.
    """
    performance = {
        "overall": None,
        "by_system": [],
        "by_subject": [],
        "raw_found": False,
    }
    incorrect_questions = []

    for url, data in intercepted.items():
        url_lower = url.lower()

        # Skip tiny responses (likely not data)
        data_str = json.dumps(data)
        if len(data_str) < 100:
            continue

        # ── Look for performance breakdown ──────────────────────────────
        if any(k in url_lower for k in ["performance", "analytics", "stats", "report"]):
            performance["raw_found"] = True
            _parse_performance_blob(data, performance)

        # ── Look for question lists ──────────────────────────────────────
        if any(k in url_lower for k in ["question", "qbank", "item", "incorrect"]):
            _parse_question_blob(data, incorrect_questions)

        # ── Try generic extraction on everything ────────────────────────
        # Sometimes the data is nested, so scan all responses
        _parse_performance_blob(data, performance)
        _parse_question_blob(data, incorrect_questions)

    # Deduplicate questions by ID
    seen_ids = set()
    unique_questions = []
    for q in incorrect_questions:
        qid = q.get("id") or q.get("qid") or q.get("questionId")
        if qid and qid not in seen_ids:
            seen_ids.add(qid)
            unique_questions.append(q)

    return performance, unique_questions


def _parse_performance_blob(data, performance):
    """Recursively look for performance-shaped data in a JSON blob."""
    if isinstance(data, dict):
        # Look for overall stats
        for key in ["overall", "total", "summary", "stats"]:
            if key in data:
                sub = data[key]
                if isinstance(sub, dict):
                    pct = sub.get("percent") or sub.get("percentage") or sub.get("score")
                    correct = sub.get("correct") or sub.get("correctCount")
                    total = sub.get("total") or sub.get("totalCount") or sub.get("count")
                    if pct is not None and performance["overall"] is None:
                        performance["overall"] = {
                            "percent": round(float(pct), 1),
                            "correct": correct,
                            "total": total,
                        }

        # Look for system/subject arrays
        for key in ["systems", "subjects", "disciplines", "categories", "topics",
                    "bySystem", "bySubject", "systemBreakdown", "subjectBreakdown"]:
            if key in data and isinstance(data[key], list):
                items = data[key]
                parsed = _parse_category_list(items)
                if parsed:
                    if "system" in key.lower() or key in ["systems", "bySystem", "systemBreakdown"]:
                        performance["by_system"].extend(parsed)
                    else:
                        performance["by_subject"].extend(parsed)

        # Recurse into nested dicts
        for v in data.values():
            if isinstance(v, (dict, list)):
                _parse_performance_blob(v, performance)

    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                _parse_performance_blob(item, performance)


def _parse_category_list(items):
    """Parse a list of category performance objects."""
    results = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = (item.get("name") or item.get("system") or item.get("subject")
                or item.get("topic") or item.get("category") or item.get("label"))
        pct = (item.get("percent") or item.get("percentage") or item.get("score")
               or item.get("correctPercent") or item.get("percentCorrect"))
        correct = item.get("correct") or item.get("correctCount")
        total = item.get("total") or item.get("totalCount") or item.get("count")

        if name and pct is not None:
            results.append({
                "name": str(name),
                "percent": round(float(pct), 1),
                "correct": correct,
                "total": total,
            })
    return results


def _parse_question_blob(data, questions):
    """Recursively find question objects in a JSON blob."""
    if isinstance(data, dict):
        # Check if this dict looks like a question
        qid = (data.get("id") or data.get("qid") or data.get("questionId")
               or data.get("question_id") or data.get("itemId"))
        status = str(data.get("status") or data.get("result") or "").lower()
        is_incorrect = "incorrect" in status or "wrong" in status or data.get("incorrect") is True

        if qid and is_incorrect:
            questions.append({
                "id": str(qid),
                "system": data.get("system") or data.get("bodySystem"),
                "subject": data.get("subject") or data.get("discipline"),
                "topic": data.get("topic") or data.get("concept"),
                "subtopic": data.get("subtopic") or data.get("subTopic"),
            })
        elif qid and not status:
            # Might be in a list of incorrect questions without explicit status
            questions.append({
                "id": str(qid),
                "system": data.get("system") or data.get("bodySystem"),
                "subject": data.get("subject") or data.get("discipline"),
                "topic": data.get("topic") or data.get("concept"),
                "subtopic": data.get("subtopic") or data.get("subTopic"),
            })

        # Recurse
        for v in data.values():
            if isinstance(v, (dict, list)):
                _parse_question_blob(v, questions)

    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                _parse_question_blob(item, questions)


# ─── SUMMARIZE FOR CLAUDE ─────────────────────────────────────────────────────

def build_performance_summary(performance, incorrect_questions):
    """Build a readable summary of performance data for Claude."""
    lines = []

    # Overall
    if performance["overall"]:
        o = performance["overall"]
        lines.append(f"Overall: {o['percent']}% correct ({o.get('correct', '?')}/{o.get('total', '?')} questions)")
    else:
        lines.append("Overall: Not found in API data")

    # By system
    if performance["by_system"]:
        systems = sorted(performance["by_system"], key=lambda x: x["percent"])
        lines.append(f"\nPerformance by System ({len(systems)} systems):")
        for s in systems:
            total_str = f" ({s['correct']}/{s['total']})" if s.get("total") else ""
            lines.append(f"  {s['percent']:5.1f}%  {s['name']}{total_str}")

    # By subject
    if performance["by_subject"]:
        subjects = sorted(performance["by_subject"], key=lambda x: x["percent"])
        lines.append(f"\nPerformance by Subject ({len(subjects)} subjects):")
        for s in subjects[:20]:  # Top 20 weakest
            total_str = f" ({s['correct']}/{s['total']})" if s.get("total") else ""
            lines.append(f"  {s['percent']:5.1f}%  {s['name']}{total_str}")

    return "\n".join(lines)


def build_incorrect_summary(incorrect_questions):
    """Summarize incorrect questions by system/topic for Claude."""
    if not incorrect_questions:
        return "No incorrect question data extracted."

    # Group by system
    by_system = defaultdict(list)
    by_topic = defaultdict(int)

    for q in incorrect_questions:
        system = q.get("system") or "Unknown System"
        topic = q.get("topic") or q.get("subject") or "Unknown Topic"
        by_system[system].append(q)
        by_topic[topic] += 1

    lines = [f"Total incorrect: {len(incorrect_questions)}"]
    lines.append("\nIncorrect by System:")
    for system, qs in sorted(by_system.items(), key=lambda x: -len(x[1])):
        lines.append(f"  {len(qs):3d}x  {system}")

    lines.append("\nTop Recurring Topics in Incorrects:")
    for topic, count in sorted(by_topic.items(), key=lambda x: -x[1])[:30]:
        lines.append(f"  {count:3d}x  {topic}")

    # Sample of question IDs
    qids = [q["id"] for q in incorrect_questions[:50] if q.get("id")]
    if qids:
        lines.append(f"\nSample Question IDs: {', '.join(qids[:20])}")

    return "\n".join(lines)


# ─── CLAUDE ANALYSIS ─────────────────────────────────────────────────────────

def analyze_with_claude(performance_summary: str, incorrect_summary: str, incorrect_count: int) -> str:
    """Send data to Claude and get weakness analysis."""
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    prompt = ANALYSIS_PROMPT.format(
        performance_data=performance_summary,
        incorrect_count=incorrect_count,
        incorrect_summary=incorrect_summary,
    )

    print("\n  🤖 Asking Claude to analyze your weak spots...")
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


# ─── OUTPUT ──────────────────────────────────────────────────────────────────

def save_report(analysis: str, incorrect_questions: list, performance: dict, output_path: Path):
    """Save the analysis as a Markdown file."""
    output_path.parent.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    content = f"""# UWorld Weakness Analysis
Generated: {timestamp}

---

{analysis}

---

## Raw Data

### Incorrect Question IDs
```
{', '.join(q['id'] for q in incorrect_questions if q.get('id'))}
```

### Incorrect Questions with Topics
| QID | System | Subject | Topic |
|-----|--------|---------|-------|
"""
    for q in incorrect_questions[:200]:
        content += f"| {q.get('id','?')} | {q.get('system','?')} | {q.get('subject','?')} | {q.get('topic','?')} |\n"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    # Also save JSON for use with MedTools UWorld Review tool
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated": timestamp,
            "performance": performance,
            "incorrect_questions": incorrect_questions,
            "question_ids": [q["id"] for q in incorrect_questions if q.get("id")],
        }, f, indent=2)

    print(f"\n  💾 Report saved to: {output_path}")
    print(f"  💾 JSON data saved to: {json_path}")
    print(f"     (You can paste the question IDs from the JSON into MedTools UWorld Review)")


# ─── FALLBACK: ANKI-ONLY MODE ─────────────────────────────────────────────────

async def run_anki_only_analysis(output_path: Path):
    """
    Skip UWorld login entirely.
    Pull struggling cards from AnkiConnect and analyze with Claude.
    """
    print("\n📚 Anki-only mode: pulling weak AnKing/UWorld cards from local Anki...")
    all_qids, struggling_qids = await get_anki_weak_cards()

    if not all_qids and not struggling_qids:
        print("❌ No AnKing UWorld cards found. Make sure:")
        print("   1. Anki is open")
        print("   2. AnkiConnect is installed")
        print("   3. The AnKing deck is loaded")
        return

    performance = {
        "overall": None,
        "by_system": [],
        "by_subject": [],
        "note": "Pulled from AnkiConnect - no UWorld login data"
    }

    incorrect_questions = [{"id": qid, "system": None, "subject": None, "topic": None}
                           for qid in struggling_qids]

    performance_summary = (
        f"Source: AnkiConnect (local Anki data)\n"
        f"Total UWorld-tagged AnKing cards: {len(all_qids)}\n"
        f"Cards with at least 1 lapse (you've struggled with): {len(struggling_qids)}\n"
        f"\nNote: Full system/topic breakdown not available without UWorld login."
    )
    incorrect_summary = build_incorrect_summary(incorrect_questions)
    incorrect_summary += f"\n\nAnki QIDs with lapses: {', '.join(struggling_qids[:50])}"

    analysis = analyze_with_claude(performance_summary, incorrect_summary, len(struggling_qids))
    save_report(analysis, incorrect_questions, performance, output_path)

    print("\n" + "="*60)
    print(analysis)
    print("="*60)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

async def main():
    args = parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output) if args.output else OUTPUT_DIR / f"report_{timestamp}.md"

    print("\n" + "="*60)
    print("  UWorld Auto-Analyzer")
    print("="*60)

    # ── Anki-only mode ───────────────────────────────────────────────
    if args.anki_only:
        await run_anki_only_analysis(output_path)
        return

    # ── Full UWorld scrape mode ──────────────────────────────────────
    email, password = get_credentials(args)

    if not email or not password:
        print("❌ Email and password are required.")
        print("   Set UWORLD_EMAIL and UWORLD_PASSWORD in your .env file, or pass --email and --password")
        sys.exit(1)

    # Also check for AnkiConnect data in parallel (used to enrich the analysis)
    print("\n  🔍 Checking AnkiConnect for additional context...")
    anki_qids_task = asyncio.create_task(get_anki_weak_cards())

    # Scrape UWorld
    try:
        print(f"\n  🌐 Opening UWorld{'(headless)' if args.headless else ' (browser window will open)'}...")
        if not args.headless:
            print("  ℹ️  The browser will open — you can watch it work. Don't click anything.")
            print("  ℹ️  If there's a 2FA prompt, complete it manually.")

        intercepted = await scrape_uworld(
            email=email,
            password=password,
            headless=args.headless,
            debug=args.debug,
        )

        print(f"\n  📦 Captured {len(intercepted)} API responses from UWorld")
        print("  API endpoints hit:")
        for url in sorted(intercepted.keys()):
            print(f"    {url[:120]}")

        if not intercepted:
            print("  ⚠️  No API responses captured. UWorld may have blocked the session.")
            print("  💡 Try running without --headless so you can see what happened.")
            print("  💡 Or try: python uworld_auto.py --anki-only")
            sys.exit(1)

        # Extract structured data
        performance, incorrect_questions = extract_performance_data(intercepted)

        print(f"\n  📊 Extracted:")
        print(f"      Overall: {performance['overall']}")
        print(f"      Systems: {len(performance['by_system'])} found")
        print(f"      Subjects: {len(performance['by_subject'])} found")
        print(f"      Incorrect Qs: {len(incorrect_questions)} found")

    except Exception as e:
        print(f"\n  ❌ UWorld scraping failed: {e}")
        print("\n  Falling back to Anki-only mode...")
        await run_anki_only_analysis(output_path)
        return

    # Wait for Anki data and enrich if available
    try:
        anki_all, anki_weak = await anki_qids_task
        if anki_weak and not incorrect_questions:
            print(f"\n  ℹ️  No incorrect questions from UWorld API — using {len(anki_weak)} weak Anki cards instead")
            incorrect_questions = [{"id": qid, "system": None, "subject": None, "topic": None}
                                   for qid in anki_weak]
    except Exception:
        pass

    if not incorrect_questions and not performance["by_system"]:
        print("\n  ⚠️  Could not extract usable data from UWorld's API.")
        print("  This usually means UWorld changed their API structure.")
        print("  Run with --debug to save raw API responses for inspection.")
        print("  Or try: python uworld_auto.py --anki-only")
        sys.exit(1)

    # Build summaries and analyze
    performance_summary = build_performance_summary(performance, incorrect_questions)
    incorrect_summary = build_incorrect_summary(incorrect_questions)

    print("\n" + "─"*60)
    print("PERFORMANCE SUMMARY:")
    print(performance_summary)

    analysis = analyze_with_claude(performance_summary, incorrect_summary, len(incorrect_questions))
    save_report(analysis, incorrect_questions, performance, output_path)

    print("\n" + "="*60)
    print(analysis)
    print("="*60)
    print(f"\n✅ Done! Report saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
