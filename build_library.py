#!/usr/bin/env python3
"""
build_library.py — Build a local UWorld question library (ID → Subject/System/Topic).

Strategy:
  1. Log in with stealth patches so UWorld doesn't detect automation.
  2. Navigate to the Search page inside the qbank.
  3. Type search terms into the input via JS and capture the API responses
     the Angular app itself fires (no fetch() calls — no CORS issues).
  4. Save incrementally to uworld_library.json.

RUN:
    python build_library.py              # full run
    python build_library.py --headless   # no visible browser (may trigger detection)
    python build_library.py --fresh      # wipe and start over
    python build_library.py --show       # print stats and exit
    python build_library.py --debug      # save screenshots at each step
"""

import asyncio, json, os, re, sys, argparse
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

LIBRARY_PATH  = Path("uworld_library.json")
PROGRESS_PATH = Path("uworld_library_progress.json")
APPS_BASE     = "https://apps.uworld.com/courseapp/usmle/v53/en-US"

SEARCH_TERMS = [
    "cardiovascular","renal","pulmonary","gastrointestinal",
    "endocrine","neurology","hematology","infectious",
    "obstetrics","gynecology","pediatrics","psychiatry",
    "dermatology","musculoskeletal","oncology","surgery",
    "emergency","biostatistics","pharmacology","immunology",
    "fever","chest","abdominal","headache","fatigue",
    "hypertension","diabetes","pregnancy","trauma","blood",
    "fracture","weight","nausea","seizure","rash",
    "vision","urinary","alcohol","screening","ethics",
    "vaccine","antibiotic","pain","cough","weakness",
]

# ── Stealth JS injected before every page load ───────────────────────────────
# Hides the most common Playwright / headless Chrome fingerprints.
STEALTH_SCRIPT = """
// 1. Hide webdriver flag
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});

// 2. Add a realistic Chrome object
if (!window.chrome) {
    window.chrome = {
        app: {isInstalled: false},
        runtime: {},
        csi: () => {},
        loadTimes: () => {},
    };
}

// 3. Add fake plugins so navigator.plugins isn't empty
Object.defineProperty(navigator, 'plugins', {
    get: () => [
        {name:'Chrome PDF Plugin', filename:'internal-pdf-viewer'},
        {name:'Chrome PDF Viewer', filename:'mhjfbmdgcfjbbpaeojofohoefgiehjai'},
        {name:'Native Client', filename:'internal-nacl-plugin'},
    ]
});

// 5. Realistic languages
Object.defineProperty(navigator, 'languages', {get: () => ['en-US','en']});
"""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--headless", action="store_true")
    p.add_argument("--fresh",    action="store_true")
    p.add_argument("--show",     action="store_true")
    p.add_argument("--debug",    action="store_true")
    return p.parse_args()

def load_library():
    return json.loads(LIBRARY_PATH.read_text(encoding="utf-8")) if LIBRARY_PATH.exists() else {}

def save_library(lib):
    LIBRARY_PATH.write_text(json.dumps(lib, indent=2), encoding="utf-8")

def load_progress():
    return json.loads(PROGRESS_PATH.read_text(encoding="utf-8")) if PROGRESS_PATH.exists() \
           else {"completed_terms": [], "sub_id": None}

def save_progress(p):
    PROGRESS_PATH.write_text(json.dumps(p, indent=2), encoding="utf-8")

def show_stats(lib):
    if not lib: print("Library is empty."); return
    subjects, systems = {}, {}
    for q in lib.values():
        s = q.get("subject","?"); subjects[s] = subjects.get(s, 0)+1
        s = q.get("system", "?"); systems [s] = systems .get(s, 0)+1
    print(f"\n{'='*50}\n  UWorld Library: {len(lib)} questions\n{'='*50}")
    print("\nBy Subject:")
    for s,n in sorted(subjects.items(), key=lambda x:-x[1]): print(f"  {n:4d}  {s}")
    print("\nBy System (top 20):")
    for s,n in sorted(systems.items(),  key=lambda x:-x[1])[:20]: print(f"  {n:4d}  {s}")
    missing = sum(1 for q in lib.values() if not q.get("topic"))
    print(f"\n  Missing topic: {missing} questions")


# ── Parse question records from any JSON shape ────────────────────────────────
def parse_questions(data):
    results = []
    def _ex(obj):
        if not isinstance(obj, dict): return
        qid = (obj.get("id") or obj.get("qid") or obj.get("questionId")
               or obj.get("itemId") or obj.get("uwId"))
        if not qid: return
        subject = (obj.get("subject") or obj.get("subjectName") or obj.get("discipline"))
        system  = (obj.get("system")  or obj.get("systemName")  or obj.get("bodySystem"))
        topic   = (obj.get("topic")   or obj.get("topicName")   or obj.get("concept"))
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
            _ex(obj)
            for v in obj.values(): _walk(v)
        elif isinstance(obj, list):
            for v in obj: _walk(v)
    _walk(data)
    return results


# ── Main browser session ───────────────────────────────────────────────────────
async def run(args):
    from playwright.async_api import async_playwright

    lib      = load_library()
    progress = load_progress()

    email    = os.getenv("UWORLD_EMAIL")    or input("UWorld email: ").strip()
    password = os.getenv("UWORLD_PASSWORD") or __import__("getpass").getpass("UWorld password: ")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=args.headless,
            args=["--disable-blink-features=AutomationControlled"]
        )
        # ONE context for everything — stealth applied, native Playwright
        # fill() handles Angular forms fine regardless of stealth patches.
        ctx = await browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            locale="en-US",
            timezone_id="America/New_York",
        )
        await ctx.add_init_script(STEALTH_SCRIPT)

        sub_id_box = {}
        async def capture_subs(resp):
            if "GetAllSubscriptions" in resp.url or "GetPayments" in resp.url:
                try: sub_id_box[resp.url] = await resp.json()
                except Exception: pass

        page = await ctx.new_page()
        page.on("response", capture_subs)

        # ── Login ─────────────────────────────────────────────────────────
        print("  🔐 Logging in...")
        await page.goto("https://www.uworld.com/app/index.html#/login/",
                        wait_until="networkidle", timeout=30000)
        await asyncio.sleep(2)

        try: await page.wait_for_selector("#login-email", timeout=12000)
        except Exception: await page.wait_for_selector('input[type="email"]', timeout=5000)

        # Fill credentials using JS directly — this is what worked reliably
        # before. Playwright's fill() fires 'input' but AngularJS also needs
        # 'change' and 'blur' to update its ng-model binding.
        await page.evaluate("""([em, pw]) => {
            function fill(el, val) {
                if (!el) return;
                el.value = val;
                el.dispatchEvent(new Event('input',  {bubbles:true}));
                el.dispatchEvent(new Event('change', {bubbles:true}));
                el.dispatchEvent(new Event('blur',   {bubbles:true}));
            }
            fill(document.getElementById('login-email'), em);
            fill(document.getElementById('login-password'), pw);
        }""", [email, password])
        await asyncio.sleep(0.5)

        if args.debug:
            await page.screenshot(path="debug_01b_pre_submit.png")
            print("     Saved debug_01b_pre_submit.png — check fields are filled")

        # Click submit via JS — Playwright click has overlay issues, JS click does not
        await page.evaluate(
            "() => { const b = document.getElementById('login-submit')"
            " || document.querySelector('button[type=submit]')"
            " || [...document.querySelectorAll('button')]"
            ".find(b => /login|sign in/i.test(b.textContent));"
            " if (b) b.click(); }"
        )
        await asyncio.sleep(1)
        # If still on login page, try Enter as backup
        try:
            if "login" in (await page.evaluate("window.location.hash")).lower():
                await page.keyboard.press("Enter")
        except Exception:
            pass

        try: await page.wait_for_function(
            "!window.location.hash.includes('login')", timeout=25000)
        except Exception: pass
        await asyncio.sleep(3)

        current_hash = await page.evaluate("window.location.hash")
        if "login" in current_hash.lower():
            await page.screenshot(path="debug_login_failed.png")  # always save
            # Check for error message on the page
            err_text = await page.evaluate(
                "() => document.querySelector('[class*=error],[class*=alert],"
                "[class*=message]')?.textContent?.trim() || '(no error text found)'"
            )
            raise RuntimeError(
                f"Login failed. Page error: {err_text}\n"
                "Check debug_login_failed.png to see the login page state.\n"
                "If you see 'Unable to authenticate', try logging into uworld.com\n"
                "manually first — the account may be temporarily rate-limited."
            )
        print(f"  ✅ Logged in!")
        if args.debug: await page.screenshot(path="debug_01_loggedin.png")

        # ── Find subscription ID ──────────────────────────────────────────
        await asyncio.sleep(2)
        sub_id = progress.get("sub_id")
        if not sub_id:
            all_subs = next((v for k,v in sub_id_box.items() if "GetAllSubscriptions" in k), None)
            if all_subs and isinstance(all_subs, list):
                for item in all_subs:
                    if not isinstance(item, dict): continue
                    if item.get("IsSim") or item.get("FormId"): continue
                    name = str(item.get("CourseName") or "").lower()
                    if any(x in name for x in ["self-assessment","free trial"]): continue
                    sid = item.get("SubscriptionId")
                    if sid: sub_id = str(sid); break
            if not sub_id:
                ids = sorted({re.search(r'GetPaymentsForSubscription/(\d+)', u).group(1)
                              for u in sub_id_box
                              if re.search(r'GetPaymentsForSubscription/(\d+)', u)}, key=int)
                sub_id = ids[1] if len(ids) > 1 else (ids[0] if ids else None)
            if not sub_id: raise RuntimeError("Could not find subscription ID.")
            progress["sub_id"] = sub_id
            save_progress(progress)
        print(f"  📋 Subscription ID: {sub_id}")

        # ── Navigate into the qbank via Launch button ─────────────────────
        # This is the critical step — clicking Launch (or following its href)
        # triggers the auth handoff that sets apps.uworld.com cookies.
        print("  🚀 Launching qbank...")
        qb_page = None

        # Strategy 1: get the href directly and navigate to it
        launch_href = await page.evaluate(
            "() => { const b=[...document.querySelectorAll('a,button')]"
            ".find(e=>/launch/i.test(e.textContent));"
            " return b&&b.href?b.href:null; }"
        )
        if launch_href and "apps.uworld.com" in launch_href:
            qb_page = await ctx.new_page()
            await qb_page.goto(launch_href, wait_until="domcontentloaded", timeout=25000)
            await asyncio.sleep(4)
            if args.debug:
                await qb_page.screenshot(path="debug_02_qbank.png")
                print(f"     qbank URL: {qb_page.url}")

        # Strategy 2: click Launch and catch the new tab
        if not qb_page or "apps.uworld.com" not in qb_page.url:
            try:
                async with ctx.expect_page(timeout=8000) as pg_info:
                    await page.evaluate(
                        "() => { const b=[...document.querySelectorAll('a,button')]"
                        ".find(e=>/launch/i.test(e.textContent)); if(b)b.click(); }"
                    )
                qb_page = await pg_info.value
                await qb_page.wait_for_load_state("domcontentloaded", timeout=20000)
                await asyncio.sleep(4)
                if args.debug:
                    await qb_page.screenshot(path="debug_02_qbank.png")
                    print(f"     qbank URL: {qb_page.url}")
            except Exception as e:
                if args.debug: print(f"     Launch click: {e}")

        # Strategy 3: direct navigation (last resort)
        if not qb_page or "apps.uworld.com" not in qb_page.url:
            if args.debug: print("     Falling back to direct navigation")
            qb_page = await ctx.new_page()
            await qb_page.goto(f"{APPS_BASE}/dashboard/{sub_id}",
                               wait_until="domcontentloaded", timeout=25000)
            await asyncio.sleep(4)

        if args.debug: print(f"  In qbank: {qb_page.url}")

        # ── Open search page ──────────────────────────────────────────────
        search_url = f"{APPS_BASE}/search/{sub_id}/false"
        print(f"  🔍 Opening search page...")
        await qb_page.goto(search_url, wait_until="networkidle", timeout=30000)
        await asyncio.sleep(6)  # Angular needs time to fully initialize

        if args.debug:
            await qb_page.screenshot(path="debug_03_search_page.png")
            print(f"     URL: {qb_page.url}")

        # Report all inputs found — critical for debugging
        all_inputs = await qb_page.evaluate(
            "() => [...document.querySelectorAll('input')].map(el=>({"
            "type:el.type,placeholder:el.placeholder,id:el.id,"
            "name:el.name,visible:el.offsetParent!==null,"
            "cls:el.className.substring(0,50)}))"
        )
        if args.debug:
            print(f"     Inputs found: {len(all_inputs)}")
            for inp in all_inputs: print(f"       {inp}")
        elif len(all_inputs) == 0:
            print("  ⚠️  No inputs found on search page — page may not be authenticated.")
            print("     Run with --debug and check debug_03_search_page.png")

        # Dismiss any modal/overlay (one-liner JS, no string concatenation issues)
        await qb_page.evaluate(
            "() => { document.querySelectorAll('[class*=modal] button,[class*=dialog] button,"
            "[class*=alert] button').forEach(b=>b.click());"
            " document.querySelectorAll('[class*=overlay],[class*=backdrop]')"
            ".forEach(el=>{if(el.style)el.style.pointerEvents='none';}); }"
        )
        await asyncio.sleep(1)

        # ── Set up response interceptor on the search page ────────────────
        # We capture ALL JSON responses — the Angular app will fire real
        # gateway-api calls when we type in the search box.
        captured_items = []
        async def on_search_response(resp):
            ct = resp.headers.get("content-type","")
            if "json" not in ct: return
            # Only care about responses that look like question data
            url = resp.url
            if not any(x in url for x in ["gateway-api","search","question","item","qbank"]):
                return
            try:
                body = await resp.json()
                items = parse_questions(body)
                if items:
                    captured_items.extend(items)
                    if args.debug:
                        print(f"       📡 {len(items)} questions from {url[:70]}")
            except Exception:
                pass

        qb_page.on("response", on_search_response)

        # ── Search loop ───────────────────────────────────────────────────
        terms_to_run = [t for t in SEARCH_TERMS if t not in progress["completed_terms"]]
        print(f"\n  Running {len(terms_to_run)} search terms...\n")

        for i, term in enumerate(terms_to_run):
            print(f"  [{i+1}/{len(terms_to_run)}] '{term}'", end=" ", flush=True)
            captured_items.clear()

            # Fill the search input using JS (works even if Playwright can't "see" it)
            filled = await qb_page.evaluate(f"""() => {{
                // Try multiple strategies to find the search input
                const candidates = [
                    ...document.querySelectorAll('input[type=text]'),
                    ...document.querySelectorAll('input[type=search]'),
                    ...document.querySelectorAll('input:not([type=hidden])'),
                ];
                const inp = candidates.find(el => el.offsetParent !== null)
                         || candidates[0];  // fallback: first input even if hidden

                if (!inp) return false;

                // Focus and fill
                inp.focus();
                inp.value = {json.dumps(term)};
                // Fire events Angular listens to
                inp.dispatchEvent(new Event('focus',   {{bubbles:true}}));
                inp.dispatchEvent(new Event('input',   {{bubbles:true}}));
                inp.dispatchEvent(new Event('change',  {{bubbles:true}}));
                inp.dispatchEvent(new KeyboardEvent('keydown',{{key:'Enter',keyCode:13,bubbles:true}}));
                inp.dispatchEvent(new KeyboardEvent('keypress',{{key:'Enter',keyCode:13,bubbles:true}}));
                inp.dispatchEvent(new KeyboardEvent('keyup',  {{key:'Enter',keyCode:13,bubbles:true}}));
                return inp.placeholder || inp.id || inp.name || '(found)';
            }}""")

            # Also press Enter via Playwright keyboard (more reliable for form submit)
            await qb_page.keyboard.press("Enter")
            await asyncio.sleep(6)  # wait for API response

            if args.debug:
                await qb_page.screenshot(path=f"debug_search_{term[:8]}.png")
                print(f"\n       input filled: {filled}")

            # Merge captured results into library
            new_this = 0
            for q in captured_items:
                qid = q["id"]
                if qid not in lib:
                    lib[qid] = q; new_this += 1
                else:
                    for f in ["subject","system","topic","pct_correct"]:
                        if not lib[qid].get(f) and q.get(f):
                            lib[qid][f] = q[f]

            print(f"→ {len(captured_items)} results, {new_this} new  (total: {len(lib)})")

            # Save after every term
            save_library(lib)
            progress["completed_terms"].append(term)
            save_progress(progress)

            # Clear search box for next term
            await qb_page.evaluate("""() => {
                const inp = document.querySelector('input[type=text],input[type=search],input:not([type=hidden])');
                if(inp){inp.value='';inp.dispatchEvent(new Event('input',{bubbles:true}));}
            }""")
            await asyncio.sleep(0.5)

        await browser.close()

    print(f"\n{'='*50}")
    print(f"  Done! {len(lib)} questions in library.")
    print(f"  Saved to: {LIBRARY_PATH}")
    show_stats(lib)


async def main():
    args = parse_args()
    if args.show:
        show_stats(load_library()); return
    if args.fresh:
        LIBRARY_PATH.unlink(missing_ok=True)
        PROGRESS_PATH.unlink(missing_ok=True)
        print("  🗑️  Wiped existing library.")

    lib = load_library()
    prog = load_progress()
    print(f"\n{'='*50}\n  UWorld Library Builder\n{'='*50}")
    print(f"  Library: {len(lib)} questions")
    print(f"  Completed: {len(prog['completed_terms'])}/{len(SEARCH_TERMS)} terms")

    await run(args)


if __name__ == "__main__":
    asyncio.run(main())
