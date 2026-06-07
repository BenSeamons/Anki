#!/usr/bin/env python3
"""
build_library.py — Build a local UWorld question library (ID -> Subject/System/Topic).

AUTH STRATEGY — Persistent Chrome Profile:
  - Uses launch_persistent_context() with a saved profile directory.
  - First run: tries automated login; if reCAPTCHA blocks it, waits for
    the user to log in manually in the visible browser window. The user
    only needs to do this ONCE — the session is saved in the profile.
  - Subsequent runs: profile already has valid session cookies → no login needed.

SEARCH STRATEGY:
  - Navigates to apps.uworld.com/search via the Launch button.
  - Intercepts gateway-api.uworld.com JSON responses (avoids CORS issues).
  - Types 65 search terms, captures question ID → Subject/System/Topic mappings.
  - Saves incrementally after each term.

RUN:
    python build_library.py              # resume where you left off
    python build_library.py --debug      # verbose + screenshots
    python build_library.py --fresh      # wipe library and restart
    python build_library.py --show       # print library stats and exit
    python build_library.py --resetprofile   # delete saved Chrome profile (force re-login)
"""

import asyncio, json, os, re, sys, argparse, time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

LIBRARY_PATH  = Path("uworld_library.json")
PROGRESS_PATH = Path("uworld_library_progress.json")
PROFILE_DIR   = Path("uworld_pw_profile")   # persisted across runs (Playwright Chromium)
APPS_BASE     = "https://apps.uworld.com/courseapp/usmle/v53/en-US"

SEARCH_TERMS = [
    "cardiovascular", "renal", "pulmonary", "gastrointestinal",
    "endocrine", "neurology", "hematology", "infectious",
    "obstetrics", "gynecology", "pediatrics", "psychiatry",
    "dermatology", "musculoskeletal", "oncology", "surgery",
    "emergency", "biostatistics", "pharmacology", "immunology",
    "fever", "chest", "abdominal", "headache", "fatigue",
    "hypertension", "diabetes", "pregnancy", "trauma", "blood",
    "fracture", "weight", "nausea", "seizure", "rash",
    "vision", "urinary", "alcohol", "screening", "ethics",
    "vaccine", "antibiotic", "pain", "cough", "weakness",
    "heart", "lung", "kidney", "liver", "brain",
    "cancer", "infection", "stroke", "failure", "syndrome",
    "therapy", "diagnosis", "management", "prevention", "drug",
    "test", "patient", "woman", "man", "child",
]


# ── Utility functions ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--headless",      action="store_true", help="Run headless (only works if already logged in)")
    p.add_argument("--fresh",         action="store_true", help="Wipe library and restart")
    p.add_argument("--show",          action="store_true", help="Print stats and exit")
    p.add_argument("--debug",         action="store_true", help="Verbose output + screenshots")
    p.add_argument("--resetprofile",  action="store_true", help="Delete saved Chrome profile")
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
    if not lib:
        print("Library is empty."); return
    subjects, systems = {}, {}
    for q in lib.values():
        s = q.get("subject", "?"); subjects[s] = subjects.get(s, 0) + 1
        s = q.get("system",  "?"); systems[s]  = systems.get(s, 0)  + 1
    print(f"\n{'='*50}\n  UWorld Library: {len(lib)} questions\n{'='*50}")
    print("\nBy Subject:")
    for s, n in sorted(subjects.items(), key=lambda x: -x[1]):
        print(f"  {n:4d}  {s}")
    print("\nBy System (top 20):")
    for s, n in sorted(systems.items(), key=lambda x: -x[1])[:20]:
        print(f"  {n:4d}  {s}")
    print(f"\n  Missing topic: {sum(1 for q in lib.values() if not q.get('topic'))}")

def parse_questions(data):
    results = []
    def _ex(obj):
        if not isinstance(obj, dict): return
        qid = (obj.get("id") or obj.get("qid") or obj.get("questionId")
               or obj.get("itemId") or obj.get("uwId"))
        if not qid: return
        subject = obj.get("subject") or obj.get("subjectName") or obj.get("discipline")
        system  = obj.get("system")  or obj.get("systemName")  or obj.get("bodySystem")
        topic   = obj.get("topic")   or obj.get("topicName")   or obj.get("concept")
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
            _ex(obj); [_walk(v) for v in obj.values()]
        elif isinstance(obj, list):
            [_walk(v) for v in obj]
    _walk(data)
    return results

async def wait_for_content(page, timeout=60, debug=False):
    deadline = time.time() + timeout
    while time.time() < deadline:
        await asyncio.sleep(2)
        c = await page.evaluate("""() => ({
            nav:    document.querySelectorAll('nav a,[class*=sidebar] a').length,
            inputs: document.querySelectorAll('input:not([type=hidden])').length,
            btns:   document.querySelectorAll('button').length,
        })""")
        if debug: print(f"     [wait] {c}")
        if (c.get("inputs",0) + c.get("btns",0) + c.get("nav",0)) >= 3:
            return True
    return False


# ── Main run ───────────────────────────────────────────────────────────────────

async def run(args):
    from playwright.async_api import async_playwright
    from playwright_stealth import Stealth

    lib      = load_library()
    progress = load_progress()
    email    = os.getenv("UWORLD_EMAIL")    or input("UWorld email: ").strip()
    password = os.getenv("UWORLD_PASSWORD") or __import__("getpass").getpass()

    sub_id_box     = {}
    captured_items = []

    PROFILE_DIR.mkdir(exist_ok=True)

    # Remove stale lockfile from a previous crashed run
    for lockname in ("lockfile", "SingletonLock", "SingletonSocket"):
        lf = PROFILE_DIR / lockname
        if lf.exists():
            try: lf.unlink(); print(f"  [CLEAN]  Removed stale {lockname}")
            except Exception: pass

    async with async_playwright() as pw:

        # ── LAUNCH PERSISTENT CHROMIUM ─────────────────────────────────────
        # Uses Playwright's bundled Chromium (not real Chrome) to avoid lockfile
        # conflicts with the user's running Chrome browser.
        # launch_persistent_context() saves session cookies/storage between runs.
        # First run: user logs in manually (one-time). Future runs: cookies valid.
        print(f"  [BROWSER] Launching Chromium with profile: {PROFILE_DIR.resolve()}")
        ctx = await pw.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR),
            headless=args.headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
            viewport={"width": 1440, "height": 900},
            locale="en-US",
            timezone_id="America/New_York",
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/136.0.0.0 Safari/537.36"
            ),
        )

        await Stealth(chrome_runtime=False).apply_stealth_async(ctx)

        new_pages = []
        ctx.on("page", lambda p: new_pages.append(p))

        # Get or create the first tab
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()

        # Response interceptor (shared for both login and qbank tabs)
        async def on_resp(resp):
            url = resp.url
            if "GetAllSubscriptions" in url or "GetPaymentsForSubscription" in url:
                try: sub_id_box[url] = await resp.json()
                except Exception: pass
            if args.debug and "userapi/auth" in url.lower():
                try:
                    body = (await resp.body())[:150].decode(errors="replace")
                    print(f"     [AUTH] {resp.status} {url[:70]}: {body[:100]}")
                except Exception: pass

        page.on("response", on_resp)

        # ── CHECK IF ALREADY LOGGED IN ─────────────────────────────────────
        print("  [LOGIN] Checking existing session...")
        try:
            await page.goto("https://www.uworld.com/app/index.html#/subscriptions/",
                            wait_until="networkidle", timeout=20000)
            await asyncio.sleep(3)
        except Exception:
            pass

        if "#/login" in page.url or "/login" in page.url.lower():
            print("  [LOGIN] No saved session — logging in...")

            # Dismiss cookie banner
            await page.evaluate("""() => {
                const b = [...document.querySelectorAll('button')]
                    .find(b => b.textContent.includes('Allow All'));
                if (b) b.click();
            }""")
            await asyncio.sleep(1)

            # Try automated login
            auto_login_ok = False
            try:
                await page.wait_for_selector("#login-email", timeout=8000)

                # Click label to focus input (label overlays the input element)
                await page.locator("label[for='login-email']").click(timeout=4000)
                await asyncio.sleep(0.3)
                await page.keyboard.type(email, delay=35)
                await asyncio.sleep(0.4)
                await page.locator("label[for='login-password']").click(timeout=4000)
                await asyncio.sleep(0.3)
                await page.keyboard.type(password, delay=35)
                await asyncio.sleep(1.0)

                # Submit
                btn = page.locator("button[type=submit]")
                if await btn.count() == 0:
                    btn = page.locator("button").filter(
                        has_text=re.compile("login", re.I)).first
                await btn.click(timeout=5000)
                print("  [LOGIN] Submitted credentials — waiting for redirect...")

                try:
                    await page.wait_for_function(
                        "!window.location.hash.includes('login')", timeout=20000)
                except Exception:
                    pass
                await asyncio.sleep(3)

                if "#/login" not in page.url:
                    auto_login_ok = True
                    print("  [LOGIN] Automated login succeeded!")

            except Exception as e:
                print(f"  [LOGIN] Automated login error: {e}")

            if not auto_login_ok:
                # ── MANUAL LOGIN FALLBACK ──────────────────────────────────
                # reCAPTCHA blocked automated login. Open the login page and wait
                # for the user to log in manually. They only need to do this ONCE —
                # the session is saved in the profile for all future runs.
                print("\n" + "="*55)
                print("  MANUAL LOGIN REQUIRED (one-time setup)")
                print("="*55)
                print("  The UWorld login page is open in Chrome.")
                print("  Please log in with your credentials manually.")
                print("  The script will continue automatically once")
                print("  you're logged in.")
                print("  Waiting up to 3 hours...")
                print("="*55 + "\n")

                # Make sure we're on the login page
                try:
                    await page.goto("https://www.uworld.com/app/index.html#/login/",
                                    wait_until="networkidle", timeout=15000)
                except Exception:
                    pass

                # Wait up to 3 hours for manual login (user may be asleep)
                for secs in range(10800):
                    await asyncio.sleep(1)
                    try:
                        curr_url = page.url
                        if "#/login" not in curr_url and "login" not in curr_url.lower():
                            print(f"\n  [LOGIN] Manual login detected! URL: {curr_url[:80]}")
                            break
                        if secs % 30 == 0 and secs > 0:
                            print(f"  [WAIT]  Still waiting for login... ({secs}s elapsed)")
                    except Exception:
                        break
                else:
                    print("  [FAIL] Timed out waiting for manual login.")
                    await ctx.close()
                    return

        if "#/login" in page.url:
            print("  [FAIL] Login failed — session still on login page.")
            await ctx.close()
            return

        print(f"  [OK]   Session active! URL: {page.url[:80]}")
        if args.debug:
            await page.screenshot(path="debug_03_loggedin.png")

        # Let subscriptions page finish loading
        await asyncio.sleep(5)

        # ── SUBSCRIPTION ID ────────────────────────────────────────────────
        sub_id = progress.get("sub_id")
        if not sub_id:
            all_subs = next(
                (v for k, v in sub_id_box.items() if "GetAllSubscriptions" in k), None)
            if all_subs and isinstance(all_subs, list):
                for item in all_subs:
                    if not isinstance(item, dict): continue
                    if item.get("IsSim") or item.get("FormId"): continue
                    name = str(item.get("CourseName") or "").lower()
                    if any(x in name for x in ["self-assessment","free trial","sat ","act "]): continue
                    sid = item.get("SubscriptionId")
                    if sid:
                        sub_id = str(sid)
                        print(f"  [SUB]  Found: {item.get('CourseName')} -> {sub_id}")
                        break
            if not sub_id:
                ids = sorted(
                    {re.search(r'GetPaymentsForSubscription/(\d+)', u).group(1)
                     for u in sub_id_box
                     if re.search(r'GetPaymentsForSubscription/(\d+)', u)}, key=int)
                sub_id = ids[1] if len(ids) > 1 else (ids[0] if ids else "15778134")
            if not sub_id:
                sub_id = "15778134"
            progress["sub_id"] = sub_id
            save_progress(progress)

        print(f"  [SUB]  Subscription ID: {sub_id}")

        # ── INSPECT & CLICK LAUNCH ─────────────────────────────────────────
        launch_info = await page.evaluate("""() =>
            [...document.querySelectorAll('a,button')]
            .filter(e => /^launch$/i.test(e.textContent.trim()))
            .map(e => ({
                tag: e.tagName, href: e.href||'', target: e.getAttribute('target')||'',
                parentTxt: (e.parentElement?.textContent?.trim()?.substring(0,50)||'')
            }))
        """)
        print(f"  [LAUNCH] Launch buttons found: {json.dumps(launch_info)}")

        print("  [LAUNCH] Clicking QBank Launch button...")
        try:
            launch_btn = page.locator("a, button").filter(
                has_text=re.compile(r"^launch$", re.I)
            ).first
            await launch_btn.click(timeout=6000)
            print("  [LAUNCH] Clicked via Playwright.")
        except Exception as e:
            print(f"  [WARN]  Playwright click failed ({e}), JS fallback")
            await page.evaluate("""() => {
                const all = [...document.querySelectorAll('a,button')];
                const btn = all.find(e => /^launch$/i.test(e.textContent.trim()));
                if (btn) btn.click();
            }""")

        # Wait for new tab
        qb_page = None
        print("  [LAUNCH] Waiting for apps.uworld.com tab...")
        for _ in range(25):
            await asyncio.sleep(1)
            if new_pages:
                qb_page = new_pages[-1]
                print(f"  [LAUNCH] New tab: {qb_page.url[:80]}")
                break

        if not qb_page and "apps.uworld.com" in page.url:
            qb_page = page
            print(f"  [LAUNCH] Same-tab: {page.url[:80]}")

        if not qb_page or "apps.uworld.com" not in qb_page.url:
            # Try following Launch href directly
            launch_href = next(
                (i["href"] for i in launch_info if "uworld.com" in i.get("href","")), None)
            if launch_href:
                print(f"  [LAUNCH] Following href: {launch_href[:80]}")
                await page.goto(launch_href, wait_until="networkidle", timeout=30000)
                await asyncio.sleep(5)
                if "apps.uworld.com" in page.url:
                    qb_page = page

        if not qb_page or "apps.uworld.com" not in qb_page.url:
            print("  [WARN]  Launch strategies failed — direct nav as last resort")
            qb_page = await ctx.new_page()
            await qb_page.goto(f"{APPS_BASE}/dashboard/{sub_id}",
                               wait_until="networkidle", timeout=30000)
            await asyncio.sleep(10)

        if args.debug:
            await qb_page.screenshot(path="debug_04_qbank.png")
            print(f"  [LAUNCH] qb_page URL: {qb_page.url[:80]}")

        # ── WAIT FOR ANGULAR APP ───────────────────────────────────────────
        print("  [WAIT]  Waiting for Angular app to initialize...")
        qb_page.on("response", on_resp)
        loaded = await wait_for_content(qb_page, timeout=60, debug=args.debug)
        print(f"  [WAIT]  Content loaded: {loaded}  URL: {qb_page.url[:80]}")

        if args.debug:
            await qb_page.screenshot(path="debug_04b_after_wait.png")

        # ── NAVIGATE TO SEARCH ─────────────────────────────────────────────
        search_path = f"/courseapp/usmle/v53/en-US/search/{sub_id}/false"
        search_url  = f"https://apps.uworld.com{search_path}"
        print(f"  [SEARCH] Navigating to search page...")

        if loaded:
            await qb_page.evaluate(f"""() => {{
                const p = {json.dumps(search_path)};
                window.history.pushState(null, '', p);
                window.dispatchEvent(new PopStateEvent('popstate', {{state:null}}));
            }}""")
            await asyncio.sleep(5)

        visible_inputs = await qb_page.evaluate(
            "() => [...document.querySelectorAll('input')].filter(el=>el.offsetParent!==null).length"
        )
        print(f"  [SEARCH] Visible inputs after Angular nav: {visible_inputs}")

        if visible_inputs == 0:
            print("  [SEARCH] Trying page.goto for search URL...")
            await qb_page.goto(search_url, wait_until="networkidle", timeout=30000)
            await asyncio.sleep(10)
            loaded2 = await wait_for_content(qb_page, timeout=30)
            visible_inputs = await qb_page.evaluate(
                "() => [...document.querySelectorAll('input')].filter(el=>el.offsetParent!==null).length"
            )
            print(f"  [SEARCH] After page.goto: inputs={visible_inputs}, loaded={loaded2}")

        if args.debug:
            await qb_page.screenshot(path="debug_05_search_page.png")
            all_inp = await qb_page.evaluate(
                "() => [...document.querySelectorAll('input')].map(el=>({"
                "type:el.type,id:el.id,placeholder:el.placeholder,visible:el.offsetParent!==null}))"
            )
            print(f"  [SEARCH] All inputs: {json.dumps(all_inp)}")

        # ── RESPONSE INTERCEPTOR ───────────────────────────────────────────
        async def on_search_resp(resp):
            ct  = resp.headers.get("content-type", "")
            url = resp.url
            if "json" not in ct: return
            if not any(x in url for x in
                       ["gateway-api","search","question","item","qbank",
                        "GetQuestions","GetItems","GetSearch","Search"]):
                return
            try:
                body  = await resp.json()
                items = parse_questions(body)
                if items:
                    captured_items.extend(items)
                    if args.debug:
                        print(f"       [API] {len(items)} Qs <- {url[:80]}")
            except Exception:
                pass

        qb_page.on("response", on_search_resp)

        # Dismiss modals
        await qb_page.evaluate("""() => {
            document.querySelectorAll('[class*=modal] button,[class*=dialog] button,[class*=alert] button')
                    .forEach(b => b.click());
            document.querySelectorAll('[class*=overlay],[class*=backdrop]')
                    .forEach(el => { if(el.style) el.style.pointerEvents='none'; });
        }""")
        await asyncio.sleep(0.5)

        # ── SEARCH LOOP ────────────────────────────────────────────────────
        terms_to_run = [t for t in SEARCH_TERMS if t not in progress["completed_terms"]]
        print(f"\n  Running {len(terms_to_run)} search terms...\n")

        for i, term in enumerate(terms_to_run):
            print(f"  [{i+1}/{len(terms_to_run)}] '{term}'", end=" ", flush=True)
            captured_items.clear()

            filled = await qb_page.evaluate(f"""() => {{
                const candidates = [
                    ...document.querySelectorAll('input[type=text]'),
                    ...document.querySelectorAll('input[type=search]'),
                    ...document.querySelectorAll('input:not([type=hidden])'),
                ];
                const inp = candidates.find(el => el.offsetParent !== null) || candidates[0];
                if (!inp) return false;
                inp.focus();
                inp.value = {json.dumps(term)};
                ['focus','input','change'].forEach(ev =>
                    inp.dispatchEvent(new Event(ev, {{bubbles:true}})));
                ['keydown','keypress','keyup'].forEach(ev =>
                    inp.dispatchEvent(new KeyboardEvent(ev,
                        {{key:'Enter',keyCode:13,bubbles:true}})));
                return inp.placeholder || inp.id || inp.name || '(found)';
            }}""")

            await qb_page.keyboard.press("Enter")
            await asyncio.sleep(6)

            if args.debug:
                await qb_page.screenshot(path=f"debug_search_{term[:10]}.png")
                print(f"\n       filled: {filled}")

            new_this = 0
            for q in captured_items:
                qid = q["id"]
                if qid not in lib:
                    lib[qid] = q; new_this += 1
                else:
                    for f in ["subject","system","topic","pct_correct"]:
                        if not lib[qid].get(f) and q.get(f): lib[qid][f] = q[f]

            print(f"-> {len(captured_items)} results, {new_this} new  (total: {len(lib)})")

            save_library(lib)
            progress["completed_terms"].append(term)
            save_progress(progress)

            # Clear input for next term
            await qb_page.evaluate("""() => {
                const inp = document.querySelector(
                    'input[type=text],input[type=search],input:not([type=hidden])');
                if (inp){ inp.value=''; inp.dispatchEvent(new Event('input',{bubbles:true})); }
            }""")
            await asyncio.sleep(0.5)

        await ctx.close()

    print(f"\n{'='*50}")
    print(f"  Done! {len(lib)} questions in library.")
    print(f"  Saved to: {LIBRARY_PATH}")
    show_stats(lib)


async def main():
    args = parse_args()

    if args.show:
        show_stats(load_library()); return

    if args.resetprofile:
        import shutil
        if PROFILE_DIR.exists():
            shutil.rmtree(PROFILE_DIR)
            print(f"  Deleted Chrome profile: {PROFILE_DIR}")
        LIBRARY_PATH.unlink(missing_ok=True)
        PROGRESS_PATH.unlink(missing_ok=True)
        print("  Reset complete. You'll need to log in manually on next run.")
        return

    if args.fresh:
        LIBRARY_PATH.unlink(missing_ok=True)
        PROGRESS_PATH.unlink(missing_ok=True)
        print("  [WIPE]  Library wiped (Chrome profile kept — no re-login needed).")

    lib  = load_library()
    prog = load_progress()
    print(f"\n{'='*50}\n  UWorld Library Builder\n{'='*50}")
    print(f"  Library : {len(lib)} questions")
    print(f"  Progress: {len(prog['completed_terms'])}/{len(SEARCH_TERMS)} terms done")
    print(f"  Profile : {PROFILE_DIR.resolve()}")

    await run(args)


if __name__ == "__main__":
    asyncio.run(main())
