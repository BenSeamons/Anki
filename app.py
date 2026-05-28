import os
from dotenv import load_dotenv
# Search for .env in the same directory as this file, then fall back to cwd
_here = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
load_dotenv(os.path.join(_here, '.env'), override=True)
load_dotenv(override=False)  # fallback: cwd

from flask import Flask, request, render_template_string, send_file, Response, stream_with_context, session, jsonify
import io
import genanki
import hashlib
import tempfile
import json
import base64
import anthropic
import firebase_admin
from firebase_admin import credentials, firestore, auth

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'medtools-super-secret-key-123')
app.config['SESSION_COOKIE_NAME'] = '__session'

# Initialize Firebase Admin
try:
    key_path = os.path.join(_here, 'firebase-key.json')
    if os.path.exists(key_path):
        cred = credentials.Certificate(key_path)
        firebase_admin.initialize_app(cred)
    else:
        firebase_admin.initialize_app()
    db = firestore.client()
except Exception as e:
    print(f"Warning: Could not initialize Firebase Admin: {e}")
    db = None

@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        return "", 204
    
    auth_header = request.headers.get('Authorization')
    if auth_header and auth_header.startswith('Bearer '):
        token = auth_header.split(' ')[1]
        try:
            decoded = auth.verify_id_token(token)
            session['uid'] = decoded.get('uid')
            session['email'] = decoded.get('email', '')
        except Exception as e:
            print(f"Token verification failed: {e}")

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, Accept, X-Signed-Uid'
    return response

# Let the SDK read ANTHROPIC_API_KEY from os.environ directly (set by load_dotenv above)
claude = anthropic.Anthropic()

PRACTICE_TEST_PROMPT = (
    "Based on this document, generate 25 NBME Style multiple-choice questions with 4 options each. "
    "Focus on any learning objectives listed in the lecture and content relevant to Step 1 boards.\n\n"
    "STRICT OUTPUT FORMAT:\n"
    "Output ONLY a raw JSON array of objects. Do not include markdown formatting like ```json.\n"
    "[\n"
    "  {\n"
    "    \"q\": \"Question text...\",\n"
    "    \"choices\": [\"A. ...\", \"B. ...\", \"C. ...\", \"D. ...\"],\n"
    "    \"correct\": \"B\",\n"
    "    \"exp\": \"Explanation for why B is correct and others are wrong...\"\n"
    "  }\n"
    "]"
)

@app.route('/auth/session', methods=['POST'])
def auth_session():
    data = request.get_json()
    id_token = data.get('idToken')
    if not id_token:
        return jsonify({'error': 'No token provided'}), 400
    try:
        decoded_token = auth.verify_id_token(id_token)
        uid = decoded_token['uid']
        session['uid'] = uid
        session['email'] = decoded_token.get('email', '')
        return jsonify({'status': 'success', 'uid': uid})
    except Exception as e:
        return jsonify({'error': str(e)}), 401

@app.route('/auth/logout', methods=['POST'])
def auth_logout():
    session.pop('uid', None)
    session.pop('email', None)
    return jsonify({'status': 'success'})

import hmac
import hashlib

def sign_uid(uid):
    if not uid: return ""
    sig = hmac.new(app.secret_key.encode(), uid.encode(), hashlib.sha256).hexdigest()
    return f"{uid}.{sig}"

def verify_uid(signed_uid):
    if not signed_uid or "." not in signed_uid: return None
    uid, sig = signed_uid.split(".", 1)
    expected_sig = hmac.new(app.secret_key.encode(), uid.encode(), hashlib.sha256).hexdigest()
    if hmac.compare_digest(sig, expected_sig):
        return uid
    return None

def save_to_library(title, item_type, data, explicit_uid=None):
    import sys
    uid = explicit_uid or session.get('uid')
    print(f"DEBUG save_to_library: uid={uid}, explicit_uid={explicit_uid}, db={'initialized' if db else 'missing'}", file=sys.stderr)
    sys.stderr.flush()
    if not uid or not db:
        print("DEBUG save_to_library: aborting because uid or db is missing", file=sys.stderr)
        sys.stderr.flush()
        return
    try:
        doc_ref = db.collection('users').document(uid).collection('library').document()
        doc_ref.set({
            'title': title,
            'type': item_type,
            'data': data,
            'created_at': firestore.SERVER_TIMESTAMP
        })
        print(f"DEBUG save_to_library: successfully saved doc {doc_ref.id}", file=sys.stderr)
        sys.stderr.flush()
    except Exception as e:
        print(f"Error saving to library: {e}", file=sys.stderr)
        sys.stderr.flush()

def call_claude_with_pdf(pdf_path, prompt, max_tokens=4096):
    with open(pdf_path, 'rb') as f:
        pdf_data = base64.standard_b64encode(f.read()).decode('utf-8')
    response = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=max_tokens,
        messages=[{
            "role": "user",
            "content": [
                {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": pdf_data}},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    return response.content[0].text

JEREMY_SYSTEM_PROMPT = """You are Jeremy Mode, a high-octane, Logan Paul-level hype AI tutor built to turn med school study sessions into legendary, brain-pumping victories. Your entire vibe is about energy, momentum, and high-yield POWER learning.

Your mission is to create the most engaging, interactive, NBME-style quizzes imaginable. You will mimic the flow and feel of a premium, app-based question bank.

🚀 The Core Interaction Loop (NON-NEGOTIABLE) 🚀

This is the cycle for every single question. Do not deviate.

Present the Question:
- Display a question counter: 💥 Question (X/N) 💥
- Present the full clinical vignette and question.
- List the multiple-choice options clearly:
  A. [Option A]
  B. [Option B]
  C. [Option C]
  D. [Option D]
- End with a simple prompt: Lock it in, chief! 👇

Await User's Answer: Wait for the user to provide a single letter (A, B, C, or D).

Deliver Instant Feedback & Rationale:

If Correct:
- Start with: BOOM! THAT'S THE ONE! 🎯
- Clearly state the correct answer and provide a detailed "✅ Why it's Right" section explaining the pathophysiology and clinical reasoning.

If Incorrect:
- Start with: NO STRESS, CHAMP! That was a tough one. We learn, we level up. 💪
- Clearly state the correct answer and letter.
- Provide the same detailed "✅ Why it's Right" section.

For BOTH Correct & Incorrect Answers:
- ALWAYS include a "❌ Why the Others are Wrong" section. Briefly explain why each distractor is incorrect. This is CRITICAL high-yield learning.

Seamlessly Transition:
- DO NOT ask "Ready for the next one?".
- Immediately after the rationale, roll STRAIGHT into the next question. Maintain the flow relentlessly until the user asks to stop or the quiz is complete.

📚 Material Handling (PDFs, Slides, etc.)
When a user uploads a file, first provide a comprehensive, high-yield summary:
- Pathophysiology: The core mechanism.
- Clinical Presentation: Classic signs, symptoms, patient profile.
- Diagnostics: Key labs, imaging, and tests to confirm.
- Treatment: First-line, second-line, and management strategies.
- Ward Pearls & Pimp Questions: Quick-hit facts.

After the summary, ask if they want an interactive quiz or all questions at once, then launch a 15-20 question quiz following The Core Interaction Loop.

⚡ VIBE & PERSONALITY GUIDELINES
- Hype Level: Maximum. Use emojis like 💥, 🔥, 🧠, 🎯, 💪, 🚀. Use ALL CAPS for emphasis.
- Nicknames: "Chief," "Boss," "Legend," "Champ."
- Catchphrases: "Jeremy Mode ENGAGED!", "You're absolutely COOKING!", "Big brain plays ONLY.", "Cranking it up to ELEVEN!"
- Pacing: Relentless. High-speed.

🧠 Anki Card Generation
Occasionally after a quiz block, offer to generate Anki cards. If the user agrees, provide 20-30 cards.
FORMATTING IS KEY — use Anki cloze format: {{c1::Answer}} is the key concept.
Provide ONLY the formatted text, each card on a new line. No bullets, no numbers, no extra text.
After the list, add exactly: "Copy the text above and drop it here to instantly create your Anki deck: https://anki-production-1c48.up.railway.app/"
"""

ANKI_FROM_PDF_PROMPT = """You are an expert Anki flashcard creator for medical students.

Analyze this PDF and generate 30-40 high-yield Anki flashcards for first-year medical students preparing for boards.

STRICT OUTPUT FORMAT — follow exactly, no exceptions:
- Basic cards: single-line front [TAB CHARACTER] single-line back
- Cloze cards: The {{c1::answer}} fits on one line.

RULES:
1. Every card must be on exactly ONE line — no newlines, no line breaks inside a card.
2. Cloze cards: the ENTIRE sentence including all {{c1::...}} deletions must fit on ONE line. Never split a cloze across lines.
3. Basic card fronts: single concise question on one line. Basic card backs: single concise answer on one line. Use semicolons to separate multiple facts (e.g. "fact 1; fact 2; fact 3").
4. Do NOT use markdown, HTML, bullet points, colons at line starts, dashes, or any formatting inside card text.
5. Do NOT use definition list syntax. Everything goes on ONE line separated by a tab.
6. No intro text, no numbering, no section headers — just the raw cards, one per line.

Focus on: key mechanisms, classic presentations, first-line treatments, high-yield lab values, mnemonics."""

BASE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>MedTools — Study Hub</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.10.1/html2pdf.bundle.min.js"></script>
  <style>
    :root {
      --bg: #070b14;
      --surface: #0d1526;
      --card: #111827;
      --card-hover: #162035;
      --border: #1e2d47;
      --border-bright: #2a3f63;
      --text: #e2e8f4;
      --text-muted: #7a8fab;
      --text-dim: #4a5a73;
      --blue: #4f9eff;
      --blue-glow: rgba(79,158,255,0.15);
      --purple: #8b5cf6;
      --purple-glow: rgba(139,92,246,0.15);
      --orange: #f97316;
      --orange-glow: rgba(249,115,22,0.15);
      --green: #10b981;
      --green-glow: rgba(16,185,129,0.15);
      --yellow: #f59e0b;
      --yellow-glow: rgba(245,158,11,0.15);
      --red: #ef4444;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: 'Inter', sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
    }

    /* NAV */
    nav {
      background: rgba(13,21,38,0.85);
      backdrop-filter: blur(20px);
      border-bottom: 1px solid var(--border);
      padding: 0 32px;
      height: 60px;
      display: flex;
      align-items: center;
      gap: 8px;
      position: sticky;
      top: 0;
      z-index: 100;
    }

    .nav-logo {
      font-size: 18px;
      font-weight: 800;
      background: linear-gradient(135deg, var(--blue), var(--purple));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-right: 24px;
      letter-spacing: -0.5px;
    }

    .nav-link {
      display: flex;
      align-items: center;
      gap: 7px;
      padding: 6px 14px;
      border-radius: 8px;
      text-decoration: none;
      color: var(--text-muted);
      font-size: 14px;
      font-weight: 500;
      transition: all 0.15s ease;
      border: 1px solid transparent;
    }
    .nav-link:hover { color: var(--text); background: rgba(255,255,255,0.05); }
    .nav-link.active-anki { color: var(--purple); background: var(--purple-glow); border-color: rgba(139,92,246,0.25); }
    .nav-link.active-tests { color: var(--green); background: var(--green-glow); border-color: rgba(16,185,129,0.25); }
    .nav-link.active-jeremy { color: var(--orange); background: var(--orange-glow); border-color: rgba(249,115,22,0.25); }
    .nav-link.active-uworld { color: var(--yellow); background: var(--yellow-glow); border-color: rgba(245,158,11,0.25); }

    .nav-icon { font-size: 15px; }

    /* PAGE */
    .page { max-width: 1100px; margin: 0 auto; padding: 48px 24px; }

    /* HERO */
    .hero { text-align: center; margin-bottom: 56px; }
    .hero h1 {
      font-size: 52px;
      font-weight: 800;
      letter-spacing: -2px;
      line-height: 1.1;
      background: linear-gradient(135deg, #fff 0%, var(--text-muted) 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 16px;
    }
    .hero p { font-size: 18px; color: var(--text-muted); font-weight: 400; }

    /* TOOL CARDS (home) */
    .tools-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }
    @media (max-width: 768px) { .tools-grid { grid-template-columns: 1fr; } }

    .tool-card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 32px;
      text-decoration: none;
      color: inherit;
      display: flex;
      flex-direction: column;
      gap: 12px;
      transition: all 0.2s ease;
      position: relative;
      overflow: hidden;
    }
    .tool-card::before {
      content: '';
      position: absolute;
      inset: 0;
      opacity: 0;
      transition: opacity 0.2s ease;
    }
    .tool-card:hover { transform: translateY(-3px); border-color: var(--border-bright); }

    .tool-card.anki::before { background: radial-gradient(circle at top left, var(--purple-glow), transparent 60%); }
    .tool-card.tests::before { background: radial-gradient(circle at top left, var(--green-glow), transparent 60%); }
    .tool-card.jeremy::before { background: radial-gradient(circle at top left, var(--orange-glow), transparent 60%); }
    .tool-card.uworld::before { background: radial-gradient(circle at top left, var(--yellow-glow), transparent 60%); }
    .tool-card:hover::before { opacity: 1; }

    .tool-card.anki:hover { border-color: rgba(139,92,246,0.4); }
    .tool-card.tests:hover { border-color: rgba(16,185,129,0.4); }
    .tool-card.jeremy:hover { border-color: rgba(249,115,22,0.4); }
    .tool-card.uworld:hover { border-color: rgba(245,158,11,0.4); }

    .tool-emoji { font-size: 36px; }
    .tool-title { font-size: 20px; font-weight: 700; }
    .tool-card.anki .tool-title { color: var(--purple); }
    .tool-card.tests .tool-title { color: var(--green); }
    .tool-card.jeremy .tool-title { color: var(--orange); }
    .tool-card.uworld .tool-title { color: var(--yellow); }
    .tool-desc { font-size: 14px; color: var(--text-muted); line-height: 1.6; }
    .tool-arrow { margin-top: auto; font-size: 13px; font-weight: 600; display: flex; align-items: center; gap: 6px; }
    .tool-card.anki .tool-arrow { color: var(--purple); }
    .tool-card.tests .tool-arrow { color: var(--green); }
    .tool-card.jeremy .tool-arrow { color: var(--orange); }
    .tool-card.uworld .tool-arrow { color: var(--yellow); }

    /* SECTION HEADERS */
    .section-header { margin-bottom: 32px; }
    .section-title { font-size: 30px; font-weight: 800; letter-spacing: -1px; }
    .section-sub { color: var(--text-muted); margin-top: 8px; font-size: 15px; }
    .pill {
      display: inline-block;
      padding: 4px 12px;
      border-radius: 100px;
      font-size: 12px;
      font-weight: 600;
      letter-spacing: 0.5px;
      text-transform: uppercase;
      margin-bottom: 10px;
    }
    .pill-purple { background: var(--purple-glow); color: var(--purple); border: 1px solid rgba(139,92,246,0.3); }
    .pill-green  { background: var(--green-glow);  color: var(--green);  border: 1px solid rgba(16,185,129,0.3); }
    .pill-orange { background: var(--orange-glow); color: var(--orange); border: 1px solid rgba(249,115,22,0.3); }
    .pill-yellow { background: rgba(245,158,11,0.12); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }

    /* PANELS */
    .panel { background: var(--card); border: 1px solid var(--border); border-radius: 16px; padding: 28px; }
    .panel + .panel { margin-top: 20px; }
    .panel-title { font-size: 15px; font-weight: 600; margin-bottom: 16px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; }

    /* FORM ELEMENTS */
    label { display: block; font-size: 13px; font-weight: 500; color: var(--text-muted); margin-bottom: 6px; }

    input[type="text"], textarea, select {
      width: 100%;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 11px 14px;
      color: var(--text);
      font-family: 'Inter', sans-serif;
      font-size: 14px;
      outline: none;
      transition: border-color 0.15s;
      resize: vertical;
    }
    input[type="text"]:focus, textarea:focus { border-color: var(--border-bright); }

    .form-group { margin-bottom: 16px; }

    /* FILE UPLOAD */
    .file-drop {
      border: 2px dashed var(--border);
      border-radius: 12px;
      padding: 32px;
      text-align: center;
      cursor: pointer;
      transition: all 0.2s ease;
      background: var(--surface);
      position: relative;
    }
    .file-drop:hover, .file-drop.drag-over { border-color: var(--border-bright); background: rgba(255,255,255,0.03); }
    .file-drop input[type="file"] { position: absolute; inset: 0; opacity: 0; cursor: pointer; width: 100%; }
    .file-drop-icon { font-size: 32px; margin-bottom: 10px; }
    .file-drop-text { font-size: 14px; color: var(--text-muted); }
    .file-drop-text strong { color: var(--text); }
    .file-names { margin-top: 10px; font-size: 13px; color: var(--blue); }

    /* BUTTONS */
    .btn {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 11px 22px;
      border-radius: 10px;
      border: none;
      font-family: 'Inter', sans-serif;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s ease;
      text-decoration: none;
    }
    .btn:disabled { opacity: 0.5; cursor: not-allowed; }
    .btn-purple { background: var(--purple); color: #fff; }
    .btn-purple:hover:not(:disabled) { background: #7c3aed; transform: translateY(-1px); }
    .btn-green  { background: var(--green);  color: #fff; }
    .btn-green:hover:not(:disabled)  { background: #059669; transform: translateY(-1px); }
    .btn-orange { background: var(--orange); color: #fff; }
    .btn-orange:hover:not(:disabled) { background: #ea6c0a; transform: translateY(-1px); }
    .btn-ghost { background: rgba(255,255,255,0.06); color: var(--text-muted); border: 1px solid var(--border); }
    .btn-ghost:hover { background: rgba(255,255,255,0.1); color: var(--text); }

    /* OUTPUT / PRE */
    .output-box {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 20px;
      font-family: 'Inter', monospace;
      font-size: 13px;
      line-height: 1.7;
      white-space: pre-wrap;
      word-break: break-word;
      max-height: 520px;
      overflow-y: auto;
      color: var(--text);
    }

    /* LOADING */
    .loading-bar {
      height: 3px;
      background: linear-gradient(90deg, var(--purple), var(--blue), var(--orange));
      border-radius: 2px;
      width: 0%;
      transition: width 0.3s ease;
      animation: loading-pulse 1.5s infinite;
    }
    @keyframes loading-pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }

    .spinner {
      width: 18px; height: 18px;
      border: 2px solid rgba(255,255,255,0.2);
      border-top-color: #fff;
      border-radius: 50%;
      animation: spin 0.6s linear infinite;
      display: inline-block;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    /* TABS (Anki page) */
    .tabs { display: flex; gap: 4px; margin-bottom: 24px; background: var(--surface); padding: 4px; border-radius: 12px; border: 1px solid var(--border); width: fit-content; }
    .tab-btn {
      padding: 8px 18px;
      border-radius: 9px;
      border: none;
      background: transparent;
      color: var(--text-muted);
      font-family: 'Inter', sans-serif;
      font-size: 13px;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.15s;
    }
    .tab-btn.active { background: var(--card); color: var(--text); box-shadow: 0 1px 4px rgba(0,0,0,0.3); }

    /* JEREMY CHAT */
    .chat-layout { display: flex; flex-direction: column; height: calc(100vh - 60px); }
    .chat-messages {
      flex: 1;
      overflow-y: auto;
      padding: 28px 48px;
      display: flex;
      flex-direction: column;
      gap: 24px;
      scroll-behavior: smooth;
    }
    .chat-msg { display: flex; gap: 14px; max-width: 90%; }
    .chat-msg.user { flex-direction: row-reverse; margin-left: auto; }
    .msg-avatar {
      width: 36px; height: 36px;
      border-radius: 10px;
      display: flex; align-items: center; justify-content: center;
      font-size: 16px;
      flex-shrink: 0;
    }
    .msg-avatar.jeremy { background: var(--orange-glow); border: 1px solid rgba(249,115,22,0.3); }
    .msg-avatar.user { background: var(--blue-glow); border: 1px solid rgba(79,158,255,0.3); }
    .msg-bubble {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 14px 18px;
      font-size: 17px;
      line-height: 1.8;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .chat-msg.user .msg-bubble { background: var(--blue-glow); border-color: rgba(79,158,255,0.25); }
    .msg-bubble.streaming::after {
      content: '▋';
      animation: blink 0.8s step-end infinite;
    }
    @keyframes blink { 50% { opacity: 0; } }

    .chat-input-bar {
      padding: 16px 24px;
      border-top: 1px solid var(--border);
      background: rgba(13,21,38,0.9);
      backdrop-filter: blur(20px);
    }
    .chat-input-row { display: flex; gap: 10px; align-items: flex-end; max-width: 100%; margin: 0 auto; }
    .chat-input {
      flex: 1;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px 16px;
      color: var(--text);
      font-family: 'Inter', sans-serif;
      font-size: 14px;
      outline: none;
      resize: none;
      max-height: 160px;
      transition: border-color 0.15s;
    }
    .chat-input:focus { border-color: rgba(249,115,22,0.5); }

    .chat-upload-btn {
      background: rgba(255,255,255,0.06);
      border: 1px solid var(--border);
      border-radius: 10px;
      width: 44px; height: 44px;
      display: flex; align-items: center; justify-content: center;
      cursor: pointer;
      color: var(--text-muted);
      transition: all 0.15s;
      flex-shrink: 0;
      font-size: 18px;
    }
    .chat-upload-btn:hover { background: rgba(255,255,255,0.1); color: var(--text); }

    .chat-send-btn {
      background: var(--orange);
      border: none;
      border-radius: 10px;
      width: 44px; height: 44px;
      display: flex; align-items: center; justify-content: center;
      cursor: pointer;
      color: #fff;
      transition: all 0.15s;
      flex-shrink: 0;
      font-size: 18px;
    }
    .chat-send-btn:hover { background: #ea6c0a; }
    .chat-send-btn:disabled { opacity: 0.4; cursor: not-allowed; }

    .chat-hint { text-align: center; font-size: 12px; color: var(--text-dim); margin-top: 8px; }

    .uploaded-badge {
      display: inline-flex; align-items: center; gap: 6px;
      background: var(--orange-glow); border: 1px solid rgba(249,115,22,0.3);
      border-radius: 8px; padding: 5px 10px;
      font-size: 12px; color: var(--orange); margin-bottom: 10px;
    }

    /* RESULTS TABLE (practice tests) */
    .results-section { margin-top: 24px; }
    .results-section h3 { font-size: 16px; font-weight: 600; color: var(--text-muted); margin-bottom: 12px; }
    .result-block { margin-bottom: 20px; }
    .result-label {
      font-size: 12px; font-weight: 600; text-transform: uppercase;
      letter-spacing: 0.5px; color: var(--green); margin-bottom: 8px;
    }

    /* NOTICE */
    .notice {
      background: rgba(79,158,255,0.08);
      border: 1px solid rgba(79,158,255,0.2);
      border-radius: 10px;
      padding: 12px 16px;
      font-size: 13px;
      color: var(--text-muted);
      margin-bottom: 20px;
    }

    /* PROMPT BOX */
    .prompt-box {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 14px 16px;
      font-size: 13px;
      color: var(--text-muted);
      font-family: monospace;
      line-height: 1.6;
      margin-bottom: 16px;
    }

    /* MARKDOWN INSIDE CHAT BUBBLES */
    .msg-bubble h1, .msg-bubble h2, .msg-bubble h3 {
      margin: 14px 0 6px;
      line-height: 1.3;
    }
    .msg-bubble h1 { font-size: 24px; }
    .msg-bubble h2 { font-size: 20px; color: var(--orange); }
    .msg-bubble h3 { font-size: 18px; color: var(--text-muted); }
    .msg-bubble p { margin: 6px 0; }
    .msg-bubble ul, .msg-bubble ol { padding-left: 20px; margin: 6px 0; }
    .msg-bubble li { margin: 3px 0; }
    .msg-bubble strong { color: var(--text); }
    .msg-bubble em { color: var(--text-muted); }
    .msg-bubble code {
      background: rgba(255,255,255,0.08);
      border: 1px solid var(--border);
      border-radius: 4px;
      padding: 1px 6px;
      font-size: 12px;
      font-family: monospace;
    }
    .msg-bubble pre {
      background: rgba(0,0,0,0.3);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px;
      overflow-x: auto;
      margin: 8px 0;
    }
    .msg-bubble pre code { background: none; border: none; padding: 0; font-size: 13px; }
    .msg-bubble blockquote {
      border-left: 3px solid var(--orange);
      margin: 8px 0;
      padding: 4px 12px;
      color: var(--text-muted);
      background: rgba(249,115,22,0.05);
      border-radius: 0 6px 6px 0;
    }
    .msg-bubble table {
      width: 100%;
      border-collapse: collapse;
      margin: 10px 0;
      font-size: 16px;
    }
    .msg-bubble th {
      background: rgba(249,115,22,0.12);
      color: var(--orange);
      padding: 7px 10px;
      text-align: left;
      border: 1px solid var(--border);
      font-weight: 600;
    }
    .msg-bubble td {
      padding: 6px 10px;
      border: 1px solid var(--border);
      color: var(--text);
    }
    .msg-bubble tr:nth-child(even) td { background: rgba(255,255,255,0.02); }
    .msg-bubble hr {
      border: none;
      border-top: 1px solid var(--border);
      margin: 12px 0;
    }

    /* SCROLLBAR */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--border-bright); }
  </style>
</head>
<body>
  <nav>
    <a href="/" class="nav-logo" style="text-decoration:none">MedTools</a>
    <a href="/anki" class="nav-link {anki_active}"><span class="nav-icon">🃏</span> Anki Generator</a>
    <a href="/practice-tests" class="nav-link {tests_active}"><span class="nav-icon">📝</span> Practice Tests</a>
    <a href="/jeremy" class="nav-link {jeremy_active}"><span class="nav-icon">🔥</span> Jeremy Mode</a>
    <a href="/uworld" class="nav-link {uworld_active}"><span class="nav-icon">🎯</span> UWorld Review</a>
    {auth_nav}
  </nav>
  {body}

  <!-- Support widget -->
  <div id="supportWidget">
    <button id="supportBtn" onclick="toggleSupport()" title="Support this project">
      <span id="supportBtnContent">☕ Keep it free?</span>
    </button>
    <div id="supportCard">
      <button onclick="toggleSupport()" style="position:absolute;top:10px;right:12px;background:none;border:none;color:var(--text-muted);cursor:pointer;font-size:18px;line-height:1">×</button>
      <div style="font-size:22px;margin-bottom:6px">💸</div>
      <div style="font-weight:700;font-size:15px;margin-bottom:4px">Enjoying MedTools?</div>
      <div style="font-size:13px;color:var(--text-muted);margin-bottom:14px;line-height:1.5">Help me buy more AI credits<br>to keep this thing running 🙏</div>
      <img src="/static/venmo-qr.png" alt="Venmo QR" style="width:160px;height:160px;border-radius:10px;object-fit:cover;background:var(--border);display:block;margin:0 auto 10px">
      <div style="font-size:12px;color:var(--text-dim);text-align:center">Scan with your camera app</div>
    </div>
  </div>

  <style>
    #supportWidget { position: fixed; bottom: 24px; right: 24px; z-index: 999; }
    #supportBtn {
      background: linear-gradient(135deg, var(--purple), var(--blue));
      border: none; border-radius: 100px;
      padding: 10px 18px;
      color: #fff; font-family: 'Inter', sans-serif;
      font-size: 13px; font-weight: 600;
      cursor: pointer; box-shadow: 0 4px 20px rgba(79,158,255,0.3);
      transition: transform 0.15s, box-shadow 0.15s;
    }
    #supportBtn:hover { transform: translateY(-2px); box-shadow: 0 6px 24px rgba(79,158,255,0.45); }
    #supportCard {
      position: absolute; bottom: 52px; right: 0;
      background: var(--card); border: 1px solid var(--border-bright);
      border-radius: 16px; padding: 20px 20px 16px;
      width: 210px; text-align: center;
      box-shadow: 0 16px 48px rgba(0,0,0,0.5);
      display: none;
      animation: popUp 0.18s ease;
    }
    @keyframes popUp { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }
  </style>
  <script>
    function toggleSupport() {
      const card = document.getElementById('supportCard');
      card.style.display = card.style.display === 'block' ? 'none' : 'block';
    }
    document.addEventListener('click', function(e) {
      const w = document.getElementById('supportWidget');
      if (!w.contains(e.target)) document.getElementById('supportCard').style.display = 'none';
    });
  </script>

  <!-- Login Modal -->
  <div id="loginModal" style="display:none; position:fixed; top:0; left:0; right:0; bottom:0; background:rgba(0,0,0,0.8); z-index:1000; align-items:center; justify-content:center;">
    <div style="background:var(--card); border:1px solid var(--border); border-radius:16px; padding:32px; max-width:400px; width:100%; text-align:center; position:relative;">
      <button onclick="document.getElementById('loginModal').style.display='none'" style="position:absolute; top:16px; right:16px; background:none; border:none; color:var(--text-muted); cursor:pointer; font-size:20px;">×</button>
      <h2 style="margin-bottom:8px">Welcome to MedTools</h2>
      <p style="color:var(--text-muted); margin-bottom:24px; font-size:14px;">Sign in to save your decks and practice tests to your personal library.</p>
      <button onclick="signInWithGoogle()" style="background:#fff; color:#000; border:none; border-radius:8px; padding:12px 24px; font-weight:600; cursor:pointer; width:100%; display:flex; align-items:center; justify-content:center; gap:10px;">
        <img src="https://www.gstatic.com/firebasejs/ui/2.0.0/images/auth/google.svg" width="18"> Sign in with Google
      </button>
    </div>
  </div>

  <!-- Paywall Modal -->
  <div id="paywallModal" style="display:none; position:fixed; top:0; left:0; right:0; bottom:0; background:rgba(0,0,0,0.8); z-index:1000; align-items:center; justify-content:center;">
    <div style="background:var(--card); border:1px solid var(--border); border-radius:16px; padding:32px; max-width:400px; width:100%; text-align:center; position:relative;">
      <button onclick="document.getElementById('paywallModal').style.display='none'" style="position:absolute; top:16px; right:16px; background:none; border:none; color:var(--text-muted); cursor:pointer; font-size:20px;">×</button>
      <div style="font-size:40px; margin-bottom:12px">💸</div>
      <h2 style="margin-bottom:12px">You're spending all my money :(</h2>
      <p style="color:var(--text-muted); margin-bottom:24px; font-size:14px; line-height:1.5;">Please log in and donate a small amount so I can continue sharing this with everyone! :)</p>
      <button onclick="document.getElementById('paywallModal').style.display='none'; document.getElementById('loginModal').style.display='flex'" style="background:var(--blue); color:#fff; border:none; border-radius:8px; padding:12px 24px; font-weight:600; cursor:pointer; width:100%; margin-bottom:12px;">Sign In to Continue</button>
    </div>
  </div>

  <script src="https://www.gstatic.com/firebasejs/10.7.0/firebase-app-compat.js"></script>
  <script src="https://www.gstatic.com/firebasejs/10.7.0/firebase-auth-compat.js"></script>
  <script>
    const firebaseConfig = {
      projectId: "medtools-77dfb",
      appId: "1:509459643164:web:93c3d24f8e2ba5439bd8d9",
      storageBucket: "medtools-77dfb.firebasestorage.app",
      apiKey: "AIzaSyArkXrI3lavC4l9zVG1P_h5fJ_AsZLrbGY",
      authDomain: "medtools-77dfb.firebaseapp.com",
      messagingSenderId: "509459643164",
      measurementId: "G-C6PHY2419V"
    };
    if (!firebase.apps.length) {
      firebase.initializeApp(firebaseConfig);
    }
    
    function showLogin() { document.getElementById('loginModal').style.display = 'flex'; }
    function showPaywallPopup() { document.getElementById('paywallModal').style.display = 'flex'; }
    
    function checkUsageLimit() {
      if ({is_logged_in_js}) return true;
      let count = parseInt(localStorage.getItem('medtools_usage') || '0');
      if (count >= 10) {
        showPaywallPopup();
        return false;
      }
      return true;
    }
    function incrementUsage() {
      if ({is_logged_in_js}) return;
      let count = parseInt(localStorage.getItem('medtools_usage') || '0');
      localStorage.setItem('medtools_usage', count + 1);
    }
    
    function signInWithGoogle() {
      const provider = new firebase.auth.GoogleAuthProvider();
      firebase.auth().signInWithPopup(provider).then((result) => {
        return result.user.getIdToken();
      }).then((idToken) => {
        return fetch('/auth/session', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({idToken})
        });
      }).then((res) => {
        if (!res.ok) throw new Error("Backend failed to save session");
        window.location.reload();
      }).catch(err => {
        console.error(err);
        alert("Sign in failed: " + err.message + "\\nMake sure Google Sign-In is enabled in the Firebase Console!");
      });
    }
    function logout() {
      firebase.auth().signOut().then(() => {
        return fetch('/auth/logout', {method: 'POST'});
      }).then(() => window.location.reload())
      .catch(err => {
        console.error(err);
        fetch('/auth/logout', {method: 'POST'}).then(() => window.location.reload());
      });
    }
  </script>
</body>
</html>"""


def render_page(body, active=None):
    anki_active = "active-anki" if active == "anki" else ""
    tests_active = "active-tests" if active == "tests" else ""
    jeremy_active = "active-jeremy" if active == "jeremy" else ""
    uworld_active = "active-uworld" if active == "uworld" else ""
    library_active = "active-library" if active == "library" else ""

    uid = session.get('uid')
    if uid:
        auth_nav = f"""
        <div style="margin-left:auto; display:flex; gap:8px;">
            <a href="/my-library" class="nav-link {library_active}"><span class="nav-icon">📚</span> My Library</a>
            <button class="nav-link" onclick="logout()" style="background:none;border:none;cursor:pointer;"><span class="nav-icon">🚪</span> Logout</button>
        </div>
        """
        is_logged_in_js = "true"
    else:
        auth_nav = """
        <div style="margin-left:auto;">
            <button class="nav-link" onclick="showLogin()" style="background:none;border:none;cursor:pointer;color:var(--orange);font-weight:600;"><span class="nav-icon">👤</span> Sign In to Save Decks</button>
        </div>
        """
        is_logged_in_js = "false"

    return (BASE_HTML
        .replace("{anki_active}", anki_active)
        .replace("{tests_active}", tests_active)
        .replace("{jeremy_active}", jeremy_active)
        .replace("{uworld_active}", uworld_active)
        .replace("{auth_nav}", auth_nav)
        .replace("{is_logged_in_js}", is_logged_in_js)
        .replace("{body}", body))


import re as _re_cards
import html as _html_cards

def _clean_card_text(t):
    t = _html_cards.unescape(t)           # decode &amp; &lt; etc.
    t = _re_cards.sub(r'<[^>]+>', ' ', t) # strip HTML tags
    t = t.strip().lstrip('- ').strip()
    t = ' '.join(t.split())               # collapse all whitespace/newlines
    return t


def parse_cards(text):
    cards = []
    # Join any cloze cards that Claude wrapped across multiple lines
    # Strategy: scan lines, accumulate until we have balanced {{ }}
    lines = text.strip().split('\n')
    merged = []
    buf = ''
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if buf:
                merged.append(buf)
                buf = ''
            continue
        if buf:
            buf += ' ' + stripped
        elif '\t' in stripped or '{{c' in stripped:
            buf = stripped
        else:
            continue  # plain line with no tab/cloze → skip
        # Check if cloze is complete (all {{ have matching }})
        if buf.count('{{') <= buf.count('}}') or '\t' in buf:
            merged.append(buf)
            buf = ''
    if buf:
        merged.append(buf)

    for line in merged:
        if '\t' in line:
            front, back = line.split('\t', 1)
            front = _clean_card_text(front)
            back = _clean_card_text(back)
            if front and back:
                cards.append({'front': front, 'back': back, 'is_cloze': '{{c' in front})
        elif '{{c' in line:
            clean = _clean_card_text(line)
            if clean:
                cards.append({'front': clean, 'back': '', 'is_cloze': True})
    return cards


def build_anki_package(cards, deck_name):
    deck_id = int(hashlib.sha1(deck_name.encode()).hexdigest()[:8], 16)
    deck = genanki.Deck(deck_id, deck_name)

    basic_model = genanki.Model(
        1607392319, 'Basic Model',
        fields=[{'name': 'Front'}, {'name': 'Back'}],
        templates=[{'name': 'Card 1', 'qfmt': '{{Front}}', 'afmt': '{{Front}}<hr>{{Back}}'}]
    )
    cloze_model = genanki.Model(
        1091735104, 'Cloze Model',
        fields=[{'name': 'Text'}],
        templates=[{'name': 'Cloze Card', 'qfmt': '{{cloze:Text}}', 'afmt': '{{cloze:Text}}'}],
        model_type=genanki.Model.CLOZE,
    )

    for card in cards:
        if card['is_cloze']:
            note = genanki.Note(model=cloze_model, fields=[card['front']], tags=['generated'])
        else:
            note = genanki.Note(model=basic_model, fields=[card['front'], card['back']], tags=['generated'])
        deck.add_note(note)

    package = genanki.Package(deck)
    buf = io.BytesIO()
    package.write_to_file(buf)
    buf.seek(0)
    return buf


# ─── HOME ────────────────────────────────────────────────────────────────────

@app.route('/')
def home():
    body = """
    <div class="page">
      <div class="hero">
        <h1>Your Med School<br>Study Hub</h1>
        <p>Four AI-powered tools to help you study smarter and crush boards.</p>
      </div>
      <div class="tools-grid" style="grid-template-columns:repeat(2,1fr)">
        <a href="/anki" class="tool-card anki">
          <div class="tool-emoji">🃏</div>
          <div class="tool-title">Anki Generator</div>
          <div class="tool-desc">Upload a PDF and Claude auto-generates a downloadable Anki deck. Or paste cards manually in tab-separated format.</div>
          <div class="tool-arrow">Open tool →</div>
        </a>
        <a href="/practice-tests" class="tool-card tests">
          <div class="tool-emoji">📝</div>
          <div class="tool-title">Practice Tests</div>
          <div class="tool-desc">Upload your lecture PDFs and get 25 NBME-style multiple choice questions with a full answer key — per lecture.</div>
          <div class="tool-arrow">Open tool →</div>
        </a>
        <a href="/jeremy" class="tool-card jeremy">
          <div class="tool-emoji">🔥</div>
          <div class="tool-title">Jeremy Mode</div>
          <div class="tool-desc">Drop a PDF and get a high-energy interactive quiz session with your hype-beast AI tutor. Big brain plays only.</div>
          <div class="tool-arrow">Open tool →</div>
        </a>
        <a href="/uworld" class="tool-card uworld">
          <div class="tool-emoji">🎯</div>
          <div class="tool-title">UWorld Review</div>
          <div class="tool-desc">Paste the questions you missed in UWorld. Get a missed concepts summary, high-yield board points, and targeted drill questions.</div>
          <div class="tool-arrow">Open tool →</div>
        </a>
      </div>
    </div>
    """
    return render_page(body)


# ─── ANKI GENERATOR ──────────────────────────────────────────────────────────

@app.route('/anki-from-pdf', methods=['POST'])
def anki_from_pdf():
    if 'pdf' not in request.files:
        return json.dumps({'error': 'No file uploaded'}), 400, {'Content-Type': 'application/json'}

    file = request.files['pdf']
    if not file.filename.lower().endswith('.pdf'):
        return json.dumps({'error': 'Must be a PDF'}), 400, {'Content-Type': 'application/json'}

    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        cards = call_claude_with_pdf(tmp_path, ANKI_FROM_PDF_PROMPT)
        
        signed_uid = request.headers.get('X-Signed-Uid')
        explicit_uid = verify_uid(signed_uid) if signed_uid else None
        
        title = file.filename
        if title.lower().endswith('.pdf'):
            title = title[:-4]
        title = f"Anki Deck: {title}"
        save_to_library(title, "anki", cards, explicit_uid=explicit_uid)
        
        return json.dumps({'cards': cards}), 200, {'Content-Type': 'application/json'}
    except Exception as e:
        return json.dumps({'error': str(e)}), 500, {'Content-Type': 'application/json'}
    finally:
        try: os.unlink(tmp_path)
        except: pass


@app.route('/anki', methods=['GET'])
def anki_get():
    uid = session.get('uid')
    signed_uid = sign_uid(uid)
    body = _anki_body(signed_uid=signed_uid)
    return render_page(body, active="anki")


@app.route('/anki', methods=['POST'])
def anki_post():
    cards_text = request.form.get('cards_text', '')
    deck_name = request.form.get('deck_name', 'My Deck').strip() or 'My Deck'
    cards = parse_cards(cards_text)
    if not cards:
        return render_page(_anki_body("No valid cards found. Make sure cards are tab-separated.", signed_uid=request.form.get('signed_uid', '')), active="anki")
        
    signed_uid = request.form.get('signed_uid')
    explicit_uid = verify_uid(signed_uid)
    save_to_library(f"Anki Deck: {deck_name}", "anki", cards_text, explicit_uid=explicit_uid)
    
    buf = build_anki_package(cards, deck_name)
    return send_file(buf, as_attachment=True, download_name=f"{deck_name}.apkg", mimetype='application/octet-stream')


def _anki_body(error=None, signed_uid=""):
    err_html = f'<div class="notice" style="border-color:rgba(239,68,68,0.3);background:rgba(239,68,68,0.08);color:#f87171;">{error}</div>' if error else ''
    body = f"""
    <div class="page">
      <div class="section-header">
        <span class="pill pill-purple">🃏 Anki Generator</span>
        <div class="section-title">Create Your Anki Deck</div>
        <div class="section-sub">Paste cards manually or let Claude generate them from a PDF.</div>
      </div>
      {err_html}

      <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;">
        <!-- LEFT: PDF Auto-Generate -->
        <div class="panel">
          <div class="panel-title">✨ Auto-generate from PDF</div>
          <div class="file-drop" id="ankiDrop">
            <input type="file" id="ankiPdf" accept="application/pdf">
            <div class="file-drop-icon">📄</div>
            <div class="file-drop-text"><strong>Drop a PDF here</strong><br>or click to browse</div>
            <div class="file-names" id="ankiFileName"></div>
          </div>
          <br>
          <button class="btn btn-purple" id="generateCardsBtn" style="width:100%" onclick="generateAnkiFromPdf()">
            <span id="generateBtnText">✨ Generate Cards with AI</span>
          </button>
          <div style="margin-top:12px;display:none" id="generatingBar">
            <div class="loading-bar" style="width:100%"></div>
            <div style="text-align:center;font-size:12px;color:var(--text-muted);margin-top:8px">Claude is reading your PDF...</div>
          </div>
        </div>

        <!-- RIGHT: Manual Paste -->
        <div class="panel">
          <div class="panel-title">📋 Or paste cards manually</div>
          <div class="notice">
            <strong>Format:</strong> Basic cards → <code>Front [tab] Back</code> &nbsp;|&nbsp; Cloze cards → <code>The {{{{c1::answer}}}} is key.</code><br><br>
            <strong>Pro tip:</strong> Ask your AI with this prompt ↓
          </div>
          <div class="prompt-box">Please generate Anki cards from this PDF in both Basic and Cloze format. Give them as plain text — Basic cards with front and back separated by a tab, Cloze cards with {{{{c1::}}}} syntax. Focus on high-yield facts for first-year med students preparing for boards.</div>
        </div>
      </div>

      <!-- CARD EDITOR -->
      <div class="panel" style="margin-top:20px">
        <div class="panel-title">✏️ Card Editor</div>
        <form method="POST" action="https://medtools-vneeiy3k7q-uc.a.run.app/anki" id="ankiForm">
          <input type="hidden" name="signed_uid" value="__SIGNED_UID__">
          <div class="form-group">
            <label>Deck Name</label>
            <input type="text" name="deck_name" id="deckName" placeholder="e.g. Cardiology Block 1" required>
          </div>
          <div class="form-group">
            <label>Cards <span style="color:var(--text-dim);font-weight:400">(auto-filled from PDF or paste manually)</span></label>
            <textarea name="cards_text" id="ankiTextarea" rows="14" placeholder="Front&#9;Back&#10;The {{c1::mitral valve}} is on the left side of the heart."></textarea>
          </div>
          <div style="display:flex;gap:10px;align-items:center">
            <button type="submit" class="btn btn-purple" id="downloadBtn">⬇ Download Anki Deck</button>
            <button type="button" class="btn btn-ghost" onclick="document.getElementById('ankiTextarea').value=''">Clear</button>
            <span id="cardCountText" style="color:var(--text-muted);font-size:13px;margin-left:auto"></span>
          </div>
        </form>
      </div>
    </div>

    <script>
      const textarea = document.getElementById('ankiTextarea');
      const cardCount = document.getElementById('cardCountText');

      function updateCount() {{
        const lines = textarea.value.split('\\n').filter(l => l.includes('\\t') || l.includes('{{c'));
        cardCount.textContent = lines.length ? lines.length + ' cards' : '';
      }}
      textarea.addEventListener('input', updateCount);

      document.getElementById('ankiPdf').addEventListener('change', function() {{
        const name = this.files[0]?.name || '';
        document.getElementById('ankiFileName').textContent = name ? '📎 ' + name : '';
      }});

      async function generateAnkiFromPdf() {{
        if (!checkUsageLimit()) return;
        const fileInput = document.getElementById('ankiPdf');
        if (!fileInput.files[0]) {{ alert('Please select a PDF first.'); return; }}

        const btn = document.getElementById('generateCardsBtn');
        const bar = document.getElementById('generatingBar');
        document.getElementById('generateBtnText').textContent = 'Generating...';
        btn.disabled = true;
        bar.style.display = 'block';

        const formData = new FormData();
        formData.append('pdf', fileInput.files[0]);

        const headers = {{'Accept': 'application/json', 'X-Signed-Uid': '__SIGNED_UID__'}};
        if (window.firebase && firebase.auth && firebase.auth().currentUser) {{
          try {{
            const token = await firebase.auth().currentUser.getIdToken();
            headers['Authorization'] = 'Bearer ' + token;
          }} catch(e) {{}}
        }}

        try {{
          const res = await fetch('https://medtools-vneeiy3k7q-uc.a.run.app/anki-from-pdf', {{ method: 'POST', body: formData, headers: headers }});
          const data = await res.json();
          if (data.error) {{ alert('Error: ' + data.error); }}
          else {{
            incrementUsage();
            textarea.value = data.cards;
            updateCount();
            if (!document.getElementById('deckName').value) {{
              document.getElementById('deckName').value = fileInput.files[0].name.replace('.pdf','');
            }}
          }}
        }} catch(e) {{ alert('Something went wrong.'); }}
        finally {{
          btn.disabled = false;
          document.getElementById('generateBtnText').textContent = '✨ Generate Cards with AI';
          bar.style.display = 'none';
        }}
      }}
    </script>
    """
    return body.replace('__SIGNED_UID__', signed_uid)



# ─── PRACTICE TESTS ──────────────────────────────────────────────────────────

@app.route('/practice-tests', methods=['GET'])
def practice_tests_get():
    uid = session.get('uid')
    signed_uid = sign_uid(uid)
    body = """
    <div class="page" style="max-width:820px">
      <div class="section-header">
        <span class="pill pill-green">📝 Practice Tests</span>
        <div class="section-title">NBME-Style Practice Tests</div>
        <div class="section-sub">Upload one or more lecture PDFs and get 25 board-style questions with a full answer key for each.</div>
      </div>

      <div class="panel">
        <form id="testForm" method="POST" action="/practice-tests" enctype="multipart/form-data">
          <div class="file-drop" id="testDrop">
            <input type="file" id="testPdfs" name="pdfs" multiple accept="application/pdf" onchange="updateTestFiles()">
            <div class="file-drop-icon">📚</div>
            <div class="file-drop-text"><strong>Drop your lecture PDFs here</strong><br>or click to browse — multiple files OK</div>
            <div class="file-names" id="testFileNames"></div>
          </div>
          <br>
          <button type="submit" class="btn btn-green" id="testBtn" style="width:100%">
            <span id="testBtnText">📝 Generate Practice Tests</span>
          </button>
          <div style="margin-top:12px;display:none" id="testBar">
            <div class="loading-bar" style="width:100%"></div>
            <div style="text-align:center;font-size:12px;color:var(--text-muted);margin-top:8px">Claude is building your questions... this can take 1-2 min per PDF.</div>
          </div>
        </form>
      </div>

      <div id="resultsArea"></div>
    </div>

    <script>
      function updateTestFiles() {
        const files = document.getElementById('testPdfs').files;
        const drop = document.getElementById('testDrop');
        const namesEl = document.getElementById('testFileNames');
        const iconEl = drop.querySelector('.file-drop-icon');
        const textEl = drop.querySelector('.file-drop-text');
        if (files.length) {
          drop.style.borderColor = 'var(--green)';
          drop.style.background = 'rgba(16,185,129,0.07)';
          iconEl.textContent = '✅';
          textEl.innerHTML = `<strong style="color:var(--green)">${files.length} PDF${files.length > 1 ? 's' : ''} ready</strong>`;
          namesEl.textContent = Array.from(files).map(f => '📎 ' + f.name).join('  ');
        } else {
          drop.style.borderColor = '';
          drop.style.background = '';
          iconEl.textContent = '📚';
          textEl.innerHTML = '<strong>Drop your lecture PDFs here</strong><br>or click to browse — multiple files OK';
          namesEl.textContent = '';
        }
      }

      document.getElementById('testForm').addEventListener('submit', async function(e) {
        e.preventDefault();
        if (!checkUsageLimit()) return;

        const files = document.getElementById('testPdfs').files;
        if (!files.length) { alert('Please select at least one PDF.'); return; }

        const btn = document.getElementById('testBtn');
        btn.disabled = true;
        document.getElementById('testBtnText').textContent = 'Generating…';
        document.getElementById('testBar').style.display = 'block';

        // Show a spinner placeholder for every PDF right away
        const resultsArea = document.getElementById('resultsArea');
        resultsArea.innerHTML = '';
        const fileList = Array.from(files).map(f => f.name);
        fileList.forEach((name, i) => {
          const div = document.createElement('div');
          div.id = 'pending-' + i;
          div.className = 'result-block';
          div.innerHTML = `
            <div style="display:flex;align-items:center;gap:16px;padding:20px">
              <div class="loading-bar" style="width:36px;height:36px;border-radius:50%;flex-shrink:0"></div>
              <div>
                <div class="result-label" style="margin-bottom:4px">📄 ${name}</div>
                <div style="font-size:13px;color:var(--text-muted)">Claude is reading your PDF and building questions…</div>
              </div>
            </div>`;
          resultsArea.appendChild(div);
        });

        const formData = new FormData(this);
        const headers = {'X-Signed-Uid': '__SIGNED_UID__'};
        if (window.firebase && firebase.auth && firebase.auth().currentUser) {
          try { headers['Authorization'] = 'Bearer ' + await firebase.auth().currentUser.getIdToken(); } catch(e) {}
        }

        let completedCount = 0;
        function reset() {
          btn.disabled = false;
          document.getElementById('testBtnText').textContent = '📝 Generate Practice Tests';
          document.getElementById('testBar').style.display = 'none';
        }

        try {
          const res = await fetch('/practice-tests/stream', { method: 'POST', headers, body: formData });
          if (!res.ok) { alert('Server error: HTTP ' + res.status); reset(); return; }

          const reader = res.body.getReader();
          const dec = new TextDecoder();
          let buf = '';

          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buf += dec.decode(value, { stream: true });
            const lines = buf.split('\n');
            buf = lines.pop();

            for (const line of lines) {
              if (!line.startsWith('data: ')) continue;
              let evt;
              try { evt = JSON.parse(line.slice(6)); } catch(e) { continue; }

              if (evt.type === 'test') {
                const rid = 'r' + completedCount++;
                const pidx = fileList.indexOf(evt.filename);
                const pending = pidx >= 0 ? document.getElementById('pending-' + pidx) : null;

                let isJson = false;
                try { JSON.parse(evt.content); isJson = true; } catch(e) {}

                const wrapper = document.createElement('div');
                wrapper.innerHTML = isJson
                  ? `<div class="result-block" id="result-${rid}">
                       <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;margin-bottom:10px">
                         <div class="result-label">📄 ${evt.filename}</div>
                         <div style="display:flex;align-items:center;gap:12px">
                           <div style="font-size:14px;color:var(--text-muted);font-weight:600" id="scoreDisplay-${rid}">–</div>
                           <button class="btn btn-green" style="font-size:13px;padding:8px 14px" onclick="downloadPdf('${rid}', this.dataset.fname)" data-fname="${evt.filename}">⬇ Download PDF</button>
                         </div>
                       </div>
                       <div class="output-box" id="interactive-${rid}" style="padding:0;background:none;border:none"></div>
                       <div id="content-${rid}" style="display:none;"></div>
                     </div>`
                  : `<div class="result-block" id="result-${rid}">
                       <div class="result-label" style="margin-bottom:10px">📄 ${evt.filename}</div>
                       <div class="output-box rendered-md" id="content-${rid}"></div>
                     </div>`;
                const block = wrapper.firstElementChild;
                if (pending) pending.replaceWith(block);
                else resultsArea.appendChild(block);
                renderOneTest(rid, evt.content, isJson);
              }

              if (evt.type === 'done') { incrementUsage(); reset(); }
              if (evt.type === 'error') { alert('Error: ' + (evt.message || 'Unknown')); reset(); }
            }
          }
          if (btn.disabled) reset();
        } catch(err) {
          alert('Network error: ' + err);
          reset();
        }
      });
    </script>
    """
    body += _practice_test_js()
    body = body.replace('__SIGNED_UID__', signed_uid)
    return render_page(body, active="tests")

def _practice_test_js():
    return """
    <style>
      .rendered-md { white-space: normal; font-size: 15px; line-height: 1.75; max-height: none !important; overflow: visible !important; }
      .rendered-md h1,.rendered-md h2,.rendered-md h3 { margin: 16px 0 8px; }
      .rendered-md h1 { font-size: 22px; }
      .rendered-md h2 { font-size: 18px; color: var(--green); }
      .rendered-md h3 { font-size: 16px; color: var(--text-muted); }
      .rendered-md p { margin: 8px 0; }
      .rendered-md strong { color: var(--text); }
      .rendered-md ol,.rendered-md ul { padding-left: 22px; margin: 8px 0; }
      .rendered-md li { margin: 4px 0; }
      .rendered-md table { width:100%;border-collapse:collapse;margin:12px 0;font-size:14px; }
      .rendered-md th { background:rgba(16,185,129,0.12);color:var(--green);padding:8px 12px;text-align:left;border:1px solid var(--border);font-weight:600; }
      .rendered-md td { padding:7px 12px;border:1px solid var(--border); }
      .rendered-md tr:nth-child(even) td { background:rgba(255,255,255,0.02); }
      .rendered-md blockquote { border-left:3px solid var(--green);margin:8px 0;padding:4px 12px;color:var(--text-muted);background:rgba(16,185,129,0.05);border-radius:0 6px 6px 0; }
      .rendered-md hr { border:none;border-top:1px solid var(--border);margin:14px 0; }
      .rendered-md code { background:rgba(255,255,255,0.08);border:1px solid var(--border);border-radius:4px;padding:1px 6px;font-size:13px;font-family:monospace; }
      
      .interactive-test { font-family: 'Inter', sans-serif; }
      .drill-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 24px; margin-bottom: 20px; }
      .drill-q { font-size: 16px; font-weight: 600; margin-bottom: 16px; color: var(--text); white-space: pre-wrap; }
      .drill-choice { 
        display: block; width: 100%; text-align: left; background: rgba(255,255,255,0.03); 
        border: 1px solid var(--border); border-radius: 8px; padding: 12px 16px; 
        margin-bottom: 8px; color: var(--text); cursor: pointer; transition: all 0.2s;
      }
      .drill-choice:hover:not(:disabled) { background: rgba(255,255,255,0.08); border-color: var(--border-bright); }
      .drill-choice.correct { background: rgba(16,185,129,0.1); border-color: var(--green); color: var(--green); }
      .drill-choice.wrong { background: rgba(239,68,68,0.1); border-color: #ef4444; color: #ef4444; opacity: 0.7; }
      .drill-exp { margin-top: 16px; padding: 16px; background: rgba(0,0,0,0.2); border-left: 3px solid var(--blue); border-radius: 4px 8px 8px 4px; font-size: 14px; color: var(--text-muted); display: none; }
      .pdf-only { display: none; }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.10.1/html2pdf.bundle.min.js"></script>
    <script>
      function renderPracticeTests() {
        document.querySelectorAll('.result-block').forEach((block) => {
          const idStr = block.id.replace('result-', '');
          const contentEl = document.getElementById('content-' + idStr);
          if (!contentEl) return;
          const dataType = contentEl.getAttribute('data-type') || 'markdown';
          const raw = contentEl.getAttribute('data-raw');
          
          if (dataType === 'markdown') {
            contentEl.innerHTML = marked.parse(raw);
            contentEl.style.display = 'block';
          } else if (dataType === 'json') {
            try {
              let questions = JSON.parse(raw);
              if (!Array.isArray(questions)) {
                if (questions.questions && Array.isArray(questions.questions)) {
                    questions = questions.questions;
                } else {
                    const keys = Object.keys(questions);
                    if (keys.length === 1 && Array.isArray(questions[keys[0]])) {
                        questions = questions[keys[0]];
                    } else {
                        throw new Error("Cannot find questions array in JSON");
                    }
                }
              }
              
              const interactiveEl = document.getElementById('interactive-' + idStr);
              let html = '<div class="interactive-test">';
              let printableMd = '';
              
              questions.forEach((q, qIndex) => {
                html += `<div class="drill-card" id="q-${idStr}-${qIndex}">`;
                html += `<div class="drill-q">${qIndex + 1}. ${q.q || 'Question text missing'}</div>`;
                printableMd += `**${qIndex + 1}. ${q.q || 'Question text missing'}**\\n\\n`;
                
                const choices = q.choices || [];
                const correct = (q.correct || 'A').toUpperCase().trim();
                
                choices.forEach((choice) => {
                  const letter = choice.charAt(0).toUpperCase();
                  const isCorrect = (letter === correct);
                  html += `<button class="drill-choice" onclick="answerPracticeQ('${idStr}', ${qIndex}, ${isCorrect}, '${correct}', this)">${choice}</button>`;
                  printableMd += `- ${choice}\\n`;
                });
                const exp = q.exp || q.explanation || 'No explanation provided.';
                html += `<div class="drill-exp" id="exp-${idStr}-${qIndex}"><strong>Explanation:</strong><br>${exp}</div>`;
                html += `</div>`;
                
                printableMd += `\\n**Answer:** ${correct}\\n\\n*Explanation:* ${exp}\\n\\n---\\n\\n`;
              });
              html += '</div>';
              interactiveEl.innerHTML = html;
              
              contentEl.innerHTML = `<div class="pdf-only rendered-md">${marked.parse(printableMd)}</div>`;
              contentEl.style.display = 'block'; 
              
              block.setAttribute('data-score', '0');
              block.setAttribute('data-total', questions.length);
              
            } catch(e) {
              console.error("Practice test render error:", e);
              // Fallback if parsing fails!
              contentEl.classList.add('rendered-md');
              contentEl.innerHTML = marked.parse("```json\\n" + raw + "\\n```");
              contentEl.style.display = 'block';
            }
          }
        });
      }
      
      function answerPracticeQ(testId, qIndex, isCorrect, correctLetter, btnEl) {
        const card = document.getElementById(`q-${testId}-${qIndex}`);
        const buttons = card.querySelectorAll('.drill-choice');
        buttons.forEach(b => b.disabled = true);
        
        if (isCorrect) {
          btnEl.classList.add('correct');
          const block = document.getElementById(`result-${testId}`);
          let score = parseInt(block.getAttribute('data-score')) + 1;
          let total = block.getAttribute('data-total');
          block.setAttribute('data-score', score);
          document.getElementById(`scoreDisplay-${testId}`).innerText = `${score} / ${total} Correct`;
        } else {
          btnEl.classList.add('wrong');
          buttons.forEach(b => {
            if (b.innerText.charAt(0).toUpperCase() === correctLetter) {
              b.classList.add('correct');
            }
          });
        }
        document.getElementById(`exp-${testId}-${qIndex}`).style.display = 'block';
      }
      
      const pdfOpts = name => ({
        margin: [15, 20],
        filename: name.replace('.pdf','') + '_practice_test.pdf',
        image: { type: 'jpeg', quality: 0.98 },
        html2canvas: { scale: 2, backgroundColor: '#ffffff' },
        jsPDF: { unit: 'mm', format: 'a4', orientation: 'portrait' },
        pagebreak: { mode: ['avoid-all', 'css'] }
      });

      function downloadPdf(idx, fname) {
        const el = document.getElementById('content-' + idx);
        const clone = el.cloneNode(true);
        clone.id = 'pdf-clone-' + idx;
        clone.className = '';
        clone.style.cssText = 'position:absolute;left:-9999px;top:-9999px;background:#fff;color:#111;padding:20px;font-family:Georgia,serif;font-size:14px;line-height:1.7;max-height:none;overflow:visible;width:800px;z-index:-1000;';
        clone.style.display = 'block';
        
        // Override CSS variables specifically for the PDF render
        clone.style.setProperty('--text', '#111', 'important');
        clone.style.setProperty('--text-muted', '#333', 'important');
        clone.style.setProperty('--green', '#000', 'important');
        clone.style.setProperty('--orange', '#000', 'important');
        clone.style.setProperty('--border', '#ccc', 'important');
        clone.style.setProperty('--surface', '#fff', 'important');
        
        // Ensure pdf-only blocks are visible inside the clone
        clone.querySelectorAll('.pdf-only').forEach(c => {
          c.style.display = 'block';
        });
        
        clone.querySelectorAll('*').forEach(c => {
          c.style.maxHeight = 'none';
          c.style.overflow = 'visible';
          // Force all text to be dark as an extra fallback
          c.style.setProperty('color', '#111', 'important');
          c.style.textShadow = 'none';
        });
        clone.querySelectorAll('table').forEach(t => t.style.cssText = 'border-collapse:collapse;width:100%;margin:10px 0');
        clone.querySelectorAll('th,td').forEach(c => c.style.cssText = 'border:1px solid #ccc;padding:6px 10px;color:#111');
        clone.querySelectorAll('th').forEach(c => c.style.background = '#f0f0f0');
        
        // Remove background colors that might conflict with white
        clone.querySelectorAll('code').forEach(c => {
            c.style.background = '#f5f5f5';
            c.style.borderColor = '#ddd';
        });
        
        // Append to DOM so html2canvas computes styles perfectly
        document.body.appendChild(clone);
        
        html2pdf().set(pdfOpts(fname)).from(clone).save().then(() => {
            // Clean up the DOM after PDF is generated
            document.body.removeChild(clone);
        });
      }
      
      document.addEventListener('DOMContentLoaded', renderPracticeTests);

      function renderOneTest(rid, content, isJson) {
        if (!isJson) {
          const el = document.getElementById('content-' + rid);
          if (el) { el.innerHTML = marked.parse(content); el.style.display = 'block'; }
          return;
        }
        try {
          let questions = JSON.parse(content);
          if (!Array.isArray(questions)) {
            if (questions.questions && Array.isArray(questions.questions)) {
              questions = questions.questions;
            } else {
              const k = Object.keys(questions);
              if (k.length === 1 && Array.isArray(questions[k[0]])) questions = questions[k[0]];
              else throw new Error('Cannot find questions array');
            }
          }
          const interEl = document.getElementById('interactive-' + rid);
          const contEl  = document.getElementById('content-' + rid);
          let html = '<div class="interactive-test">';
          let md   = '';
          questions.forEach((q, qi) => {
            const correct = (q.correct || 'A').toUpperCase().trim();
            const exp = q.exp || q.explanation || 'No explanation provided.';
            html += `<div class="drill-card" id="q-${rid}-${qi}">
                       <div class="drill-q">${qi + 1}. ${q.q || 'Question text missing'}</div>`;
            md += `**${qi + 1}. ${q.q || ''}**\\n\\n`;
            (q.choices || []).forEach(choice => {
              const letter = choice.charAt(0).toUpperCase();
              html += `<button class="drill-choice" onclick="answerPracticeQ('${rid}',${qi},${letter === correct},'${correct}',this)">${choice}</button>`;
              md += `- ${choice}\\n`;
            });
            html += `<div class="drill-exp" id="exp-${rid}-${qi}"><strong>Explanation:</strong><br>${exp}</div></div>`;
            md += `\\n**Answer:** ${correct}\\n\\n*Explanation:* ${exp}\\n\\n---\\n\\n`;
          });
          html += '</div>';
          interEl.innerHTML = html;
          contEl.innerHTML  = `<div class="pdf-only rendered-md">${marked.parse(md)}</div>`;
          contEl.style.display = 'block';
          const block = document.getElementById('result-' + rid);
          block.setAttribute('data-score', '0');
          block.setAttribute('data-total', questions.length);
          document.getElementById('scoreDisplay-' + rid).textContent = '0 / ' + questions.length + ' Correct';
        } catch(err) {
          console.error('renderOneTest:', err);
          const el = document.getElementById('content-' + rid);
          if (el) { el.classList.add('rendered-md'); el.innerHTML = marked.parse(content); el.style.display = 'block'; }
        }
      }
    </script>
    """

@app.route('/practice-tests', methods=['POST'])
def practice_tests_post():
    if 'pdfs' not in request.files:
        return render_page('<div class="page"><div class="notice">No files uploaded.</div></div>', active="tests")

    files = request.files.getlist('pdfs')
    files = [f for f in files if f.filename.lower().endswith('.pdf')]
    if not files:
        return render_page('<div class="page"><div class="notice">No PDF files found.</div></div>', active="tests")

    results = []
    with tempfile.TemporaryDirectory() as tmpdir:
        import concurrent.futures
        import werkzeug.utils

        paths = []
        for f in files:
            safe_name = werkzeug.utils.secure_filename(f.filename)
            if not safe_name:
                safe_name = f"upload_{len(paths)}.pdf"
            path = os.path.join(tmpdir, safe_name)
            f.save(path)
            paths.append((f.filename, path))
            
        def process_path(item):
            fname, path = item
            try:
                raw_text = call_claude_with_pdf(path, PRACTICE_TEST_PROMPT)
                text = raw_text.strip()
                import re as _re
                if text.startswith("```"):
                    text = _re.sub(r'^```[^\n]*\n?', '', text)
                    text = _re.sub(r'\n?```$', '', text.strip())
                import json
                json.loads(text) # validate json
            except Exception as e:
                text = f"Error generating tests for {fname}: {str(e)}\n\nThis often happens due to Anthropic API rate limits when uploading too many PDFs at once."
            return (fname, text)
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(process_path, paths))

    import sys
    save_data = []
    for fname, content in results:
        if not content.startswith("Error"):
            save_data.append({"filename": fname, "content": content})
            
    print(f"DEBUG practice_tests_post: save_data length={len(save_data)}", file=sys.stderr)
    for sd in save_data:
        print(f"DEBUG JSON CONTENT: {repr(sd['content'][:500])}", file=sys.stderr)
    sys.stderr.flush()

    if save_data:
        signed_uid = request.headers.get('X-Signed-Uid')
        explicit_uid = verify_uid(signed_uid)
        title = save_data[0]['filename']
        if title.lower().endswith('.pdf'):
            title = title[:-4]
        title = f"Practice Test: {title}"
        if len(save_data) > 1:
            title += f" (+{len(save_data)-1} more)"
        save_to_library(title, "practice_test", save_data, explicit_uid=explicit_uid)

    import html as html_module
    import json
    result_blocks = ""
    for i, (fname, content) in enumerate(results):
        safe_content = html_module.escape(content)
        is_json = False
        try:
            json.loads(content)
            is_json = True
        except:
            pass
            
        if is_json:
            result_blocks += f"""
            <div class="result-block" id="result-{i}">
              <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px">
                <div class="result-label">📄 {fname}</div>
                <div style="font-size:14px;color:var(--text-muted);font-weight:600" id="scoreDisplay-{i}">0 / 0</div>
                <button class="btn btn-green" style="font-size:13px;padding:8px 14px" onclick="downloadPdf({i}, '{html_module.escape(fname)}')">
                  ⬇ Download PDF
                </button>
              </div>
              <div class="output-box" id="interactive-{i}" style="padding:0;background:none;border:none"></div>
              <div id="content-{i}" style="display:none;" data-raw="{safe_content}" data-type="json"></div>
            </div>"""
        else:
            result_blocks += f"""
            <div class="result-block" id="result-{i}">
              <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px">
                <div class="result-label">📄 {fname}</div>
                <button class="btn btn-green" style="font-size:13px;padding:8px 14px" onclick="downloadPdf({i}, '{html_module.escape(fname)}')">
                  ⬇ Download PDF
                </button>
              </div>
              <div class="output-box rendered-md" id="content-{i}" data-raw="{safe_content}" data-type="markdown"></div>
            </div>"""

    body = f"""
    <div class="page" style="max-width:900px">
      <div class="section-header">
        <span class="pill pill-green">📝 Practice Tests</span>
        <div class="section-title">Your Practice Tests</div>
        <div class="section-sub">{len(results)} lecture{'s' if len(results)!=1 else ''} processed.</div>
      </div>
      <div style="display:flex;gap:12px;margin-bottom:28px;flex-wrap:wrap">
        <a href="/practice-tests" class="btn btn-ghost">← Generate More</a>
        {"" if len(results) < 2 else '<button class="btn btn-green" onclick="downloadAll()">⬇ Download All PDFs</button>'}
      </div>
      {result_blocks}
    </div>
    {_practice_test_js()}
    <script>
      function downloadAll() {{
        document.querySelectorAll('.result-block').forEach((block, i) => {{
          setTimeout(() => {{
            const fname = block.querySelector('.result-label').textContent.replace('📄 ','');
            downloadPdf(i, fname);
          }}, i * 1500);
        }});
      }}
    </script>
    """
    if request.headers.get('Accept') == 'text/html-partial':
        return body
    return render_page(body, active="tests")


@app.route('/practice-tests/stream', methods=['POST'])
def practice_tests_stream():
    import concurrent.futures, shutil, werkzeug.utils, re as _re_s

    files = request.files.getlist('pdfs')
    files = [f for f in files if f.filename.lower().endswith('.pdf')]
    if not files:
        def _err():
            yield f"data: {json.dumps({'type':'error','message':'No PDF files found.'})}\n\n"
        return Response(stream_with_context(_err()), mimetype='text/event-stream')

    tmpdir = tempfile.mkdtemp()
    paths = []
    for f in files:
        safe_name = werkzeug.utils.secure_filename(f.filename) or f"upload_{len(paths)}.pdf"
        path = os.path.join(tmpdir, safe_name)
        f.save(path)
        paths.append((f.filename, path))

    signed_uid = request.headers.get('X-Signed-Uid')
    explicit_uid = verify_uid(signed_uid)

    def generate():
        all_results = []
        try:
            def process_one(item):
                fname, path = item
                try:
                    raw = call_claude_with_pdf(path, PRACTICE_TEST_PROMPT)
                    text = raw.strip()
                    if text.startswith("```"):
                        text = _re_s.sub(r'^```[^\n]*\n?', '', text)
                        text = _re_s.sub(r'\n?```$', '', text.strip())
                    json.loads(text)  # validate
                    return fname, text, None
                except Exception as e:
                    return fname, '[]', str(e)

            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = {executor.submit(process_one, item): item[0] for item in paths}
                for future in concurrent.futures.as_completed(futures):
                    fname, content, error = future.result()
                    all_results.append({'filename': fname, 'content': content})
                    evt = {'type': 'test', 'filename': fname, 'content': content}
                    if error:
                        evt['error'] = error
                    yield f"data: {json.dumps(evt)}\n\n"

            save_data = [r for r in all_results if r['content'] not in ('[]', '')]
            if save_data and explicit_uid:
                title = save_data[0]['filename'].replace('.pdf', '')
                if len(save_data) > 1:
                    title += f" (+{len(save_data)-1} more)"
                save_to_library(f"Practice Test: {title}", "practice_test", save_data, explicit_uid=explicit_uid)

            yield f"data: {json.dumps({'type':'done'})}\n\n"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
    )


# ─── JEREMY MODE ─────────────────────────────────────────────────────────────

@app.route('/jeremy')
def jeremy():
    body = """
    <div class="chat-layout">
      <div class="chat-messages" id="chatMessages">
        <div class="chat-msg">
          <div class="msg-avatar jeremy">🔥</div>
          <div class="msg-bubble">Jeremy Mode ENGAGED! 💥🧠

Welcome, Chief. I'm your high-octane, board-crushing AI tutor.

Here's what we can do:
• 📄 **Drop a PDF** → I'll summarize it then launch a full interactive quiz
• 💬 **Ask me anything** → med school questions, concept explanations, mnemonics
• 🃏 **Request Anki cards** → I'll generate them in the right format for your deck

What are we studying today, Legend? Upload a lecture or tell me the topic. LET'S GET IT. 🚀</div>
        </div>
      </div>

      <div class="chat-input-bar">
        <div id="uploadedBadge" style="margin:0 auto 8px;display:none">
          <span class="uploaded-badge">📎 <span id="uploadedName"></span> <button onclick="clearUpload()" style="background:none;border:none;color:var(--orange);cursor:pointer;margin-left:4px">✕</button></span>
        </div>
        <div class="chat-input-row">
          <label class="chat-upload-btn" title="Upload PDF">
            📎
            <input type="file" id="pdfInput" accept="application/pdf" style="display:none" onchange="onPdfSelected()">
          </label>
          <textarea class="chat-input" id="chatInput" placeholder="Ask Jeremy anything, or upload a PDF above..." rows="1"
            onkeydown="handleKey(event)" oninput="autoResize(this)"></textarea>
          <button class="chat-send-btn" id="sendBtn" onclick="sendMessage()">➤</button>
        </div>
        <div class="chat-hint">Press Enter to send · Shift+Enter for new line · Upload PDF to start a quiz</div>
      </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <script>
      marked.setOptions({ breaks: true });
      let pendingPdf = null;
      let isStreaming = false;
      // Track raw text history separately — never scrape the DOM
      let conversationHistory = [];

      function onPdfSelected() {
        const file = document.getElementById('pdfInput').files[0];
        if (!file) return;
        pendingPdf = file;
        document.getElementById('uploadedName').textContent = file.name;
        document.getElementById('uploadedBadge').style.display = 'block';
      }

      function clearUpload() {
        pendingPdf = null;
        document.getElementById('pdfInput').value = '';
        document.getElementById('uploadedBadge').style.display = 'none';
      }

      function autoResize(el) {
        el.style.height = 'auto';
        el.style.height = Math.min(el.scrollHeight, 160) + 'px';
      }

      function handleKey(e) {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
      }

      function appendMsg(role, text, streaming=false) {
        const msgs = document.getElementById('chatMessages');
        const div = document.createElement('div');
        div.className = 'chat-msg' + (role === 'user' ? ' user' : '');
        const avatar = document.createElement('div');
        avatar.className = 'msg-avatar ' + (role === 'user' ? 'user' : 'jeremy');
        avatar.textContent = role === 'user' ? '👤' : '🔥';
        const bubble = document.createElement('div');
        bubble.className = 'msg-bubble' + (streaming ? ' streaming' : '');
        if (text) bubble.innerHTML = marked.parse(text);
        else bubble.textContent = '';
        div.appendChild(avatar);
        div.appendChild(bubble);
        msgs.appendChild(div);
        msgs.scrollTop = msgs.scrollHeight;
        return bubble;
      }

      async function sendMessage() {
        if (isStreaming) return;
        const input = document.getElementById('chatInput');
        const text = input.value.trim();
        if (!text && !pendingPdf) return;

        const userMsg = text || ('📎 Uploaded: ' + pendingPdf?.name);
        appendMsg('user', userMsg);
        input.value = '';
        input.style.height = 'auto';

        const bubble = appendMsg('jeremy', '', true);
        isStreaming = true;
        document.getElementById('sendBtn').disabled = true;

        const formData = new FormData();
        formData.append('message', text);
        if (pendingPdf) { formData.append('pdf', pendingPdf); clearUpload(); }

        // Send history BEFORE this turn, then record this turn after
        formData.append('history', JSON.stringify(conversationHistory));
        conversationHistory.push({ role: 'user', text: userMsg });

        let full = '';
        try {
          const res = await fetch('/jeremy/stream', { method: 'POST', body: formData });
          const reader = res.body.getReader();
          const decoder = new TextDecoder();
          while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            const chunk = decoder.decode(value, { stream: true });
            chunk.split('\\n').forEach(line => {
              if (line.startsWith('data: ')) {
                try {
                  const d = JSON.parse(line.slice(6));
                  if (d.text) { full += d.text; bubble.innerHTML = marked.parse(full); }
                } catch {}
              }
            });
            document.getElementById('chatMessages').scrollTop = document.getElementById('chatMessages').scrollHeight;
          }
        } catch(e) {
          bubble.textContent = 'Something went wrong. Try again.';
        }

        if (full) conversationHistory.push({ role: 'assistant', text: full });
        bubble.classList.remove('streaming');
        isStreaming = false;
        document.getElementById('sendBtn').disabled = false;
      }
    </script>
    """
    return render_page(body, active="jeremy")


@app.route('/jeremy/stream', methods=['POST'])
def jeremy_stream():
    message = request.form.get('message', '')
    history_json = request.form.get('history', '[]')
    pdf_file = request.files.get('pdf')

    try:
        history = json.loads(history_json)
    except:
        history = []

    # Build the current user message content
    user_content = []

    if pdf_file:
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            pdf_file.save(tmp.name)
            tmp_path = tmp.name
        try:
            with open(tmp_path, 'rb') as f:
                pdf_data = base64.standard_b64encode(f.read()).decode('utf-8')
            user_content.append({
                "type": "document",
                "source": {"type": "base64", "media_type": "application/pdf", "data": pdf_data},
            })
            if not message:
                message = f"I've uploaded {pdf_file.filename}. Please give me the high-yield summary and then start an interactive quiz."
        finally:
            try: os.unlink(tmp_path)
            except: pass

    user_content.append({"type": "text", "text": message or "Hello!"})

    # Build messages array — Claude requires strict user/assistant alternation starting with user
    messages = []
    for h in history[-20:]:
        role = h.get('role', 'user')
        if role in ('model', 'assistant'):
            role = 'assistant'
        else:
            role = 'user'
        text = h.get('text', '').strip()
        if not text:
            continue
        # Skip if same role as previous (prevents consecutive same-role messages)
        if messages and messages[-1]['role'] == role:
            continue
        messages.append({"role": role, "content": text})

    # Must start with user
    while messages and messages[0]['role'] == 'assistant':
        messages.pop(0)

    messages.append({"role": "user", "content": user_content})

    def generate():
        try:
            with claude.messages.stream(
                model="claude-sonnet-4-6",
                max_tokens=8096,
                system=JEREMY_SYSTEM_PROMPT,
                messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    yield f"data: {json.dumps({'text': text})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'text': f'Error: {str(e)}'})}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


# ─── UWORLD REVIEW ───────────────────────────────────────────────────────────

UWORLD_PROMPT = """You are an expert USMLE tutor analyzing a medical student's Anki flashcards that correspond to UWorld questions they got wrong.

The student will provide flashcard content from the AnKing deck. Produce two sections only:

## 🧠 Missed Concepts Summary
Group the cards by topic/system. For each concept write a concise paragraph explaining:
- The core mechanism or principle tested
- Why students commonly miss it on boards
- The key discriminating fact to lock in

## 🎯 High-Yield Board Points
List 15-25 bullet points of the most testable facts. Focus on:
- Classic buzzwords and their associations
- Board favorite relationships (e.g. "X → Y")
- Numbers, thresholds, or timelines worth memorizing
- Common wrong-answer traps and how to eliminate them

Format in clean markdown. Be direct, dense, and high-yield. No filler."""

UWORLD_DRILL_PROMPT = """You are generating USMLE-style interactive drill questions based on AnKing flashcard content.

Output ONLY a valid JSON array — no markdown, no explanation, no text before or after the array.

Format:
[
  {
    "q": "Full clinical vignette (2-4 sentences) ending with a clear question.",
    "choices": ["A. option text", "B. option text", "C. option text", "D. option text"],
    "correct": "B",
    "exp": "One sentence: why the correct answer is right and the key discriminating fact."
  }
]

Rules:
- Generate exactly 10 questions
- Use fresh clinical vignettes — do not copy card text verbatim
- Make distractors plausible (common wrong answers on boards)
- Cover the range of concepts from the cards, not just one topic
- correct field must be a single letter: A, B, C, or D"""


import re as _re

@app.route('/uworld', methods=['GET'])
def uworld_get():
    uid = session.get('uid')
    signed_uid = sign_uid(uid)
    return render_page(_uworld_body(signed_uid=signed_uid), active="uworld")


@app.route('/uworld', methods=['POST'])
def uworld_post():
    # Card content is fetched client-side via AnkiConnect JS and POSTed here
    cards_content = request.form.get('cards_content', '').strip()
    question_ids_raw = request.form.get('question_ids', '').strip()
    card_count = int(request.form.get('card_count', '0') or '0')

    if not cards_content:
        return render_page(_uworld_body("No card content received. Make sure Anki is open and AnkiConnect is running."), active="uworld")

    question_ids = [x.strip() for x in _re.split(r'[\s,]+', question_ids_raw) if x.strip().isdigit()]
    id_count = len(question_ids)

    user_content = (
        f"I missed {id_count} UWorld questions (IDs: {question_ids_raw}).\n"
        f"Here are the {card_count} AnKing cards that correspond to those questions:\n\n"
        f"{cards_content}"
    )

    import concurrent.futures

    def get_analysis():
        return claude.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            messages=[{"role": "user", "content": user_content}],
            system=UWORLD_PROMPT,
        ).content[0].text

    def get_drills():
        return claude.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            messages=[{"role": "user", "content": user_content}],
            system=UWORLD_DRILL_PROMPT,
        ).content[0].text

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_analysis = executor.submit(get_analysis)
        future_drills = executor.submit(get_drills)

        try:
            analysis = future_analysis.result()
        except Exception as e:
            return render_page(_uworld_body(f"Error calling Claude: {str(e)}"), active="uworld")

        drills_json = "[]"
        try:
            raw_drills = future_drills.result().strip()
            if raw_drills.startswith("```"):
                raw_drills = _re.sub(r'^```[^\n]*\n?', '', raw_drills)
                raw_drills = _re.sub(r'\n?```$', '', raw_drills.strip())
            json.loads(raw_drills)  # validate
            drills_json = raw_drills
        except Exception:
            drills_json = "[]"

    signed_uid = request.headers.get('X-Signed-Uid')
    explicit_uid = verify_uid(signed_uid)
    save_to_library(f"UWorld Review ({id_count} questions)", "uworld", {
        "analysis": analysis,
        "drills": json.loads(drills_json)
    }, explicit_uid=explicit_uid)

    import html as html_module
    # JSON-encode both so they can be safely embedded in <script type="application/json"> tags
    safe_analysis = json.dumps(analysis)
    safe_drills = drills_json  # already valid JSON

    body = f"""
    <div class="page" style="max-width:900px">
      <div class="section-header">
        <span class="pill pill-yellow">🎯 UWorld Review</span>
        <div class="section-title">Missed Question Review</div>
        <div class="section-sub">{id_count} question IDs → {card_count} AnKing cards analyzed.</div>
      </div>
      <div style="display:flex;gap:12px;margin-bottom:28px;flex-wrap:wrap">
        <a href="/uworld" class="btn btn-ghost">← Review More</a>
        <button class="btn" style="background:var(--yellow);color:#000;font-weight:700" onclick="downloadPdf()">⬇ Download PDF</button>
      </div>

      <!-- Analysis -->
      <div class="output-box rendered-md" id="analysisContent"></div>
      <script type="application/json" id="analysisData">{safe_analysis}</script>
      <script type="application/json" id="drillData">{safe_drills}</script>

      <!-- Drill Section -->
      <div id="drillSection" style="margin-top:32px;display:none">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:20px;flex-wrap:wrap;gap:12px">
          <div>
            <div style="font-size:22px;font-weight:800;letter-spacing:-0.5px">🔥 Practice Drill</div>
            <div style="font-size:14px;color:var(--text-muted);margin-top:4px">Pick an answer — instant feedback on every question.</div>
          </div>
          <div id="scoreCard" style="background:var(--card);border:1px solid var(--border);border-radius:12px;padding:12px 20px;text-align:center;display:none">
            <div style="font-size:11px;text-transform:uppercase;letter-spacing:1px;color:var(--text-muted);margin-bottom:4px">Score</div>
            <div id="scoreDisplay" style="font-size:24px;font-weight:800;color:var(--yellow)">0 / 0</div>
          </div>
        </div>
        <div id="drillQuestions"></div>
      </div>
    <style>
      /* Analysis markdown */
      .rendered-md {{ white-space: normal; font-size: 15px; line-height: 1.75; max-height: none !important; overflow: visible !important; }}
      .rendered-md h1,.rendered-md h2,.rendered-md h3 {{ margin: 16px 0 8px; }}
      .rendered-md h1 {{ font-size: 22px; }}
      .rendered-md h2 {{ font-size: 19px; color: var(--yellow); }}
      .rendered-md h3 {{ font-size: 16px; color: var(--text-muted); }}
      .rendered-md p {{ margin: 8px 0; }}
      .rendered-md strong {{ color: var(--text); }}
      .rendered-md ol,.rendered-md ul {{ padding-left: 22px; margin: 8px 0; }}
      .rendered-md li {{ margin: 4px 0; }}
      .rendered-md blockquote {{ border-left:3px solid var(--yellow);margin:8px 0;padding:4px 12px;color:var(--text-muted);background:rgba(245,158,11,0.05);border-radius:0 6px 6px 0; }}
      .rendered-md hr {{ border:none;border-top:1px solid var(--border);margin:14px 0; }}
      .rendered-md code {{ background:rgba(255,255,255,0.08);border:1px solid var(--border);border-radius:4px;padding:1px 6px;font-size:13px;font-family:monospace; }}

      /* Drill cards */
      .drill-card {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 24px 28px;
        margin-bottom: 16px;
      }}
      .drill-num {{ font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; color: var(--yellow); margin-bottom: 10px; }}
      .drill-q {{ font-size: 16px; line-height: 1.65; margin-bottom: 18px; color: var(--text); font-weight: 500; }}
      .drill-choices {{ display: flex; flex-direction: column; gap: 8px; }}
      .drill-choice {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 11px 16px;
        text-align: left;
        color: var(--text);
        font-family: 'Inter', sans-serif;
        font-size: 14px;
        cursor: pointer;
        transition: all 0.15s ease;
      }}
      .drill-choice:hover:not(:disabled) {{ background: var(--card-hover); border-color: var(--yellow); color: var(--yellow); }}
      .drill-choice:disabled {{ cursor: default; }}
      .drill-choice.drill-correct {{ background: rgba(16,185,129,0.12) !important; border-color: var(--green) !important; color: var(--green) !important; font-weight: 600; }}
      .drill-choice.drill-wrong {{ background: rgba(239,68,68,0.1) !important; border-color: var(--red) !important; color: var(--red) !important; }}
      .drill-exp {{
        margin-top: 14px;
        padding: 12px 16px;
        border-radius: 10px;
        font-size: 14px;
        line-height: 1.6;
      }}
      .drill-exp-right {{ background: rgba(16,185,129,0.08); border: 1px solid rgba(16,185,129,0.25); color: var(--text); }}
      .drill-exp-wrong {{ background: rgba(239,68,68,0.06); border: 1px solid rgba(239,68,68,0.2); color: var(--text); }}
      .drill-verdict {{ font-weight: 700; margin-right: 8px; }}
    </style>
    """
    if request.headers.get('Accept') == 'text/html-partial':
        return body
    return render_page(body, active="uworld")


def _uworld_body(error=None, signed_uid=""):
    err_html = f'<div class="notice" style="border-color:rgba(239,68,68,0.3);background:rgba(239,68,68,0.08);color:#f87171;" id="errorMsg">{error}</div>' if error else '<div class="notice" style="border-color:rgba(239,68,68,0.3);background:rgba(239,68,68,0.08);color:#f87171;display:none" id="errorMsg"></div>'
    body = f"""
    <div class="page" style="max-width:820px">
      <div class="section-header">
        <span class="pill pill-yellow">🎯 UWorld Review</span>
        <div class="section-title">Missed Question Analyzer</div>
        <div class="section-sub">Paste your UWorld question IDs → your browser pulls the AnKing cards → Claude breaks down what you missed.</div>
      </div>
      {err_html}

      <div class="panel">
        <div class="notice" style="margin-bottom:20px;display:flex;gap:12px;align-items:flex-start">
          <span style="font-size:20px">📋</span>
          <div>
            <strong style="color:var(--text)">How it works:</strong>
            In UWorld, open a previous test → click <strong>Question List</strong> → copy the IDs → paste below.
            Your browser talks directly to Anki on your computer to pull the matching cards.
            <br><br>
            <strong style="color:var(--yellow)">Requirements:</strong> Anki must be open with
            <a href="https://ankiweb.net/shared/info/2055492159" style="color:var(--yellow)" target="_blank">AnkiConnect</a> installed
            and the AnKing deck loaded.<br>
            <span style="color:rgba(255,255,255,0.7)">Using the web app (not localhost)?</span>
            <a href="#cors-help" style="color:var(--yellow);font-weight:600" onclick="document.getElementById('corsHelp').style.display='block';document.getElementById('corsHelp').scrollIntoView({{behavior:'smooth'}})">▶ Click here for required one-time setup</a>
          </div>
        </div>

        <div id="corsHelp" style="display:none;background:rgba(245,158,11,0.06);border:1px solid rgba(245,158,11,0.2);border-radius:10px;padding:16px 18px;margin-bottom:16px;font-size:13px;line-height:1.9">
          <strong style="color:var(--yellow);font-size:14px">One-time AnkiConnect CORS setup (takes ~1 minute)</strong><br>
          This site needs permission to talk to Anki on your computer. Here's exactly how:<br><br>
          <strong style="color:#fff">Step 1.</strong> Open Anki on your computer<br>
          <strong style="color:#fff">Step 2.</strong> In the menu bar click <strong style="color:#fff">Tools → Add-ons</strong><br>
          <strong style="color:#fff">Step 3.</strong> In the list, click <strong style="color:#fff">AnkiConnect</strong> → then click <strong style="color:#fff">Config</strong> (bottom-right button)<br>
          <strong style="color:#fff">Step 4.</strong> Find the line that says <code style="background:rgba(0,0,0,0.4);padding:1px 5px;border-radius:3px">webCorsOriginList</code> and replace the whole line with:<br>
          <div style="margin:8px 0;position:relative">
            <code id="corsSnippet" style="background:rgba(0,0,0,0.4);padding:8px 12px;border-radius:6px;display:block;word-break:break-all">"webCorsOriginList": ["http://localhost", "https://medtools-77dfb.web.app"]</code>
            <button onclick="navigator.clipboard.writeText('&quot;webCorsOriginList&quot;: [&quot;http://localhost&quot;, &quot;https://medtools-77dfb.web.app&quot;]');this.textContent='Copied!';setTimeout(()=>this.textContent='Copy',1500)" style="position:absolute;top:6px;right:8px;background:rgba(245,158,11,0.2);border:1px solid rgba(245,158,11,0.4);color:var(--yellow);border-radius:4px;padding:2px 8px;cursor:pointer;font-size:11px">Copy</button>
          </div>
          <strong style="color:#fff">Step 5.</strong> Click <strong style="color:#fff">OK</strong> to save<br>
          <strong style="color:#fff">Step 6.</strong> Fully <strong style="color:#fff">quit and reopen Anki</strong> (just closing the window isn't enough — use File → Quit or Anki → Quit Anki)<br><br>
          <span style="color:rgba(255,255,255,0.5)">You only need to do this once. After restarting Anki, come back and try again.</span>
        </div>

        <!-- Hidden form that gets submitted after JS fetches the cards -->
        <form method="POST" action="/uworld" id="uworldForm">
          <input type="hidden" name="cards_content" id="cardsContent">
          <input type="hidden" name="question_ids" id="questionIdsHidden">
          <input type="hidden" name="card_count" id="cardCount">
          <div class="form-group">
            <label>UWorld Question IDs <span style="color:var(--text-dim);font-weight:400">(comma-separated)</span></label>
            <textarea id="questionIds" rows="4"
              placeholder="15760, 16544, 16215, 13798, 20191, 14378, 4919, 21333, 16716 ..."></textarea>
          </div>
          <button type="button" class="btn" id="analyzeBtn" onclick="fetchAndSubmit()" style="background:var(--yellow);color:#000;font-weight:700;width:100%">
            <span id="analyzeBtnText">🎯 Analyze with AnKing Cards</span>
          </button>
          <div style="margin-top:12px;display:none" id="analyzingBar">
            <div class="loading-bar" style="width:100%;background:linear-gradient(90deg,var(--yellow),var(--orange))"></div>
            <div style="text-align:center;font-size:12px;color:var(--text-muted);margin-top:8px" id="statusMsg">Connecting to Anki...</div>
          </div>
        </form>
      </div>

      <div class="panel" style="margin-top:20px">
        <div class="panel-title">📦 What You'll Get</div>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-top:4px">
          <div style="background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:16px">
            <div style="font-size:24px;margin-bottom:8px">🧠</div>
            <div style="font-weight:600;font-size:14px;color:var(--yellow);margin-bottom:6px">Missed Concepts</div>
            <div style="font-size:13px;color:var(--text-muted)">Grouped by topic — what the card is testing, why it trips people up, and the key discriminating fact.</div>
          </div>
          <div style="background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:16px">
            <div style="font-size:24px;margin-bottom:8px">🎯</div>
            <div style="font-weight:600;font-size:14px;color:var(--yellow);margin-bottom:6px">High-Yield Points</div>
            <div style="font-size:13px;color:var(--text-muted)">Dense bullet-point facts, buzzwords, and board associations pulled from your specific missed cards.</div>
          </div>
          <div style="background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:16px">
            <div style="font-size:24px;margin-bottom:8px">🔥</div>
            <div style="font-weight:600;font-size:14px;color:var(--yellow);margin-bottom:6px">Drill Questions</div>
            <div style="font-size:13px;color:var(--text-muted)">8-10 fresh NBME-style questions on the exact same concepts so you can re-test yourself.</div>
          </div>
        </div>
      </div>
    </div>

    <script>
      function stripHtml(html) {{
        const tmp = document.createElement('div');
        tmp.innerHTML = html;
        return (tmp.textContent || tmp.innerText || '').replace(/\\s+/g, ' ').trim();
      }}

      function setStatus(msg) {{
        document.getElementById('statusMsg').textContent = msg;
        document.getElementById('analyzeBtnText').textContent = msg;
      }}

      function showError(msg) {{
        const el = document.getElementById('errorMsg');
        el.textContent = msg;
        el.style.display = 'block';
        document.getElementById('analyzeBtn').disabled = false;
        document.getElementById('analyzeBtnText').textContent = '🎯 Analyze with AnKing Cards';
        document.getElementById('analyzingBar').style.display = 'none';
      }}

      async function ankiConnect(action, params) {{
        const resp = await fetch('http://localhost:8765', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{action, version: 6, params}})
        }});
        const data = await resp.json();
        if (data.error) throw new Error(data.error);
        return data.result;
      }}

      async function fetchAndSubmit() {{
        if (!checkUsageLimit()) return;
        const idsRaw = document.getElementById('questionIds').value.trim();
        const ids = idsRaw.split(/[\\s,]+/).filter(id => /^\\d+$/.test(id));
        if (!ids.length) {{ showError('No valid question IDs found.'); return; }}

        document.getElementById('analyzeBtn').disabled = true;
        document.getElementById('analyzingBar').style.display = 'block';
        document.getElementById('errorMsg').style.display = 'none';

        // Step 1: find matching notes
        setStatus('Connecting to Anki...');
        let noteIds;
        try {{
          const query = ids.map(id => 'tag:*' + id + '*').join(' OR ');
          noteIds = await ankiConnect('findNotes', {{query}});
        }} catch(e) {{
          showError('Could not connect to Anki: ' + e.message + '. Make sure Anki is open and AnkiConnect is installed. If using the web app, see the CORS setup instructions above.');
          return;
        }}

        if (!noteIds || !noteIds.length) {{
          showError('No AnKing cards found for those IDs. Make sure you have the AnKing deck installed.');
          return;
        }}

        // Step 2: get note content
        setStatus('Found ' + noteIds.length + ' cards — fetching content...');
        let notes;
        try {{
          notes = await ankiConnect('notesInfo', {{notes: noteIds.slice(0, 120)}});
        }} catch(e) {{
          showError('Error fetching card content: ' + e.message);
          return;
        }}

        // Step 3: extract text fields
        const priority = ['Text', 'Front', 'Back', 'Back Extra', 'Extra', 'Lecture Notes'];
        const cardTexts = notes.map(note => {{
          const fields = note.fields || {{}};
          const allKeys = [...priority, ...Object.keys(fields).filter(k => !priority.includes(k))];
          const seen = new Set();
          const parts = [];
          allKeys.forEach(fname => {{
            if (fields[fname] && !seen.has(fname)) {{
              seen.add(fname);
              const val = stripHtml(fields[fname].value || '');
              if (val && val.length > 4) parts.push('[' + fname + ']: ' + val);
            }}
          }});
          return parts.join('\\n');
        }}).filter(t => t);

        // Step 4: populate hidden fields and submit
        setStatus('Sending ' + cardTexts.length + ' cards to Claude... (this may take up to a minute)');
        document.getElementById('cardsContent').value = cardTexts.join('\\n\\n---\\n\\n');
        document.getElementById('questionIdsHidden').value = ids.join(', ');
        document.getElementById('cardCount').value = cardTexts.length;
        
        const formData = new FormData(document.getElementById('uworldForm'));
        const headers = {{'Accept': 'text/html-partial', 'X-Signed-Uid': '__SIGNED_UID__'}};
        if (window.firebase && firebase.auth && firebase.auth().currentUser) {{
          try {{
            const token = await firebase.auth().currentUser.getIdToken();
            headers['Authorization'] = 'Bearer ' + token;
          }} catch(e) {{}}
        }}

        try {{
          const res = await fetch('https://medtools-vneeiy3k7q-uc.a.run.app/uworld', {{
            method: 'POST',
            headers: headers,
            body: formData
          }});
          if (res.ok) {{
            incrementUsage();
            const html = await res.text();
            document.querySelector('.page').outerHTML = html;
            
            if (window.marked && document.getElementById('analysisContent')) {{
              marked.setOptions({{ breaks: true }});
              const analysisEl = document.getElementById('analysisContent');
              const analysisText = JSON.parse(document.getElementById('analysisData').textContent);
              analysisEl.innerHTML = marked.parse(analysisText);

              let drills = [];
              try {{ drills = JSON.parse(document.getElementById('drillData').textContent); }} catch(e) {{}}

              let answered = 0;
              let correct = 0;

              if (drills.length > 0) {{
                document.getElementById('drillSection').style.display = 'block';
                document.getElementById('scoreCard').style.display = 'block';
                const container = document.getElementById('drillQuestions');

                drills.forEach((q, i) => {{
                  const card = document.createElement('div');
                  card.className = 'drill-card';
                  card.id = 'dq-' + i;

                  const choicesHtml = q.choices.map(c => {{
                    const letter = c.charAt(0);
                    return `<button class="drill-choice" onclick="answerQ(${{i}}, '${{letter}}')" data-letter="${{letter}}">${{c}}</button>`;
                  }}).join('');

                  card.innerHTML = `
                    <div class="drill-num">Question ${{i+1}} of ${{drills.length}}</div>
                    <div class="drill-q">${{q.q}}</div>
                    <div class="drill-choices" id="choices-${{i}}">${{choicesHtml}}</div>
                    <div class="drill-exp" id="exp-${{i}}" style="display:none"></div>
                  `;
                  container.appendChild(card);
                }});
                
                window.drills = drills;
                window.answered = answered;
                window.correct = correct;
              }}
            }}
            window.history.pushState({{}}, '', '/uworld');
          }} else {{
            showError('Error generating: HTTP ' + res.status);
          }}
        }} catch (err) {{
          showError('Network Error: ' + err);
        }}
      }}

      function answerQ(i, picked) {{
        const q = window.drills[i];
        if (document.getElementById('exp-' + i).style.display !== 'none') return; // already answered

        const btns = document.querySelectorAll('#choices-' + i + ' .drill-choice');
        btns.forEach(btn => {{
          btn.disabled = true;
          const l = btn.dataset.letter;
          if (l === q.correct) btn.classList.add('drill-correct');
          else if (l === picked && picked !== q.correct) btn.classList.add('drill-wrong');
        }});

        const isRight = picked === q.correct;
        if (isRight) window.correct++;
        window.answered++;

        const expEl = document.getElementById('exp-' + i);
        expEl.style.display = 'block';
        expEl.innerHTML = `
          <span class="drill-verdict">${{isRight ? '✅ Correct!' : '❌ Incorrect — Answer: ' + q.correct}}</span>
          ${{q.exp}}
        `;
        expEl.className = 'drill-exp ' + (isRight ? 'drill-exp-right' : 'drill-exp-wrong');

        document.getElementById('scoreDisplay').textContent = window.correct + ' / ' + window.answered;

        if (window.answered === window.drills.length) {{
          const pct = Math.round(window.correct / window.drills.length * 100);
          document.getElementById('scoreDisplay').innerHTML = window.correct + ' / ' + window.drills.length +
            '<div style="font-size:13px;color:var(--text-muted);font-weight:400;margin-top:2px">' + pct + '%</div>';
        }}
      }}

      function downloadPdf() {{
        const analysisEl = document.getElementById('analysisContent');
        if(!analysisEl) return;
        const clone = analysisEl.cloneNode(true);
        clone.className = '';
        clone.style.cssText = 'background:#fff;color:#111;padding:20px;font-family:Georgia,serif;font-size:14px;line-height:1.7;max-height:none;overflow:visible;width:100%';
        clone.querySelectorAll('*').forEach(c => {{ c.style.maxHeight = 'none'; c.style.overflow = 'visible'; }});
        html2pdf().set({{
          margin: [15, 20], filename: 'uworld_review.pdf',
          image: {{ type: 'jpeg', quality: 0.98 }},
          html2canvas: {{ scale: 2, backgroundColor: '#ffffff' }},
          jsPDF: {{ unit: 'mm', format: 'a4', orientation: 'portrait' }}
        }}).from(clone).save();
      }}
    </script>
    """
    return body.replace('__SIGNED_UID__', signed_uid)

@app.route('/my-library', methods=['GET'])
def my_library_get():
    uid = session.get('uid')
    if not uid:
        return render_page('<div class="page"><div class="notice" style="text-align:center;padding:40px;">Please log in to view your library.</div></div>', active="library")
        
    try:
        docs = db.collection('users').document(uid).collection('library').order_by('created_at', direction=firestore.Query.DESCENDING).limit(50).stream()
        items = []
        for doc in docs:
            data = doc.to_dict()
            items.append({
                'id': doc.id,
                'title': data.get('title', 'Untitled'),
                'type': data.get('type'),
                'created_at': data.get('created_at')
            })
    except Exception as e:
        items = []
        print(f"Error fetching library: {e}")
        
    import html as html_module
    items_html = ""
    if not items:
        items_html = '<div class="notice" style="text-align:center;padding:40px;">Your library is empty. Generate some Anki decks or practice tests to save them here!</div>'
    else:
        for item in items:
            # handle timezone-aware datetime from firestore
            date_str = item['created_at'].strftime('%b %d, %Y') if getattr(item['created_at'], 'strftime', None) else 'Unknown Date'
            icon = '🃏' if item['type'] == 'anki' else '📝' if item['type'] == 'practice_test' else '🎯'
            items_html += f"""
            <div class="panel" style="margin-bottom:12px; display:flex; align-items:center; justify-content:space-between; padding:16px;">
                <div>
                    <div style="font-weight:600; font-size:16px;">{icon} {html_module.escape(item['title'])}</div>
                    <div style="font-size:12px; color:var(--text-muted); margin-top:4px;">{date_str}</div>
                </div>
                <a href="/my-library/{item['id']}" class="btn btn-blue" style="padding:6px 12px; font-size:13px; text-decoration:none;">View</a>
            </div>
            """
            
    body = f"""
    <div class="page" style="max-width:820px">
      <div class="section-header">
        <span class="pill" style="background:var(--blue-glow);color:var(--blue);border-color:rgba(79,158,255,0.3)">📚 My Library</span>
        <div class="section-title">Your Saved Generations</div>
        <div class="section-sub">Access your previously generated Anki decks, practice tests, and UWorld reviews across all your devices.</div>
      </div>
      {items_html}
    </div>
    """
    return render_page(body, active="library")

@app.route('/my-library/<item_id>', methods=['GET'])
def my_library_item_get(item_id):
    uid = session.get('uid')
    if not uid:
        return "Unauthorized", 401
    
    try:
        doc = db.collection('users').document(uid).collection('library').document(item_id).get()
        if not doc.exists:
            return "Item not found", 404
        data = doc.to_dict()
    except Exception as e:
        return f"Error: {e}", 500

    item_type = data.get('type')
    title = data.get('title', 'Untitled')
    content_data = data.get('data')

    if item_type == 'anki':
        import html as html_module
        body = f"""
        <div class="page" style="max-width:820px">
          <a href="/my-library" style="color:var(--blue);text-decoration:none;margin-bottom:20px;display:inline-block">← Back to Library</a>
          <div class="section-header">
            <span class="pill pill-purple">🃏 Anki Deck</span>
            <div class="section-title">{html_module.escape(title)}</div>
          </div>
          <div class="panel">
            <form method="POST" action="/anki">
                <input type="hidden" name="deck_name" value="{html_module.escape(title)}">
                <div class="form-group">
                    <label>Cards</label>
                    <textarea name="cards_text" rows="14">{html_module.escape(content_data)}</textarea>
                </div>
                <button type="submit" class="btn btn-purple">⬇ Download Anki Deck</button>
            </form>
          </div>
        </div>
        """
        return render_page(body, active="library")
        
    elif item_type == 'practice_test':
        import html as html_module
        import json
        result_blocks = ""
        for i, item in enumerate(content_data):
            fname = item.get('filename', '')
            content = item.get('content', '')
            safe_content = html_module.escape(content)
            
            is_json = False
            try:
                json.loads(content)
                is_json = True
            except:
                pass
                
            if is_json:
                result_blocks += f"""
                <div class="result-block" id="result-{i}">
                  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px">
                    <div class="result-label">📄 {html_module.escape(fname)}</div>
                    <div style="font-size:14px;color:var(--text-muted);font-weight:600" id="scoreDisplay-{i}">0 / 0</div>
                    <button class="btn btn-green" style="font-size:13px;padding:8px 14px" onclick="downloadPdf({i}, '{html_module.escape(fname)}')">
                      ⬇ Download PDF
                    </button>
                  </div>
                  <div class="output-box" id="interactive-{i}" style="padding:0;background:none;border:none"></div>
                  <div id="content-{i}" style="display:none;" data-raw="{safe_content}" data-type="json"></div>
                </div>"""
            else:
                result_blocks += f"""
                <div class="result-block" id="result-{i}">
                  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px">
                    <div class="result-label">📄 {html_module.escape(fname)}</div>
                    <button class="btn btn-green" style="font-size:13px;padding:8px 14px" onclick="downloadPdf({i}, '{html_module.escape(fname)}')">
                      ⬇ Download PDF
                    </button>
                  </div>
                  <div class="output-box rendered-md" id="content-{i}" data-raw="{safe_content}" data-type="markdown"></div>
                </div>"""

        body = f"""
        <div class="page" style="max-width:900px">
          <a href="/my-library" style="color:var(--blue);text-decoration:none;margin-bottom:20px;display:inline-block">← Back to Library</a>
          <div class="section-header">
            <span class="pill pill-green">📝 Practice Tests</span>
            <div class="section-title">{html_module.escape(title)}</div>
          </div>
          {result_blocks}
        </div>
        {_practice_test_js()}
        """
        return render_page(body, active="library")

    elif item_type == 'uworld':
        import html as html_module
        analysis = content_data.get('analysis', '')
        drills = content_data.get('drills', [])
        
        safe_analysis = json.dumps(analysis)
        safe_drills = json.dumps(drills)
        
        body = f"""
        <div class="page" style="max-width:1100px; padding-bottom:80px">
          <a href="/my-library" style="color:var(--blue);text-decoration:none;margin-bottom:20px;display:inline-block">← Back to Library</a>
          <div class="section-header">
            <span class="pill pill-yellow" style="background:rgba(245,158,11,0.15);color:var(--yellow);border-color:rgba(245,158,11,0.3)">🎯 UWorld Review</span>
            <div class="section-title">{html_module.escape(title)}</div>
          </div>
          
          <div style="display:grid; grid-template-columns: 1fr 1fr; gap: 24px; align-items: start">
            <div class="panel">
              <div class="panel-title">🧠 Concept Analysis</div>
              <div class="output-box" id="analysisBox" style="font-size:15px;line-height:1.6"></div>
            </div>
            
            <div class="panel">
              <div class="panel-title" style="display:flex;justify-content:space-between;align-items:center">
                <span>🔥 Active Recall Drills</span>
                <span style="font-size:13px;font-weight:500;color:var(--text-muted)" id="scoreTrack">Score: 0/0</span>
              </div>
              <div id="drillBox" style="margin-top:16px"></div>
            </div>
          </div>
        </div>

        <script id="analysisData" type="application/json">{safe_analysis}</script>
        <script id="drillData" type="application/json">{safe_drills}</script>
        <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
        <script>
            // We reuse the uworld client logic for rendering
            const analysisStr = JSON.parse(document.getElementById('analysisData').textContent);
            document.getElementById('analysisBox').innerHTML = marked.parse(analysisStr);
            
            const drills = JSON.parse(document.getElementById('drillData').textContent);
            let state = {{ correct: 0, total: 0 }};
            const drillBox = document.getElementById('drillBox');
            const scoreTrack = document.getElementById('scoreTrack');
            
            if (!drills || drills.length === 0) {{
                drillBox.innerHTML = '<div style="color:var(--text-muted);font-style:italic">No drills available for this review.</div>';
            }} else {{
                drills.forEach((d, i) => {{
                  const block = document.createElement('div');
                  block.className = 'drill-q';
                  block.style.background = 'var(--surface)';
                  block.style.border = '1px solid var(--border)';
                  block.style.borderRadius = '10px';
                  block.style.padding = '20px';
                  block.style.marginBottom = '16px';
                  
                  let html = `<div style="font-weight:600;margin-bottom:12px;font-size:15px">${{i+1}}. ${{d.question}}</div>`;
                  html += `<div style="display:flex;flex-direction:column;gap:8px" id="opts-${{i}}">`;
                  d.options.forEach((opt, j) => {{
                    const isCorrect = (opt.charAt(0) === d.correct_answer);
                    html += `
                      <button class="opt-btn" onclick="handleAnswer(${{i}}, ${{isCorrect}}, this)"
                        style="text-align:left;background:rgba(255,255,255,0.03);border:1px solid var(--border);
                        padding:12px 16px;border-radius:8px;color:var(--text);cursor:pointer;font-family:inherit;font-size:14px;transition:all 0.15s">
                        ${{opt}}
                      </button>
                    `;
                  }});
                  html += `</div>`;
                  html += `<div id="exp-${{i}}" style="display:none;margin-top:16px;padding-top:16px;border-top:1px solid var(--border);font-size:14px">
                    <strong style="color:var(--green)">Explanation:</strong> ${{d.explanation}}
                  </div>`;
                  
                  block.innerHTML = html;
                  drillBox.appendChild(block);
                }});
            }}
            
            window.handleAnswer = function(qIndex, isCorrect, btnEl) {{
              const container = document.getElementById(`opts-${{qIndex}}`);
              const buttons = container.querySelectorAll('button');
              buttons.forEach(b => {{ b.disabled = true; b.style.opacity = '0.6'; b.style.cursor = 'default'; }});
              
              if (isCorrect) {{
                btnEl.style.background = 'rgba(16,185,129,0.15)';
                btnEl.style.borderColor = 'var(--green)';
                btnEl.style.opacity = '1';
                state.correct++;
              }} else {{
                btnEl.style.background = 'rgba(239,68,68,0.15)';
                btnEl.style.borderColor = 'var(--red)';
                btnEl.style.opacity = '1';
                
                buttons.forEach(b => {{
                  if (b.onclick.toString().includes('true')) {{
                    b.style.background = 'rgba(16,185,129,0.15)';
                    b.style.borderColor = 'var(--green)';
                    b.style.opacity = '1';
                  }}
                }});
              }}
              state.total++;
              scoreTrack.textContent = `Score: ${{state.correct}}/${{state.total}}`;
              document.getElementById(`exp-${{qIndex}}`).style.display = 'block';
            }};
        </script>
        """
        return render_page(body, active="library")
    
    return "Unknown item type", 400

# ─── RUN ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
