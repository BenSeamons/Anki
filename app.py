from flask import Flask, request, render_template_string, send_file, jsonify
import sqlite3, zipfile, tempfile
import io
import genanki
import os
import hashlib
import tempfile
import pandas as pd
from PyPDF2 import PdfReader
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
import numpy as np
from just_PDFs import generate_practice_test_return_text

app = Flask(__name__)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")


# Simple parser to convert pasted text into cards list
def parse_cards(text):
    cards = []
    lines = text.strip().split('\n')
    for line in lines:
        # Allow cloze lines without a tab; otherwise require front/back
        if '\t' in line:
            front, back = line.split('\t', 1)
        elif '{{c' in line:
            front, back = line, ''
        else:
            continue

        front = front.strip()
        back = back.strip()

        # Skip completely empty lines
        if not front:
            continue

        # Detect cloze by {{c1::...}} style in front
        is_cloze = '{{c' in front

        cards.append({
            'front': front,
            'back': back,
            'is_cloze': is_cloze
        })
    return cards


@app.route('/practice-tests', methods=['GET', 'POST'])
def practice_tests():
    if request.method == 'POST':
       # Get files from both inputs
        folder_files = request.files.getlist('folder_pdfs')
        individual_files = request.files.getlist('individual_pdfs')

        all_files = []

        if folder_files:
            all_files.extend(folder_files)
        if individual_files:
            all_files.extend(individual_files)

        if not all_files or all_files[0].filename == '':
            return "No PDFs selected", 400

        results = []
        with tempfile.TemporaryDirectory() as tmpdir:
            for file in all_files:
                if not file.filename.lower().endswith('.pdf'):
                    continue  # skip non-PDFs

                pdf_path = os.path.join(tmpdir, file.filename)
                os.makedirs(os.path.dirname(pdf_path), exist_ok=True)  # Ensure directory exists
                file.save(pdf_path)

                # Call the wrapper function that returns generated text
                test_text = generate_practice_test_return_text(pdf_path)
                if not test_text:
                    test_text = "(No content generated or error occurred.)"
                results.append((file.filename, test_text))

        # Return results HTML even if results is empty
        html = "<h1>Generated Practice Tests</h1>"
        if not results:
            html += "<p>No valid PDF files were uploaded or processed.</p>"
        else:
            for fname, content in results:
                html += f"<h2>{fname}</h2><pre>{content}</pre><hr>"
        return html


def load_apkg_to_df(apkg_file):
    """
    Takes an uploaded .apkg file (Werkzeug FileStorage from Flask),
    extracts collection.anki2, reads notes table, and returns a DataFrame
    with noteId, Front, Back.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        apkg_path = os.path.join(tmpdir, apkg_file.filename)
        apkg_file.save(apkg_path)

        # Unzip .apkg
        with zipfile.ZipFile(apkg_path, "r") as z:
            z.extractall(tmpdir)

        # Find collection.anki2
        db_path = os.path.join(tmpdir, "collection.anki2")
        if not os.path.exists(db_path):
            raise ValueError("No collection.anki2 found in apkg")

        # Read SQLite database
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT id, flds FROM notes")
        rows = cur.fetchall()
        conn.close()

        # Build dataframe
        data = []
        for note_id, flds in rows:
            fields = flds.split("\x1f")  # Anki separator
            front = fields[0] if len(fields) > 0 else ""
            back = fields[1] if len(fields) > 1 else ""
            data.append({"noteId": note_id, "Front": front, "Back": back})

        return pd.DataFrame(data)


@app.route("/decksurf", methods=["POST"])
def decksurf():
    try:
        # --- 1. Validate deck upload ---
        if "deck_file" not in request.files or request.files["deck_file"].filename == "":
            return jsonify({"error": "No deck file uploaded"}), 400

        deck_file = request.files["deck_file"]

        # --- 2. Load deck ---
        if deck_file.filename.endswith(".apkg"):
            deck_df = load_apkg_to_df(deck_file)
        elif deck_file.filename.endswith(".txt"):
            deck_df = pd.read_csv(deck_file, sep="\t", engine="python", on_bad_lines="skip")
        else:
            deck_df = pd.read_csv(deck_file, engine="python", on_bad_lines="skip")

        if "noteId" not in deck_df.columns:
            deck_df["noteId"] = range(len(deck_df))

        required_cols = {"noteId", "Front", "Back"}
        if not required_cols.issubset(set(deck_df.columns)):
            return jsonify({"error": "Deck file must have noteId, Front, Back columns"}), 400

        # --- 3. Convert to card dicts ---
        deck_cards = []
        for _, row in deck_df.iterrows():
            card_text = f"{row.get('Front','')} {row.get('Back','')}"
            deck_cards.append({
                "note_id": int(row.get("noteId")),
                "front": str(row.get("Front","")),
                "back": str(row.get("Back","")),
                "tags": row.get("Tags",""),
                "text": card_text
            })

        # --- 4. Parse learning objectives ---
        los = []
        if "los_file" in request.files and request.files["los_file"].filename:
            f = request.files["los_file"]
            fname = f.filename.lower()
            if fname.endswith(".pdf"):
                reader = PdfReader(f)
                raw_text = "\n".join(page.extract_text() or "" for page in reader.pages)
                los = [line.strip() for line in raw_text.split("\n") if 20 < len(line.strip()) < 250]
            elif fname.endswith(".csv"):
                df = pd.read_csv(f)
                col = df.columns[0]
                los = df[col].dropna().astype(str).tolist()
            elif fname.endswith(".txt"):
                los = [line.strip() for line in f.read().decode("utf-8").splitlines() if line.strip()]
        elif "text" in request.form:
            los = [line.strip() for line in request.form["text"].splitlines() if line.strip()]

        if not los:
            return jsonify({"error": "No learning objectives found"}), 400

        # --- 5. Embed cards ---
        card_texts = [c["text"] for c in deck_cards]
        card_embeddings = embed_model.encode(card_texts, normalize_embeddings=True)

        results = []
        alpha = float(request.form.get("alpha", 0.85))  # default weight for embeddings

        # --- 6. Match each LO ---
        for lo in los:
            lo_vec = embed_model.encode([lo], normalize_embeddings=True)[0]
            emb_scores = card_embeddings @ lo_vec
            fz_scores = np.array([
                0.5 * fuzz.token_set_ratio(lo, c["text"]) / 100.0 +
                0.3 * fuzz.partial_ratio(lo, c["text"]) / 100.0 +
                0.2 * fuzz.token_sort_ratio(lo, c["text"]) / 100.0
                for c in deck_cards
            ])
            combo_scores = alpha * emb_scores + (1 - alpha) * fz_scores
            idxs = np.argsort(-combo_scores)[:3]

            matches = []
            note_ids = []
            for i in idxs:
                c = deck_cards[i]
                matches.append({
                    "note_id": c["note_id"],
                    "preview": (c["front"][:100] + " ...") if len(c["front"]) > 100 else c["front"],
                    "score": float(combo_scores[i])
                })
                note_ids.append(c["note_id"])

            results.append({
                "learning_objective": lo,
                "matches": matches,
                "search_query": " OR ".join([f"nid:{nid}" for nid in note_ids])
            })

        # --- 7. Build new deck with unsuspended matches ---
        deck_name = request.form.get("deck_name", "Uploaded Deck")
        deck_id = int(hashlib.sha1(deck_name.encode('utf-8')).hexdigest()[:8], 16)

        gen_deck = genanki.Deck(deck_id, deck_name)
        basic_model = genanki.Model(
            1607392319,
            'Basic Model',
            fields=[{'name': 'Front'}, {'name': 'Back'}],
            templates=[{
                'name': 'Card 1',
                'qfmt': '{{Front}}',
                'afmt': '{{Front}}<br>{{Back}}'
            }]
        )

        matched_ids = {m["note_id"] for r in results for m in r["matches"]}

        for c in deck_cards:
            note = genanki.Note(
                model=basic_model,
                fields=[c["front"], c["back"]],
                tags=["unsuspended"] if c["note_id"] in matched_ids else ["suspended"]
            )
            gen_deck.add_note(note)

        package = genanki.Package(gen_deck)
        deck_data = io.BytesIO()
        package.write_to_file(deck_data)
        deck_data.seek(0)

        return send_file(
            deck_data,
            as_attachment=True,
            download_name=f"{deck_name}_unsuspended.apkg",
            mimetype="application/octet-stream"
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return render_template_string('''
    <html>
      <head>
        <title>Study Tools Hub</title>
      </head>
      <body>
        <h1>Welcome to Your Study Tools Hub</h1>
        <p>Select a tool below:</p>
        <ul>
          <li><a href="/anki-generator">Anki Deck Generator (Paste Cards)</a></li>
        </ul>
      </body>
    </html>
    ''')
@app.route('/anki-generator', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        cards_text = request.form.get('cards_text', '')
        deck_name = request.form.get('deck_name', 'Custom Deck').strip()
        if not deck_name:
            deck_name = 'Custom Deck'
        cards = parse_cards(cards_text)

        # Generate deck_id from deck_name (using hashlib)
        deck_id = int(hashlib.sha1(deck_name.encode('utf-8')).hexdigest()[:8], 16)

        deck = genanki.Deck(
            deck_id,  # unique deck ID (change if needed)
            deck_name
        )

        # Basic model
        basic_model = genanki.Model(
            1607392319,
            'Basic Model',
            fields=[
                {'name': 'Front'},
                {'name': 'Back'},
            ],
            templates=[
                {
                    'name': 'Card 1',
                    'qfmt': '{{Front}}',
                    'afmt': '{{Front}}<br>{{Back}}',
                },
            ])

        # Cloze model
        cloze_model = genanki.Model(
            1091735104,
            'Cloze Model',
            fields=[
                {'name': 'Text'},
            ],
            templates=[
                {
                    'name': 'Cloze Card',
                    'qfmt': '{{cloze:Text}}',
                    'afmt': '{{cloze:Text}}',
                },
            ],
            model_type=genanki.Model.CLOZE,
        )

        for card in cards:
            if card['is_cloze']:
                note = genanki.Note(
                    model=cloze_model,
                    fields=[card['front']],
                    tags=['generated']
                )
            else:
                note = genanki.Note(
                    model=basic_model,
                    fields=[card['front'], card['back']],
                    tags=['generated']
                )
            deck.add_note(note)

        package = genanki.Package(deck)
        deck_data = io.BytesIO()
        package.write_to_file(deck_data)
        deck_data.seek(0)

        return send_file(
            deck_data,
            as_attachment=True,
            download_name='custom_deck.apkg',
            mimetype='application/octet-stream'
        )

    # GET request returns HTML form
    return render_template_string('''
    <html>
    <head>
      <title>Study Tools Combined</title>
      <style>
        body {
          font-family: Arial, sans-serif;
          margin: 20px;
        }
        .container {
          display: flex;
          gap: 40px;
        }
        .half {
              flex: 1;
              min-width: 300px;
              border: 1px solid #ccc;
              padding: 20px;
              box-sizing: border-box;
              height: 90vh;
              overflow-y: auto;
        }
        textarea {
          width: 100%;
          box-sizing: border-box;
        }
        input[type="text"] {
          width: 100%;
          box-sizing: border-box;
          padding: 6px;
          margin-bottom: 10px;
        }
        button {
          padding: 10px 15px;
          font-size: 1em;
          cursor: pointer;
        }
        pre {
          background: #f0f0f0;
          padding: 10px;
          border-radius: 5px;
          font-family: monospace;
          white-space: pre-wrap;
        }
      </style>
    </head>
    <body>
      <h1>Study Tools Hub</h1>
      <div class="container">
        <!-- Anki Deck Generator -->
        <div class="half">
          <h2>Anki Deck Generator</h2>
          <h3>How to generate good cards for this app</h3>
          <p>To get the best cards, you can ask your AI or note-taking tool using a prompt like this:</p>
          <pre>
    Please generate Anki cards for me from this (pdf/powerpoint) in both the Basic and Cloze format.
    Please give them to me as plain text with the Basic cards front and back separated by a tab.
    Focus on giving me high yield flashcards that would help a first year med student pass their board exams.
          </pre>
          <hr>
          <p>Or, if you want to generate cards automatically, you can try <a href="https://chatgpt.com/g/g-683f7d2e85348191b074c1875dc79ca7-anki-card-generator" target="_blank" rel="noopener noreferrer">this GPT-powered Anki card generator</a> — <em>use at your own risk!</em></p>
          <hr>
          <form method="POST" action="/anki-generator">
            <label for="deck_name">Deck Name:</label>
            <input type="text" id="deck_name" name="deck_name" placeholder="Enter deck name" required>
            <textarea name="cards_text" rows="15" placeholder="Front [tab] Back"></textarea><br>
            <button type="submit">Generate Anki Deck</button>
          </form>
          <p>Use <code>{&#123;&#123;c1::cloze deletion&#125;&#125;}</code> syntax for cloze cards in the Front field. Back can be empty for cloze cards.</p>
        </div>

        <!-- Practice Test Generator -->
        <div class="half">
          <h2>Practice Test Generator</h2>
          <form method="POST" action="/practice-tests" enctype="multipart/form-data">
            <label for="pdfs">Upload a Folder of PDFs (Chrome, Edge, Opera support folder upload):</label><br>
            <p><strong>Note:</strong> Processing your PDFs can take up to <em>10 minutes</em>. Please be patient after submitting the form.</p>
            <input type="file" id="pdfs" name="pdfs" multiple webkitdirectory accept="application/pdf"><br><br>
            <label for="individual_pdfs">Or upload Individual PDF files:</label><br>
            <input type="file" id="individual_pdfs" name="individual_pdfs" multiple accept="application/pdf"><br><br>

            <button type="submit">Generate Practice Tests</button>
          </form>
          <p><small>Note: Folder upload works only in Chrome, Edge, and Opera. Firefox does not support folder upload.</small></p>
        </div>
      </div>
    </body>
    </html>
       <!-- DeckSurfer Mapper -->
        <div class="half">
          <h2>DeckSurfer Mapper</h2>
          <form id="decksurf-form" enctype="multipart/form-data">
            <label>Upload Deck CSV:</label><br>
            <input type="file" id="deck_file" name="deck_file" accept=".apkg,.csv,.txt" required>
        
            <label>Upload Lecture Objectives (PDF/CSV/TXT):</label><br>
            <input type="file" id="los_file" name="los_file" accept=".pdf,.csv,.txt"><br><br>
        
            <label>Or paste objectives:</label><br>
            <textarea id="los_text" name="text" rows="8" placeholder="One objective per line"></textarea><br><br>

            <label for="alpha">Semantic Weight (α):</label><br>
            <input type="range" id="alpha" name="alpha" min="0" max="1" step="0.05" value="0.85"
                   oninput="alphaValue.value = this.value">
            <output id="alphaValue">0.85</output><br><br>

        
            <button type="submit">Run DeckSurf</button>
          </form>
        
          <div id="decksurf-results" style="margin-top:20px;"></div>
          <button id="copy-queries-btn" style="display:none;margin-top:10px;">Copy All Search Queries</button>
        </div>
        
        <script>
        document.getElementById("decksurf-form").addEventListener("submit", async (e) => {
          e.preventDefault();
        
          const formData = new FormData();
          const deckFile = document.getElementById("deck_file").files[0];
          const losFile = document.getElementById("los_file").files[0];
          const losText = document.getElementById("los_text").value.trim();
        
          if (!deckFile) {
            alert("Please upload a deck CSV file");
            return;
          }
          formData.append("deck_file", deckFile);
          if (losFile) formData.append("los_file", losFile);
          if (losText) formData.append("text", losText);
        
          const resultsBox = document.getElementById("decksurf-results");
          resultsBox.innerHTML = "<p>Processing... ⏳</p>";
        
          const response = await fetch("/decksurf", { method: "POST", body: formData });
          const data = await response.json();
        
          if (data.error) {
            resultsBox.innerHTML = `<p style="color:red;">Error: ${data.error}</p>`;
            return;
          }
        
          // Build results table
          let html = `<h3>Results</h3>`;
          let allQueries = [];
        
          data.results.forEach((res, idx) => {
            html += `
              <div style="border:1px solid #ccc;padding:10px;margin-bottom:10px;">
                <b>LO ${idx+1}: ${res.learning_objective}</b><br>
                <ul>
            `;
            res.matches.forEach(m => {
              html += `<li><code>nid:${m.note_id}</code> — ${m.preview} (score: ${m.score.toFixed(2)})</li>`;
            });
            html += `
                </ul>
                <p><b>Search Query:</b> <code>${res.search_query}</code></p>
                <button onclick="navigator.clipboard.writeText('${res.search_query}')">Copy Query</button>
              </div>
            `;
            allQueries.push(res.search_query);
          });
        
          resultsBox.innerHTML = html;
        
          // Show global copy button
          const copyBtn = document.getElementById("copy-queries-btn");
          copyBtn.style.display = "inline-block";
          copyBtn.onclick = () => {
            const combined = allQueries.join(" OR ");
            navigator.clipboard.writeText(combined);
            alert("All queries copied to clipboard ✅");
          };
        });
        </script>
    ''')


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
