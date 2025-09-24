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

def create_filtered_deck_genanki(deck_name, cards):
    """Create filtered deck using pure genanki (replaces ApkgWriter)"""
    deck_id = int(hashlib.sha1(deck_name.encode('utf-8')).hexdigest()[:8], 16)
    deck = genanki.Deck(deck_id, deck_name)
    
    # Basic model for cards
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
    
    # Add cards to deck
    for front, back in cards:
        note = genanki.Note(
            model=basic_model,
            fields=[front, back],
            tags=['filtered']
        )
        deck.add_note(note)
    
    # Create package
    package = genanki.Package(deck)
    deck_data = io.BytesIO()
    package.write_to_file(deck_data)
    deck_data.seek(0)
    return deck_data

def extract_deck_metadata(apkg_file):
    """Extract deck name and ID from .apkg file"""
    with tempfile.TemporaryDirectory() as tmpdir:
        apkg_path = os.path.join(tmpdir, apkg_file.filename)
        apkg_file.save(apkg_path)

        with zipfile.ZipFile(apkg_path, "r") as z:
            z.extractall(tmpdir)

        db_path = os.path.join(tmpdir, "collection.anki2")
        if not os.path.exists(db_path):
            raise ValueError("No collection.anki2 found in apkg")

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        
        # Get deck info
        cur.execute("SELECT decks FROM col")
        decks_json = cur.fetchone()[0]
        conn.close()
        
        import json
        decks = json.loads(decks_json)
        
        # Find the main deck (not default)
        for deck_id, deck_info in decks.items():
            if deck_info['name'] != 'Default':
                return deck_info['name'], int(deck_id)
        
        # Fallback
        return "Imported Deck", 1

@app.route('/practice-tests', methods=['GET', 'POST'])
def practice_tests():
    if request.method == 'GET':
        return render_template_string('''
        <html>
        <head>
            <title>Practice Test Generator</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
                .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                .back-link { display: inline-block; margin-bottom: 20px; color: #007bff; text-decoration: none; }
                button { background-color: #007bff; color: white; padding: 12px 24px; border: none; border-radius: 6px; cursor: pointer; font-size: 16px; margin-top: 20px; }
            </style>
        </head>
        <body>
            <div class="container">
                <a href="/" class="back-link">← Back to Hub</a>
                <h1>Practice Test Generator</h1>
                <form method="POST" enctype="multipart/form-data">
                    <p><strong>Note:</strong> Processing can take up to 10 minutes.</p>
                    <label>Upload PDF files:</label><br>
                    <input type="file" name="individual_pdfs" multiple accept="application/pdf"><br><br>
                    <button type="submit">Generate Practice Tests</button>
                </form>
            </div>
        </body>
        </html>
        ''')
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
    """Extract notes from an uploaded .apkg -> DataFrame(noteId, Front, Back)."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            apkg_path = os.path.join(tmpdir, apkg_file.filename)
            apkg_file.save(apkg_path)
            
            print(f"DEBUG: Saved .apkg to {apkg_path}, size: {os.path.getsize(apkg_path)} bytes")

            # Extract the zip file
            with zipfile.ZipFile(apkg_path, "r") as z:
                print(f"DEBUG: Files in .apkg: {z.namelist()}")
                z.extractall(tmpdir)

            # Check what was extracted
            extracted_files = os.listdir(tmpdir)
            print(f"DEBUG: Extracted files: {extracted_files}")

            # Look for the database file - could be collection.anki2 or collection.anki21b
            db_path = None
            for filename in ["collection.anki2", "collection.anki21b"]:
                potential_path = os.path.join(tmpdir, filename)
                if os.path.exists(potential_path):
                    db_path = potential_path
                    print(f"DEBUG: Found database at {filename}")
                    break
            
            if not db_path:
                raise ValueError(f"No Anki database found. Extracted files: {extracted_files}")

            print(f"DEBUG: Database size: {os.path.getsize(db_path)} bytes")
            
            # Connect to the database
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            
            # Check database structure
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cur.fetchall()
            print(f"DEBUG: Available tables: {[t[0] for t in tables]}")
            
            # Check if we're dealing with a newer format
            cur.execute("PRAGMA table_info(notes)")
            columns = cur.fetchall()
            print(f"DEBUG: Notes table columns: {[col[1] for col in columns]}")
            
            # Count total notes
            cur.execute("SELECT COUNT(*) FROM notes")
            total_notes = cur.fetchone()[0]
            print(f"DEBUG: Total notes in database: {total_notes}")
            
            # Get a few sample notes to understand the structure
            cur.execute("SELECT id, mid, flds FROM notes LIMIT 5")
            sample_rows = cur.fetchall()
            print("DEBUG: Sample notes:")
            for note_id, model_id, flds in sample_rows:
                fields = flds.split("\x1f") if flds else []
                print(f"  Note {note_id} (model {model_id}): {len(fields)} fields")
                for i, field in enumerate(fields[:3]):  # Show first 3 fields
                    print(f"    Field {i}: '{field[:100]}{'...' if len(field) > 100 else ''}'")
            
            # Now get all notes
            cur.execute("SELECT id, mid, flds FROM notes")
            rows = cur.fetchall()
            print(f"DEBUG: Retrieved {len(rows)} rows from notes table")
            
            # Also check what models/notetypes exist
            try:
                cur.execute("SELECT id, name FROM notetypes")
                models = cur.fetchall()
                print(f"DEBUG: Available models: {models}")
            except:
                # Fallback for older Anki versions
                cur.execute("SELECT models FROM col")
                models_json = cur.fetchone()[0]
                print(f"DEBUG: Models JSON length: {len(models_json)} chars")
            
            conn.close()

            # Process the notes
            data = []
            for note_id, model_id, flds in rows:
                fields = flds.split("\x1f") if flds else []
                
                # For cloze cards, the content is usually in the first field
                # For basic cards, front=field[0], back=field[1]
                if len(fields) == 1:
                    # Likely a cloze card
                    front = fields[0]
                    back = ""
                elif len(fields) >= 2:
                    # Basic card or other format
                    front = fields[0]
                    back = fields[1]
                else:
                    front = ""
                    back = ""
                
                print(f"DEBUG: Note {note_id}: Front='{front[:50]}...', Back='{back[:50]}...'")
                data.append({"noteId": note_id, "Front": front, "Back": back})

            print(f"DEBUG: Successfully processed {len(data)} notes")
            return pd.DataFrame(data)
            
    except Exception as e:
        print(f"DEBUG: Exception in load_apkg_to_df: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(columns=["noteId", "Front", "Back"])
@app.route('/')
def home():
    return render_template_string('''
    <html>
      <head>
        <title>Study Tools Hub</title>
        <style>
          body {
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
          }
          .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
          }
          h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
          }
          .tool-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 30px;
          }
          .tool-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #e9ecef;
            text-decoration: none;
            color: inherit;
            transition: transform 0.2s, box-shadow 0.2s;
          }
          .tool-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            text-decoration: none;
            color: inherit;
          }
          .tool-title {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
            color: #495057;
          }
          .tool-description {
            font-size: 14px;
            color: #6c757d;
            line-height: 1.4;
          }
        </style>
      </head>
      <body>
        <div class="container">
          <h1>Study Tools Hub</h1>
          <p style="text-align: center; color: #666; margin-bottom: 40px;">
            Choose from our collection of study tools designed to help students succeed
          </p>
          
          <div class="tool-grid">
            <a href="/anki-generator" class="tool-card">
              <div class="tool-title">Anki Deck Generator</div>
              <div class="tool-description">
                Create custom Anki decks by pasting your cards. Supports both basic and cloze deletion formats.
              </div>
            </a>
            
            <a href="/decksurf" class="tool-card">
              <div class="tool-title">DeckSurfer Mapper</div>
              <div class="tool-description">
                Upload an Anki deck and learning objectives to find relevant cards and create filtered study decks.
              </div>
            </a>
            
            <a href="/practice-tests" class="tool-card">
              <div class="tool-title">Practice Test Generator</div>
              <div class="tool-description">
                Upload PDF lectures to automatically generate practice tests and study questions.
              </div>
            </a>
          </div>
        </div>
      </body>
    </html>
    ''')

@app.route('/anki-generator', methods=['GET', 'POST'])
def anki_generator():
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
    return render_template_string(ANKI_GENERATOR_TEMPLATE)

# Template for DeckSurfer
DECKSURF_TEMPLATE = '''
<html>
<head>
  <title>DeckSurfer - Smart Card Mapper</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      margin: 20px;
      background-color: #f5f5f5;
    }
    .container {
      max-width: 900px;
      margin: 0 auto;
      background: white;
      padding: 30px;
      border-radius: 10px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    h1 {
      color: #333;
      text-align: center;
    }
    .form-section {
      margin-bottom: 25px;
      padding: 20px;
      background: #f8f9fa;
      border-radius: 8px;
    }
    .form-section h3 {
      margin-top: 0;
      color: #495057;
    }
    input[type="file"], textarea {
      width: 100%;
      padding: 8px;
      margin: 5px 0;
      border: 1px solid #ccc;
      border-radius: 4px;
      box-sizing: border-box;
    }
    .slider-container {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    input[type="range"] {
      flex-grow: 1;
    }
    .button-group {
      display: flex;
      gap: 10px;
      margin-top: 20px;
    }
    button {
      padding: 12px 24px;
      font-size: 16px;
      border: none;
      border-radius: 6px;
      cursor: pointer;
      transition: background-color 0.2s;
    }
    .btn-primary {
      background-color: #007bff;
      color: white;
    }
    .btn-primary:hover {
      background-color: #0056b3;
    }
    .btn-success {
      background-color: #28a745;
      color: white;
    }
    .btn-success:hover {
      background-color: #1e7e34;
    }
    .btn-secondary {
      background-color: #6c757d;
      color: white;
    }
    .btn-secondary:hover {
      background-color: #545b62;
    }
    #decksurf-results {
      margin-top: 30px;
    }
    .match-result {
      border: 1px solid #dee2e6;
      border-radius: 8px;
      padding: 15px;
      margin-bottom: 15px;
      background: white;
    }
    .match-result h4 {
      margin-top: 0;
      color: #495057;
    }
    .card-match {
      background: #e9ecef;
      padding: 8px;
      margin: 5px 0;
      border-radius: 4px;
      font-family: monospace;
    }
    .search-query {
      background: #f8f9fa;
      padding: 10px;
      border-radius: 4px;
      border: 1px solid #e9ecef;
      margin-top: 10px;
    }
    .stats-box {
      background: #d4edda;
      border: 1px solid #c3e6cb;
      padding: 15px;
      border-radius: 8px;
      margin-bottom: 20px;
    }
    .back-link {
      display: inline-block;
      margin-bottom: 20px;
      color: #007bff;
      text-decoration: none;
    }
    .back-link:hover {
      text-decoration: underline;
    }
  </style>
</head>
<body>
  <div class="container">
    <a href="/" class="back-link">← Back to Hub</a>
    <h1>DeckSurfer - Smart Card Mapper</h1>
    <p style="text-align: center; color: #666;">
      Upload your Anki deck and learning objectives to automatically find and extract relevant cards
    </p>

    <form id="decksurf-form" enctype="multipart/form-data">
      <div class="form-section">
        <h3>Upload Your Anki Deck</h3>
        <label>Select deck file (.apkg, .csv, .txt):</label>
        <input type="file" id="deck_file" name="deck_file" accept=".apkg,.csv,.txt" required>
        <small style="color: #6c757d;">
          Supports Anki packages (.apkg), CSV files, or tab-separated text files
        </small>
      </div>

      <div class="form-section">
        <h3>Learning Objectives</h3>
        <label>Upload objectives file (PDF/CSV/TXT):</label>
        <input type="file" id="los_file" name="los_file" accept=".pdf,.csv,.txt">
        
        <label style="margin-top: 15px; display: block;">Or paste objectives directly:</label>
        <textarea id="los_text" name="text" rows="6" placeholder="Enter one learning objective per line..."></textarea>
        <small style="color: #6c757d;">
          You can either upload a file or paste objectives directly (one per line)
        </small>
      </div>

      <div class="form-section">
        <h3>Matching Settings</h3>
        <label>Semantic vs Fuzzy Matching Balance:</label>
        <div class="slider-container">
          <span>Fuzzy</span>
          <input type="range" id="alpha" name="alpha" min="0" max="1" step="0.05" value="0.85"
                 oninput="document.getElementById('alphaValue').textContent = this.value">
          <span>Semantic</span>
          <output id="alphaValue" style="font-weight: bold;">0.85</output>
        </div>
        <small style="color: #6c757d;">
          Higher values favor semantic similarity, lower values favor exact text matching
        </small>
      </div>

      <div class="button-group">
        <button type="button" id="run-decksurf" class="btn-primary">Find Matching Cards</button>
        <button type="button" id="download-deck" class="btn-success">Download Filtered Deck</button>
      </div>
    </form>

    <div id="decksurf-results"></div>
    <button id="copy-queries-btn" class="btn-secondary" style="display:none;margin-top:10px;">
      Copy All Search Queries
    </button>
  </div>

  <script>
  async function runDeckSurf(mode="json") {
    const formData = new FormData();
    const deckFile = document.getElementById("deck_file").files[0];
    const losFile = document.getElementById("los_file").files[0];
    const losText = document.getElementById("los_text").value.trim();
    const alpha = document.getElementById("alpha").value;

    if (!deckFile) {
      alert("Please upload a deck file");
      return;
    }
    
    if (!losFile && !losText) {
      alert("Please either upload a learning objectives file or paste objectives in the text area");
      return;
    }

    formData.append("deck_file", deckFile);
    if (losFile) formData.append("los_file", losFile);
    if (losText) formData.append("text", losText);
    formData.append("alpha", alpha);
    formData.append("mode", mode);

    if (mode === "apkg") {
      // Show loading state
      const btn = document.getElementById("download-deck");
      const originalText = btn.textContent;
      btn.textContent = "Generating...";
      btn.disabled = true;

      try {
        const response = await fetch("/decksurf", { method: "POST", body: formData });
        if (!response.ok) {
          const error = await response.json();
          alert(`Error: ${error.error || 'Unknown error'}`);
          return;
        }
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = response.headers.get('Content-Disposition')?.split('filename=')[1] || "filtered_deck.apkg";
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);
      } finally {
        btn.textContent = originalText;
        btn.disabled = false;
      }
      return;
    }

    // Show loading for results
    const resultsBox = document.getElementById("decksurf-results");
    resultsBox.innerHTML = "<p style='text-align: center;'>Processing and matching cards... ⏳</p>";

    try {
      const response = await fetch("/decksurf", { method: "POST", body: formData });
      const data = await response.json();

      if (data.error) {
        resultsBox.innerHTML = `<p style="color:red; text-align: center;">❌ Error: ${data.error}</p>`;
        return;
      }

      // Build results
      let html = `
        <div class="stats-box">
          <h3>Matching Results</h3>
          <p><strong>Deck:</strong> ${data.stats.deck_name}</p>
          <p><strong>Total Cards:</strong> ${data.stats.deck_size} | <strong>Learning Objectives:</strong> ${data.stats.total_objectives} | <strong>Matched Cards:</strong> ${data.stats.matched_cards}</p>
        </div>
      `;

      let allQueries = [];

      data.results.forEach((res, idx) => {
        html += `
          <div class="match-result">
            <h4>Objective ${idx+1}</h4>
            <p><strong>"${res.learning_objective}"</strong></p>
            <div style="margin: 10px 0;">
              <strong>Top Matching Cards:</strong>
        `;
        res.matches.forEach((m, i) => {
          html += `
            <div class="card-match">
              ${i+1}. <code>nid:${m.note_id}</code> — ${m.preview} 
              <span style="color: #28a745; font-weight: bold;">(${(m.score * 100).toFixed(1)}%)</span>
            </div>
          `;
        });
        html += `
            </div>
            <div class="search-query">
              <strong>Anki Search Query:</strong> 
              <code>${res.search_query}</code>
              <button onclick="navigator.clipboard.writeText('${res.search_query}')" 
                      style="margin-left: 10px; padding: 4px 8px; font-size: 12px;" 
                      class="btn-secondary">Copy</button>
            </div>
          </div>
        `;
        allQueries.push(res.search_query);
      });

      resultsBox.innerHTML = html;

      // Show copy all button
      const copyBtn = document.getElementById("copy-queries-btn");
      copyBtn.style.display = "inline-block";
      copyBtn.onclick = () => {
        const combined = allQueries.join(" OR ");
        navigator.clipboard.writeText(combined);
        alert("✅ All search queries copied to clipboard!");
      };

    } catch (error) {
      resultsBox.innerHTML = `<p style="color:red; text-align: center;">❌ Error: ${error.message}</p>`;
    }
  }

  // Hook up buttons
  document.getElementById("run-decksurf").addEventListener("click", () => runDeckSurf("json"));
  document.getElementById("download-deck").addEventListener("click", () => runDeckSurf("apkg"));
  </script>
</body>
</html>
'''

# Template for Anki Generator  
ANKI_GENERATOR_TEMPLATE = '''
<html>
<head>
  <title>Anki Deck Generator</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      margin: 20px;
      background-color: #f5f5f5;
    }
    .container {
      max-width: 800px;
      margin: 0 auto;
      background: white;
      padding: 30px;
      border-radius: 10px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    h1 {
      color: #333;
      text-align: center;
    }
    .help-section {
      background: #e7f3ff;
      padding: 20px;
      border-radius: 8px;
      margin-bottom: 25px;
    }
    .help-section h3 {
      margin-top: 0;
      color: #0066cc;
    }
    label {
      display: block;
      margin-bottom: 5px;
      font-weight: bold;
      color: #333;
    }
    input[type="text"], textarea {
      width: 100%;
      padding: 10px;
      border: 1px solid #ddd;
      border-radius: 4px;
      font-size: 14px;
      box-sizing: border-box;
    }
    textarea {
      resize: vertical;
      min-height: 200px;
    }
    button {
      background-color: #007bff;
      color: white;
      padding: 12px 24px;
      border: none;
      border-radius: 6px;
      cursor: pointer;
      font-size: 16px;
      margin-top: 20px;
    }
    button:hover {
      background-color: #0056b3;
    }
    .example {
      background: #f8f9fa;
      padding: 10px;
      border-left: 4px solid #007bff;
      margin: 10px 0;
      font-family: monospace;
    }
    .back-link {
      display: inline-block;
      margin-bottom: 20px;
      color: #007bff;
      text-decoration: none;
    }
    .back-link:hover {
      text-decoration: underline;
    }
  </style>
</head>
<body>
  <div class="container">
    <a href="/" class="back-link">← Back to Hub</a>
    <h1>Anki Deck Generator</h1>
    
    <div class="help-section">
      <h3>How to Format Your Cards</h3>
      <p><strong>Basic Cards:</strong> Use a tab to separate front and back</p>
      <div class="example">What is the capital of France?	Paris</div>
      
      <p><strong>Cloze Deletion:</strong> Use {{c1::answer}} format (no tab needed)</p>
      <div class="example">{%raw%}The capital of France is {{c1::Paris}}{%endraw%}</div>
      
      <p><strong>Multiple Clozes:</strong> Use c1, c2, etc. for different deletions</p>
      <div class="example">{%raw%}{{c1::Napoleon}} was born in {{c2::1769}} in {{c3::Corsica}}{%endraw%}</div>
    </div>

    <form method="POST">
      <label for="deck_name">Deck Name:</label>
      <input type="text" id="deck_name" name="deck_name" value="Custom Deck" required>

      <label for="cards_text">Your Cards (one per line):</label>
      <textarea id="cards_text" name="cards_text" placeholder="Enter your cards here, one per line..." required></textarea>

      <button type="submit">Generate Anki Deck</button>
    </form>
  </div>
</body>
</html>
'''

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
