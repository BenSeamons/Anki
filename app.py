from flask import Flask, request, render_template_string, send_file
import io
import genanki
import os
import hashlib
import tempfile
import pandas as pd
from PyPDF2 import PdfReader
from just_PDFs import generate_practice_test_return_text

app = Flask(__name__)



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

@app.route('/decksurf', methods=['POST'])
def decksurf():
    # Accept either file upload or pasted text
    los = []
    if 'file' in request.files and request.files['file'].filename:
        f = request.files['file']
        filename = f.filename.lower()
        if filename.endswith('.pdf'):
            reader = PdfReader(f)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            # crude LO split
            los = [line.strip() for line in text.split("\n") if 20 < len(line.strip()) < 250]
        elif filename.endswith('.csv'):
            df = pd.read_csv(f)
            col = df.columns[0]
            los = df[col].dropna().astype(str).tolist()
        elif filename.endswith('.txt'):
            los = [line.strip() for line in f.read().decode('utf-8').splitlines() if line.strip()]
    elif request.form.get('text'):
        los = [line.strip() for line in request.form['text'].splitlines() if line.strip()]

    if not los:
        return "No learning objectives found.", 400

    # For now, just echo them back (you can plug in Deck_Sorter here later)
    html = "<h2>Extracted Learning Objectives</h2><ul>"
    for lo in los[:30]:  # cap for readability
        html += f"<li>{lo}</li>"
    html += "</ul>"
    return html


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
          <form method="POST" action="/decksurf" enctype="multipart/form-data">
            <label for="decksurf_file">Upload PDF/CSV/TXT:</label><br>
            <input type="file" id="decksurf_file" name="file" accept=".pdf,.csv,.txt"><br><br>
            <label for="decksurf_text">Or paste objectives:</label><br>
            <textarea id="decksurf_text" name="text" rows="10" placeholder="One objective per line"></textarea><br><br>
            <button type="submit">Run DeckSurf</button>
          </form>
        </div>
    ''')


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
