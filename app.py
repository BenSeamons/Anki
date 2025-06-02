from flask import Flask, request, render_template_string, send_file
import io
import genanki
import os
import hashlib

app = Flask(__name__)

# Simple parser to convert pasted text into cards list
def parse_cards(text):
    cards = []
    lines = text.strip().split('\n')
    for line in lines:
        # Expect tab-separated front and back
        if '\t' in line:
            front, back = line.split('\t', 1)
            front = front.strip()
            back = back.strip()

            # Detect cloze by {{c1::...}} style in front
            is_cloze = '{{c' in front

            cards.append({
                'front': front,
                'back': back,
                'is_cloze': is_cloze
            })
    return cards

@app.route('/', methods=['GET', 'POST'])
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
            <head><title>Anki Deck Generator</title></head>
            <body>
                <h1>Paste Your Cards Here (tab-separated Front and Back)</h1>

                <!-- Put your prompt example instructions here -->
                <h2>How to generate good cards for this app</h2>
                <p>To get the best cards, you can ask your AI or note-taking tool using a prompt like this:</p>
                <pre style="background:#f0f0f0; padding:10px; border-radius:5px; font-family: monospace;">
                Please generate Anki cards for me from this (pdf/powerpoint) in both the Basic and Cloze format.  
                Please give them to me as plain text with the Basic cards front and back separated by a tab.  
                Focus on giving me high yield flashcards that would help a first year med student pass their board exams.
                </pre>

    
                <form method="POST">
                    <label for="deck_name">Deck Name:</label>
                    <input type="text" id="deck_name" name="deck_name" placeholder="Enter deck name" required><br><br>
                    <textarea name="cards_text" rows="15" cols="80" placeholder="Front [tab] Back"></textarea><br>
                    <button type="submit">Generate Anki Deck</button>
                </form>
                <p>Use <code>{&#123;&#123;c1::cloze deletion&#125;&#125;}</code> syntax for cloze cards in the Front field. Back field can be empty for cloze.</p>
            </body>
        </html>
    ''')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

