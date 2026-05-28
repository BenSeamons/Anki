import csv
import requests
import os

# Constants
ANKI_ENDPOINT = "http://localhost:8765"
DEFAULT_DECK = "MICRO-GI"
MODEL_BASIC = "Basic"
MODEL_CLOZE = "Cloze"



def read_csv_cards(file_path):
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        cards = []
        for row in reader:
            front = row.get('Front', '').strip()
            back = row.get('Back', '').strip()
            deck = row.get('Deck', DEFAULT_DECK)
            tags = row.get('Tags', 'JeremyMode').split()

            # Detect if this is a cloze card by checking for cloze syntax
            is_cloze = '{{c' in front

            if front and back:
                cards.append({
                    'front': front,
                    'back': back,
                    'deck': deck,
                    'tags': tags
                })
    return cards


def send_to_anki(cards):
    for card in cards:
        if card['is_cloze']:
            #Cloze card structure
            payload = {
                "action": "addNote",
                "version": 6,
                "params": {
                    "note": {
                        "deckName": card['deck'],
                        "modelName": MODEL_CLOZE,
                        "fields": {
                            "Text": card['front'] #Cloze text goes here
                        },
                        "options": {
                            "allowDuplicate": False
                        },
                        "tags": card['tags']
                    }
                }
            }
        else:
            #basic card structure
            payload = {
                "action": "addNote",
                "version": 6,
                "params": {
                    "note": {
                        "deckName": card['deck'],
                        "modelName": MODEL_BASIC,
                        "fields": {
                            "Front": card['front'],
                            "Back": card['back']
                        },
                        "options": {
                            "allowDuplicate": False
                        },
                        "tags": card['tags']
                    }
                }
            }

        response = requests.post(ANKI_ENDPOINT, json=payload).json()
        print(f"Sent: {card['front'][:40]}... → {response}")


if __name__ == "__main__":
    file_path = "flashcards.csv"  # Replace with your CSV file path
    cards = read_csv_cards(file_path)
    send_to_anki(cards)
    os.remove(file_path)
