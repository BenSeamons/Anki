import types
import os, sys

# Stub external modules used by app.py so it can be imported without deps
flask_stub = types.ModuleType("flask")
class FakeFlask:
    def __init__(self, *a, **k):
        pass
    def route(self, *a, **k):
        def decorator(f):
            return f
        return decorator
    def run(self, *a, **k):
        pass
flask_stub.Flask = FakeFlask
flask_stub.render_template_string = lambda *a, **k: ""
flask_stub.request = types.SimpleNamespace()
flask_stub.send_file = lambda *a, **k: None
sys.modules["flask"] = flask_stub

# genanki is not required for parse_cards logic
sys.modules["genanki"] = types.ModuleType("genanki")

# just_PDFs is referenced but not needed here
justpdfs_stub = types.ModuleType("just_PDFs")
justpdfs_stub.generate_practice_test_return_text = lambda x: ""
sys.modules["just_PDFs"] = justpdfs_stub

# Ensure repo root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from app import parse_cards


def test_basic_card_parsing():
    text = "Front of card\tBack of card"
    result = parse_cards(text)
    assert result == [{
        'front': 'Front of card',
        'back': 'Back of card',
        'is_cloze': False
    }]


def test_cloze_detection():
    text = "{{c1::Capital of France}} is Paris\tParis"
    result = parse_cards(text)
    assert len(result) == 1
    card = result[0]
    assert card['front'] == "{{c1::Capital of France}} is Paris"
    assert card['back'] == 'Paris'
    assert card['is_cloze'] is True


def test_ignore_non_cloze_without_tab():
    text = "Front\tBack\nInvalid line without tab\nAnother front\tAnother back"
    result = parse_cards(text)
    assert len(result) == 2
    fronts = [c['front'] for c in result]
    assert 'Front' in fronts
    assert 'Another front' in fronts


def test_cloze_without_tab():
    text = "Front\tBack\n{{c1::Capital of France}} is Paris\nAnother front\tAnother back"
    result = parse_cards(text)
    assert len(result) == 3
    cloze_card = [c for c in result if c['is_cloze']][0]
    assert cloze_card['front'] == "{{c1::Capital of France}} is Paris"
    assert cloze_card['back'] == ''
