#!/usr/bin/env python3
"""
Say It Right - Phrasebook

A JSON-backed lookup of user-corrected phoneme targets.
Used to override the CAMeL G2P output when the auto-generated
target contains errors (e.g. unwanted case endings).

File location: phrasebook.json (in project root, gitignored).

Schema:
    {
        "كتاب": ["k", "i", "t", "aa", "b"],
        "مدرسة": ["m", "a", "d", "r", "a", "s", "a"]
    }
"""

import json
from pathlib import Path


DEFAULT_PATH = Path(__file__).parent / "phrasebook.json"


def load_phrasebook(path: Path = DEFAULT_PATH) -> dict:
    """Load the phrasebook. Returns {} if file is missing or invalid."""
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        # Validate: every value must be a list of strings
        cleaned = {}
        for k, v in data.items():
            if isinstance(v, list) and all(isinstance(x, str) for x in v):
                cleaned[k] = v
        return cleaned
    except (json.JSONDecodeError, OSError):
        return {}


def save_entry(text: str, phonemes: list, path: Path = DEFAULT_PATH) -> None:
    """Insert or update a single entry, preserving the rest of the file."""
    book = load_phrasebook(path)
    book[text] = list(phonemes)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(book, f, ensure_ascii=False, indent=2)


def lookup(text: str, path: Path = DEFAULT_PATH):
    """Return the saved phoneme list for text, or None if not present."""
    book = load_phrasebook(path)
    return book.get(text)
