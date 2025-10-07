"""
Fuzzy matching helper to map user food input to closest food in food_master.csv
Improvements:
 - Normalize text (lowercase, remove diacritics) for more robust matching
 - If full-string match fails, try token-level matching (useful for inputs like "trứng gà")
"""

import pandas as pd
import unicodedata
from rapidfuzz import process, fuzz


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    # remove diacritics
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    # collapse whitespace
    s = ' '.join(s.split())
    return s


food_master = pd.read_csv("data/food_master.csv")
# build normalized choices -> original key mapping
raw_choices = food_master["food_key"].tolist()
normalized_choices = [normalize_text(c) for c in raw_choices]
choices_map = {nc: rc for nc, rc in zip(normalized_choices, raw_choices)}
choices = list(choices_map.keys())


def map_food(text: str, threshold: int = 80):
    if not text:
        return None

    key = normalize_text(text)

    # Try full-string match first
    best = process.extractOne(key, choices, scorer=fuzz.WRatio)
    if best and best[1] >= threshold:
        return choices_map[best[0]]

    # Fallback: try token-level matches (split input and try each token)
    tokens = key.split()
    for tok in tokens:
        if not tok:
            continue
        b = process.extractOne(tok, choices, scorer=fuzz.WRatio)
        if b and b[1] >= threshold:
            return choices_map[b[0]]

    return None


if __name__ == "__main__":
    print(map_food("banh mi thit"))
