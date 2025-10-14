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


# Load Vietnamese food dataset
food_master = pd.read_csv("data/food_nutrition_data_final.csv")

# Create normalized choices from Vietnamese food names
food_names = food_master["Tên thực phẩm"].fillna("").astype(str).tolist()
normalized_names = [normalize_text(name) for name in food_names]
# Map normalized names back to original names for lookup
choices_map = {norm: orig for norm, orig in zip(normalized_names, food_names)}
choices = list(choices_map.keys())


def map_food(text: str, threshold: int = 60):  # Lower threshold for Vietnamese matching
    if not text:
        return None

    key = normalize_text(text)
    
    # Try full-string match first
    best = process.extractOne(key, choices, scorer=fuzz.WRatio)
    if best and best[1] >= threshold:
        matched_name = choices_map[best[0]]
        # Find row index for this food name
        idx = food_master[food_master["Tên thực phẩm"] == matched_name].index
        if len(idx) > 0:
            return matched_name

    # Fallback: try token-level matches
    tokens = key.split()
    for tok in tokens:
        if len(tok) < 3:  # Skip very short tokens
            continue
        b = process.extractOne(tok, choices, scorer=fuzz.WRatio)
        if b and b[1] >= threshold:
            matched_name = choices_map[b[0]]
            idx = food_master[food_master["Tên thực phẩm"] == matched_name].index
            if len(idx) > 0:
                return matched_name

    return None


if __name__ == "__main__":
    print(map_food("banh mi thit"))
