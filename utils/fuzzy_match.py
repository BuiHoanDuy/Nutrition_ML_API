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
food_master = pd.read_csv("data/food_nutrition_data_final.csv", encoding='utf-8')

# Create normalized choices from Vietnamese food names
food_names = food_master["Tên thực phẩm"].fillna("").astype(str).tolist()
normalized_names = [normalize_text(name) for name in food_names]
# Map normalized names back to original names for lookup
choices_map = {norm: orig for norm, orig in zip(normalized_names, food_names)}
choices = list(choices_map.keys())


def map_food(text: str, nutrition_df: pd.DataFrame, threshold: int = 60):  # Lower threshold for Vietnamese matching
    if not text:
        return None

    key = normalize_text(text)
    
    # Try full-string match first
    best = process.extractOne(key, choices, scorer=fuzz.WRatio)
    if best and best[1] >= threshold:
        matched_name = choices_map[best[0]]
        # Find row for this food name
        matched_row = nutrition_df[nutrition_df["Tên thực phẩm"] == matched_name]
        if len(matched_row) > 0:
            row = matched_row.iloc[0]
            return {
                "name": matched_name,
                "calories": float(row["Năng lượng"]) if pd.notna(row["Năng lượng"]) else 0.0,
                "protein": float(row["Protein"]) if pd.notna(row["Protein"]) else 0.0,
                "carbs": float(row["Glucid"]) if pd.notna(row["Glucid"]) else 0.0,
                "fat": float(row["Lipid"]) if pd.notna(row["Lipid"]) else 0.0,
                "fiber": float(row["Celluloza"]) if pd.notna(row["Celluloza"]) else 0.0,
                "weight_g": 100.0  # Default weight per serving
            }

    # Fallback: try token-level matches
    tokens = key.split()
    for tok in tokens:
        if len(tok) < 3:  # Skip very short tokens
            continue
        b = process.extractOne(tok, choices, scorer=fuzz.WRatio)
        if b and b[1] >= threshold:
            matched_name = choices_map[b[0]]
            matched_row = nutrition_df[nutrition_df["Tên thực phẩm"] == matched_name]
            if len(matched_row) > 0:
                row = matched_row.iloc[0]
                return {
                    "name": matched_name,
                    "calories": float(row["Năng lượng"]) if pd.notna(row["Năng lượng"]) else 0.0,
                    "protein": float(row["Protein"]) if pd.notna(row["Protein"]) else 0.0,
                    "carbs": float(row["Glucid"]) if pd.notna(row["Glucid"]) else 0.0,
                    "fat": float(row["Lipid"]) if pd.notna(row["Lipid"]) else 0.0,
                    "fiber": float(row["Celluloza"]) if pd.notna(row["Celluloza"]) else 0.0,
                    "weight_g": 100.0  # Default weight per serving
                }

    return None


if __name__ == "__main__":
    print(map_food("banh mi thit"))
