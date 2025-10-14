"""
Inference: parse text -> fuzzy match -> estimate calories.
"""

import os
import sys
import pandas as pd
import joblib

# Ensure project root is on sys.path so imports like `scripts.*` work when running
# scripts from the project root (or from the services/ folder)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from scripts.fuzzy_match import map_food, normalize_text
from scripts.parser_rule_based import parse_input

# Load Vietnamese nutrition dataset
data_path = os.path.join(HERE, "data", "food_nutrition_data_final.csv")
food_master = pd.read_csv(data_path)

# Create food_key from normalized food names
if not food_master.empty and "food_key" not in food_master.columns:
    from scripts.fuzzy_match import normalize_text
    food_master["food_key"] = food_master["Tên thực phẩm"].fillna("").astype(str).apply(normalize_text)

rf = joblib.load(os.path.join(HERE, "services", "models", "calorie_from_macro_rf.pkl"))

def _extract_nutrition_from_row(row: pd.Series, qty: float = 1.0):
    """Return a dict of nutrition values scaled by quantity.
    
    Args:
        row: DataFrame row with nutrition values (per 100g)
        qty: Quantity in grams
    
    Returns:
        Dict with scaled nutrition values based on actual grams consumed
    """
    # Convert qty to actual grams if needed
    grams = qty
    
    # Scale factor (data is per 100g)
    scale = grams / 100.0
    
    # Map Vietnamese column names to standard output names
    keys = {
        "calories": "Năng lượng",     # Energy in kcal
        "protein_g": "Protein",      # Protein in g
        "carbs_g": "Glucid",        # Carbohydrates in g
        "fat_g": "Lipid",           # Fat in g
        "fiber_g": "Celluloza",     # Fiber in g
        "water_g": "Nước",          # Water content in g
        "ash_g": "Tro",            # Ash content in g
    }
    out = {}
    for out_key, col in keys.items():
        if col in row.index:
            try:
                val = float(row[col])
                # Scale from per 100g to actual grams consumed
                out[out_key] = round(val * scale, 2)
            except Exception:
                out[out_key] = None
        else:
            out[out_key] = None
    return out


def infer(text: str):
    """Parse input text and return matched food with nutrition breakdown.

    Input: text (str)
    Output: dict with keys: food, food_key, quantity, unit, found_in_master, nutrition: {calories, protein_g, carbs_g, fat_g, fiber_g, sugars_g, sodium_mg, cholesterol_mg}
    """
    p = parse_input(text)
    food_key = map_food(p["food"])
    qty = p.get("quantity") or 1
    unit = p.get("unit")
    if food_key:
        # Find row for matched food name
        matches = food_master[food_master["Tên thực phẩm"] == food_key]
        if len(matches) > 0:
            row = matches.iloc[0]
            nutrition = _extract_nutrition_from_row(row, qty)
            return {
                "food": row["Tên thực phẩm"],  # Original Vietnamese food name
                "food_key": normalize_text(row["Tên thực phẩm"]),  # Normalized for matching
                "quantity": qty,
                "unit": unit,
                "nutrition": nutrition,
                "found_in_master": True,
            }
    return {"food": p["food"], "found_in_master": False, "message": "Không tìm thấy món ăn.", "parsed": p}

if __name__ == "__main__":
    # print(infer("Tôi ăn 3 cái bánh mì thịt"))
    print(infer("Tôi ăn 300 bánh mì"))


