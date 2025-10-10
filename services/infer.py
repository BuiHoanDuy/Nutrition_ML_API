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

from scripts.fuzzy_match import map_food
from scripts.parser_rule_based import parse_input

# Resolve data/model paths relative to project root
food_master = pd.read_csv(os.path.join(HERE, "data", "food_master.csv"))
rf = joblib.load(os.path.join(HERE, "services", "models", "calorie_from_macro_rf.pkl"))

def _extract_nutrition_from_row(row: pd.Series, qty: float = 1.0):
    """Return a dict of nutrition values scaled by quantity."""
    # Columns expected in food_master: Calories_kcal, Protein_g, Carbohydrates_g, Fat_g, Fiber_g, Sugars_g, Sodium_mg, Cholesterol_mg
    keys = {
        "calories": "Calories_kcal",
        "protein_g": "Protein_g",
        "carbs_g": "Carbohydrates_g",
        "fat_g": "Fat_g",
        "fiber_g": "Fiber_g",
        "sugars_g": "Sugars_g",
        "sodium_mg": "Sodium_mg",
        "cholesterol_mg": "Cholesterol_mg",
    }
    out = {}
    for out_key, col in keys.items():
        if col in row.index:
            try:
                val = float(row[col])
            except Exception:
                val = None
        else:
            val = None
        out[out_key] = round(val * qty, 2) if val is not None else None
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
        row = food_master[food_master["food_key"] == food_key].iloc[0]
        nutrition = _extract_nutrition_from_row(row, qty)
        return {
            "food": row.get("Food_Item", p["food"]),
            "food_key": food_key,
            "quantity": qty,
            "unit": unit,
            "nutrition": nutrition,
            "found_in_master": True,
        }
    return {"food": p["food"], "found_in_master": False, "message": "Không tìm thấy món ăn.", "parsed": p}

if __name__ == "__main__":
    # print(infer("Tôi ăn 3 cái bánh mì thịt"))
    print(infer("Tôi ăn 3 kí ức gà"))

