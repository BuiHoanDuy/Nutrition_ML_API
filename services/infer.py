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

def infer(text: str):
    p = parse_input(text)
    food_key = map_food(p["food"])
    if food_key:
        row = food_master[food_master["food_key"] == food_key].iloc[0]
        kcal = row["Calories_kcal"]
        qty = p["quantity"] if p["quantity"] else 1
        return {
            "food": row["Food_Item"],
            "quantity": qty,
            "unit": p["unit"],
            "calories_estimated": round(kcal * qty, 2),
            "found_in_master": True
        }
    return {"food": p["food"], "found_in_master": False, "message": "Không tìm thấy món ăn."}

if __name__ == "__main__":
    print(infer("Tôi ăn 3 cái bánh mì thịt"))
