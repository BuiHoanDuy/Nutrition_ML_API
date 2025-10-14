"""
Aggregate unique foods to build a master list with mean nutrition values.
"""

import pandas as pd
from pathlib import Path

def main():
    # Use the new FOOD-DATA.csv if available, otherwise fall back to cleaned_logs
    try:
        df = pd.read_csv("data/FOOD-DATA.csv")
    except Exception:
        df = pd.read_csv("data/cleaned_logs.csv")

    group = (
        df.groupby("Food_norm")
        .agg({
            "Food_Item": "first",
            "Calories_kcal": "mean",
            "Protein_g": "mean",
            "Carbohydrates_g": "mean",
            "Fat_g": "mean",
            "Fiber_g": "mean",
            "Sugars_g": "mean",
            "Sodium_mg": "mean",
            "Cholesterol_mg": "mean"
        })
        .reset_index()
        .rename(columns={"Food_norm": "food_key"})
    )

    out = Path("data/food_master.csv")
    group.to_csv(out, index=False)
    print(f"✅ Saved food master ({len(group)} items) to {out}")

if __name__ == "__main__":
    main()
