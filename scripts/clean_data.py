"""
Clean raw nutrition CSV file:
- Normalize column names
- Strip whitespace
- Add normalized text field for food items
"""

import pandas as pd
import unicodedata
from pathlib import Path

def normalize_text(s: str):
    if pd.isna(s):
        return s
    s = str(s).strip()
    s = " ".join(s.split())
    return s

def remove_marks(text: str):
    """Remove Vietnamese tone marks for fuzzy matching."""
    if pd.isna(text):
        return text
    text = unicodedata.normalize('NFKD', text)
    return "".join([c for c in text if not unicodedata.combining(c)]).lower()

def main():
    raw_path = Path("data\daily_food_nutrition_dataset_vi.csv")
    df = pd.read_csv(raw_path)

    # Normalize column names
    df.columns = (
        df.columns.str.strip()
        .str.replace(" ", "_")
        .str.replace("(", "")
        .str.replace(")", "")
        .str.replace("-", "_")
        .str.replace("/", "_")
    )

    # Clean Food_Item
    df["Food_Item"] = df["Food_Item"].astype(str).map(normalize_text)
    df["Food_norm"] = df["Food_Item"].map(remove_marks)

    out_path = Path("data/cleaned_logs.csv")
    df.to_csv(out_path, index=False)
    print(f"✅ Saved cleaned data to {out_path} with {len(df)} rows.")

if __name__ == "__main__":
    main()
