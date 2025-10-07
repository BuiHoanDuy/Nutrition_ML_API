"""
Generate synthetic NER dataset for training from food_master.csv.
"""

import pandas as pd
import json
import random
from pathlib import Path

def main():
    foods = pd.read_csv("data/food_master.csv")["Food_Item"].dropna().unique().tolist()
    units = ["cái", "bát", "muỗng", "g", "ml", "phần", "chén", 
             "tô", "hũ", "ly", "đĩa", "miếng", "quả", "viên", "cốc", "gói", "tách", "chai", "ống", "hộp"
             , "bánh", "cây", "bịch", "tảng", "miếng", "thỏi", "bó", "con", "củ", "nhánh", "lá", "quyển", 
             "bông", "cọng", "bắp", "bầu", "bịch", "bó", "cây", "củ", "gói", "hộp", "khúc", "lạng", "mẩu",
             "nắm", "nải", "tảng", "túi", "vỉ", "vá", "viên", "xấp", "xâu", "đôi",
             "đĩa", "đoạn"]
    quantities = ["1", "2", "0.5", "3"]

    templates = [
        "Tôi ăn {q} {u} {f} vào buổi sáng.",
        "Hôm nay mình ăn {q} {u} {f}.",
        "Ăn {q} {u} {f} cho bữa trưa.",
        "{q} {u} {f}",
    ]

    examples = []
    for f in foods:
        for _ in range(6):
            q = random.choice(quantities)
            u = random.choice(units)
            t = random.choice(templates).format(q=q, u=u, f=f)
            start_food = t.find(f)
            end_food = start_food + len(f)
            start_q = t.find(q)
            end_q = start_q + len(q)
            entities = []
            if start_q >= 0:
                entities.append({"start": start_q, "end": end_q, "label": "QUANTITY"})
            if start_food >= 0:
                entities.append({"start": start_food, "end": end_food, "label": "FOOD"})
            examples.append({"text": t, "entities": entities})

    out = Path("data/ner_generated.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"✅ Saved {len(examples)} NER samples to {out}")

if __name__ == "__main__":
    main()
