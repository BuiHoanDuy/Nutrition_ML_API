"""
Simple rule-based parser to extract quantity, unit, and food name from text.
"""

import re


def parse_input(text: str):
    text = text.lower()

    # common Vietnamese unit words (add or extend as needed)
    units = [
        'gram', 'g',
        'cái', 'quả', 'chiếc', 'miếng', 'viên',
        'bát', 'chén', 'muỗng', 'thìa', 'ly', 'ml', 'phần'
    ]
    unit_pattern = r"(?:" + r"|".join(re.escape(u) for u in units) + r")"

    # find quantity + optional unit
    m = re.search(rf"(\d+(?:\.\d+)?)(\s*)({unit_pattern})?\b", text)
    quantity = None
    unit = None
    if m:
        quantity = float(m.group(1))
        unit = m.group(3)

    # remove quantity and units from text
    t = re.sub(r'(tôi|mình)\s+ăn', '', text)
    t = re.sub(rf"(\d+(?:\.\d+)?)(\s*){unit_pattern}\b", '', t)
    # also remove any leftover standalone unit words
    t = re.sub(rf"\b{unit_pattern}\b", '', t)

    food = t.strip()

    return {"quantity": quantity, "unit": unit, "food": food}

if __name__ == "__main__":
    print(parse_input("Tôi ăn 2 cái bánh mì thịt"))
