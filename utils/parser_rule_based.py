"""
Simple rule-based parser to extract quantity, unit, and food name from text.
"""

import re


def parse_input(text: str):
    text = text.lower()

    # common Vietnamese unit words (add or extend as needed)
    # Note: order matters for multi-word units; put longer ones first
    units = [
        'muỗng cà phê', 'muong ca phe', 'thìa cà phê', 'thia ca phe', 'muỗng',
        'muỗng canh', 'muong canh', 'thìa canh', 'thia canh', 'thìa',
        'gram', 'gam', 'g', 'kg', 'mg', 'kí', 'kilogam', 'kilo', 'ký',
        'lạng', 'lang', 'ta',
        'cốc', 'tách', 'ly', 'ml', 'ca', 'bát', 'chén',
        'bát', 'chén',
        'cái', 'quả', 'trái', 'chiếc', 'miếng', 'viên',
        'phần'
    ]
    unit_pattern = r"(?:" + r"|".join(re.escape(u) for u in units) + r")"

    # find quantity + optional unit (supports multi-word units)
    m = re.search(rf"(\d[\d\.,]*)(\s*)({unit_pattern})?\b", text)
    quantity = None
    unit = None
    if m:
        quantity = float(m.group(1))
        unit = m.group(3)

    # remove leading pronouns + auxiliaries + verbs (ăn/uống/dùng)
    # examples: "tôi đã ăn", "mình đang uống", "tôi mới dùng"
    t = re.sub(r"\b(tôi|mình)\s+(?:đã|dang|đang|vừa|mới)?\s*(ăn|uống|dùng)\b", '', text, flags=re.IGNORECASE)
    # also remove bare forms at start
    t = re.sub(r"^\s*(?:tôi|mình)?\s*(?:đã|dang|đang|vừa|mới)?\s*(ăn|uống|dùng)\b", '', t, flags=re.IGNORECASE)
    
    # remove quantity and units from text
    t = re.sub(rf"(\d[\d\.,]*)(\s*){unit_pattern}\b", '', t)
    # also remove any leftover standalone unit words
    t = re.sub(rf"\b{unit_pattern}\b", '', t)

    food = t.strip()

    return {"quantity": quantity, "unit": unit, "food": food}

if __name__ == "__main__":
    print(parse_input("Tôi ăn 2 cái bánh mì thịt"))
