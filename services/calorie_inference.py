"""
Calorie inference service.
"""

import os
import sys
import pandas as pd

# Ensure project root is on sys.path
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from utils.fuzzy_match import map_food_candidates, normalize_text
from utils.parser_rule_based import parse_input

# Load Vietnamese nutrition dataset
data_path = os.path.join(HERE, "data", "food_nutrition_data_final.csv")
nutrition_df = pd.read_csv(data_path, encoding='utf-8')

def convert_to_grams(quantity, unit, food_text: str) -> float | None:
    """Convert quantity+unit to grams when possible.
    Returns grams or None if not convertible generically.
    """
    if quantity is None:
        return None

    if not unit:
        return None

    u = unit.strip().lower()
    # Normalize ASCII variants
    ascii_map = {
        'muong ca phe': 'muỗng cà phê',
        'thia ca phe': 'thìa cà phê',
        'muong canh': 'muỗng canh',
        'thia canh': 'thìa canh',
        'lang': 'lạng'
    }
    u = ascii_map.get(u, u)

    # Direct mass units
    if u in {'g', 'gram', 'gam'}:
        return float(quantity)
    if u == 'mg':
        return float(quantity) / 1000.0
    if u == 'kg' or u == 'kí' or u == 'kilogam' or u == 'kilo' or u == 'ký':
        return float(quantity) * 1000.0
    if u in {'lạng', 'lang', 'ta'}:  # VN traditional ~100g
        return float(quantity) * 100.0

    # Volume-ish units (approx water density)
    if u == 'ml':
        return float(quantity)  # ~1g/ml for water-like
    if u in {'cốc', 'tách', 'ly', 'ca'}:
        return float(quantity) * 240.0  # 1 cup ~ 240ml

    # Spoons
    if u in {'muỗng cà phê', 'thìa cà phê', 'muỗng'}:
        return float(quantity) * 5.0
    if u in {'muỗng canh', 'thìa canh', 'thìa'}:
        return float(quantity) * 15.0

    # Bowls
    if u in {'bát', 'chén'}:
        # Heuristic: 1 bowl cooked rice ~ 220-250g
        return float(quantity) * 240.0

    # Piece-based units require food heuristics
    if u in {'cái', 'quả', 'trái', 'chiếc', 'miếng', 'viên'}:
        ft = (food_text or '').lower()
        heuristics = [
            # common items
            (['trứng', 'trung', 'egg'], 50.0),
            (['chuối', 'chuoi', 'banana'], 118.0),
            (['táo', 'tao', 'apple'], 182.0),
            (['cam', 'orange'], 130.0),
            (['bưởi', 'buoi', 'grapefruit'], 230.0),
            (['cà chua', 'ca chua', 'tomato'], 120.0),
            (['khoai tây', 'khoai tay', 'potato'], 170.0),
            (['bánh mì', 'banh mi', 'ổ bánh mì', 'ổ bánh mỳ'], 55.0),
            (['đùi gà', 'dui ga', 'cánh gà', 'canh ga', 'gà', 'ga'], 80.0),
        ]
        for keywords, avg_g in heuristics:
            if any(k in ft for k in keywords):
                return float(quantity) * avg_g
        # Fallback generic piece weight
        return float(quantity) * 50.0

    # Portion not convertible here (handled by serving weight)
    if u == 'phần':
        return None

    return None

def infer(text: str) -> dict:
    """
    Parse text input and return calorie estimation.
    
    Args:
        text: Input text containing food information
        
    Returns:
        Dictionary with food info and calorie estimation
    """
    try:
        # Parse the input text
        parsed = parse_input(text)
        
        if not parsed or not parsed.get("food"):
            return {
                "success": False,
                "message": "Không thể tìm thấy thông tin món ăn trong câu hỏi của bạn.",
                "found_in_master": False
            }
        
        food_text = parsed["food"]
        quantity = parsed.get("quantity")
        unit = parsed.get("unit")
        
        # Normalize and map food vào bảng dinh dưỡng gốc
        normalized_food = normalize_text(food_text)
        candidate_foods = map_food_candidates(normalized_food, nutrition_df, top_n=5)

        if not candidate_foods:
            return {
                "success": False,
                "message": f"Không tìm thấy thông tin dinh dưỡng cho '{food_text}'",
                "found_in_master": False,
                "food": food_text,
                "quantity": quantity,
                "unit": unit,
                "matches": []
            }

        grams = convert_to_grams(quantity, unit, food_text)
        matches = []
        for candidate in candidate_foods:
            calories_per_100g = float(candidate.get("calories") or 0.0)
            if calories_per_100g <= 0.0:
                estimated_calories = None
            elif grams is not None:
                estimated_calories = calories_per_100g * grams / 100.0
            elif quantity:
                default_serving_g = float(candidate.get("weight_g", 100.0))
                estimated_calories = (calories_per_100g * default_serving_g / 100.0) * float(quantity)
            else:
                estimated_calories = calories_per_100g * float(candidate.get("weight_g", 100.0)) / 100.0

            matches.append(
                {
                    "food": candidate["name"],
                    "score": candidate.get("score"),
                    "calories_per_100g": calories_per_100g,
                    "estimated_calories": round(estimated_calories, 2) if estimated_calories is not None else None,
                    "weight_g": grams if grams is not None else float(candidate.get("weight_g", 100.0)),
                    "image_url": candidate.get("image_url"),
                    "nutrition_info": {
                        "protein": candidate.get("protein", 0.0),
                        "carbs": candidate.get("carbs", 0.0),
                        "fat": candidate.get("fat", 0.0),
                        "fiber": candidate.get("fiber", 0.0),
                    },
                }
            )

        return {
            "success": True,
            "found_in_master": True,
            "query": food_text,
            "quantity": quantity,
            "unit": unit,
            "matches": matches
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Lỗi khi xử lý: {str(e)}",
            "found_in_master": False
        }
