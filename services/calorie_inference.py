"""
Calorie inference service.
"""

import os
import sys
import pandas as pd
import joblib

# Ensure project root is on sys.path
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from utils.fuzzy_match import map_food, normalize_text
from utils.parser_rule_based import parse_input

# Load Vietnamese nutrition dataset
data_path = os.path.join(HERE, "data", "food_nutrition_data_final.csv")
nutrition_df = pd.read_csv(data_path, encoding='utf-8')

# Load calorie prediction model
model_path = os.path.join(HERE, "models", "calorie", "calorie_from_macro_rf.pkl")
calorie_model = joblib.load(model_path)

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
        quantity = parsed.get("quantity", 1)
        unit = parsed.get("unit", "phần")
        
        # Normalize and map food
        normalized_food = normalize_text(food_text)
        mapped_food = map_food(normalized_food, nutrition_df)
        
        if mapped_food is None:
            return {
                "success": False,
                "message": f"Không tìm thấy thông tin dinh dưỡng cho '{food_text}'",
                "found_in_master": False,
                "food": food_text,
                "quantity": quantity,
                "unit": unit
            }
        
        # Calculate calories
        calories_per_100g = float(mapped_food["calories"])  # per 100g
        default_serving_g = float(mapped_food["weight_g"])  # our default serving size

        grams = convert_to_grams(quantity, unit, food_text)

        if grams is None:
            # Use serving-based calculation (e.g., portion)
            total_calories = (calories_per_100g * default_serving_g / 100.0) * float(quantity if quantity else 1)
        else:
            total_calories = (calories_per_100g * grams / 100.0)
        
        return {
            "success": True,
            "found_in_master": True,
            "food": food_text,
            "quantity": quantity,
            "unit": unit,
            "calories": round(total_calories, 2),
            "calories_per_100g": calories_per_100g,
            "weight_g": grams if grams is not None else default_serving_g,
            "nutrition_info": {
                "protein": mapped_food["protein"],
                "carbs": mapped_food["carbs"],
                "fat": mapped_food["fat"],
                "fiber": mapped_food["fiber"]
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Lỗi khi xử lý: {str(e)}",
            "found_in_master": False
        }
