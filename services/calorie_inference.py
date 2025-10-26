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
        calories_per_100g = mapped_food["calories"]
        weight_g = mapped_food["weight_g"]
        
        # Calculate total calories based on quantity
        if unit == "phần":
            total_calories = (calories_per_100g * weight_g / 100) * quantity
        else:
            # Assume quantity is in grams
            total_calories = (calories_per_100g * quantity / 100)
        
        return {
            "success": True,
            "found_in_master": True,
            "food": food_text,
            "quantity": quantity,
            "unit": unit,
            "calories": round(total_calories, 2),
            "calories_per_100g": calories_per_100g,
            "weight_g": weight_g,
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
