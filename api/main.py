import json
import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import routers
from api.routers import obesity

from services.calorie_inference import infer  # For calorie prediction
from services.meal_plan_inference import (
    generate_natural_response_from_recommendations,
    parse_meal_plan_question,
    recommend_meal_plan,
)
from services.question_classifier import classify_question
# from services.infer_meal_sequence import generate_full_day_menu, suggest_next_meal # For HMM meal sequence

# --- Setup Logging for user questions ---
LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Create a specific logger for meal plan requests
request_logger = logging.getLogger("meal_plan_requests")
request_logger.setLevel(logging.INFO)

# Prevent logs from propagating to the root logger (which uvicorn uses for console)
request_logger.propagate = False

# Create a file handler which logs messages to a file
handler = RotatingFileHandler(
    LOGS_DIR / "meal_plan_requests.log", maxBytes=10_000_000, backupCount=5, encoding="utf-8"
)
# Set formatter to handle Unicode properly
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
request_logger.addHandler(handler)

app = FastAPI(title="Nutrition Inference API")
INTENT_CONFIDENCE_THRESHOLD = float(os.getenv("INTENT_CONFIDENCE_THRESHOLD", "0.9"))

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include routers
app.include_router(obesity.router)


class PredictRequest(BaseModel):
    content: str


@app.post("/predict")
async def predict(req: PredictRequest) -> Dict[str, Any]:
    text = req.content
    if not text or not isinstance(text, str):
        raise HTTPException(status_code=400, detail="Field 'content' is required and must be a string")
    try:
        result = infer(text)
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class MealPlanRequest(BaseModel):
    question: str


class AskRequest(BaseModel):
    question: str


def _format_calorie_answer(result: Dict[str, Any]) -> str:
    if not result.get("success"):
        return result.get("message", "Không thể xử lý câu hỏi này.")

    food = result.get("food", "món ăn")
    quantity = result.get("quantity")
    unit = result.get("unit")
    calories = result.get("calories")
    nutrition = result.get("nutrition_info", {})

    parts = [
        f"{food.strip().title()}",
        f"khối lượng ~ {result.get('weight_g', 'N/A')}g" if result.get("weight_g") else "",
        f"ước tính {calories} kcal" if calories is not None else ""
    ]
    header = ", ".join([p for p in parts if p])

    macros = []
    if nutrition:
        for key, label in {
            "protein": "protein",
            "carbs": "carb",
            "fat": "fat",
            "fiber": "chất xơ"
        }.items():
            if nutrition.get(key) is not None:
                macros.append(f"{label}: {nutrition[key]}g")

    macros_str = ", ".join(macros)
    serving = f"Số lượng bạn nhập: {quantity} {unit}" if quantity else ""

    return ". ".join(filter(None, [header, macros_str, serving]))


async def _generate_meal_plan_answer(question: str) -> tuple[bool, str]:
    try:
        # Log the user's question for future training
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "question": question
        }
        try:
            request_logger.info(json.dumps(log_entry, ensure_ascii=False))
        except UnicodeEncodeError:
            # Fallback: encode question to ASCII with replacement
            safe_question = question.encode('ascii', 'replace').decode('ascii')
            log_entry["question"] = safe_question
            request_logger.info(json.dumps(log_entry, ensure_ascii=False))

        # Parse the natural language question to extract parameters
        parsed_params = await parse_meal_plan_question(question) # Await the async function

        # SỬA LỖI: Truyền toàn bộ dictionary `parsed_params` thay vì các key riêng lẻ.
        recommendations = recommend_meal_plan(
            original_question=question,
            parsed_params=parsed_params
        )

        if not recommendations:
            return False, "Rất tiếc, tôi không tìm thấy thực đơn nào phù hợp với yêu cầu của bạn."

        # New Step: Generate a natural language response from the recommendations
        natural_response = await generate_natural_response_from_recommendations(
            question=question,
            parsed_params=parsed_params,
            recommendations=recommendations
        )
 
        # Return the generated text instead of the raw JSON
        # Ensure response is properly encoded for Windows console
        try:
            return True, natural_response
        except UnicodeEncodeError:
            # Fallback: encode to ASCII with replacement for problematic characters
            safe_response = natural_response.encode('ascii', 'replace').decode('ascii')
            return True, safe_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting meal plan recommendations: {e}")


@app.post("/recommend_meal_plan")
async def get_meal_plan_recommendations(
    req: MealPlanRequest
) -> Dict[str, Any]: # The response will now be a natural language string
    """
    Recommends meal plans based on user's health status, goal, and nutrition intent.
    """
    success, natural_response = await _generate_meal_plan_answer(req.question)
    return {"success": success, "response": natural_response}


@app.post("/ask")
async def ask_nutrition_assistant(req: AskRequest) -> Dict[str, Any]:
    """
    Classify the user's intent and route to the appropriate module,
    returning a unified response structure.
    """
    classification = classify_question(req.question) or {}
    raw_intent = classification.get("label", "unknown")
    confidence = classification.get("confidence", 0.0)
    is_reliable = confidence >= INTENT_CONFIDENCE_THRESHOLD and raw_intent != "unknown"

    intent = raw_intent if is_reliable else "unknown"
    answer = None
    answer_available = False

    if intent == "meal_plan":
        success, answer = await _generate_meal_plan_answer(req.question)
        answer_available = success
        if not success:
            answer = "Xin lỗi, tôi chưa tìm thấy thực đơn nào phù hợp. Bạn có thể mô tả chi tiết hơn giúp mình nhé!"
    elif intent == "nutrition_fact":
        calorie_result = infer(req.question)
        answer = _format_calorie_answer(calorie_result)
        answer_available = bool(answer)
    elif intent == "general_questions":
        answer = "Mình là trợ lý dinh dưỡng AI, chuyên trả lời về thực đơn và dinh dưỡng. Bạn thử hỏi về bữa ăn hoặc món cụ thể nhé!"
        answer_available = True
    else:
        answer = "Xin lỗi, tôi chưa đủ tự tin để phân loại câu hỏi này. Bạn có thể diễn đạt lại rõ hơn liên quan đến dinh dưỡng được không?"

    final_intent = intent if (answer_available and is_reliable) else "unknown"
    final_answer = answer if (answer_available and is_reliable) else None

    return {
        "intent": final_intent,
        "answer": final_answer
    }


# --- HMM-based Meal Sequence Endpoints ---

# @app.get("/generate_daily_menu", summary="Generate a full day menu using HMM")
# async def get_daily_menu(num_meals: int = 3) -> Dict[str, Any]:
#     """
#     Generates a random but probable full-day meal plan using the HMM.
#     This simulates a likely sequence of meal types (e.g., breakfast -> lunch -> dinner).
#     """
#     try:
#         menu = generate_full_day_menu(num_meals=num_meals)
#         if "error" in menu:
#             raise HTTPException(status_code=500, detail=menu["error"])
#         return {"success": True, "menu": menu}
#     except Exception as e:
#         # Catch exceptions from within the service function as well
#         raise HTTPException(status_code=500, detail=f"Error generating daily menu: {e}")


# class SuggestNextMealRequest(BaseModel):
#     question: str


# @app.post("/suggest_next_meal", summary="Suggest next meal based on previous one using HMM")
# async def get_next_meal_suggestion(
#     req: SuggestNextMealRequest,
#     num_suggestions: int = 3
# ) -> Dict[str, Any]:
#     """
#     Suggests possible next meals based on a user's question about what they just ate.
#     This endpoint will:
#     1. Parse the question to extract the food item.
#     2. Use the extracted food item to get suggestions from the HMM.
#     """
#     try:
#         # Step 1: Parse the question to extract the food item.
#         # We can reuse the `infer` function which is designed for this.
#         parsed_info = infer(req.question)

#         if not parsed_info or not parsed_info.get("found_in_master"):
#             # If no food is found in the question, return a helpful message.
#             return {
#                 "success": False,
#                 "message": "Không thể tìm thấy món ăn trong câu hỏi của bạn. Vui lòng thử lại, ví dụ: 'Tôi vừa ăn phở bò'."
#             }

#         # Step 2: Use the extracted food to get suggestions from the HMM.
#         suggestions = suggest_next_meal(
#             previous_meal_text=parsed_info["food"],
#             num_suggestions=num_suggestions
#         )
#         if suggestions and isinstance(suggestions[0], dict) and "error" in suggestions[0]:
#             raise HTTPException(status_code=500, detail=suggestions[0]["error"])
#         return {"success": True, "suggestions": suggestions}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error suggesting next meal: {e}")
