from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import json
from datetime import datetime

# Import routers
from api.routers import obesity

from services.calorie_inference import infer # For calorie prediction
from services.meal_plan_inference import recommend_meal_plan, parse_meal_plan_question, generate_natural_response_from_recommendations # For meal plan recommendation and parsing
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


@app.post("/recommend_meal_plan")
async def get_meal_plan_recommendations(
    req: MealPlanRequest
) -> Dict[str, Any]: # The response will now be a natural language string
    """
    Recommends meal plans based on user's health status, goal, and nutrition intent.
    """
    try:
        # Log the user's question for future training
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "question": req.question
        }
        try:
            request_logger.info(json.dumps(log_entry, ensure_ascii=False))
        except UnicodeEncodeError:
            # Fallback: encode question to ASCII with replacement
            safe_question = req.question.encode('ascii', 'replace').decode('ascii')
            log_entry["question"] = safe_question
            request_logger.info(json.dumps(log_entry, ensure_ascii=False))

        # Parse the natural language question to extract parameters
        parsed_params = await parse_meal_plan_question(req.question) # Await the async function

        # SỬA LỖI: Truyền toàn bộ dictionary `parsed_params` thay vì các key riêng lẻ.
        recommendations = recommend_meal_plan(
            original_question=req.question,
            parsed_params=parsed_params
        )

        # New Step: Generate a natural language response from the recommendations
        # SỬA LỖI: Truyền toàn bộ dictionary `parsed_params`
        natural_response = await generate_natural_response_from_recommendations(
            question=req.question,
            parsed_params=parsed_params,
            recommendations=recommendations
        )
 
        # Return the generated text instead of the raw JSON
        # Ensure response is properly encoded for Windows console
        try:
            return {"success": True, "response": natural_response}
        except UnicodeEncodeError:
            # Fallback: encode to ASCII with replacement for problematic characters
            safe_response = natural_response.encode('ascii', 'replace').decode('ascii')
            return {"success": True, "response": safe_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting meal plan recommendations: {e}")


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
