"""
Meal plan inference service.
"""

import os
import pandas as pd
import joblib
from pathlib import Path
import scipy.sparse
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import httpx
import json
import logging
from logging.handlers import RotatingFileHandler

# Define paths
HERE = Path(__file__).parent.parent
MODEL_DIR = HERE / "models" / "meal_plan"

# --- Setup Logging for LLM interactions ---
LOGS_DIR = HERE / "logs"
LOGS_DIR.mkdir(exist_ok=True)

llm_logger = logging.getLogger("llm_interactions")
llm_logger.setLevel(logging.INFO)
llm_logger.propagate = False
llm_handler = RotatingFileHandler(
    LOGS_DIR / "llm_interactions.log", maxBytes=10_000_000, backupCount=5, encoding="utf-8"
)
llm_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
llm_handler.setFormatter(llm_formatter)
llm_logger.addHandler(llm_handler)

# Global variables to store loaded artifacts
meal_plans_df = None
user_feature_encoder = None
meal_tfidf_vectorizer = None
user_features_encoded_matrix = None
meal_features_tfidf_matrix = None
user_features_cols = ['tình_trạng_sức_khỏe', 'mục_tiêu', 'intent_dinh_dưỡng']

def load_recommender_artifacts():
    """Loads all necessary artifacts for the meal plan recommender."""
    global meal_plans_df, user_feature_encoder, meal_tfidf_vectorizer, \
           user_features_encoded_matrix, meal_features_tfidf_matrix

    if meal_plans_df is not None:
        return

    print("[INFO] Loading meal plan recommender artifacts...")
    try:
        meal_plans_df = pd.read_csv(MODEL_DIR / "meal_plans_data.csv", encoding='utf-8-sig')
        user_feature_encoder = joblib.load(MODEL_DIR / "user_feature_encoder.pkl")
        meal_tfidf_vectorizer = joblib.load(MODEL_DIR / "meal_tfidf_vectorizer.pkl")
        user_features_encoded_matrix = scipy.sparse.load_npz(MODEL_DIR / "user_features_encoded.npz")
        meal_features_tfidf_matrix = scipy.sparse.load_npz(MODEL_DIR / "meal_features_tfidf.npz")
        
        print("[SUCCESS] Meal plan recommender artifacts loaded successfully.")

    except FileNotFoundError as e:
        print(f"[ERROR] Error loading meal plan recommender artifacts: {e}")
        print("Please ensure meal plan data has been processed successfully.")
        meal_plans_df = None
        user_feature_encoder = None
        meal_tfidf_vectorizer = None
        user_features_encoded_matrix = None
        meal_features_tfidf_matrix = None
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred while loading artifacts: {e}")
        meal_plans_df = None
        user_feature_encoder = None
        meal_tfidf_vectorizer = None
        user_features_encoded_matrix = None
        meal_features_tfidf_matrix = None

# --- OpenRouter Integration ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct-v0.2")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

async def call_openrouter_llm(prompt: str, model: str = OPENROUTER_MODEL, json_mode: bool = True) -> dict | str | None:
    """Calls the OpenRouter LLM API to get a structured response."""
    if not OPENROUTER_API_KEY:
        llm_logger.warning("OPENROUTER_API_KEY not set. Skipping LLM call.")
        return None

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1 if json_mode else 0.7,
    }
    if json_mode:
        data["response_format"] = {"type": "json_object"}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(OPENROUTER_URL, headers=headers, json=data, timeout=30.0)
            response.raise_for_status()
            response_json = response.json()
            
            llm_response_content = response_json["choices"][0]["message"]["content"]
            try:
                llm_logger.info(f"LLM call successful. Model: {model}, Prompt: {prompt[:100]}..., Raw Response: {llm_response_content}")
            except UnicodeEncodeError:
                safe_prompt = prompt[:100].encode('ascii', 'replace').decode('ascii')
                safe_response = llm_response_content.encode('ascii', 'replace').decode('ascii')
                llm_logger.info(f"LLM call successful. Model: {model}, Prompt: {safe_prompt}..., Raw Response: {safe_response}")

            if json_mode:
                parsed_llm_output = json.loads(llm_response_content)
                return parsed_llm_output
            else:
                return llm_response_content.strip()
    except httpx.RequestError as e:
        llm_logger.error(f"HTTPX Request Error calling OpenRouter LLM: {e}")
        return None
    except httpx.HTTPStatusError as e:
        llm_logger.error(f"HTTP Status Error from OpenRouter LLM: {e.response.status_code} - {e.response.text}")
        return None
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        llm_logger.error(f"Error parsing LLM response or malformed response: {e}. Response: {response.text if 'response' in locals() else 'N/A'}")
        return None
    except Exception as e:
        llm_logger.error(f"An unexpected error occurred during LLM call: {e}")
        return None

async def parse_meal_plan_question(question: str) -> dict:
    """Parses a natural language question to extract health_status, goal, and nutrition_intent."""
    llm_prompt = f"""Bạn là một trợ lý dinh dưỡng. Hãy phân tích câu hỏi sau của người dùng và trích xuất các thông tin sau vào định dạng JSON:
- `health_status`: Tình trạng sức khỏe (ví dụ: 'béo phì', 'tiểu đường', 'bình thường', 'không có').
- `goal`: Mục tiêu (ví dụ: 'giảm cân', 'tăng cân', 'tăng cơ', 'giữ cân', 'không có').
- `nutrition_intent`: Ý định dinh dưỡng (ví dụ: 'ăn ít tinh bột', 'ăn nhiều protein', 'ăn chay', 'ổn định đường huyết', 'không có').
- `requested_meals`: Danh sách các bữa ăn được yêu cầu (ví dụ: ['bữa_sáng', 'bữa_trưa', 'bữa_tối', 'bữa_phụ']). Nếu không rõ, mặc định là tất cả.

Nếu không tìm thấy thông tin cụ thể, hãy sử dụng giá trị 'không có' cho các trường `health_status`, `goal`, `nutrition_intent`.
Đối với `requested_meals`, nếu không có yêu cầu cụ thể, hãy mặc định là `['bữa_sáng', 'bữa_trưa', 'bữa_tối', 'bữa_phụ']`.

Câu hỏi của người dùng: '{question}'

Vui lòng chỉ trả về một đối tượng JSON hợp lệ.
"""
    llm_parsed_params = await call_openrouter_llm(llm_prompt)

    if llm_parsed_params:
        try:
            llm_logger.info(f"LLM successfully parsed question: {question} -> {llm_parsed_params}")
        except UnicodeEncodeError:
            safe_question = question.encode('ascii', 'replace').decode('ascii')
            llm_logger.info(f"LLM successfully parsed question: {safe_question} -> {llm_parsed_params}")
        
        final_params = {
            "health_status": llm_parsed_params.get("health_status", "không có"),
            "goal": llm_parsed_params.get("goal", "không có"),
            "nutrition_intent": llm_parsed_params.get("nutrition_intent", "không có"),
            "requested_meals": llm_parsed_params.get("requested_meals", ["bữa_sáng", "bữa_trưa", "bữa_tối", "bữa_phụ"])
        }
        
        if not isinstance(final_params["requested_meals"], list):
            llm_logger.warning(f"LLM returned non-list for requested_meals. Defaulting. Raw: {llm_parsed_params.get('requested_meals')}")
            final_params["requested_meals"] = ["bữa_sáng", "bữa_trưa", "bữa_tối", "bữa_phụ"]
        return final_params
    else:
        try:
            llm_logger.warning(f"LLM parsing failed for question: {question}. Falling back to keyword parsing.")
        except UnicodeEncodeError:
            safe_question = question.encode('ascii', 'replace').decode('ascii')
            llm_logger.warning(f"LLM parsing failed for question: {safe_question}. Falling back to keyword parsing.")
        
        # Fallback to keyword matching
        question_lower = question.lower()
        extracted_params = {
            "health_status": "không có",
            "goal": "không có",
            "nutrition_intent": "không có",
            "requested_meals": ["bữa_sáng", "bữa_trưa", "bữa_tối", "bữa_phụ"]
        }

        # Define keywords for each category
        health_status_map = {
            "béo phì": ["béo phì", "thừa cân"],
            "cao huyết áp": ["cao huyết áp", "huyết áp cao"],
            "tiểu đường": ["tiểu đường", "đường huyết cao"],
            "gầy yếu": ["gầy yếu", "quá gầy"],
            "bình thường": ["bình thường", "người bình thường"]
        }

        goal_map = {
            "giảm cân": ["giảm cân", "muốn giảm cân"],
            "tăng cân": ["tăng cân", "muốn tăng cân"],
            "tăng cơ": ["tăng cơ", "xây dựng cơ bắp"],
            "giữ cân": ["giữ cân", "duy trì cân nặng"],
            "duy trì sức khỏe": ["duy trì sức khỏe", "sức khỏe tốt"],
            "ăn lành mạnh": ["ăn lành mạnh", "chế độ ăn lành mạnh"]
        }

        nutrition_intent_map = {
            "ăn ít tinh bột": ["ít tinh bột", "low carb"],
            "ăn nhiều protein": ["nhiều protein", "high protein"],
            "ổn định đường huyết": ["ổn định đường huyết"],
            "ăn chay": ["ăn chay", "chay"],
            "giảm cân": ["giảm cân"],
            "tăng cân": ["tăng cân"],
            "tăng cơ": ["tăng cơ"],
            "giữ cân": ["giữ cân"],
            "duy trì sức khỏe": ["duy trì sức khỏe"],
            "ăn lành mạnh": ["ăn lành mạnh"]
        }

        meal_map = {
            "bữa_sáng": ["bữa sáng", "buổi sáng", "sáng"],
            "bữa_trưa": ["bữa trưa", "buổi trưa", "trưa"],
            "bữa_tối": ["bữa tối", "buổi tối", "tối"],
            "bữa_phụ": ["bữa phụ", "ăn vặt", "ăn nhẹ"],
            "cả_ngày": ["cả ngày", "một ngày", "1 ngày"]
        }

        def find_keyword(text, keyword_map):
            for category, kws in keyword_map.items():
                kws_sorted = sorted(kws, key=len, reverse=True)
                for kw in kws_sorted:
                    if kw in text:
                        return category
            return "không có"

        def find_requested_meals(text, meal_keyword_map):
            found_meals = []
            for meal_key, kws in meal_keyword_map.items():
                for kw in kws:
                    if kw in text:
                        if meal_key == "cả_ngày":
                            return ["bữa_sáng", "bữa_trưa", "bữa_tối", "bữa_phụ"]
                        if meal_key not in found_meals:
                            found_meals.append(meal_key)
            return found_meals if found_meals else None

        extracted_params["health_status"] = find_keyword(question_lower, health_status_map)
        extracted_params["goal"] = find_keyword(question_lower, goal_map)
        extracted_params["nutrition_intent"] = find_keyword(question_lower, nutrition_intent_map)
        requested = find_requested_meals(question_lower, meal_map)

        if extracted_params["nutrition_intent"] == "không có":
            if extracted_params["goal"] in ["giảm cân", "tăng cân", "tăng cơ", "giữ cân", "duy trì sức khỏe", "ăn lành mạnh"]:
                extracted_params["nutrition_intent"] = extracted_params["goal"]

        if requested:
            extracted_params["requested_meals"] = requested

        return extracted_params

def recommend_meal_plan(
    health_status: str,
    goal: str,
    nutrition_intent: str,
    requested_meals: list[str] = None
) -> list[dict]:
    """Recommends the single best meal plan based on user's health status, goal, and nutrition intent."""
    load_recommender_artifacts()

    if meal_plans_df is None or user_feature_encoder is None:
        return []

    if not requested_meals:
        requested_meals = ['bữa_sáng', 'bữa_trưa', 'bữa_tối', 'bữa_phụ']

    # Create a DataFrame for the user's input to encode
    user_input_df = pd.DataFrame([{
        'tình_trạng_sức_khỏe': health_status,
        'mục_tiêu': goal,
        'intent_dinh_dưỡng': nutrition_intent
    }])
    
    try:
        user_input_encoded = user_feature_encoder.transform(user_input_df[user_features_cols])
    except ValueError as e:
        print(f"[ERROR] Error encoding user input: {e}")
        return []

    # Calculate similarity between user input and all existing user profiles
    user_similarity_scores = cosine_similarity(user_input_encoded, user_features_encoded_matrix).flatten()
    
    # Get the index of the single best match (highest similarity score)
    best_match_index = user_similarity_scores.argmax()
    
    # Retrieve the corresponding meal plan
    best_plan_raw = meal_plans_df.iloc[best_match_index:best_match_index+1].to_dict(orient='records')

    # Format the output to include only relevant meal plan details
    output_recommendations = []
    for plan in best_plan_raw:
        recommendation = {
            "tình_trạng_sức_khỏe": plan.get('tình_trạng_sức_khỏe', 'N/A'),
            "mục_tiêu": plan.get('mục_tiêu', 'N/A'),
            "intent_dinh_dưỡng": plan.get('intent_dinh_dưỡng', 'N/A'),
            "tổng_calo": plan.get('tổng_calo', 'N/A'),
            "ghi_chú": plan.get('ghi_chú', 'N/A')
        }
        for meal_key in requested_meals:
            recommendation[meal_key] = plan.get(meal_key) or "Không có gợi ý"
        output_recommendations.append(recommendation)
    
    return output_recommendations

async def generate_natural_response_from_recommendations(question: str, recommendations: list[dict]) -> str:
    """Uses an LLM via OpenRouter to generate a natural, conversational response based on the structured meal recommendations."""
    if not recommendations:
        return "Rất tiếc, tôi không tìm thấy thực đơn nào phù hợp với yêu cầu của bạn. Bạn có thể thử lại với một yêu cầu khác nhé."

    # Convert recommendations to a string format for the prompt
    recs_str = ""
    if recommendations:
        rec = recommendations[0]
        for meal_key in ['bữa_sáng', 'bữa_trưa', 'bữa_tối', 'bữa_phụ']:
            if meal_key in rec and rec[meal_key] and rec[meal_key] != "Không có gợi ý":
                recs_str += f"- {meal_key.replace('_', ' ').capitalize()}: {rec[meal_key]}\n"
        recs_str += f"- Tổng calo (ước tính): {rec.get('tổng_calo', 'N/A')} kcal\n"
        recs_str += f"- Ghi chú: {rec.get('ghi_chú', 'Không có')}\n"

    prompt = f"""Bạn là một chuyên gia dinh dưỡng AI thân thiện và am hiểu.
Người dùng vừa hỏi: "{question}"

Dựa trên dữ liệu, đây là gợi ý thực đơn phù hợp nhất cho họ:
{recs_str}
Nhiệm vụ của bạn là diễn giải những gợi ý trên thành một câu trả lời tự nhiên, mượt mà, và mang tính tư vấn cho người dùng.

**Yêu cầu:**
1.  **Bắt đầu thân thiện**: Chào hỏi và tóm tắt lại yêu cầu của người dùng (ví dụ: "Chào bạn, để giúp bạn tăng cơ hiệu quả...").
2.  **Trình bày rõ ràng**: Liệt kê các món ăn gợi ý cho từng bữa một cách mạch lạc.
3.  **Giải thích ngắn gọn**: Nêu lý do tại sao thực đơn này phù hợp (ví dụ: "Thực đơn này tập trung vào các món giàu protein như cá và ức gà, rất tốt cho việc xây dựng cơ bắp...").
4.  **Thêm lời khuyên**: Dựa vào phần "Ghi chú" để đưa ra lời khuyên bổ sung.
5.  **Kết thúc động viên**: Kết thúc bằng một lời chúc hoặc động viên tích cực.

**Quan trọng**: Câu trả lời của bạn phải là một đoạn văn hoàn chỉnh, không sử dụng các ký tự gạch đầu dòng (`-`) hay xuống dòng không cần thiết (`\n`). Hãy viết như một chuyên gia đang trò chuyện trực tiếp với người dùng.
"""
    
    natural_response = await call_openrouter_llm(
        prompt,
        model=os.getenv("OPENROUTER_MODEL_CHAT", "openai/gpt-3.5-turbo"),
        json_mode=False
    )

    if natural_response and isinstance(natural_response, str):
        try:
            cleaned_response = natural_response.replace("\n", " ").replace("- ", "").strip()
            return cleaned_response
        except UnicodeEncodeError:
            safe_response = natural_response.encode('ascii', 'replace').decode('ascii')
            return safe_response.replace("\n", " ").replace("- ", "").strip()
    
    llm_logger.warning("LLM response for natural language generation was not in the expected format. Falling back to raw data.")
    fallback_response = f"Dưới đây là các gợi ý dành cho bạn: {recs_str}"
    cleaned_fallback = fallback_response.replace('- ', '').replace('\n', ', ').strip().rstrip(',').strip()
    return cleaned_fallback
