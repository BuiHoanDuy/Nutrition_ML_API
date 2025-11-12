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
import random
import logging
import asyncio # Import asyncio for sleep
from logging.handlers import RotatingFileHandler
from transformers import AutoTokenizer, AutoModel
import torch 
# ✅ Thêm phần này để load file .env
from dotenv import load_dotenv

# --- Load environment variables from .env file ---
# Xác định đường dẫn tuyệt đối đến file .env trong thư mục gốc của dự án.
env_path = Path(__file__).parent.parent / ".env"
# Nạp các biến môi trường từ file .env. Nếu file không tồn tại, nó sẽ không báo lỗi.
load_dotenv(dotenv_path=env_path)
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

class MealPlanRecommender:
    """
    A class to handle meal plan recommendations, encapsulating all models and data.
    """
    def __init__(self):
        self.meal_plans_df = None
        self.user_feature_encoder = None
        self.user_features_encoded_matrix = None
        self.meal_features_phobert_matrix = None
        self.phobert_model = None
        self.phobert_tokenizer = None
        self.user_features_cols = ['tinh_trang_suc_khoe', 'muc_tieu', 'che_do_an']
        self.last_recommendation_cache = {}
        self._load_artifacts()

    def _load_artifacts(self):
        """Loads all necessary artifacts for the recommender."""
        if self.meal_plans_df is not None:
            return

        print("[INFO] Loading meal plan recommender artifacts...")
        try:
            # Load data and encoders
            self.meal_plans_df = pd.read_csv(MODEL_DIR / "meal_plans_data.csv", encoding='utf-8-sig')
            self.user_feature_encoder = joblib.load(MODEL_DIR / "user_feature_encoder.pkl")
            self.user_features_encoded_matrix = scipy.sparse.load_npz(MODEL_DIR / "user_features_encoded.npz")
            
            # Load PhoBERT embeddings
            self.meal_features_phobert_matrix = np.load(MODEL_DIR / "meal_features_phobert.npy")

            # Load PhoBERT model and tokenizer from local files
            print("[INFO] Loading local PhoBERT model for inference...")
            self.phobert_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR / "phobert_tokenizer")
            self.phobert_model = AutoModel.from_pretrained(MODEL_DIR / "phobert_model")
            self.phobert_model.eval() # Set model to evaluation mode
            
            print("[SUCCESS] Meal plan recommender artifacts loaded successfully.")

        except FileNotFoundError as e:
            print(f"[ERROR] A required model file was not found: {e}")
            print("Please ensure the meal plan model has been trained by running `scripts/train_meal_plan_model.py`.")
            self._reset_artifacts()
        except Exception as e:
            print(f"[ERROR] An unexpected error occurred while loading artifacts: {e}")
            self._reset_artifacts()

    def _reset_artifacts(self):
        """Resets all artifacts to None in case of a loading error."""
        self.meal_plans_df = None
        self.user_feature_encoder = None
        self.user_features_encoded_matrix = None
        self.meal_features_phobert_matrix = None
        self.phobert_model = None
        self.phobert_tokenizer = None

    def is_ready(self) -> bool:
        """Check if all artifacts are loaded and the recommender is ready."""
        return self.meal_plans_df is not None and self.phobert_model is not None

# --- Instantiate the recommender ---
# This creates a single instance that will be reused across requests.
recommender = MealPlanRecommender()

# --- OpenRouter Integration ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-small-latest")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# --- NEW: Instantiate OpenAI client for OpenRouter ---
# This client is configured to use OpenRouter's API endpoint and handles retries automatically.
from openai import AsyncOpenAI, RateLimitError, APIError

if OPENROUTER_API_KEY:
    llm_client = AsyncOpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        max_retries=3, # Automatically retry up to 3 times on transient errors
    )
else:
    llm_client = None
    # Thêm cảnh báo quan trọng nếu không tìm thấy API Key
    print("="*80)
    print("[CẢNH BÁO] Biến môi trường OPENROUTER_API_KEY không được tìm thấy hoặc không hợp lệ.")
    print("Hãy đảm bảo bạn đã tạo file .env và đặt API key vào đó.")
    print("="*80)

async def _call_openrouter_llm(prompt: str, model: str = OPENROUTER_MODEL, json_mode: bool = True) -> dict | str | None:
    """Calls the OpenRouter LLM API using the openai library to get a structured response."""
    if not llm_client:
        llm_logger.warning("OPENROUTER_API_KEY not set. Skipping LLM call.")
        return None

    request_params = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1 if json_mode else 0.7
    }
    if json_mode:
        request_params["response_format"] = {"type": "json_object"}

    try:
        response = await llm_client.chat.completions.create(**request_params)
        llm_response_content = response.choices[0].message.content

        try:
            llm_logger.info(f"LLM call successful. Model: {model}, Prompt: {prompt[:100]}..., Raw Response: {llm_response_content}")
        except UnicodeEncodeError:
            safe_prompt = prompt[:100].encode('ascii', 'replace').decode('ascii')
            safe_response = llm_response_content.encode('ascii', 'replace').decode('ascii')
            llm_logger.info(f"LLM call successful. Model: {model}, Prompt: {safe_prompt}..., Raw Response: {safe_response}")

        if json_mode:
            return json.loads(llm_response_content)
        else:
            return llm_response_content.strip()

    except RateLimitError as e:
        llm_logger.error(f"Rate limit hit (429) calling OpenRouter LLM. The library's retries were exhausted. Error: {e}")
        return None
    except APIError as e:
        llm_logger.error(f"API Error from OpenRouter LLM: {e.status_code} - {e.message}")
        return None
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        llm_logger.error(f"Error parsing LLM response or malformed response: {e}")
        return None
    except Exception as e:
        llm_logger.error(f"An unexpected error occurred during LLM call: {e}")
        return None

async def parse_meal_plan_question(question: str) -> dict:
    """Parses a natural language question to extract health_status, goal, and nutrition_intent."""
    llm_prompt = f"""Bạn là một trợ lý dinh dưỡng. Hãy phân tích câu hỏi sau của người dùng và trích xuất các thông tin sau vào định dạng JSON:
- `health_status`: Tình trạng sức khỏe (ví dụ: 'béo phì', 'tiểu đường', 'bình thường', 'không có').
- `goal`: Mục tiêu (ví dụ: 'giảm cân', 'tăng cân', 'tăng cơ', 'giữ cân', 'không có').
- `diet_type`: Loại chế độ ăn đặc biệt nếu có (ví dụ: 'chay', 'keto', 'paleo', 'không có'). Nếu trong câu hỏi có từ 'chay' hoặc 'ăn chay', hãy đặt giá trị này là 'chay'.
- `requested_meals`: Danh sách các bữa ăn được yêu cầu (ví dụ: ['Bữa sáng', 'Bữa trưa', 'Bữa tối', 'Bữa phụ']). Nếu không rõ, mặc định là tất cả.

Nếu không tìm thấy thông tin cụ thể, hãy sử dụng giá trị 'không có' cho các trường `health_status` và `goal`.
Đối với `requested_meals`, nếu không có yêu cầu cụ thể, hãy mặc định là `['Bữa sáng', 'Bữa trưa', 'Bữa tối', 'Bữa phụ']`. Nếu người dùng hỏi về "cả ngày", hãy trả về tất cả các bữa.

Câu hỏi của người dùng: '{question}'

Vui lòng chỉ trả về một đối tượng JSON hợp lệ.
"""
    llm_parsed_params = await _call_openrouter_llm(llm_prompt)

    if llm_parsed_params and isinstance(llm_parsed_params, dict):
        try:
            llm_logger.info(f"LLM successfully parsed question: {question} -> {llm_parsed_params}")
        except UnicodeEncodeError:
            safe_question = question.encode('ascii', 'replace').decode('ascii')
            llm_logger.info(f"LLM successfully parsed question: {safe_question} -> {llm_parsed_params}")
        
        final_params = {
            "health_status": llm_parsed_params.get("health_status", "không có"),
            "goal": llm_parsed_params.get("goal", "không có"),
            "diet_type": llm_parsed_params.get("diet_type", "không có"),
            "requested_meals": llm_parsed_params.get("requested_meals", ["Bữa sáng", "Bữa trưa", "Bữa tối", "Bữa phụ"])
        }
        
        if not isinstance(final_params["requested_meals"], list):
            llm_logger.warning(f"LLM returned non-list for requested_meals. Defaulting. Raw: {llm_parsed_params.get('requested_meals')}")
            final_params["requested_meals"] = ["Bữa sáng", "Bữa trưa", "Bữa tối", "Bữa phụ"]
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
            "diet_type": "không có",
            "requested_meals": ["Bữa sáng", "Bữa trưa", "Bữa tối", "Bữa phụ"]
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

        diet_type_map = {
            "chay": ["chay", "ăn chay", "thuần chay", "vegan"],
        }

        meal_map = {
            "Bữa sáng": ["bữa sáng", "buổi sáng", "sáng"],
            "Bữa trưa": ["bữa trưa", "buổi trưa", "trưa"],
            "Bữa tối": ["bữa tối", "buổi tối", "tối"],
            "Bữa phụ": ["bữa phụ", "ăn vặt", "ăn nhẹ"],
            "cả_ngày": ["cả ngày", "một ngày", "1 ngày"]
        }

        def find_keyword(text, keyword_map):
            for category, kws in keyword_map.items():
                kws_sorted = sorted(kws, key=len, reverse=True)
                for kw in kws_sorted: # Use word boundaries for more precise matching
                    if kw in text:
                        return category
            return "không có"

        def find_requested_meals(text, meal_keyword_map):
            found_meals = []
            for meal_key, kws in meal_keyword_map.items():
                for kw in kws:
                    if kw in text:
                        if meal_key == "cả_ngày":
                            return ["Bữa sáng", "Bữa trưa", "Bữa tối", "Bữa phụ"]
                        if meal_key not in found_meals:
                            found_meals.append(meal_key)
            return found_meals if found_meals else None

        extracted_params["health_status"] = find_keyword(question_lower, health_status_map)
        extracted_params["goal"] = find_keyword(question_lower, goal_map)
        extracted_params["diet_type"] = find_keyword(question_lower, diet_type_map)
        requested = find_requested_meals(question_lower, meal_map)

        if requested:
            extracted_params["requested_meals"] = requested

        return extracted_params

def _get_query_embedding(text: str, model, tokenizer) -> np.ndarray:
    """Generates a PhoBERT embedding for a single query text."""
    if not model or not tokenizer:
        raise RuntimeError("PhoBERT model is not loaded. Call load_recommender_artifacts() first.")
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

def recommend_meal_plan(
    original_question: str, # Thêm câu hỏi gốc làm tham số
    health_status: str,
    goal: str,
    diet_type: str,
    requested_meals: list[str] = None
) -> list[dict]:
    """Recommends a suitable meal plan based on user's query and profile."""
    if not recommender.is_ready():
        print("[ERROR] Recommender is not ready. Artifacts may have failed to load.")
        return []

    if not requested_meals:
        requested_meals = ['Bữa sáng', 'Bữa trưa', 'Bữa tối', 'Bữa phụ']

    # --- NEW: Early exit if no meaningful parameters are extracted ---
    # If the question is irrelevant (e.g., "Tôi bị khùng"), all parameters will be 'không có'.
    # In this case, we can confidently say we can't find a suitable meal plan.
    if health_status == 'không có' and goal == 'không có' and diet_type == 'không có':
        print("[INFO] No relevant health, goal, or diet keywords found in the question. Returning no results.")
        return []

    # Create a DataFrame for the user's input to encode
    user_input_df = pd.DataFrame([{
        'Tình trạng sức khỏe': health_status,
        'Mục tiêu': goal,
        'Chế độ ăn': diet_type
    }])
    
    try:
        # Rename columns to match the names the encoder was trained on
        df_to_encode = user_input_df.rename(columns={
            'Tình trạng sức khỏe': 'tinh_trang_suc_khoe',
            'Mục tiêu': 'muc_tieu',
            'Chế độ ăn': 'che_do_an'
        })
        # This is not used for ranking anymore, but kept for potential future use
        user_input_encoded = recommender.user_feature_encoder.transform(df_to_encode[recommender.user_features_cols])
    except ValueError as e:
        print(f"[ERROR] Error encoding user input: {e}")
        return []

    # --- NEW LOGIC: Filter-then-Rank ---
    # 1. Hard-filter the dataset based on extracted attributes
    print("[INFO] Applying hard filters based on extracted attributes...")
    filtered_df = recommender.meal_plans_df.copy()

    if diet_type and diet_type != 'không có':
        filtered_df = filtered_df[filtered_df['Chế độ ăn'].str.lower().str.strip() == diet_type.lower()]

    if health_status and health_status != 'không có':
        filtered_df = filtered_df[filtered_df['Tình trạng sức khỏe'].str.lower().str.strip() == health_status.lower()]

    if goal and goal != 'không có':
        filtered_df = filtered_df[filtered_df['Mục tiêu'].str.lower().str.strip() == goal.lower()]

    if filtered_df.empty:
        print("[INFO] No meal plans found after applying hard filters.")
        return []
    
    print(f"[INFO] Found {len(filtered_df)} candidates after filtering.")

    # 2. Perform semantic search on the filtered subset
    query_text = original_question
    query_embedding = _get_query_embedding(query_text, recommender.phobert_model, recommender.phobert_tokenizer)

    # Get embeddings only for the filtered candidates
    filtered_indices = filtered_df.index
    filtered_embeddings = recommender.meal_features_phobert_matrix[filtered_indices]

    # Calculate similarity scores against the filtered subset
    similarity_scores = cosine_similarity(query_embedding, filtered_embeddings).flatten()

    # 3. Rank the filtered candidates
    top_indices = similarity_scores.argsort()[::-1]

    SIMILARITY_THRESHOLD = 0.3 
    best_score = similarity_scores[top_indices[0]]

    if best_score < SIMILARITY_THRESHOLD:
        print(f"[INFO] Best match score ({best_score:.2f}) is below threshold ({SIMILARITY_THRESHOLD}). Returning no results.")
        return [] # Return empty list if no good match is found

    # Filter the meal plans to only those that have a suggestion for at least one of the requested meals.
    # This ensures we don't return an empty plan.
    # Map local filtered indices back to original DataFrame indices.
    # `top_indices` are the positions within the `filtered_df`. We use them to get the original indices.
    original_relevant_indices = [filtered_indices[i] for i in top_indices]
    
    if not original_relevant_indices:
        return [] # No suitable plan found after filtering and ranking
    
    # --- NEW LOGIC: Handle first vs. subsequent requests for the same question ---
    # Check if this is a subsequent request for the same question
    if original_question in recommender.last_recommendation_cache:
        # This is a subsequent request. Try to find the *next best* result.
        last_shown_index = recommender.last_recommendation_cache[original_question]

        # Find all other candidates, excluding the one shown last time
        other_candidates = [
            idx for idx in original_relevant_indices 
            if idx != last_shown_index
        ]
        
        if other_candidates:
            # --- LOGIC CHANGE: Randomly select from other candidates ---
            best_match_index = random.choice(other_candidates)
            print(f"[INFO] Subsequent request. Found {len(other_candidates)} other candidates. Randomly selecting index: {best_match_index}")
        else:
            # No other options available, so we can show the same one again.
            best_match_index = last_shown_index
            print(f"[INFO] Subsequent request. No other candidates found. Returning previous best index: {best_match_index}")
    else:
        # --- LOGIC CHANGE: Randomly select from all relevant candidates for the first request ---
        best_match_index = random.choice(original_relevant_indices)
        print(f"[INFO] First request. Found {len(original_relevant_indices)} candidates. Randomly selecting index: {best_match_index}")

    # Update the cache with the index we are about to show
    recommender.last_recommendation_cache[original_question] = best_match_index
    
    # Retrieve the corresponding meal plan
    best_plans_raw = recommender.meal_plans_df.iloc[best_match_index:best_match_index+1].to_dict(orient='records')

    # Format the output to include only relevant meal plan details
    output_recommendations = []
    for plan in best_plans_raw:
        recommendation = {}
        for meal_key in requested_meals:
            meal_value = plan.get(meal_key)
            # Check if meal_value is NaN or None
            if pd.isna(meal_value):
                recommendation[meal_key] = "Không có gợi ý"
            else:
                recommendation[meal_key] = meal_value
        output_recommendations.append(recommendation)
    
    return output_recommendations

async def generate_natural_response_from_recommendations(
    question: str,
    health_status: str, # Thêm tham số health_status
    goal: str,           # Thêm tham số goal
    recommendations: list[dict]
) -> str:
    """Uses an LLM via OpenRouter to generate a natural, conversational response based on the structured meal recommendations."""
    if not recommendations:
        return "Rất tiếc, tôi không tìm thấy thực đơn nào phù hợp với yêu cầu của bạn. Bạn có thể thử lại với một yêu cầu khác nhé."

    # Convert recommendations to a string format for the prompt
    recs_parts = []
    if recommendations:
        rec = recommendations[0]
        for meal_key in ['Bữa sáng', 'Bữa trưa', 'Bữa tối', 'Bữa phụ']:
            if meal_key in rec and rec[meal_key] and rec[meal_key] != "Không có gợi ý":
                # --- NEW: Replace semicolons with commas for better readability ---
                meal_content = rec[meal_key].replace(';', ',')
                recs_parts.append(f"{meal_key}: {meal_content}")
    recs_str = "; ".join(recs_parts)
    if not recs_str:
        recs_str = "Không có gợi ý cụ thể cho các bữa ăn."

    # Construct a more informative prompt for the LLM
    health_goal_context = ""
    if health_status != 'không có' and goal != 'không có':
        health_goal_context = f"Họ có tình trạng sức khỏe là '{health_status}' và mục tiêu là '{goal}'."
    elif health_status != 'không có':
        health_goal_context = f"Họ có tình trạng sức khỏe là '{health_status}'."
    elif goal != 'không có':
        health_goal_context = f"Họ có mục tiêu là '{goal}'."

    prompt = f"""Bạn là một chuyên gia dinh dưỡng AI thân thiện và am hiểu, có khả năng tư vấn cá nhân hóa.
Người dùng vừa hỏi: "{question}"
${health_goal_context}

Dựa trên dữ liệu, đây là gợi ý thực đơn phù hợp nhất cho họ:
{recs_str}
Nhiệm vụ của bạn là diễn giải những gợi ý trên thành một câu trả lời tự nhiên, mượt mà, và mang tính tư vấn cho người dùng.

**Yêu cầu:**
1.  **Bắt đầu thân thiện**: Chào hỏi và tóm tắt lại yêu cầu của người dùng, bao gồm tình trạng sức khỏe và mục tiêu của họ (nếu có). Ví dụ: "Chào bạn, với tình trạng sức khỏe [tình trạng] và mục tiêu [mục tiêu], tôi gợi ý...". Nếu không có thông tin cụ thể về sức khỏe/mục tiêu, hãy bắt đầu bằng cách chào hỏi và đề cập đến câu hỏi chung của họ.
2.  **Trình bày rõ ràng**: Liệt kê các món ăn gợi ý cho từng bữa một cách mạch lạc.
3.  **Giải thích ngắn gọn**: Nêu lý do tại sao thực đơn này phù hợp với tình trạng sức khỏe và mục tiêu của họ (nếu có). Ví dụ: "Thực đơn này tập trung vào các món giàu protein nạc và rau xanh, rất tốt cho việc xây dựng cơ bắp...". Nếu không có thông tin gì thêm, chỉ cần trình bày thực đơn.
4.  **Thêm lời khuyên (nếu có)**: Nếu có thông tin bổ sung, hãy đưa ra lời khuyên.
5.  **Kết thúc động viên**: Kết thúc bằng một lời chúc hoặc động viên tích cực.

**Quan trọng**: Câu trả lời của bạn phải là một đoạn văn hoàn chỉnh, không sử dụng các ký tự gạch đầu dòng (`-`) hay xuống dòng không cần thiết (`\n`). Hãy viết như một chuyên gia đang trò chuyện trực tiếp với người dùng.
"""
    
    natural_response = await _call_openrouter_llm(
        prompt,
        model=os.getenv("OPENROUTER_MODEL_CHAT", "google/gemma-7b-it"),
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
    # --- CRITICAL FIX: Improve fallback response formatting ---
    # The original recs_str is "Bữa sáng: Món A; Bữa trưa: Món B". We want to make it more readable.
    # Replace the first colon with a space and subsequent semicolons with newlines for better display.
    readable_recs = recs_str.replace(":", " ", 1).replace(";", ". ")
    fallback_response = f"Dưới đây là các gợi ý dành cho bạn: {readable_recs}."
    return fallback_response
