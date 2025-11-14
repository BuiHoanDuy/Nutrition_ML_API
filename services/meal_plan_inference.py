"""
Meal plan inference service.
"""
# Standard library imports
import os
import json
import random
import logging
import ssl
import certifi
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Third-party imports
import joblib
import numpy as np
import pandas as pd
import scipy.sparse
import google.generativeai as genai
import torch
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel

# --- SSL Certificate Configuration ---
# Set SSL_CERT_FILE to use certifi's certificate bundle
# This is a robust way to handle SSL verification errors in corporate/firewalled environments.
os.environ['SSL_CERT_FILE'] = certifi.where()

# --- Constants ---
SIMILARITY_THRESHOLD = 0.4
DEFAULT_MEALS = ['Bữa sáng', 'Bữa trưa', 'Bữa tối', 'Bữa phụ']

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

# --- Setup Logging for Meal Plan Requests ---
meal_plan_logger = logging.getLogger("meal_plan_requests")
meal_plan_logger.setLevel(logging.INFO)
meal_plan_logger.propagate = False
meal_plan_handler = RotatingFileHandler(
    LOGS_DIR / "meal_plan_requests.log", maxBytes=10_000_000, backupCount=5, encoding="utf-8"
)
meal_plan_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
meal_plan_handler.setFormatter(meal_plan_formatter)
meal_plan_logger.addHandler(meal_plan_handler)

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
recommender = MealPlanRecommender()

# --- Gemini Integration ---
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") # SỬA LỖI: Thống nhất tên biến môi trường
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash") # Cảnh báo: Model này có thể chưa có sẵn qua API.

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    llm_client = genai.GenerativeModel(
        GEMINI_MODEL_NAME,
        generation_config={"temperature": 0.1},
        safety_settings={ # Disable all safety settings for more control
            'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
            'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
            'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
            'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
        }
    )
else:
    llm_client = None
    print("="*80)
    print("[CẢNH BÁO] Biến môi trường GOOGLE_API_KEY không được tìm thấy hoặc không hợp lệ.")
    print("Hãy đảm bảo bạn đã tạo file .env và đặt API key vào đó.")
    print("="*80)

async def _call_gemini_llm(prompt: str, json_mode: bool = False) -> dict | str | None:
    """Calls the Gemini LLM API to get a structured response."""
    if not llm_client:
        llm_logger.warning("GOOGLE_API_KEY not set. Skipping LLM call.")
        return None

    # Gemini uses generation_config for parameters
    request_params = {
        "generation_config": {"temperature": 0.2 if json_mode else 0.7}
    }
    if json_mode:
        # For Gemini, we specify the response format in generation_config
        request_params["generation_config"]["response_mime_type"] = "application/json"

    try:
        # Use the correct async method for Gemini
        response = await llm_client.generate_content_async(
            contents=prompt,
            generation_config=request_params["generation_config"]
        )
        llm_response_content = response.text

        try:
            llm_logger.info(f"LLM call successful. Model: {GEMINI_MODEL_NAME}, Prompt: {prompt[:100]}..., Raw Response: {llm_response_content}")
        except UnicodeEncodeError:
            safe_prompt = prompt[:100].encode('ascii', 'replace').decode('ascii')
            safe_response = llm_response_content.encode('ascii', 'replace').decode('ascii')
            llm_logger.info(f"LLM call successful. Model: {GEMINI_MODEL_NAME}, Prompt: {safe_prompt}..., Raw Response: {safe_response}")

        if json_mode:
            return json.loads(llm_response_content)
        else:
            return llm_response_content.strip()

    except (json.JSONDecodeError, ValueError) as e:
        llm_logger.error(f"Error parsing LLM response or malformed response: {e}")
        return None
    except Exception as e:
        # Catching Gemini's rate limit error
        if "Resource has been exhausted" in str(e) or "429" in str(e):
            llm_logger.error(f"Rate limit hit calling Gemini LLM. Error: {e}")
        else:
            llm_logger.error(f"An unexpected error occurred during LLM call: {e}")
        return None

def _parse_question_with_keywords(question_lower: str) -> dict:
    """
    Parse a question using advanced keyword matching algorithm.
    Supports multiple health conditions and goals in a single query.
    """
    extracted_params = {
        "health_status": "không có",
        "goal": "không có",
        "diet_type": "không có",
        "requested_meals": DEFAULT_MEALS
    }

    # Define comprehensive keywords for each category
    # Updated with the complete list of health conditions
    health_status_map = {
        "Táo bón": ["táo bón", "bị táo bón", "mắc táo bón", "táo bón mãn tính"],
        "Béo phì": ["béo phì", "thừa cân", "béo", "bị béo phì", "mắc béo phì", "béo phì độ"],
        "Tim mạch": ["tim mạch", "bệnh tim", "vấn đề tim mạch", "bệnh tim mạch", "tim"],
        "Tăng huyết áp": ["tăng huyết áp", "cao huyết áp", "huyết áp cao", "bị cao huyết áp", "mắc cao huyết áp"],
        "Tiểu đường": ["tiểu đường", "đái tháo đường", "đường huyết cao", "bị tiểu đường", "mắc tiểu đường"],
        "Thiếu kẽm": ["thiếu kẽm", "thiếu chất kẽm", "bị thiếu kẽm"],
        "Thiếu máu": ["thiếu máu", "bị thiếu máu", "mắc thiếu máu", "thiếu hồng cầu"],
        "Suy dinh dưỡng": ["suy dinh dưỡng", "bị suy dinh dưỡng", "mắc suy dinh dưỡng", "thiếu dinh dưỡng"],
        "Thiếu canxi": ["thiếu canxi", "thiếu chất canxi", "bị thiếu canxi", "thiếu can xi"],
        "Rối loạn mỡ máu": ["rối loạn mỡ máu", "mỡ máu cao", "cholesterol cao", "rối loạn lipid máu", "mỡ máu"],
        "Người mệt mỏi": ["mệt mỏi", "người mệt mỏi", "hay mệt mỏi", "thường xuyên mệt mỏi", "mệt"],
        "Thiếu vitamin": ["thiếu vitamin", "thiếu vi chất", "thiếu vitamin và khoáng chất", "thiếu vitamin"],
        "Loãng xương": ["loãng xương", "bị loãng xương", "mắc loãng xương", "xương yếu"],
        "Trẻ em": ["trẻ em", "trẻ nhỏ", "bé", "con nhỏ", "trẻ", "em bé"],
        "Phụ nữ mang thai": ["phụ nữ mang thai", "bà bầu", "mang thai", "có thai", "thai phụ", "mẹ bầu"],
        "Người khỏe mạnh": ["người khỏe mạnh", "bình thường", "khỏe mạnh", "người bình thường", "sức khỏe tốt"]
    }
    
    # Updated with the complete list of goals
    goal_map = {
        "kiểm soát năng lượng": ["kiểm soát năng lượng", "kiểm soát calo", "kiểm soát lượng calo", "kiểm soát calori"],
        "Giảm cân": ["giảm cân", "muốn giảm cân", "cần giảm cân", "giảm béo", "slim", "giảm trọng lượng"],
        "Tăng chất xơ": ["tăng chất xơ", "bổ sung chất xơ", "nhiều chất xơ", "ăn nhiều xơ", "chất xơ"],
        "hạn chế chất béo xấu": ["hạn chế chất béo xấu", "giảm chất béo xấu", "ít chất béo xấu", "tránh chất béo xấu", "chất béo xấu"],
        "ổn định huyết áp": ["ổn định huyết áp", "huyết áp ổn định", "kiểm soát huyết áp", "điều hòa huyết áp"],
        "Ổn định đường huyết": ["ổn định đường huyết", "đường huyết ổn định", "kiểm soát đường huyết", "điều hòa đường huyết"],
        "uống đủ nước": ["uống đủ nước", "đủ nước", "bổ sung nước", "nhiều nước", "đủ chất lỏng"],
        "Giảm muối": ["giảm muối", "ít muối", "hạn chế muối", "giảm natri", "ít natri", "giảm bớt muối", "muối"],
        "Bổ sung vitamin và khoáng chất": ["bổ sung vitamin và khoáng chất", "vitamin và khoáng chất", "bổ sung vitamin", "bổ sung khoáng chất", "vitamin khoáng chất"],
        "Giảm mỡ bão hòa": ["giảm mỡ bão hòa", "ít mỡ bão hòa", "hạn chế mỡ bão hòa", "chất béo bão hòa", "mỡ bão hòa"],
        "Tăng cường năng lượng": ["tăng cường năng lượng", "nhiều năng lượng", "bổ sung năng lượng", "tăng năng lượng", "năng lượng"],
        "Bổ sung sắt": ["bổ sung sắt", "nhiều sắt", "thêm sắt", "sắt", "thiếu sắt"],
        "tăng cường máu": ["tăng cường máu", "bổ máu", "tạo máu", "tăng hồng cầu", "máu"],
        "Tăng cân": ["tăng cân", "muốn tăng cân", "cần tăng cân", "tăng trọng lượng", "tăng ký"],
        "bổ sung năng lượng và đạm": ["bổ sung năng lượng và đạm", "năng lượng và đạm", "nhiều đạm", "protein", "đạm"],
        "Bổ sung canxi và D": ["bổ sung canxi và d", "canxi và vitamin d", "canxi d", "vitamin d và canxi", "canxi vitamin d"],
        "Hỗ trợ phát triển toàn diện": ["hỗ trợ phát triển toàn diện", "phát triển toàn diện", "phát triển", "tăng trưởng"],
        "cân bằng dinh dưỡng": ["cân bằng dinh dưỡng", "dinh dưỡng cân bằng", "đầy đủ dinh dưỡng", "dinh dưỡng"],
        "Duy trì sức khỏe": ["duy trì sức khỏe", "sức khỏe tốt", "giữ sức khỏe", "bảo vệ sức khỏe"],
        "Bổ sung dinh dưỡng cho thai kỳ": ["bổ sung dinh dưỡng cho thai kỳ", "dinh dưỡng thai kỳ", "dinh dưỡng cho bà bầu", "dinh dưỡng mang thai"]
    }
    
    diet_type_map = {
        "chay": ["chay", "ăn chay", "thuần chay", "vegan", "vegetarian"],
    }
    
    meal_map = {
        "Bữa sáng": ["bữa sáng", "buổi sáng", "sáng", "bữa ăn sáng"],
        "Bữa trưa": ["bữa trưa", "buổi trưa", "trưa", "bữa ăn trưa"],
        "Bữa tối": ["bữa tối", "buổi tối", "tối", "bữa ăn tối"],
        "Bữa phụ": ["bữa phụ", "ăn vặt", "ăn nhẹ", "snack"],
        "cả_ngày": ["cả ngày", "một ngày", "1 ngày", "trong ngày", "suốt ngày"]
    }

    def find_keywords_multiple(text, keyword_map):
        """
        Find ALL matching keywords in the text, not just the first one.
        Returns a comma-separated string of matched categories, or "không có" if none found.
        Uses longest-match-first strategy to avoid partial matches.
        """
        found_categories = set()  # Use set to avoid duplicates
        
        # Sort all keywords by length (longest first) to match longer phrases first
        # This prevents "cao" from matching when "cao huyết áp" should match
        all_keywords = []
        for category, kws in keyword_map.items():
            for kw in kws:
                all_keywords.append((category, kw, len(kw)))
        
        # Sort by length descending, then by category to ensure consistency
        all_keywords.sort(key=lambda x: (-x[2], x[0]))
        
        # Track matched positions to avoid overlapping matches
        matched_ranges = []
        
        for category, kw, kw_len in all_keywords:
            # Check if this keyword appears in text
            start_pos = text.find(kw)
            if start_pos != -1:
                end_pos = start_pos + kw_len
                
                # Check if this range significantly overlaps with already matched ranges
                # Allow some overlap (up to 30% of shorter keyword) to handle cases like
                # "cao huyết áp" and "huyết áp cao"
                overlaps = False
                for matched_start, matched_end in matched_ranges:
                    overlap_start = max(start_pos, matched_start)
                    overlap_end = min(end_pos, matched_end)
                    if overlap_end > overlap_start:
                        overlap_len = overlap_end - overlap_start
                        min_len = min(kw_len, matched_end - matched_start)
                        # If overlap is more than 30% of the shorter keyword, skip
                        if overlap_len > min_len * 0.3:
                            overlaps = True
                            break
                
                if not overlaps:
                    found_categories.add(category)
                    matched_ranges.append((start_pos, end_pos))
        
        if found_categories:
            # Return comma-separated string, sorted for consistency
            return ", ".join(sorted(found_categories))
        return "không có"

    def find_requested_meals(text, meal_keyword_map):
        found_meals = []
        for meal_key, kws in meal_keyword_map.items():
            for kw in kws:
                if kw in text:
                    if meal_key == "cả_ngày":
                        return DEFAULT_MEALS
                    if meal_key not in found_meals:
                        found_meals.append(meal_key)
        return found_meals if found_meals else None

    # Use the improved multi-keyword matching
    extracted_params["health_status"] = find_keywords_multiple(question_lower, health_status_map)
    extracted_params["goal"] = find_keywords_multiple(question_lower, goal_map)
    extracted_params["diet_type"] = find_keywords_multiple(question_lower, diet_type_map)
    
    requested = find_requested_meals(question_lower, meal_map)
    if requested:
        extracted_params["requested_meals"] = requested

    return extracted_params

async def _is_meal_plan_request(question: str) -> bool:
    """
    Uses a quick LLM call to determine if the user's query is related to meal planning.
    This acts as a guardrail to prevent irrelevant questions from being processed.
    """
    if not llm_client:
        # If LLM is not available, assume it's a valid request and let the old logic handle it.
        return True

    prompt = f"""
    Phân tích câu hỏi của người dùng. Câu hỏi này có liên quan đến thực phẩm, dinh dưỡng, bữa ăn, hoặc gợi ý thực đơn không?
    Chỉ trả lời "CÓ" hoặc "KHÔNG".

    Câu hỏi: "{question}"
    """
    try:
        # Use a simpler, faster model call if possible. Here we use the main client.
        response_text = await _call_gemini_llm(prompt, json_mode=False)
        
        if response_text and "có" in response_text.lower():
            return True
        return False
    except Exception as e:
        llm_logger.error(f"Error during intent detection for question '{question}': {e}")
        return True # Default to true to avoid blocking valid requests on error

async def parse_meal_plan_question(question: str) -> dict:
    """
    Parses a natural language question to extract health_status, goal, and nutrition_intent.
    Uses advanced keyword matching algorithm instead of LLM for extraction.
    LLM is only used for intent detection (guardrail) and response generation.
    """
    # --- GUARDRAIL: Check intent first using LLM ---
    is_valid_intent = await _is_meal_plan_request(question)
    if not is_valid_intent:
        llm_logger.info(f"Query rejected by intent detection: '{question}'")
        # Return a special flag to indicate an invalid question
        return {"invalid_intent": True}

    # --- Use keyword matching algorithm for extraction ---
    # This is more reliable and faster than LLM extraction
    parsed_params = _parse_question_with_keywords(question.lower())
    
    try:
        llm_logger.info(f"Keyword matching parsed question: {question} -> {parsed_params}")
    except UnicodeEncodeError:
        safe_question = question.encode('ascii', 'replace').decode('ascii')
        llm_logger.info(f"Keyword matching parsed question: {safe_question} -> {parsed_params}")
    
    # --- Log extracted parameters to meal_plan_requests.log ---
    try:
        meal_plan_logger.info("=" * 80)
        meal_plan_logger.info(f"Question: {question}")
        meal_plan_logger.info(f"Tình trạng sức khỏe: {parsed_params.get('health_status', 'không có')}")
        meal_plan_logger.info(f"Mục tiêu: {parsed_params.get('goal', 'không có')}")
        meal_plan_logger.info(f"Chế độ ăn: {parsed_params.get('diet_type', 'không có')}")
        meal_plan_logger.info(f"Các bữa được yêu cầu: {parsed_params.get('requested_meals', DEFAULT_MEALS)}")
    except UnicodeEncodeError:
        safe_question = question.encode('ascii', 'replace').decode('ascii')
        meal_plan_logger.info("=" * 80)
        meal_plan_logger.info(f"Question: {safe_question}")
        meal_plan_logger.info(f"Tình trạng sức khỏe: {parsed_params.get('health_status', 'không có')}")
        meal_plan_logger.info(f"Mục tiêu: {parsed_params.get('goal', 'không có')}")
        meal_plan_logger.info(f"Chế độ ăn: {parsed_params.get('diet_type', 'không có')}")
        meal_plan_logger.info(f"Các bữa được yêu cầu: {parsed_params.get('requested_meals', DEFAULT_MEALS)}")
    
    return parsed_params

def _get_query_embedding(text: str, model, tokenizer) -> np.ndarray:
    """Generates a PhoBERT embedding for a single query text."""
    if not model or not tokenizer:
        raise RuntimeError("PhoBERT model is not loaded. Call load_recommender_artifacts() first.")
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

def recommend_meal_plan(
    original_question: str,
    parsed_params: dict
) -> list[dict]:
    """Recommends a suitable meal plan based on user's query and profile."""
    # Lấy các giá trị từ dictionary, sử dụng .get() để tránh KeyError
    health_status = parsed_params.get("health_status")
    goal = parsed_params.get("goal")
    diet_type = parsed_params.get("diet_type")
    requested_meals = parsed_params.get("requested_meals", DEFAULT_MEALS)

    # Handle the case where the intent was invalid from the parsing step
    if parsed_params.get("invalid_intent"):
        return []

    if not recommender.is_ready():
        print("[ERROR] Recommender is not ready. Artifacts may have failed to load.")
        return []

    if not requested_meals:
        requested_meals = DEFAULT_MEALS

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

    conditions = []
    # SỬA LỖI LOGIC: Xử lý nhiều giá trị trong một trường (ví dụ: "béo phì, tiểu đường")
    # Thay vì tìm kiếm sự trùng khớp chính xác, chúng ta sẽ tìm kiếm sự tồn tại của bất kỳ từ khóa nào.
    if diet_type and diet_type != 'không có':
        # Chế độ ăn thường là một giá trị duy nhất, giữ nguyên logic so sánh bằng.
        conditions.append(filtered_df['Chế độ ăn'].str.lower().str.strip() == diet_type.lower().strip())

    if health_status and health_status != 'không có':
        # Tách các tình trạng sức khỏe (ví dụ: "béo phì, tiểu đường" -> ["béo phì", "tiểu đường"])
        health_keywords = [kw.strip() for kw in health_status.lower().split(',')]
        # Tạo điều kiện OR: cột phải chứa BẤT KỲ từ khóa nào trong danh sách.
        health_condition = filtered_df['Tình trạng sức khỏe'].str.lower().str.contains('|'.join(health_keywords), na=False)
        conditions.append(health_condition)

    if goal and goal != 'không có':
        # Tương tự cho mục tiêu
        goal_keywords = [kw.strip() for kw in goal.lower().split(',')]
        goal_condition = filtered_df['Mục tiêu'].str.lower().str.contains('|'.join(goal_keywords), na=False)
        conditions.append(goal_condition)

    # Only apply filter if there are specific conditions.
    if conditions:
        filtered_df = filtered_df[np.logical_and.reduce(conditions)]
        if filtered_df.empty:
            print("[INFO] No meal plans found after applying hard filters.")
            return []
    else:
        print("[INFO] Generic query. Skipping hard filters and performing semantic search on the entire dataset.")
    
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
    
    # SỬA LỖI: Nếu chỉ có 1 ứng viên, luôn trả về ứng viên đó.
    if len(original_relevant_indices) == 1:
        best_match_index = original_relevant_indices[0]
        print(f"[INFO] Only one candidate found. Consistently returning index: {best_match_index}")

    if original_question in recommender.last_recommendation_cache:
        # This is a subsequent request. Try to find the *next best* result.
        last_shown_index = recommender.last_recommendation_cache[original_question]
        try:
            # Find the position of the last shown index in our list of candidates
            current_pos = original_relevant_indices.index(last_shown_index)
            # Get the next position, wrapping around if at the end of the list
            next_pos = (current_pos + 1) % len(original_relevant_indices)
            best_match_index = original_relevant_indices[next_pos]
            print(f"[INFO] Subsequent request. Showing next candidate at index: {best_match_index}")
        except ValueError:
            # The last shown index is not in the current candidate list, so start from the beginning
            best_match_index = original_relevant_indices[0]
            print(f"[INFO] Subsequent request. Last shown index not in new candidates. Starting over at index: {best_match_index}")

    else:
        # This is the first request for this question.
        # We can take the top result directly or randomize if needed. Let's take the top one for consistency.
        if original_relevant_indices:
            best_match_index = original_relevant_indices[0]
            print(f"[INFO] First request. Selecting top candidate at index: {best_match_index}")
        else:
            return [] # Should not happen if we passed the earlier checks, but as a safeguard.

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
    
    # --- Log meal plan recommendations to meal_plan_requests.log ---
    try:
        meal_plan_logger.info("Thực đơn được gợi ý:")
        for rec in output_recommendations:
            for meal_key, meal_value in rec.items():
                if meal_value and meal_value != "Không có gợi ý":
                    meal_plan_logger.info(f"  {meal_key}: {meal_value}")
                else:
                    meal_plan_logger.info(f"  {meal_key}: Không có gợi ý")
        meal_plan_logger.info("=" * 80)
    except Exception as e:
        meal_plan_logger.error(f"Error logging meal plan recommendations: {e}")
    
    return output_recommendations

def _build_natural_response_prompt(question: str, health_status: str, goal: str, recommendations: list[dict]) -> str:
    """Constructs the prompt for the LLM to generate a natural language response."""
    # Convert recommendations to a readable string
    recs_parts = []
    if recommendations:
        rec = recommendations[0]
        for meal_key in ['Bữa sáng', 'Bữa trưa', 'Bữa tối', 'Bữa phụ']:
            if meal_key in rec and pd.notna(rec[meal_key]) and rec[meal_key] != "Không có gợi ý":
                meal_content = str(rec[meal_key]).replace(';', ',')
                recs_parts.append(f"{meal_key}: {meal_content}")
    
    recs_str = "; ".join(recs_parts) if recs_parts else "Không có gợi ý cụ thể cho các bữa ăn."

    # Construct context about user's health and goals
    health_goal_context = ""
    if health_status != 'không có' and goal != 'không có':
        health_goal_context = f"Họ có tình trạng sức khỏe là '{health_status}' và mục tiêu là '{goal}'."
    elif health_status != 'không có':
        health_goal_context = f"Họ có tình trạng sức khỏe là '{health_status}'."
    elif goal != 'không có':
        health_goal_context = f"Họ có mục tiêu là '{goal}'."

    return f"""Bạn là một chuyên gia dinh dưỡng AI thân thiện và am hiểu, có khả năng tư vấn cá nhân hóa.
Người dùng vừa hỏi: "{question}"
${health_goal_context}

Dựa trên dữ liệu, đây là gợi ý thực đơn phù hợp nhất cho họ:
{recs_str}
Nhiệm vụ của bạn là diễn giải những gợi ý trên thành một câu trả lời tự nhiên, mượt mà, và mang tính tư vấn cho người dùng và giữ nguyên lại lượng calo cho từng món ăn của tôi.

**Yêu cầu:**
1.  **Bắt đầu thân thiện**: Chào hỏi và tóm tắt lại yêu cầu của người dùng, bao gồm tình trạng sức khỏe và mục tiêu của họ (nếu có). Ví dụ: "Chào bạn, với tình trạng sức khỏe [tình trạng] và mục tiêu [mục tiêu], tôi gợi ý...". Nếu không có thông tin cụ thể về sức khỏe/mục tiêu, hãy bắt đầu bằng cách chào hỏi và đề cập đến câu hỏi chung của họ.
2.  **Trình bày rõ ràng**: Liệt kê các món ăn gợi ý cho từng bữa một cách mạch lạc.
3.  **Giải thích ngắn gọn**: Nêu lý do tại sao thực đơn này phù hợp với tình trạng sức khỏe và mục tiêu của họ (nếu có). Ví dụ: "Thực đơn này tập trung vào các món giàu protein nạc và rau xanh, rất tốt cho việc xây dựng cơ bắp...". Nếu không có thông tin gì thêm, chỉ cần trình bày thực đơn.
4.  **Kết thúc động viên**: Kết thúc bằng một lời chúc hoặc động viên tích cực.

**Quan trọng**: Câu trả lời của bạn phải là một đoạn văn hoàn chỉnh, không sử dụng các ký tự gạch đầu dòng (`-`) hay xuống dòng không cần thiết (`\n`). Hãy viết như một chuyên gia đang trò chuyện trực tiếp với người dùng.
"""

async def generate_natural_response_from_recommendations(
    question: str,
    parsed_params: dict,
    recommendations: list[dict]
) -> str:
    """Uses a LLM to generate a natural, conversational response based on the structured meal recommendations."""
    health_status = parsed_params.get("health_status", "không có")
    goal = parsed_params.get("goal", "không có")

    # SỬA LỖI LOGIC: Phải kiểm tra ý định không hợp lệ TRƯỚC TIÊN.
    # Đây là bước quan trọng nhất để xử lý các câu hỏi không liên quan.
    if parsed_params.get("invalid_intent"):
        return "Xin lỗi, tôi là một trợ lý chuyên về dinh dưỡng và thực đơn. Tôi chưa được huấn luyện để trả lời các câu hỏi ngoài lĩnh vực này. Bạn có cần tôi giúp gì về việc ăn uống không?"
    elif not recommendations:
        # Nếu ý định hợp lệ nhưng không tìm thấy kết quả, trả về thông báo này.
        return "Rất tiếc, tôi không tìm thấy thực đơn nào phù hợp với yêu cầu của bạn. Bạn có thể thử lại với một yêu cầu khác nhé."

    prompt = _build_natural_response_prompt(
        question=question,
        health_status=health_status,
        goal=goal,
        recommendations=recommendations
    )
    natural_response = await _call_gemini_llm(prompt=prompt, json_mode=False)

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
    recs_parts = []
    if recommendations:
        rec = recommendations[0]
        for meal_key, meal_value in rec.items():
            if pd.notna(meal_value) and meal_value != "Không có gợi ý":
                recs_parts.append(f"{meal_key}: {str(meal_value).replace(';', ',')}")
    
    fallback_response = f"Dưới đây là các gợi ý dành cho bạn: {'. '.join(recs_parts)}."
    return fallback_response
