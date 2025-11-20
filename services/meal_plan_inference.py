"""Meal plan inference service."""
# Standard library imports
import os
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import re
import unicodedata

# Third-party imports
import certifi
import joblib
import numpy as np
import pandas as pd
import scipy.sparse
import google.generativeai as genai
import torch
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from rapidfuzz import fuzz, process

# --- SSL Certificate Configuration ---
os.environ['SSL_CERT_FILE'] = certifi.where()

# --- Constants ---
SIMILARITY_THRESHOLD = 0.5
INTENT_MIN_SIMILARITY_FOR_WEAK_KEYWORDS = 0.6
DEFAULT_MEALS = ['Bữa sáng', 'Bữa trưa', 'Bữa tối', 'Bữa phụ']
env_path = Path(__file__).parent.parent / ".env"
# Nạp các biến môi trường từ file .env. Nếu file không tồn tại, nó sẽ không báo lỗi.
load_dotenv(dotenv_path=env_path)
# Define paths
HERE = Path(__file__).parent.parent
MODEL_DIR = HERE / "models" / "meal_plan"
KEYWORD_CONFIG_PATH = HERE / "config" / "keyword_maps.json"
VI_DICT_PATH = HERE / "data" / "vietnamese_dict.txt"

_keyword_maps_cache: dict | None = None
_vi_vocab_cache: set[str] | None = None


def load_keyword_maps() -> dict:
    """Load keyword maps for parsing from external JSON config.
    The JSON must contain: health_status_map, goal_map, diet_type_map, meal_map.
    Values are dict[label -> list of normalized (khong dau) keywords].
    """
    global _keyword_maps_cache
    if _keyword_maps_cache is not None:
        return _keyword_maps_cache

    try:
        with open(KEYWORD_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Keyword config file not found: {KEYWORD_CONFIG_PATH}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in keyword config file: {exc}") from exc

    for key in ("health_status_map", "goal_map", "diet_type_map", "meal_map"):
        data.setdefault(key, {})

    _keyword_maps_cache = data
    return data


def load_vi_vocab() -> set[str]:
    """Load a general Vietnamese vocabulary (non-accent, one word per line).

    File is optional; if missing, this step is skipped gracefully.
    """
    global _vi_vocab_cache
    if _vi_vocab_cache is not None:
        return _vi_vocab_cache

    vocab: set[str] = set()
    try:
        with open(VI_DICT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip()
                if word:
                    vocab.add(word)
    except FileNotFoundError:
        vocab = set()

    _vi_vocab_cache = vocab
    return vocab

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


def _strip_accents(s: str) -> str:
    """Remove Vietnamese accents (NFD) for robust matching."""
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )

CHAR_NORMALIZATION_MAP = {
    "@": "a",
    "0": "o",
}

def normalize_for_match(text: str) -> str:
    """Soft-normalize user text for teencode / typo tolerant matching."""
    if not text:
        return ""
    s = text.lower()
    for bad, good in CHAR_NORMALIZATION_MAP.items():
        s = s.replace(bad, good)
    s = s.replace("đ", "d").replace("Đ", "d")
    s = _strip_accents(s)
    return re.sub(r"\s+", " ", s).strip()
def _build_correction_dictionary(keyword_maps: dict) -> set[str]:
    """Build a set of normalized words from all keyword maps for fuzzy correction.

    Each keyword phrase is split into tokens ("tieu", "duong", "tim", "mach", ...)
    and used as correction targets for teencode / typo tokens.
    """
    words: set[str] = set()
    for section in ("health_status_map", "goal_map", "diet_type_map", "meal_map"):
        mapping = keyword_maps.get(section, {})
        for kw_list in mapping.values():
            for phrase in kw_list:
                for token in str(phrase).split():
                    token = token.strip()
                    if token:
                        words.add(token)
    return words


def _build_protected_tokens(keyword_maps: dict) -> set[str]:
    """Build a set of tokens that belong to domain keyword phrases.
    Những token này (vd. "tieu", "duong", "huyet", "ap") sẽ
    **không** bị sửa bởi các tầng spell-correction mạnh để tránh
    làm méo các cụm bệnh/mục tiêu/che_do_an quan trọng.
    """
    protected: set[str] = set()
    for section in ("health_status_map", "goal_map", "diet_type_map", "meal_map"):
        mapping = keyword_maps.get(section, {})
        for kw_list in mapping.values():
            for phrase in kw_list:
                for token in str(phrase).split():
                    token = token.strip()
                    if token:
                        protected.add(token)
    return protected


def correct_teencode_tokens(text: str, keyword_maps: dict, *, threshold: int = 85) -> str:

    if not text:
        return ""

    vocab = _build_correction_dictionary(keyword_maps)
    if not vocab:
        return text

    protected_tokens = _build_protected_tokens(keyword_maps)

    tokens = text.split()
    corrected_tokens: list[str] = []

    # Các ký tự thường dùng trong teencode / viết tắt
    special_chars = set("0123456789@#$%&*_+zjw")

    for tok in tokens:
        # Không sửa token rất ngắn hoặc token thuộc domain
        if len(tok) <= 2 or tok in protected_tokens:
            corrected_tokens.append(tok)
            continue

        # Chỉ coi là teencode nếu có cả chữ cái và ký tự đặc biệt/số
        has_letter = any(c.isalpha() for c in tok)
        has_special = any(c in special_chars for c in tok)
        if not (has_letter and has_special):
            corrected_tokens.append(tok)
            continue

        best = process.extractOne(tok, vocab, scorer=fuzz.ratio)
        if not best:
            corrected_tokens.append(tok)
            continue

        match_word, score, _ = best
        if score >= threshold:
            corrected_tokens.append(match_word)
        else:
            corrected_tokens.append(tok)

def strong_dict_spell_correction(text: str, keyword_maps: dict | None = None, *, threshold: int = 80) -> str:
    if not text:
        return ""

    vocab = load_vi_vocab()
    if not vocab:
        return text

    protected_tokens: set[str] = set()
    if keyword_maps is not None:
        protected_tokens = _build_protected_tokens(keyword_maps)

    tokens = text.split()
    corrected_tokens: list[str] = []

    for tok in tokens:
        # Không sửa các token domain đã được bảo vệ hoặc token quá ngắn
        if len(tok) <= 2 or tok in protected_tokens:
            corrected_tokens.append(tok)
            continue

        best = process.extractOne(tok, vocab, scorer=fuzz.ratio)
        if best and best[1] >= threshold:
            corrected_tokens.append(best[0])
        else:
            corrected_tokens.append(tok)

    return " ".join(corrected_tokens)

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
        "requested_meals": DEFAULT_MEALS,
    }

    maps = load_keyword_maps()
    health_status_map = maps["health_status_map"]
    goal_map = maps["goal_map"]
    diet_type_map = maps["diet_type_map"]
    meal_map = maps["meal_map"]

    def find_keywords_multiple(text, keyword_map):
        """Find ALL matching keyword categories, using longest-match-first and overlap control."""
        found_categories = set()
        all_keywords = [
            (category, kw, len(kw))
            for category, kws in keyword_map.items()
            for kw in kws
        ]
        all_keywords.sort(key=lambda x: (-x[2], x[0]))
        matched_ranges = []
        for category, kw, kw_len in all_keywords:
            start_pos = text.find(kw)
            if start_pos == -1:
                continue
            end_pos = start_pos + kw_len
            overlaps = False
            for matched_start, matched_end in matched_ranges:
                overlap_start = max(start_pos, matched_start)
                overlap_end = min(end_pos, matched_end)
                if overlap_end > overlap_start:
                    overlap_len = overlap_end - overlap_start
                    min_len = min(kw_len, matched_end - matched_start)
                    if overlap_len > min_len * 0.3:
                        overlaps = True
                        break
            if not overlaps:
                found_categories.add(category)
                matched_ranges.append((start_pos, end_pos))
        return ", ".join(sorted(found_categories)) if found_categories else "không có"

    def find_requested_meals(text, meal_keyword_map):
        """Detect which meals are explicitly mentioned in the text.
        - Không có keyword bữa nào -> trả về None để giữ DEFAULT_MEALS (cả ngày).
        - Có keyword "cả_ngày" -> coi như hỏi thực đơn cả ngày, trả DEFAULT_MEALS.
        - Ngược lại, trả về đúng các bữa được nhắc đến.
        """
        found_meals: list[str] = []
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

async def parse_meal_plan_question(question: str) -> dict:
    """Parse question with keyword maps; PhoBERT will gate intent via similarity.
    Pipeline:
    1) `normalize_for_match`: lower, map teencode chars, remove accents.
    2) `correct_teencode_tokens`: fuzzy-correct theo vocab domain (bỏ qua token domain).
    3) `_parse_question_with_keywords`: trích health_status/goal/diet_type/requested_meals.
    """
    # 1) Chuẩn hóa mềm câu hỏi (teencode + bỏ dấu): **luôn** là bản gốc an toàn
    base_normalized = normalize_for_match(question)
    print("[DEBUG] base_normalized:", base_normalized)

    # 2) Sửa nhẹ theo từ vựng domain nếu có teencode; nếu không, giữ nguyên base_normalized
    keyword_maps = load_keyword_maps()
    normalized_question = correct_teencode_tokens(
        base_normalized,
        keyword_maps,
        threshold=80,
    )
    if not normalized_question:
        normalized_question = base_normalized

    print("[DEBUG] normalized_question:", normalized_question)

    # 3) Parse keyword trên câu đã chuẩn hóa (nhưng vẫn rất gần base_normalized)
    parsed_params = _parse_question_with_keywords(normalized_question.lower())

    # Logging kết quả parse
    try:
        meal_plan_logger.info("=" * 80)
        meal_plan_logger.info(f"Question: {question}")
        meal_plan_logger.info(f"Question_normalized: {normalized_question}")
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

    # Trả thêm:
    # - normalized_question: bản đã chuẩn hóa + sửa teencode (nếu có)
    # - dict_corrected_base: luôn là bản chuẩn hóa mềm an toàn, không fuzzy
    parsed_params["normalized_question"] = normalized_question
    parsed_params["dict_corrected_base"] = base_normalized
    return parsed_params

def _get_query_embedding(text: str, model, tokenizer) -> np.ndarray:
    """Generates a PhoBERT embedding for a single query text."""
    if not model or not tokenizer:
        raise RuntimeError("PhoBERT model is not loaded. Call load_recommender_artifacts() first.")
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()


def _build_structured_query_text(
    health_status: str | None,
    goal: str | None,
    diet_type: str | None,
    requested_meals: list[str] | None
) -> str:
    """Creates a normalized text string from parsed attributes for embedding fallback."""
    meta_parts: list[str] = []

    if health_status and health_status != 'không có':
        meta_parts.append(f"Tình trạng sức khỏe: {health_status}")
    if goal and goal != 'không có':
        meta_parts.append(f"Mục tiêu: {goal}")
    if diet_type and diet_type != 'không có':
        meta_parts.append(f"Chế độ ăn: {diet_type}")
    if requested_meals:
        meta_parts.append("Các bữa quan tâm: " + ", ".join(requested_meals))

    return " | ".join(meta_parts)

def recommend_meal_plan(
    original_question: str,
    parsed_params: dict
) -> list[dict]:
    """Recommends a suitable meal plan based on user's query and profile.

    Intent is controlled by:
    - Keyword parsing (health_status, goal, diet_type).
    - PhoBERT similarity gate when no nutrition keywords are found.
    """
    # Lấy các giá trị từ dictionary, sử dụng .get() để tránh KeyError
    health_status = parsed_params.get("health_status")
    goal = parsed_params.get("goal")
    diet_type = parsed_params.get("diet_type")
    requested_meals = parsed_params.get("requested_meals", DEFAULT_MEALS)
    normalized_question = parsed_params.get("normalized_question") or original_question
    dict_corrected_base = parsed_params.get("dict_corrected_base")
    keyword_maps = load_keyword_maps()

    if not recommender.is_ready():
        print("[ERROR] Recommender is not ready. Artifacts may have failed to load.")
        return []

    if not requested_meals:
        requested_meals = DEFAULT_MEALS

    structured_query_text = _build_structured_query_text(
        health_status=health_status,
        goal=goal,
        diet_type=diet_type,
        requested_meals=requested_meals
    )

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
        _ = recommender.user_feature_encoder.transform(df_to_encode[recommender.user_features_cols])
    except ValueError as e:
        print(f"[ERROR] Error encoding user input: {e}")
        return []

    # --- NEW LOGIC: Filter-then-Rank ---
    # 1. Hard-filter the dataset based on extracted attributes
    print("[INFO] Applying hard filters based on extracted attributes...")
    filtered_df = recommender.meal_plans_df.copy()

    conditions = []
    def _normalize_multi_value_field(raw_value: str) -> list[str]:
        if not raw_value:
            return []
        text = str(raw_value).lower()
        for pat in (" và ", " và", "và "):
            text = text.replace(pat, ",")
        text = text.strip(" ,")
        return [p.strip() for p in text.split(",") if p.strip()]
    # SỬA LỖI LOGIC: Xử lý nhiều giá trị trong một trường (ví dụ: "béo phì, tiểu đường")
    # Thay vì tìm kiếm sự trùng khớp chính xác, chúng ta sẽ tìm kiếm sự tồn tại của bất kỳ từ khóa nào.
    if diet_type and diet_type != 'không có':
        # Chế độ ăn thường là một giá trị duy nhất, giữ nguyên logic so sánh bằng.
        conditions.append(filtered_df['Chế độ ăn'].str.lower().str.strip() == diet_type.lower().strip())

    if health_status and health_status != 'không có':
        health_keywords = _normalize_multi_value_field(health_status)
        if health_keywords:
            # OR từng keyword, không dùng regex phức tạp
            health_condition = False
            col = filtered_df['Tình trạng sức khỏe'].str.lower()
            for kw in health_keywords:
                health_condition = health_condition | col.str.contains(kw, na=False)
            conditions.append(health_condition)

    if goal and goal != 'không có':
        goal_keywords = _normalize_multi_value_field(goal)
        if goal_keywords:
            goal_condition = False
            col = filtered_df['Mục tiêu'].str.lower()
            for kw in goal_keywords:
                goal_condition = goal_condition | col.str.contains(kw, na=False)
            conditions.append(goal_condition)

    # Chỉ apply filter nếu có điều kiện cụ thể.
    if conditions:
        filtered_df = filtered_df[np.logical_and.reduce(conditions)]
        if filtered_df.empty:
            print("[INFO] No meal plans found after applying hard filters.")
            return []
    else:
        # Không có health_status / goal / diet_type -> có thể là câu hỏi chung.
        # Chỉ random thực đơn nếu câu thực sự nhắc đến bữa ăn/thực đơn.
        # Ưu tiên dùng bản đã normalize nhưng CHƯA fuzzy (dict_corrected_base) để tránh méo từ.
        base_text = (dict_corrected_base or normalized_question or original_question).lower()

        # 1) Kiểm tra keyword bữa ăn từ meal_map (Bữa sáng/trưa/tối/phụ/cả ngày)
        meal_map = keyword_maps.get("meal_map", {})
        has_meal_keyword = False
        for _, kws in meal_map.items():
            for kw in kws:
                if kw in base_text:
                    has_meal_keyword = True
                    break
            if has_meal_keyword:
                break

        # 2) Thêm một số cụm chung về thực đơn/bữa ăn trong ngày
        if not has_meal_keyword:
            generic_phrases = [
                "thuc don", "thực đơn", "an gi", "ăn gì",
                "bua an", "bữa ăn", "1 ngay", "mot ngay", "ca ngay", "trong ngay"
            ]
            has_meal_keyword = any(p in base_text for p in generic_phrases)

        if has_meal_keyword:
            print("[INFO] Generic meal-related query detected. Selecting random safe meal plan without intent gate.")
            # Thử ưu tiên các thực đơn dành cho người bình thường (nếu có)
            generic_mask = recommender.meal_plans_df['Tình trạng sức khỏe'].str.contains(
                'bình thường|khoe manh|khong benh|khong co benh', case=False, na=False
            )
            generic_df = recommender.meal_plans_df[generic_mask]
            if generic_df.empty:
                generic_df = recommender.meal_plans_df

            # Chọn ngẫu nhiên 1 dòng làm kế hoạch bữa ăn
            sampled = generic_df.sample(n=1, random_state=None)
            best_plans_raw = sampled.to_dict(orient='records')

            output_recommendations = []
            for plan in best_plans_raw:
                recommendation = {}
                for meal_key in requested_meals:
                    meal_value = plan.get(meal_key)
                    if pd.isna(meal_value):
                        recommendation[meal_key] = "Không có gợi ý"
                    else:
                        recommendation[meal_key] = meal_value
                output_recommendations.append(recommendation)

            # Log lại
            try:
                meal_plan_logger.info("Thực đơn được gợi ý (generic meal query - random):")
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
        else:
            # Không có keyword bữa ăn -> để PhoBERT + intent gate xử lý như trước.
            print("[INFO] No explicit meal keywords in a keyword-less query; falling back to semantic intent gate.")
    
    print(f"[INFO] Found {len(filtered_df)} candidates after filtering.")

    # 2. Perform semantic search on the filtered subset with graceful fallback
    filtered_indices = filtered_df.index
    filtered_embeddings = recommender.meal_features_phobert_matrix[filtered_indices]

    # Intent gate: check if any nutrition-related keywords were extracted
    has_any_keyword = any(
        v and v != 'không có'
        for v in (health_status, goal, diet_type)
    )

    # Dùng câu đã được sửa chính tả mạnh nhất nếu có, để PhoBERT bớt nhạy với lỗi.
    query_text_for_embedding = (dict_corrected_base or normalized_question or original_question).strip()
    query_embedding = _get_query_embedding(query_text_for_embedding, recommender.phobert_model, recommender.phobert_tokenizer)
    similarity_scores = cosine_similarity(query_embedding, filtered_embeddings).flatten()
    top_indices = similarity_scores.argsort()[::-1]

    if not len(top_indices):
        return []

    best_score = similarity_scores[top_indices[0]]
    print(f"[DEBUG] Best similarity score with original question: {best_score:.4f}")

    # PhoBERT-based intent gate: if no nutrition keywords and similarity is very high-level/low, treat as out-of-domain
    if not has_any_keyword and best_score < INTENT_MIN_SIMILARITY_FOR_WEAK_KEYWORDS:
        print(
            f"[INFO] Intent gate: no nutrition keywords and similarity "
            f"{best_score:.4f} < {INTENT_MIN_SIMILARITY_FOR_WEAK_KEYWORDS}. Treat as out-of-domain."
        )
        return []

    if best_score < SIMILARITY_THRESHOLD and structured_query_text:
        print("[INFO] Similarity below threshold. Retrying with structured attributes.")
        query_text_for_embedding = structured_query_text
        query_embedding = _get_query_embedding(query_text_for_embedding, recommender.phobert_model, recommender.phobert_tokenizer)
        similarity_scores = cosine_similarity(query_embedding, filtered_embeddings).flatten()
        top_indices = similarity_scores.argsort()[::-1]
        if not len(top_indices):
            return []
        best_score = similarity_scores[top_indices[0]]
        print(f"[DEBUG] Best similarity score with structured query: {best_score:.4f}")

    if best_score < SIMILARITY_THRESHOLD:
        print(f"[INFO] Best match score ({best_score:.2f}) is below threshold ({SIMILARITY_THRESHOLD}). Returning no results.")
        return [] # Return empty list if no good match is found
    original_relevant_indices = [filtered_indices[i] for i in top_indices]
    
    if not original_relevant_indices:
        return [] # No suitable plan found after filtering and ranking
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
**Lưu ý độ dài**: Toàn bộ câu trả lời nên ngắn gọn, súc tích, ưu tiên truyền đạt ý chính thay vì mô tả chi tiết từng món.
**Yêu cầu:**
1.  **Bắt đầu thân thiện**: Chào hỏi và tóm tắt lại yêu cầu của người dùng, bao gồm tình trạng sức khỏe và mục tiêu của họ (nếu có). Ví dụ: "Chào bạn, với tình trạng sức khỏe [tình trạng] và mục tiêu [mục tiêu], tôi gợi ý...". Nếu không có thông tin cụ thể về sức khỏe/mục tiêu, hãy bắt đầu bằng cách chào hỏi và đề cập đến câu hỏi chung của họ.
2.  **Trình bày rõ ràng**: Liệt kê các món ăn gợi ý cho từng bữa một cách mạch lạc.
3.  **Giải thích ngắn gọn**: Tóm tắt 1–2 câu vì sao thực đơn này phù hợp với tình trạng sức khỏe và mục tiêu của họ, tránh giải thích dài dòng.
4.  **Kết thúc động viên**: Kết thúc bằng một câu chúc ngắn gọn, tích cực.
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

    if not recommendations:
        # Không có thực đơn nào được tìm thấy (có thể do intent gate hoặc không có match phù hợp).
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
            return natural_response.replace("\n", " ").replace("- ", "").strip()
        except UnicodeEncodeError:
            safe = natural_response.encode('ascii', 'replace').decode('ascii')
            return safe.replace("\n", " ").replace("- ", "").strip()
    
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
