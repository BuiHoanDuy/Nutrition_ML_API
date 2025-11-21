"""
Meal plan inference service.
(Modified: integrate robust Vietnamese normalization + fuzzy correction directly,
 token-level correction improved with fallback scorers and debug logging)
"""
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
                    # store normalized (no-accent, lowercase) tokens
                    vocab.add(_basic_normalize(word))
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
meal_plan_logger.setLevel(logging.INFO)  # keep INFO by default; set to DEBUG during troubleshooting
meal_plan_logger.propagate = False
meal_plan_handler = RotatingFileHandler(
    LOGS_DIR / "meal_plan_requests.log", maxBytes=10_000_000, backupCount=5, encoding="utf-8"
)
meal_plan_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
meal_plan_handler.setFormatter(meal_plan_formatter)
meal_plan_logger.addHandler(meal_plan_handler)


# ---------------------------
# Normalization & Correction
# ---------------------------

def _strip_accents(s: str) -> str:
    """Remove Vietnamese accents (NFD) for robust matching."""
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


# Basic lightweight map for common teencode -> char
CHAR_NORMALIZATION_MAP = {
    "@": "a",
    "0": "o",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
    "8": "b",
    # do NOT map '1' globally (ambiguous with numbers), handle separately if needed
}


def _basic_normalize(text: str) -> str:
    """Lowercase, map teencode chars (very small set), remove accents, collapse spaces."""
    if not text:
        return ""
    s = text.lower()
    for bad, good in CHAR_NORMALIZATION_MAP.items():
        s = s.replace(bad, good)
    # normalize đ -> d
    s = s.replace("đ", "d").replace("Đ", "d")
    # strip accents (NFD)
    s = _strip_accents(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _shrink_repeated_chars(token: str, max_repeats: int = 2) -> str:
    """Shrink runs of repeated characters to at most max_repeats (to handle 'tieeeeuu')."""
    if len(token) <= 2:
        return token
    result = [token[0]]
    count = 1
    for ch in token[1:]:
        if ch == result[-1]:
            if count < max_repeats:
                result.append(ch)
            count += 1
        else:
            result.append(ch)
            count = 1
    return "".join(result)


def _tokenize_and_shrink(text: str) -> list[str]:
    tokens = _basic_normalize(text).split()
    return [_shrink_repeated_chars(t) for t in tokens]


def _build_vocabs_from_keyword_maps(keyword_maps: dict) -> tuple[set[str], set[str]]:
    """Return (token_vocab, phrase_vocab) normalized (no accents) from keyword_maps dict."""
    token_vocab: set[str] = set()
    phrase_vocab: set[str] = set()
    for section in ("health_status_map", "goal_map", "diet_type_map", "meal_map"):
        mapping = keyword_maps.get(section, {})
        for kws in mapping.values():
            for phrase in kws:
                norm = _basic_normalize(str(phrase))
                if not norm:
                    continue
                phrase_vocab.add(norm)
                for tok in norm.split():
                    token_vocab.add(tok)
    return token_vocab, phrase_vocab

def _phrase_level_correction(tokens: list[str], phrase_vocab: set[str], threshold: int = 82, overlap_min: float = 0.6) -> list[str]:
    """
    Slide over tokens and try to replace token spans with matched normalized phrase tokens,
    but accept replacement only if:
      - fuzzy score >= threshold
      - AND token-overlap_ratio >= overlap_min (common tokens / max(len(candidate), len(phrase)))
    This avoids inserting unrelated multiword phrases (e.g., 'thiếu canxi') that weren't implied by input.
    Logs PHRASE-DEBUG for visibility.
    """
    if not phrase_vocab or not tokens:
        return tokens

    i = 0
    n = len(tokens)
    out = []

    while i < n:
        matched = False
        # try larger windows first (3 then 2)
        for w in (3, 2):
            if i + w <= n:
                cand_tokens = tokens[i:i + w]
                cand = " ".join(cand_tokens)
                best = process.extractOne(cand, phrase_vocab, scorer=fuzz.ratio)
                if best:
                    matched_phrase, score = best[0], best[1]
                else:
                    matched_phrase, score = None, 0

                accept = False
                overlap_ratio = 0.0
                if matched_phrase and score >= threshold:
                    mp_tokens = matched_phrase.split()
                    # count exact token intersections
                    common = sum(1 for t in cand_tokens if t in mp_tokens)
                    denom = max(len(cand_tokens), len(mp_tokens))
                    overlap_ratio = common / denom if denom > 0 else 0.0

                    # Prevent large insertions: also ensure matched phrase length not >> candidate length
                    length_ratio = len(mp_tokens) / len(cand_tokens) if len(cand_tokens) > 0 else 1.0

                    # Accept if overlap_ratio >= overlap_min AND length_ratio <= 1.6 (avoid phrase that is much longer)
                    if overlap_ratio >= overlap_min and length_ratio <= 1.6:
                        accept = True

                # PHRASE-DEBUG log
                try:
                    meal_plan_logger.debug(
                        f"[PHRASE-DEBUG] cand='{cand}' matched_phrase='{matched_phrase}' score={score} overlap={overlap_ratio:.2f} accept={accept}"
                    )
                except Exception:
                    print(f"[PHRASE-DEBUG] cand='{cand}' matched_phrase='{matched_phrase}' score={score} overlap={overlap_ratio:.2f} accept={accept}")

                if matched_phrase and accept:
                    out.extend(matched_phrase.split())
                    i += w
                    matched = True
                    break
        if not matched:
            out.append(tokens[i])
            i += 1

    return out

def _token_level_correction(tokens: list[str],
                            token_vocab: set[str],
                            protected_tokens: set[str],
                            *,
                            threshold: int = 75,
                            short_token_threshold: int = 65) -> list[str]:
    """
    Token-level fuzzy correction with:
      - fallback scorers (partial_ratio, token_sort_ratio)
      - light pre-processing for common noisy chars (e.g., stray 'w' or 'j')
      - lower applied threshold for short tokens (short_token_threshold)
      - debug logging of top candidates when no correction applied
    """
    corrected = []
    for tok in tokens:
        # keep very short tokens and protected tokens unchanged
        if len(tok) <= 2 or tok in protected_tokens:
            corrected.append(tok)
            continue

        # prepare candidate for matching: try removing stray 'w'/'j' to improve match
        tok_candidate = tok
        if ('w' in tok_candidate or 'j' in tok_candidate) and len(tok_candidate) >= 3:
            tok_candidate = tok_candidate.replace('w', '').replace('j', '')

        # 1) primary scorer: ratio
        best = process.extractOne(tok_candidate, token_vocab, scorer=fuzz.ratio)
        best_score = best[1] if best else 0
        best_choice = best[0] if best else None

        # 2) fallback: partial_ratio (helpful for insertion/deletion)
        if best_score < threshold:
            best_partial = process.extractOne(tok_candidate, token_vocab, scorer=fuzz.partial_ratio)
            if best_partial and best_partial[1] > best_score:
                best_score = best_partial[1]
                best_choice = best_partial[0]

        # 3) fallback: token_sort_ratio (helpful for transposition)
        if best_score < threshold:
            best_sort = process.extractOne(tok_candidate, token_vocab, scorer=fuzz.token_sort_ratio)
            if best_sort and best_sort[1] > best_score:
                best_score = best_sort[1]
                best_choice = best_sort[0]

        # adjust threshold for short tokens to be more permissive
        applied_threshold = threshold
        if len(tok) <= 4:
            applied_threshold = min(applied_threshold, short_token_threshold)

        if best_choice and best_score >= applied_threshold:
            corrected.append(best_choice)
        else:
            # debug: log top-3 candidates to help tuning thresholds/dictionary
            try:
                top3 = process.extract(tok_candidate, token_vocab, scorer=fuzz.ratio, limit=3)
                meal_plan_logger.debug(f"[SPELL-DEBUG] token='{tok}' cand_top3={top3}")
            except Exception:
                # fallback to printing if logger misconfigured
                try:
                    print(f"[SPELL-DEBUG] token='{tok}' (failed to get top3)")
                except Exception:
                    pass
            corrected.append(tok)
    return corrected


def normalize_user_question(text: str,
                            keyword_maps: dict | None = None,
                            vi_wordlist_tokens: set[str] | None = None,
                            phrase_threshold: int = 60,
                            token_threshold: int = 60) -> tuple[str, str]:
    """
    Full normalization pipeline returning (dict_corrected_base, normalized_question).

    - dict_corrected_base: base safe normalization (lowercase, no-accent, teencode char map, shrink repeats)
      suitable for embedding/logging.
    - normalized_question: further corrected version (phrase-level + token-level fuzzy) used for keyword matching.
    """
    base = _basic_normalize(text)
    base_tokens = [_shrink_repeated_chars(t) for t in base.split()]

    # Build vocabs
    token_vocab: set[str] = set()
    phrase_vocab: set[str] = set()
    if keyword_maps:
        k_tokens, k_phrases = _build_vocabs_from_keyword_maps(keyword_maps)
        token_vocab.update(k_tokens)
        phrase_vocab.update(k_phrases)

    if vi_wordlist_tokens:
        token_vocab.update(vi_wordlist_tokens)

    # If token_vocab empty, skip fuzzy correction and return base
    if not token_vocab and not phrase_vocab:
        normalized = " ".join(base_tokens)
        return (" ".join(base_tokens), normalized)

    # Phrase-level correction
    tokens_after_phrase = _phrase_level_correction(base_tokens, phrase_vocab, threshold=phrase_threshold)

    # Protected tokens: tokens that are part of phrase_vocab (avoid changing them at token-level)
    protected = set()
    for p in phrase_vocab:
        for t in p.split():
            protected.add(t)

    # Token-level correction (use token_vocab if available, otherwise phrase-derived tokens)
    final_token_vocab = token_vocab if token_vocab else protected
    tokens_after_token = _token_level_correction(tokens_after_phrase,
                                                 final_token_vocab,
                                                 protected,
                                                 threshold=token_threshold,
                                                 short_token_threshold=65)

    normalized = " ".join(tokens_after_token)
    return (" ".join(base_tokens), normalized)


# ---------------------------
# MealPlanRecommender & parse
# ---------------------------

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
    1) `normalize_user_question`: returns a base normalized string and a fuzz-corrected normalized string.
    2) `_parse_question_with_keywords`: trích health_status/goal/diet_type/requested_meals.
    """
    # load keyword maps and vi vocab
    keyword_maps = load_keyword_maps()
    vi_tokens = load_vi_vocab()  # normalized tokens from vietnamese_dict.txt if present

    # 1) Normalize + fuzzy-correct using domain vocab + optional vi wordlist
    dict_corrected_base, normalized_question = normalize_user_question(
        question,
        keyword_maps=keyword_maps,
        vi_wordlist_tokens=vi_tokens,
        phrase_threshold=82,
        token_threshold=70
    )

    if not normalized_question:
        normalized_question = dict_corrected_base

    print("[DEBUG] dict_corrected_base:", dict_corrected_base)
    print("[DEBUG] normalized_question:", normalized_question)

    # 2) Parse keyword trên câu đã chuẩn hóa (no-accent)
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

    # Attach normalized strings for downstream use
    parsed_params["normalized_question"] = normalized_question
    parsed_params["dict_corrected_base"] = dict_corrected_base
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

    base_text = (dict_corrected_base or normalized_question or original_question).lower()
    meal_map = keyword_maps.get("meal_map", {})
    def _is_generic_meal_query(text: str) -> bool:
        for _, kws in meal_map.items():
            for kw in kws:
                if kw in text:
                    return True
        generic_phrases = [
            "thuc don", "thực đơn", "an gi", "ăn gì",
            "bua an", "bữa ăn", "1 ngay", "mot ngay", "ca ngay", "trong ngay",
            "goi y thuc don", "gợi ý thực đơn", "menu", "goi y an uong", "gợi ý ăn uống", "goi y bua an", "gợi ý bữa ăn", "thuc don cho toi", "thực đơn cho tôi", "thuc don hom nay", "thực đơn hôm nay"
        ]
        return any(p in text for p in generic_phrases)

    # 1. Nếu có health/goal/diet (ít nhất 1 trường khác "không có") thì luôn lọc + rank, nếu không có kết quả trả []
    has_any_keyword = any(
        v and v != 'không có'
        for v in (health_status, goal, diet_type)
    )
    if has_any_keyword:
        # --- Lọc data theo health_status, goal, diet_type như cũ ---
        structured_query_text = _build_structured_query_text(
            health_status=health_status,
            goal=goal,
            diet_type=diet_type,
            requested_meals=requested_meals
        )
        user_input_df = pd.DataFrame([{
            'Tình trạng sức khỏe': health_status,
            'Mục tiêu': goal,
            'Chế độ ăn': diet_type
        }])
        try:
            df_to_encode = user_input_df.rename(columns={
                'Tình trạng sức khỏe': 'tinh_trang_suc_khoe',
                'Mục tiêu': 'muc_tieu',
                'Chế độ ăn': 'che_do_an'
            })
            _ = recommender.user_feature_encoder.transform(df_to_encode[recommender.user_features_cols])
        except ValueError as e:
            print(f"[ERROR] Error encoding user input: {e}")
            return []
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
        if diet_type and diet_type != 'không có':
            conditions.append(filtered_df['Chế độ ăn'].str.lower().str.strip() == diet_type.lower().strip())
        if health_status and health_status != 'không có':
            health_keywords = _normalize_multi_value_field(health_status)
            if health_keywords:
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
        if conditions:
            filtered_df = filtered_df[np.logical_and.reduce(conditions)]
        if filtered_df.empty:
            print("[INFO] No meal plans found after applying hard filters.")
            return []
        # --- PhoBERT + ranking như cũ ---
        filtered_indices = filtered_df.index
        filtered_embeddings = recommender.meal_features_phobert_matrix[filtered_indices]
        query_text_for_embedding = (dict_corrected_base or normalized_question or original_question).strip()
        query_embedding = _get_query_embedding(query_text_for_embedding, recommender.phobert_model, recommender.phobert_tokenizer)
        similarity_scores = cosine_similarity(query_embedding, filtered_embeddings).flatten()
        top_indices = similarity_scores.argsort()[::-1]
        if not len(top_indices):
            return []
        best_score = similarity_scores[top_indices[0]]
        print(f"[DEBUG] Best similarity score with original question: {best_score:.4f}")
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
            return []
        original_relevant_indices = [filtered_indices[i] for i in top_indices]
        if not original_relevant_indices:
            return []
        if len(original_relevant_indices) == 1:
            best_match_index = original_relevant_indices[0]
            print(f"[INFO] Only one candidate found. Consistently returning index: {best_match_index}")
        if original_question in recommender.last_recommendation_cache:
            last_shown_index = recommender.last_recommendation_cache[original_question]
            try:
                current_pos = original_relevant_indices.index(last_shown_index)
                next_pos = (current_pos + 1) % len(original_relevant_indices)
                best_match_index = original_relevant_indices[next_pos]
                print(f"[INFO] Subsequent request. Showing next candidate at index: {best_match_index}")
            except ValueError:
                best_match_index = original_relevant_indices[0]
                print(f"[INFO] Subsequent request. Last shown index not in new candidates. Starting over at index: {best_match_index}")
        else:
            if original_relevant_indices:
                best_match_index = original_relevant_indices[0]
                print(f"[INFO] First request. Selecting top candidate at index: {best_match_index}")
            else:
                return []
        recommender.last_recommendation_cache[original_question] = best_match_index
        best_plans_raw = recommender.meal_plans_df.iloc[best_match_index:best_match_index+1].to_dict(orient='records')
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
    # 2. Nếu không có health/goal/diet nhưng là câu hỏi chung chung về thực đơn thì trả random 1 plan an toàn (bỏ qua PhoBERT)
    elif _is_generic_meal_query(base_text):
        print("[INFO] Generic meal-related query detected (no health/goal/diet). Selecting random safe meal plan.")
        generic_mask = recommender.meal_plans_df['Tình trạng sức khỏe'].str.contains(
            'bình thường|khoe manh|khong benh|khong co benh', case=False, na=False
        )
        generic_df = recommender.meal_plans_df[generic_mask]
        if generic_df.empty:
            generic_df = recommender.meal_plans_df
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
        try:
            meal_plan_logger.info("Thực đơn được gợi ý (generic meal query - random, fallback):")
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
    # 3. Nếu không có dữ liệu và không phải câu chung chung thì trả [] (để Gemini fallback)
    else:
        print("[INFO] No health/goal/diet and not a generic meal query. Returning empty for Gemini fallback.")
        return []

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

    return f"""Bạn là một chuyên gia dinh dưỡng AI, trả lời rất ngắn gọn và dễ hiểu.
Người dùng hỏi: "{question}"
{health_goal_context}

Dựa trên dữ liệu, đây là gợi ý thực đơn phù hợp nhất cho họ:
{recs_str}

Yêu cầu quan trọng:
1. Chỉ diễn giải lại thực đơn, GIỮ NGUYÊN tên món ăn đúng như dữ liệu (không tự thêm, không tự đổi tên món, không tự thêm kcal vào tên món).
2. Không liệt kê dạng gạch đầu dòng, không xuống dòng nhiều lần, không giải thích lan man.
3. Viết MỘT đoạn văn ngắn (3–5 câu), thân thiện, súc tích.
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


async def generate_answer_with_fallback(
    question: str,
    parsed_params: dict,
    recommendations: list[dict],
) -> str:
    """Generate final answer:

    - Nếu có recommendations từ data -> dùng generate_natural_response_from_recommendations.
    - Nếu KHÔNG có recommendations:
        + Nếu câu hỏi là dinh dưỡng (có keyword) -> nhờ Gemini tự đề xuất thực đơn.
        + Nếu câu hỏi ngoài dinh dưỡng          -> nhờ Gemini trả lời chung.
    """
    # 1. Nếu đã có recs từ data thì giữ nguyên luồng cũ
    if recommendations:
        return await generate_natural_response_from_recommendations(
            question=question,
            parsed_params=parsed_params,
            recommendations=recommendations,
        )

    # 2. Không có recs -> xác định xem có phải câu hỏi dinh dưỡng không
    health_status = parsed_params.get("health_status", "không có")
    goal = parsed_params.get("goal", "không có")
    diet_type = parsed_params.get("diet_type", "không có")

    has_any_keyword = any(
        v and v != "không có" for v in (health_status, goal, diet_type)
    )

    # 3. Tạo prompt linh hoạt cho Gemini
    if has_any_keyword:
        # Câu hỏi dinh dưỡng nhưng không có dữ liệu trong data
        prompt = f"""
Bạn là một chuyên gia dinh dưỡng AI.
Người dùng hỏi: "{question}"

Thông tin trích được:
- Tình trạng sức khỏe: {health_status}
- Mục tiêu: {goal}
- Chế độ ăn: {diet_type}

Trong cơ sở dữ liệu nội bộ hiện không có sẵn thực đơn phù hợp.
Hãy tự đề xuất một thực đơn 1 ngày (bữa sáng, trưa, tối, phụ) với lượng calo hợp lý cho mỗi bữa,
giải thích ngắn gọn vì sao thực đơn này phù hợp với tình trạng của người dùng.
Trả lời bằng tiếng Việt, thân thiện, ngắn gọn, dưới dạng một đoạn văn hoàn chỉnh (không dùng bullet).
Nếu nằm ngoài phạm vi về dinh dưỡng, sức khoẻ, các câu dẫn thông thường thì trả lời: "Xin lỗi, tôi là chuyên gia dinh dưỡng, hãy hỏi tôi về lĩnh vực sức khoẻ!"
"""
    else:
        # Câu hỏi ngoài lĩnh vực dinh dưỡng
        prompt = f"""
Bạn là một trợ lý AI thân thiện.
Người dùng hỏi: "{question}"

Hãy trả lời ngắn gọn, đúng trọng tâm, tích cực.
Trả lời bằng tiếng Việt, dưới dạng một đoạn văn hoàn chỉnh (không dùng bullet).
"""

    llm_answer = await _call_gemini_llm(prompt=prompt, json_mode=False)

    if isinstance(llm_answer, str) and llm_answer.strip():
        try:
            return llm_answer.replace("\n", " ").strip()
        except UnicodeEncodeError:
            safe = llm_answer.encode("ascii", "replace").decode("ascii")
            return safe.replace("\n", " ").strip()

    # Fallback cuối cùng nếu Gemini lỗi
    if has_any_keyword:
        return (
            "Hiện tôi chưa tạo được thực đơn phù hợp cho bạn. "
            "Bạn có thể thử mô tả rõ hơn tình trạng sức khỏe và mục tiêu nhé."
        )
    return "Hiện tôi chưa trả lời được câu hỏi này, bạn có thể thử hỏi lại theo cách khác nhé."
