# 📖 Giải thích chức năng của `meal_plan_inference.py`

File `services/meal_plan_inference.py` là **core service** xử lý toàn bộ logic gợi ý thực đơn dinh dưỡng dựa trên câu hỏi tự nhiên của người dùng.

## 🎯 Chức năng chính

File này thực hiện **3 chức năng chính**:

### 1. **Xử lý và chuẩn hóa câu hỏi tiếng Việt** (Vietnamese Text Normalization)
### 2. **Trích xuất thông tin từ câu hỏi** (Question Parsing & Information Extraction)
### 3. **Gợi ý thực đơn và tạo câu trả lời tự nhiên** (Meal Plan Recommendation & Natural Language Generation)

---

## 🔍 Chi tiết từng chức năng

### 1. Xử lý và chuẩn hóa câu hỏi tiếng Việt

#### 1.1. Normalization Pipeline
```python
normalize_user_question(text) -> (dict_corrected_base, normalized_question)
```

**Mục đích:** Xử lý các vấn đề phổ biến trong tiếng Việt:
- **Teencode:** `@` → `a`, `0` → `o`, `3` → `e`, `4` → `a`, `5` → `s`, `7` → `t`, `8` → `b`
- **Lỗi chính tả:** Sửa lỗi đánh máy, thiếu dấu
- **Ký tự lặp:** `tieeeeuu` → `tieeuu` (giới hạn 2 ký tự lặp)
- **Bỏ dấu:** `ăn uống` → `an uong` (để matching dễ hơn)
- **Chuẩn hóa khoảng trắng:** Loại bỏ khoảng trắng thừa

**Ví dụ:**
```
Input:  "Tôi muốn giảm cân, sáng nên ăn gì?"
Output: ("toi muon giam can sang nen an gi", "toi muon giam can sang nen an gi")
```

#### 1.2. Fuzzy Correction (Sửa lỗi chính tả thông minh)

**Token-level correction:**
- Sửa từng từ một bằng fuzzy matching
- Sử dụng nhiều scorers: `ratio`, `partial_ratio`, `token_sort_ratio`
- Threshold linh hoạt: 75 cho từ dài, 65 cho từ ngắn (≤4 ký tự)

**Phrase-level correction:**
- Sửa cụm từ (ví dụ: "giam can" → "giảm cân")
- Kiểm tra overlap ratio để tránh thay thế sai
- Chỉ chấp nhận nếu similarity ≥ 82% và overlap ≥ 60%

**Ví dụ:**
```
Input:  "toi muon giam can" (thiếu dấu)
Output: "toi muon giam can" (được match với "giảm cân" trong keyword maps)
```

---

### 2. Trích xuất thông tin từ câu hỏi

#### 2.1. Keyword Matching
```python
parse_meal_plan_question(question) -> dict
```

**Trích xuất 4 loại thông tin:**

1. **Tình trạng sức khỏe** (`health_status`):
   - Ví dụ: "Táo bón", "Béo phì", "Tiểu đường", "Tim mạch", "Huyết áp"
   - Hỗ trợ **nhiều điều kiện** trong một câu hỏi

2. **Mục tiêu** (`goal`):
   - Ví dụ: "Giảm cân", "Tăng cân", "Tăng chất xơ", "Ổn định đường huyết"
   - Hỗ trợ **nhiều mục tiêu** trong một câu hỏi

3. **Chế độ ăn** (`diet_type`):
   - Ví dụ: "Eat clean", "Keto", "Low carb", "Vegetarian"

4. **Các bữa được yêu cầu** (`requested_meals`):
   - Ví dụ: "Bữa sáng", "Bữa trưa", "Bữa tối", "Bữa phụ"
   - Mặc định: Tất cả các bữa nếu không chỉ định

**Ví dụ parsing:**
```python
Input: "Tôi bị tiểu đường và muốn giảm cân, cho tôi thực đơn eat clean buổi sáng"

Output: {
    "health_status": "Tiểu đường",
    "goal": "Giảm cân",
    "diet_type": "Eat clean",
    "requested_meals": ["Bữa sáng"],
    "normalized_question": "toi bi tieu duong va muon giam can...",
    "dict_corrected_base": "toi bi tieu duong va muon giam can..."
}
```

#### 2.2. Keyword Maps
Sử dụng file `config/keyword_maps.json` để:
- Map các từ khóa không dấu → label có dấu
- Hỗ trợ nhiều cách diễn đạt khác nhau cho cùng một ý
- Tránh overlap giữa các keyword dài và ngắn

---

### 3. Gợi ý thực đơn và tạo câu trả lời

#### 3.1. Meal Plan Recommendation
```python
recommend_meal_plan(original_question, parsed_params) -> list[dict]
```

**Quy trình:**

**Bước 1: Kiểm tra intent**
- Nếu có `health_status`, `goal`, hoặc `diet_type` → Lọc và rank
- Nếu không có nhưng là câu hỏi chung về thực đơn → Random safe plan
- Nếu không phải câu hỏi dinh dưỡng → Trả `[]` để Gemini xử lý

**Bước 2: Lọc dữ liệu (nếu có keywords)**
- Lọc `meal_plans_df` theo:
  - `health_status` (hỗ trợ nhiều điều kiện, dùng `contains`)
  - `goal` (hỗ trợ nhiều mục tiêu)
  - `diet_type` (exact match)
- Nếu không có kết quả → Trả `[]`

**Bước 3: Ranking với PhoBERT**
- Embed câu hỏi bằng PhoBERT model
- Tính cosine similarity với các meal plan embeddings
- Chọn top candidates (similarity ≥ 0.5)
- Nếu similarity thấp, thử lại với structured query text

**Bước 4: Cache và rotation**
- Cache kết quả cho cùng một câu hỏi
- Nếu user hỏi lại, trả về candidate tiếp theo (rotation)

**Bước 5: Format output**
- Chỉ trả về các bữa được yêu cầu
- Nếu bữa nào không có → "Không có gợi ý"

**Ví dụ output:**
```python
[
    {
        "Bữa sáng": "Cháo yến mạch với sữa tách béo và chuối",
        "Bữa trưa": "Salad rau xanh với ức gà nướng",
        "Bữa tối": "Cá hồi áp chảo với rau luộc",
        "Bữa phụ": "Sữa chua không đường"
    }
]
```

#### 3.2. Natural Language Generation
```python
generate_answer_with_fallback(question, parsed_params, recommendations) -> str
```

**Quy trình:**

**Nếu có recommendations:**
- Gọi `generate_natural_response_from_recommendations()`
- Tạo prompt cho Gemini với:
  - Câu hỏi của user
  - Thông tin health/goal
  - Thực đơn đã được gợi ý
- Gemini diễn giải lại thành câu trả lời tự nhiên
- Fallback: Nếu Gemini lỗi → Format thủ công

**Nếu không có recommendations:**
- Kiểm tra xem có phải câu hỏi dinh dưỡng không (có keywords)
- Nếu có → Gemini tự đề xuất thực đơn
- Nếu không → Gemini trả lời chung

**Ví dụ output:**
```
"Chào bạn! Để giúp bạn giảm cân hiệu quả, tôi gợi ý thực đơn sáng như sau: 
Bữa sáng: Cháo yến mạch với sữa tách béo và chuối. 
Thực đơn này tập trung vào protein nạc và chất xơ, giúp bạn no lâu và giảm cân an toàn."
```

---

## 🏗️ Kiến trúc và Components

### Class: `MealPlanRecommender`

**Chức năng:** Quản lý toàn bộ models và data

**Attributes:**
- `meal_plans_df`: DataFrame chứa tất cả meal plans
- `user_feature_encoder`: OneHotEncoder cho user features
- `user_features_encoded_matrix`: Ma trận encoded user features
- `meal_features_phobert_matrix`: Ma trận PhoBERT embeddings
- `phobert_model`: PhoBERT model để embed queries
- `phobert_tokenizer`: Tokenizer cho PhoBERT
- `last_recommendation_cache`: Cache để rotation

**Methods:**
- `_load_artifacts()`: Load tất cả models và data từ disk
- `is_ready()`: Kiểm tra xem đã sẵn sàng chưa

### Global Instance
```python
recommender = MealPlanRecommender()  # Singleton instance
```

---

## 🔄 Luồng xử lý hoàn chỉnh

```
User Question
    ↓
1. normalize_user_question()
   - Bỏ dấu, sửa teencode
   - Fuzzy correction (token + phrase level)
    ↓
2. parse_meal_plan_question()
   - Keyword matching
   - Trích xuất: health_status, goal, diet_type, requested_meals
    ↓
3. recommend_meal_plan()
   - Lọc data theo keywords
   - PhoBERT embedding + cosine similarity
   - Ranking và selection
    ↓
4. generate_answer_with_fallback()
   - Nếu có recommendations → Gemini diễn giải
   - Nếu không → Gemini tự đề xuất hoặc trả lời chung
    ↓
Natural Language Response
```

---

## 🔧 Các hàm hỗ trợ quan trọng

### Normalization Functions
- `_strip_accents()`: Bỏ dấu tiếng Việt
- `_basic_normalize()`: Chuẩn hóa cơ bản (lowercase, teencode, bỏ dấu)
- `_shrink_repeated_chars()`: Giảm ký tự lặp
- `_tokenize_and_shrink()`: Tokenize và shrink

### Correction Functions
- `_token_level_correction()`: Sửa lỗi từng từ
- `_phrase_level_correction()`: Sửa lỗi cụm từ
- `_build_vocabs_from_keyword_maps()`: Build vocabulary từ keyword maps

### Parsing Functions
- `_parse_question_with_keywords()`: Parse với keyword matching
- `find_keywords_multiple()`: Tìm nhiều keywords (hỗ trợ overlap control)
- `find_requested_meals()`: Tìm các bữa được yêu cầu

### Recommendation Functions
- `_get_query_embedding()`: Embed câu hỏi bằng PhoBERT
- `_build_structured_query_text()`: Tạo structured text từ parsed params
- `_is_generic_meal_query()`: Kiểm tra câu hỏi chung về thực đơn

### LLM Integration
- `_call_gemini_llm()`: Gọi Gemini API
- `_build_natural_response_prompt()`: Tạo prompt cho Gemini

---

## 📊 Logging

File này có 2 loggers:

1. **`llm_logger`**: Log tất cả interactions với Gemini
   - File: `logs/llm_interactions.log`
   - Ghi lại: prompts, responses, errors

2. **`meal_plan_logger`**: Log meal plan requests
   - File: `logs/meal_plan_requests.log`
   - Ghi lại: questions, normalized questions, parsed params, recommendations

---

## ⚙️ Configuration

### Constants
- `SIMILARITY_THRESHOLD = 0.5`: Ngưỡng similarity tối thiểu
- `INTENT_MIN_SIMILARITY_FOR_WEAK_KEYWORDS = 0.6`: Ngưỡng cho weak keywords
- `DEFAULT_MEALS`: Các bữa mặc định

### Environment Variables
- `GOOGLE_API_KEY`: API key cho Gemini (từ `.env`)
- `GEMINI_MODEL`: Model name (mặc định: `gemini-2.5-flash`)

### File Paths
- `MODEL_DIR`: `models/meal_plan/`
- `KEYWORD_CONFIG_PATH`: `config/keyword_maps.json`
- `VI_DICT_PATH`: `data/vietnamese_dict.txt` (optional)

---

## 🎯 Use Cases

### Use Case 1: Câu hỏi đầy đủ thông tin
```
Input: "Tôi bị tiểu đường và muốn giảm cân, cho tôi thực đơn eat clean"
→ Parse: health_status="Tiểu đường", goal="Giảm cân", diet_type="Eat clean"
→ Filter data → Rank → Recommend → Gemini response
```

### Use Case 2: Câu hỏi chỉ có mục tiêu
```
Input: "Tôi muốn tăng cân, nên ăn gì?"
→ Parse: goal="Tăng cân"
→ Filter data → Rank → Recommend → Gemini response
```

### Use Case 3: Câu hỏi chung chung
```
Input: "Cho tôi thực đơn hôm nay"
→ Parse: Không có keywords
→ Detect generic meal query → Random safe plan → Gemini response
```

### Use Case 4: Câu hỏi ngoài dinh dưỡng
```
Input: "Hôm nay trời đẹp quá"
→ Parse: Không có keywords, không phải meal query
→ Return [] → Gemini trả lời chung
```

---

## 🔗 Integration với API

File này được sử dụng trong `api/main.py`:

```python
from services.meal_plan_inference import (
    generate_answer_with_fallback,
    parse_meal_plan_question,
    recommend_meal_plan,
)

# Endpoint: /recommend_meal_plan
# Endpoint: /ask (khi intent = "meal_plan")
```

---

## 📝 Tóm tắt

**File `meal_plan_inference.py` là trái tim của hệ thống gợi ý thực đơn:**

1. ✅ **Xử lý tiếng Việt phức tạp** (teencode, lỗi chính tả, không dấu)
2. ✅ **Trích xuất thông tin thông minh** (health, goal, diet, meals)
3. ✅ **Gợi ý thực đơn chính xác** (filtering + ranking với PhoBERT)
4. ✅ **Tạo câu trả lời tự nhiên** (tích hợp Gemini LLM)
5. ✅ **Fallback linh hoạt** (nếu không có data → Gemini tự đề xuất)

**Đây là một hệ thống end-to-end hoàn chỉnh từ câu hỏi tự nhiên → thực đơn cụ thể → câu trả lời tự nhiên!**


