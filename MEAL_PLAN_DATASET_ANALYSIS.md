# BÁO CÁO PHÂN TÍCH DATASET_THUCDON.CSV VÀ LUỒNG XỬ LÝ MEAL PLAN

## 📋 MỤC LỤC

1. [Tổng quan Dataset](#1-tổng-quan-dataset)
2. [Cấu trúc dữ liệu](#2-cấu-trúc-dữ-liệu)
3. [Phân tích chi tiết](#3-phân-tích-chi-tiết)
4. [Luồng Training Model](#4-luồng-training-model)
5. [Luồng Inference (Gợi ý thực đơn)](#5-luồng-inference-gợi-ý-thực-đơn)
6. [Công nghệ và Thuật toán](#6-công-nghệ-và-thuật-toán)
7. [Điểm mạnh và Hạn chế](#7-điểm-mạnh-và-hạn-chế)

---

## 1. TỔNG QUAN DATASET

### 1.1. Thông tin cơ bản

- **Tên file**: `Dataset_Thucdon.csv`
- **Số lượng mẫu**: ~2,347 thực đơn (2,348 dòng bao gồm header)
- **Mục đích**: Dataset thực đơn dinh dưỡng được cá nhân hóa theo tình trạng sức khỏe và mục tiêu
- **Định dạng**: CSV, encoding UTF-8

### 1.2. Đặc điểm nổi bật

✅ **Đa dạng**: Bao gồm cả chế độ ăn chay và không chay  
✅ **Cá nhân hóa**: Mỗi thực đơn được thiết kế cho tình trạng sức khỏe cụ thể  
✅ **Đầy đủ**: Có đủ 4 bữa: sáng, trưa, tối, phụ  
✅ **Thực tế**: Món ăn Việt Nam phổ biến, dễ thực hiện  

---

## 2. CẤU TRÚC DỮ LIỆU

### 2.1. Các cột trong dataset

| Cột | Kiểu dữ liệu | Mô tả | Ví dụ |
|-----|--------------|-------|-------|
| **Bữa sáng** | String | Món ăn bữa sáng | "Khoai lang hấp" |
| **Bữa trưa** | String | Món ăn bữa trưa | "Canh gà lá giang; Miến; Trái cây ít đường" |
| **Bữa tối** | String | Món ăn bữa tối | "Phở bò; Rau luộc; Cơm trắng ít" |
| **Bữa phụ** | String | Món ăn bữa phụ | "Hạt hạnh nhân" |
| **Chế độ ăn** | String | Chế độ ăn uống | "Không chay", "chay" |
| **Tình trạng sức khỏe** | String | Các vấn đề sức khỏe (nhiều giá trị) | "Tim mạch, Suy dinh dưỡng, Thiếu kẽm, Tiểu đường" |
| **Mục tiêu** | String | Mục tiêu dinh dưỡng (nhiều giá trị) | "Giảm muối, ổn định huyết áp, Giảm mỡ bão hòa..." |
| **Táo bón** | Binary (0/1) | Có bị táo bón không | 0 hoặc 1 |
| **Béo phì** | Binary (0/1) | Có bị béo phì không | 0 hoặc 1 |
| **Tim mạch** | Binary (0/1) | Có vấn đề tim mạch không | 0 hoặc 1 |
| **Mỡ trong máu** | Binary (0/1) | Có mỡ trong máu không | 0 hoặc 1 |
| **Huyết áp** | Binary (0/1) | Có vấn đề huyết áp không | 0 hoặc 1 |
| **Thiếu máu** | Binary (0/1) | Có thiếu máu không | 0 hoặc 1 |
| **Thiếu kẽm** | Binary (0/1) | Có thiếu kẽm không | 0 hoặc 1 |
| **Thiếu canxi** | Binary (0/1) | Có thiếu canxi không | 0 hoặc 1 |
| **Suy dinh dưỡng** | Binary (0/1) | Có suy dinh dưỡng không | 0 hoặc 1 |
| **Tiểu đường** | Binary (0/1) | Có tiểu đường không | 0 hoặc 1 |

### 2.2. Cấu trúc dữ liệu đặc biệt

- **Tình trạng sức khỏe**: Có thể chứa nhiều giá trị, phân cách bằng dấu phẩy
- **Mục tiêu**: Có thể chứa nhiều mục tiêu, phân cách bằng dấu phẩy
- **Món ăn**: Có thể chứa nhiều món trong 1 bữa, phân cách bằng dấu chấm phẩy (`;`)

### 2.3. Ví dụ dữ liệu

```csv
Bữa sáng,Bữa phụ,Bữa trưa,Bữa tối,Chế độ ăn,...,Tình trạng sức khỏe,Mục tiêu
Khoai lang hấp,Hạt hạnh nhân,Canh gà lá giang; Miến; Trái cây ít đường,Phở bò; Rau luộc; Cơm trắng ít,Không chay,...,"Tim mạch, Suy dinh dưỡng, Thiếu kẽm, Tiểu đường","Giảm muối, ổn định huyết áp, Giảm mỡ bão hòa, tăng chất xơ, Tăng cân, bổ sung năng lượng và đạm, Bổ sung kẽm và đạm, Ổn định đường huyết, Tăng chất xơ hòa tan"
```

---

## 3. PHÂN TÍCH CHI TIẾT

### 3.1. Phân loại theo chế độ ăn

- **Không chay**: Phần lớn các thực đơn
- **Chay**: Một số thực đơn dành cho người ăn chay

### 3.2. Phân loại theo tình trạng sức khỏe

Các tình trạng sức khỏe phổ biến trong dataset:

1. **Tim mạch**: Thực đơn giảm muối, giảm mỡ bão hòa
2. **Béo phì**: Thực đơn giảm cân, kiểm soát năng lượng
3. **Tiểu đường**: Thực đơn ổn định đường huyết, tăng chất xơ
4. **Suy dinh dưỡng**: Thực đơn tăng cân, bổ sung năng lượng và đạm
5. **Thiếu máu**: Thực đơn bổ sung sắt
6. **Thiếu canxi**: Thực đơn bổ sung canxi và vitamin D
7. **Thiếu kẽm**: Thực đơn bổ sung kẽm và đạm
8. **Huyết áp**: Thực đơn giảm muối, ổn định huyết áp
9. **Mỡ trong máu**: Thực đơn giảm mỡ bão hòa, tăng chất xơ
10. **Táo bón**: Thực đơn tăng chất xơ, uống đủ nước

### 3.3. Phân loại theo mục tiêu

Các mục tiêu dinh dưỡng phổ biến:

- **Giảm cân**: Kiểm soát năng lượng, hạn chế chất béo xấu
- **Tăng cân**: Bổ sung năng lượng và đạm
- **Ổn định đường huyết**: Cho người tiểu đường
- **Giảm muối**: Cho người tim mạch, huyết áp
- **Bổ sung vi chất**: Sắt, canxi, kẽm
- **Tăng chất xơ**: Cho táo bón, tiểu đường

### 3.4. Đặc điểm món ăn

- **Đa dạng**: Từ món đơn giản (khoai lang hấp) đến món phức tạp (phở bò, bánh xèo)
- **Thực tế**: Món ăn Việt Nam quen thuộc, dễ tìm nguyên liệu
- **Cân bằng**: Mỗi bữa thường có đủ tinh bột, đạm, rau củ

---

## 4. LUỒNG TRAINING MODEL

### 4.1. Tổng quan

File: `scripts/train_meal_plan_model.py`

### 4.2. Các bước xử lý

#### **Bước 1: Load dữ liệu**

```python
meal_plans_df = pd.read_csv('data/Dataset_Thucdon.csv', encoding='utf-8')
```

- Đọc file CSV với encoding UTF-8
- Kết quả: DataFrame với ~2,347 thực đơn

#### **Bước 2: Chuẩn hóa tên cột**

```python
meal_plans_df = meal_plans_df.rename(columns={
    'Chế độ ăn': 'che_do_an',
    'Tình trạng sức khỏe': 'tinh_trang_suc_khoe',
    'Mục tiêu': 'muc_tieu',
    'Bữa sáng': 'bua_sang',
    'Bữa trưa': 'bua_trua',
    'Bữa tối': 'bua_toi',
    'Bữa phụ': 'bua_phu'
})
```

- Chuyển tên cột từ tiếng Việt có dấu sang không dấu, snake_case
- Mục đích: Dễ xử lý trong code, tránh lỗi encoding

#### **Bước 3: Làm sạch dữ liệu**

```python
# Loại bỏ khoảng trắng thừa
for col in ['che_do_an', 'tinh_trang_suc_khoe', 'muc_tieu']:
    meal_plans_df[col] = meal_plans_df[col].str.strip()

# Xử lý giá trị thiếu
meal_plans_df[user_features] = meal_plans_df[user_features].fillna('không có')
```

- Loại bỏ khoảng trắng thừa
- Điền giá trị thiếu bằng "không có"

#### **Bước 4: One-Hot Encoding cho User Features**

```python
user_features = ['tinh_trang_suc_khoe', 'muc_tieu', 'che_do_an']
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
user_features_encoded = encoder.fit_transform(meal_plans_df[user_features])
```

- Mã hóa các đặc trưng người dùng (tình trạng sức khỏe, mục tiêu, chế độ ăn)
- Kết quả: Ma trận sparse one-hot encoding
- Lưu encoder để dùng trong inference

#### **Bước 5: Tạo embedding text cho PhoBERT**

```python
# Kết hợp tất cả các bữa ăn
meal_plans_df['full_meal_plan'] = meal_plans_df[meal_cols].fillna('').apply(
    lambda row: ' '.join(row), axis=1
)

# Tạo text embedding bao gồm context
meal_plans_df['embedding_text'] = (
    meal_plans_df['che_do_an'] + ' ' +
    meal_plans_df['tinh_trang_suc_khoe'] + ' ' +
    meal_plans_df['muc_tieu'] + ' ' +
    meal_plans_df['full_meal_plan']
)
```

- Kết hợp nội dung các bữa ăn thành 1 chuỗi
- Thêm context (chế độ ăn, tình trạng sức khỏe, mục tiêu) vào embedding text
- Mục đích: Tạo embedding phong phú, có ngữ cảnh

#### **Bước 6: Sinh PhoBERT Embeddings**

```python
phobert_model = AutoModel.from_pretrained("vinai/phobert-base")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

def get_phobert_embeddings(texts, batch_size=32):
    # Tokenize
    inputs = tokenizer(texts, padding=True, truncation=True, 
                      return_tensors="pt", max_length=256)
    # Forward pass
    with torch.no_grad():
        outputs = phobert_model(**inputs)
    # Mean pooling
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings

meal_features_phobert = get_phobert_embeddings(meal_plan_texts)
```

- Load model PhoBERT (vinai/phobert-base) - mô hình BERT cho tiếng Việt
- Tokenize và tạo embedding cho mỗi thực đơn
- Mean pooling: Lấy trung bình của các token embeddings
- Kết quả: Ma trận embedding (N x 768) với N = số thực đơn

#### **Bước 7: Lưu artifacts**

```python
# Lưu DataFrame đã xử lý
output_df.to_csv(model_dir / "meal_plans_data.csv", ...)

# Lưu encoder
joblib.dump(encoder, model_dir / "user_feature_encoder.pkl")

# Lưu PhoBERT model và tokenizer
phobert_model.save_pretrained(model_dir / "phobert_model")
tokenizer.save_pretrained(model_dir / "phobert_tokenizer")

# Lưu ma trận embedding
scipy.sparse.save_npz(model_dir / "user_features_encoded.npz", ...)
np.save(model_dir / "meal_features_phobert.npy", meal_features_phobert)
```

**Artifacts được lưu:**

1. `meal_plans_data.csv`: Dataset đã xử lý
2. `user_feature_encoder.pkl`: OneHotEncoder đã fit
3. `phobert_model/`: PhoBERT model đã lưu
4. `phobert_tokenizer/`: Tokenizer đã lưu
5. `user_features_encoded.npz`: Ma trận one-hot encoding (sparse)
6. `meal_features_phobert.npy`: Ma trận PhoBERT embeddings (dense)

---

## 5. LUỒNG INFERENCE (GỢI Ý THỰC ĐƠN)

### 5.1. Tổng quan

File: `services/meal_plan_inference.py`

### 5.2. Sơ đồ luồng xử lý

```
User Question
    ↓
[1. Parse & Normalize]
    ↓
[2. Extract Keywords]
    ↓
[3. Filter Dataset]
    ↓
[4. PhoBERT Similarity]
    ↓
[5. Rank & Select]
    ↓
[6. Generate Response (Gemini)]
    ↓
Final Answer
```

### 5.3. Chi tiết từng bước

#### **Bước 1: Parse và Normalize câu hỏi**

**Hàm**: `parse_meal_plan_question(question: str)`

```python
# 1.1. Normalize + Fuzzy Correction
dict_corrected_base, normalized_question = normalize_user_question(
    question,
    keyword_maps=keyword_maps,
    vi_wordlist_tokens=vi_tokens,
    phrase_threshold=82,
    token_threshold=70
)
```

**Xử lý:**
- **Chuẩn hóa cơ bản**: Lowercase, bỏ dấu, xử lý teencode
- **Fuzzy correction**: Sửa lỗi chính tả ở mức phrase và token
- **Sử dụng từ điển**: `keyword_maps.json` và `vietnamese_dict.txt`

**Ví dụ:**
```
Input:  "Tôi muốn thực đơn eat clean để giảm cân"
Output: "toi muon thuc don eat clean de giam can"
```

#### **Bước 2: Extract Keywords**

**Hàm**: `_parse_question_with_keywords(question_lower: str)`

```python
parsed_params = {
    "health_status": "không có",      # Tình trạng sức khỏe
    "goal": "giảm cân",                # Mục tiêu
    "diet_type": "eat clean",          # Chế độ ăn
    "requested_meals": ["Bữa sáng", "Bữa trưa", "Bữa tối", "Bữa phụ"]
}
```

**Xử lý:**
- Tìm kiếm keywords trong `keyword_maps.json`:
  - `health_status_map`: Tim mạch, Tiểu đường, Béo phì...
  - `goal_map`: Giảm cân, Tăng cân, Ổn định đường huyết...
  - `diet_type_map`: Eat clean, Chay, Không chay...
  - `meal_map`: Bữa sáng, Bữa trưa, Bữa tối, Bữa phụ
- Hỗ trợ nhiều giá trị (ví dụ: "Tim mạch, Tiểu đường")

#### **Bước 3: Filter Dataset**

**Hàm**: `recommend_meal_plan(original_question, parsed_params)`

**Logic lọc:**

```python
# 3.1. Kiểm tra có keywords không
has_any_keyword = any(
    v and v != 'không có'
    for v in (health_status, goal, diet_type)
)

if has_any_keyword:
    # 3.2. Lọc theo chế độ ăn
    if diet_type != 'không có':
        conditions.append(
            filtered_df['Chế độ ăn'].str.lower() == diet_type.lower()
        )
    
    # 3.3. Lọc theo tình trạng sức khỏe (hỗ trợ nhiều giá trị)
    if health_status != 'không có':
        health_keywords = health_status.split(',')
        for kw in health_keywords:
            health_condition = health_condition | col.str.contains(kw)
        conditions.append(health_condition)
    
    # 3.4. Lọc theo mục tiêu (hỗ trợ nhiều giá trị)
    if goal != 'không có':
        goal_keywords = goal.split(',')
        for kw in goal_keywords:
            goal_condition = goal_condition | col.str.contains(kw)
        conditions.append(goal_condition)
    
    # 3.5. Áp dụng tất cả điều kiện (AND logic)
    filtered_df = filtered_df[np.logical_and.reduce(conditions)]
```

**Kết quả**: DataFrame đã được lọc theo tiêu chí người dùng

#### **Bước 4: PhoBERT Similarity Ranking**

```python
# 4.1. Tạo embedding cho câu hỏi
query_embedding = _get_query_embedding(
    query_text_for_embedding, 
    recommender.phobert_model, 
    recommender.phobert_tokenizer
)

# 4.2. Tính cosine similarity với các thực đơn đã lọc
similarity_scores = cosine_similarity(
    query_embedding, 
    filtered_embeddings
).flatten()

# 4.3. Sắp xếp theo độ tương đồng
top_indices = similarity_scores.argsort()[::-1]

# 4.4. Kiểm tra threshold
if best_score < SIMILARITY_THRESHOLD:  # 0.5
    # Thử lại với structured query
    query_text = "Tình trạng sức khỏe: ... | Mục tiêu: ..."
    # Hoặc trả về []
```

**Xử lý:**
- Tạo embedding cho câu hỏi bằng PhoBERT
- Tính cosine similarity với embeddings của các thực đơn đã lọc
- Sắp xếp theo độ tương đồng giảm dần
- Nếu similarity < 0.5, thử lại với structured query hoặc trả về []

#### **Bước 5: Rank & Select**

```python
# 5.1. Lấy top candidates
original_relevant_indices = [filtered_indices[i] for i in top_indices]

# 5.2. Xử lý cache (để tránh trả về cùng 1 kết quả nhiều lần)
if original_question in recommender.last_recommendation_cache:
    # Lấy kết quả tiếp theo
    last_shown_index = recommender.last_recommendation_cache[original_question]
    next_pos = (current_pos + 1) % len(original_relevant_indices)
    best_match_index = original_relevant_indices[next_pos]
else:
    # Lấy kết quả đầu tiên
    best_match_index = original_relevant_indices[0]

# 5.3. Lưu vào cache
recommender.last_recommendation_cache[original_question] = best_match_index
```

**Tính năng đặc biệt:**
- **Cache mechanism**: Tránh trả về cùng 1 thực đơn nhiều lần
- **Rotation**: Mỗi lần hỏi lại sẽ trả về thực đơn khác (nếu có)

#### **Bước 6: Generate Natural Response (Gemini)**

**Hàm**: `generate_natural_response_from_recommendations()`

```python
# 6.1. Tạo prompt cho Gemini
prompt = f"""Bạn là một chuyên gia dinh dưỡng AI...
Người dùng hỏi: "{question}"
{health_goal_context}

Dựa trên dữ liệu, đây là gợi ý thực đơn:
{recs_str}

Yêu cầu:
1. Chỉ diễn giải lại thực đơn, GIỮ NGUYÊN tên món ăn
2. Viết MỘT đoạn văn ngắn (3–5 câu), thân thiện
"""

# 6.2. Gọi Gemini API
natural_response = await _call_gemini_llm(prompt=prompt, json_mode=False)

# 6.3. Xử lý response
return natural_response.replace("\n", " ").strip()
```

**Xử lý:**
- Tạo prompt chi tiết với context về sức khỏe và mục tiêu
- Gọi Gemini API để tạo câu trả lời tự nhiên
- Xử lý và làm sạch response

### 5.4. Các trường hợp đặc biệt

#### **Trường hợp 1: Có keywords (health/goal/diet)**

```
Input: "Tôi muốn thực đơn eat clean để giảm cân"
  ↓
Extract: goal="giảm cân", diet_type="eat clean"
  ↓
Filter: Lọc dataset theo goal và diet_type
  ↓
PhoBERT: Tính similarity, rank
  ↓
Select: Chọn thực đơn tốt nhất
  ↓
Gemini: Tạo câu trả lời tự nhiên
```

#### **Trường hợp 2: Generic meal query (không có keywords)**

```
Input: "Thực đơn hôm nay là gì?"
  ↓
Detect: Generic meal query
  ↓
Filter: Lọc thực đơn "bình thường" hoặc random
  ↓
Select: Chọn ngẫu nhiên 1 thực đơn an toàn
  ↓
Gemini: Tạo câu trả lời
```

#### **Trường hợp 3: Không có keywords và không phải generic query**

```
Input: "Hôm nay trời đẹp"
  ↓
Detect: Không phải câu hỏi về dinh dưỡng
  ↓
Return: [] (empty)
  ↓
Gemini Fallback: Trả lời chung chung hoặc từ chối
```

### 5.5. Fallback mechanism

```python
async def generate_answer_with_fallback(question, parsed_params, recommendations):
    # 1. Nếu có recommendations từ data
    if recommendations:
        return await generate_natural_response_from_recommendations(...)
    
    # 2. Nếu không có recommendations
    if has_any_keyword:
        # Câu hỏi dinh dưỡng nhưng không có data
        prompt = "Tự đề xuất thực đơn dựa trên tình trạng..."
    else:
        # Câu hỏi ngoài lĩnh vực dinh dưỡng
        prompt = "Trả lời chung chung..."
    
    return await _call_gemini_llm(prompt)
```

---

## 6. CÔNG NGHỆ VÀ THUẬT TOÁN

### 6.1. Công nghệ sử dụng

| Công nghệ | Mục đích | Version/Library |
|-----------|----------|-----------------|
| **Python** | Ngôn ngữ chính | 3.8+ |
| **pandas** | Xử lý dữ liệu | - |
| **numpy** | Tính toán số học | - |
| **scikit-learn** | OneHotEncoder, cosine_similarity | - |
| **transformers** | PhoBERT model | Hugging Face |
| **torch** | Deep learning framework | PyTorch |
| **rapidfuzz** | Fuzzy string matching | - |
| **google.generativeai** | Gemini LLM API | - |
| **joblib** | Serialize models | - |
| **scipy** | Sparse matrix operations | - |

### 6.2. Thuật toán và Mô hình

#### **6.2.1. One-Hot Encoding**

- **Mục đích**: Mã hóa các đặc trưng phân loại (tình trạng sức khỏe, mục tiêu, chế độ ăn)
- **Cách hoạt động**: Tạo vector binary cho mỗi giá trị có thể
- **Ví dụ**: 
  - "chay" → [0, 1]
  - "Không chay" → [1, 0]

#### **6.2.2. PhoBERT Embeddings**

- **Model**: `vinai/phobert-base`
- **Kiến trúc**: BERT-based, được fine-tune cho tiếng Việt
- **Input**: Text (câu hỏi hoặc mô tả thực đơn)
- **Output**: Vector embedding 768 chiều
- **Pooling**: Mean pooling (trung bình các token embeddings)

**Ví dụ embedding text:**
```
"chay Tim mạch, Tiểu đường Giảm muối, ổn định huyết áp Khoai lang hấp Hạt hạnh nhân Canh gà lá giang; Miến; Trái cây ít đường Phở bò; Rau luộc; Cơm trắng ít"
```

#### **6.2.3. Cosine Similarity**

- **Công thức**: `similarity = cos(θ) = (A · B) / (||A|| × ||B||)`
- **Mục đích**: Đo độ tương đồng giữa embedding câu hỏi và embedding thực đơn
- **Range**: [-1, 1], thường dùng [0, 1] cho embeddings
- **Threshold**: 0.5 (chỉ chấp nhận similarity >= 0.5)

#### **6.2.4. Fuzzy String Matching**

- **Library**: `rapidfuzz`
- **Scorers**: 
  - `fuzz.ratio`: So sánh toàn bộ chuỗi
  - `fuzz.partial_ratio`: So sánh phần chuỗi
  - `fuzz.token_sort_ratio`: So sánh sau khi sắp xếp tokens
- **Mục đích**: Sửa lỗi chính tả, xử lý biến thể từ

#### **6.2.5. Gemini LLM**

- **Model**: `gemini-2.5-flash` (hoặc configurable)
- **Mục đích**: Tạo câu trả lời tự nhiên từ structured data
- **Temperature**: 0.7 (cho natural language), 0.2 (cho JSON mode)
- **Safety settings**: Disabled (để có control tốt hơn)

### 6.3. Pipeline Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                    │
└─────────────────────────────────────────────────────────┘

Dataset_Thucdon.csv
    ↓
[Data Cleaning & Normalization]
    ↓
[One-Hot Encoding] → user_features_encoded.npz
    ↓
[PhoBERT Embedding] → meal_features_phobert.npy
    ↓
[Save Artifacts]
    ├── meal_plans_data.csv
    ├── user_feature_encoder.pkl
    ├── phobert_model/
    ├── phobert_tokenizer/
    ├── user_features_encoded.npz
    └── meal_features_phobert.npy

┌─────────────────────────────────────────────────────────┐
│                   INFERENCE PIPELINE                    │
└─────────────────────────────────────────────────────────┘

User Question
    ↓
[Load Artifacts] (MealPlanRecommender.__init__)
    ↓
[Parse & Normalize] (parse_meal_plan_question)
    ├── normalize_user_question (fuzzy correction)
    └── _parse_question_with_keywords (extract keywords)
    ↓
[Filter Dataset] (recommend_meal_plan)
    ├── Filter by diet_type
    ├── Filter by health_status
    └── Filter by goal
    ↓
[PhoBERT Similarity] (recommend_meal_plan)
    ├── Generate query embedding
    ├── Calculate cosine similarity
    └── Rank by similarity
    ↓
[Select Best Match] (recommend_meal_plan)
    ├── Check similarity threshold
    └── Handle cache (rotation)
    ↓
[Generate Response] (generate_natural_response_from_recommendations)
    ├── Build prompt
    ├── Call Gemini API
    └── Format response
    ↓
Final Answer
```

---

## 7. ĐIỂM MẠNH VÀ HẠN CHẾ

### 7.1. Điểm mạnh

✅ **Dataset phong phú**: ~2,347 thực đơn đa dạng  
✅ **Cá nhân hóa tốt**: Mỗi thực đơn được thiết kế cho tình trạng sức khỏe cụ thể  
✅ **Xử lý tiếng Việt tốt**: PhoBERT + fuzzy matching  
✅ **Robust**: Nhiều fallback mechanisms  
✅ **User-friendly**: Cache rotation, natural language response  
✅ **Scalable**: Có thể thêm thực đơn mới vào dataset  

### 7.2. Hạn chế

⚠️ **Dataset cố định**: Không tự động cập nhật  
⚠️ **Thiếu thông tin dinh dưỡng chi tiết**: Không có calorie, macro nutrients  
⚠️ **PhoBERT inference chậm**: Cần optimize nếu scale lớn  
⚠️ **Phụ thuộc Gemini API**: Cần internet và API key  
⚠️ **Threshold cố định**: Similarity threshold = 0.5 có thể không phù hợp mọi trường hợp  

### 7.3. Khuyến nghị cải thiện

1. **Thêm thông tin dinh dưỡng**: Calorie, protein, carbs, fat cho mỗi thực đơn
2. **A/B testing threshold**: Tối ưu similarity threshold
3. **Caching embeddings**: Cache query embeddings để tăng tốc
4. **Batch processing**: Xử lý nhiều queries cùng lúc
5. **Feedback loop**: Thu thập feedback để cải thiện recommendations

---

## 8. KẾT LUẬN

Hệ thống Meal Plan Recommendation sử dụng:

- **Dataset**: `Dataset_Thucdon.csv` với ~2,347 thực đơn được cá nhân hóa
- **Training**: One-Hot Encoding + PhoBERT Embeddings
- **Inference**: Keyword parsing + Filtering + PhoBERT Similarity + Gemini LLM
- **Công nghệ**: Python, scikit-learn, Transformers (PhoBERT), Gemini API

Hệ thống có khả năng:
- Hiểu câu hỏi tiếng Việt tự nhiên
- Lọc và gợi ý thực đơn phù hợp với tình trạng sức khỏe
- Tạo câu trả lời tự nhiên, thân thiện

Đây là một hệ thống recommendation system hoàn chỉnh, kết hợp rule-based filtering và semantic similarity để đưa ra gợi ý chính xác và phù hợp.



