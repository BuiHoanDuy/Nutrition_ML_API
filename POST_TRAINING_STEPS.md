# 📋 Các bước tiếp theo sau khi train model Meal Plan

Sau khi chạy `python scripts/train_meal_plan_model.py` thành công, bạn cần thực hiện các bước sau để sử dụng model:

## ✅ Bước 1: Kiểm tra các file model đã được tạo

Sau khi train xong, kiểm tra xem các file sau đã được tạo trong thư mục `models/meal_plan/`:

```bash
models/meal_plan/
├── meal_plans_data.csv              # Dữ liệu thực đơn đã xử lý
├── user_feature_encoder.pkl         # Encoder cho user features
├── user_features_encoded.npz        # Ma trận encoded user features
├── meal_features_phobert.npy        # Ma trận embeddings PhoBERT
├── phobert_model/                   # Model PhoBERT đã lưu
│   ├── config.json
│   └── model.safetensors
└── phobert_tokenizer/              # Tokenizer PhoBERT
    ├── vocab.txt
    ├── tokenizer_config.json
    └── ...
```

**Kiểm tra nhanh:**
```bash
# Trên Windows
dir models\meal_plan

# Trên Linux/Mac
ls -la models/meal_plan/
```

## 🔑 Bước 2: Cấu hình Environment Variables

Tạo file `.env` ở thư mục gốc của project (nếu chưa có) và thêm API key cho Gemini:

```bash
# Tạo file .env
GOOGLE_API_KEY=your_google_api_key_here
GEMINI_MODEL=gemini-2.0-flash-exp  # hoặc gemini-1.5-pro
```

**Lưu ý:**
- Nếu không có `GOOGLE_API_KEY`, hệ thống vẫn hoạt động nhưng sẽ sử dụng fallback responses (không có LLM)
- Bạn có thể lấy API key tại: https://makersuite.google.com/app/apikey

## 🚀 Bước 3: Chạy server để test

Khởi động API server:

```bash
python run_server.py
```

Server sẽ chạy tại: `http://127.0.0.1:8000`

Bạn sẽ thấy thông báo:
```
[INFO] Loading meal plan recommender artifacts...
[SUCCESS] Meal plan recommender artifacts loaded successfully.
```

## 🧪 Bước 4: Test API endpoints

### 4.1. Test qua Swagger UI (Khuyến nghị)

Mở trình duyệt và truy cập:
```
http://127.0.0.1:8000/docs
```

Tại đây bạn có thể:
- Xem tất cả các endpoints
- Test trực tiếp từ giao diện web
- Xem request/response examples

### 4.2. Test qua cURL

**Test meal plan recommendation:**
```bash
curl -X POST "http://127.0.0.1:8000/recommend_meal_plan" \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"Tôi muốn giảm cân, sáng nên ăn gì?\"}"
```

**Test unified intent endpoint:**
```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"Mình muốn thực đơn eat clean để giảm cân.\"}"
```

### 4.3. Test với Python

Tạo file `test_meal_plan.py`:

```python
import requests
import json

# Test meal plan recommendation
url = "http://127.0.0.1:8000/recommend_meal_plan"
data = {
    "question": "Tôi muốn giảm cân, sáng nên ăn gì?"
}

response = requests.post(url, json=data)
print(json.dumps(response.json(), indent=2, ensure_ascii=False))
```

## 📊 Bước 5: Kiểm tra logs

Xem logs để debug và theo dõi hoạt động:

```bash
# Xem logs meal plan requests
# Trên Windows
type logs\meal_plan_requests.log

# Trên Linux/Mac
tail -f logs/meal_plan_requests.log
```

Logs sẽ ghi lại:
- Câu hỏi của người dùng
- Câu hỏi đã được normalize
- Các tham số được trích xuất (health_status, goal, diet_type)
- Thực đơn được gợi ý

## 🔍 Bước 6: Kiểm tra hoạt động của model

### 6.1. Kiểm tra model đã load thành công

Khi server khởi động, bạn sẽ thấy:
```
[INFO] Loading meal plan recommender artifacts...
[INFO] Loading local PhoBERT model for inference...
[SUCCESS] Meal plan recommender artifacts loaded successfully.
```

Nếu có lỗi, kiểm tra:
- Các file model đã được tạo đầy đủ chưa
- Đường dẫn đến thư mục `models/meal_plan/` có đúng không

### 6.2. Test với các câu hỏi khác nhau

**Câu hỏi có đầy đủ thông tin:**
```
"Tôi bị tiểu đường và muốn giảm cân, cho tôi thực đơn eat clean"
```

**Câu hỏi chỉ có mục tiêu:**
```
"Tôi muốn tăng cân, nên ăn gì?"
```

**Câu hỏi chung chung:**
```
"Cho tôi thực đơn hôm nay"
```

## ⚠️ Troubleshooting

### Lỗi: "Recommender is not ready"

**Nguyên nhân:** Model chưa được load thành công

**Giải pháp:**
1. Kiểm tra các file trong `models/meal_plan/` đã đầy đủ chưa
2. Chạy lại `python scripts/train_meal_plan_model.py`
3. Kiểm tra logs để xem lỗi cụ thể

### Lỗi: "GOOGLE_API_KEY not set"

**Nguyên nhân:** Chưa cấu hình API key cho Gemini

**Giải pháp:**
1. Tạo file `.env` ở thư mục gốc
2. Thêm `GOOGLE_API_KEY=your_key_here`
3. Restart server

**Lưu ý:** Hệ thống vẫn hoạt động không có API key, nhưng sẽ dùng fallback responses.

### Lỗi: "No meal plans found after applying hard filters"

**Nguyên nhân:** Không tìm thấy thực đơn phù hợp với điều kiện

**Giải pháp:**
1. Kiểm tra dữ liệu trong `data/Dataset_Thucdon.csv`
2. Kiểm tra keyword maps trong `config/keyword_maps.json`
3. Xem logs để biết các tham số được trích xuất

### Lỗi: UnicodeEncodeError

**Nguyên nhân:** Terminal không hỗ trợ UTF-8

**Giải pháp:**
- Trên Windows: Sử dụng PowerShell hoặc Git Bash
- Đảm bảo terminal encoding là UTF-8

## 📝 Checklist sau khi train

- [ ] Các file model đã được tạo trong `models/meal_plan/`
- [ ] File `.env` đã được tạo và có `GOOGLE_API_KEY` (nếu cần)
- [ ] Server khởi động thành công không có lỗi
- [ ] Test API endpoint `/recommend_meal_plan` hoạt động
- [ ] Logs được ghi lại đúng cách
- [ ] Model trả về kết quả phù hợp với câu hỏi

## 🎯 Bước tiếp theo (Tùy chọn)

1. **Tối ưu hóa keyword maps:** Cập nhật `config/keyword_maps.json` để cải thiện độ chính xác parsing
2. **Thêm dữ liệu:** Bổ sung thêm thực đơn vào `data/Dataset_Thucdon.csv` và train lại
3. **Tuning thresholds:** Điều chỉnh `SIMILARITY_THRESHOLD` trong `meal_plan_inference.py` nếu cần
4. **Production deployment:** Deploy lên server với Gunicorn/uWSGI

## 📚 Tài liệu tham khảo

- [README.md](README.md) - Tổng quan về project
- [SETUP.md](SETUP.md) - Hướng dẫn setup chi tiết
- [MEAL_PLAN_DATASET_ANALYSIS.md](MEAL_PLAN_DATASET_ANALYSIS.md) - Phân tích dataset

---

**Chúc bạn thành công! 🎉**


