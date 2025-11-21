# 🚀 Setup Guide - Nutrition AI Application

Hướng dẫn cài đặt và chạy ứng dụng Nutrition AI trên máy mới.

## 📋 Yêu cầu hệ thống

- Python 3.8+ (khuyến nghị Python 3.11+)
- RAM: Tối thiểu 4GB (khuyến nghị 8GB+)
- Dung lượng ổ cứng: Tối thiểu 2GB trống

## 🔧 Cài đặt

### Bước 1: Clone repository

```bash
git clone <your-repository-url>
cd nutrition-ai-app
```

### Bước 2: Tạo virtual environment

```bash
# Tạo virtual environment
python -m venv venv

# Kích hoạt virtual environment
# Trên Windows:
venv\Scripts\activate
# Trên macOS/Linux:
source venv/bin/activate
```

### Bước 3: Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### Bước 4: Chuẩn bị dữ liệu

Đảm bảo bạn có file dữ liệu `data/food_nutrition_data_final.csv` trong thư mục dự án.

### Bước 5: Train models

Chạy script setup để train tất cả models:

```bash
python scripts/setup_models.py
```

Script này sẽ:
- Tạo các thư mục cần thiết
- Train obesity prediction model  
- Train meal plan recommendation model
- Lưu models vào thư mục `models/`

### Bước 6: Cấu hình environment variables (Tùy chọn)

Tạo file `.env` để cấu hình OpenRouter API (cho meal plan recommendations):

```bash
# Tạo file .env
echo OPENROUTER_API_KEY=your_api_key_here > .env
echo OPENROUTER_MODEL=mistralai/mistral-7b-instruct-v0.2 >> .env
echo OPENROUTER_MODEL_CHAT=openai/gpt-3.5-turbo >> .env
```

**Lưu ý**: Nếu không có OpenRouter API key, meal plan recommendations vẫn hoạt động nhưng sẽ sử dụng fallback responses.

### Bước 7: Chạy ứng dụng

```bash
python run_server.py
```

Server sẽ chạy tại: `http://127.0.0.1:8000`

## 🧪 Kiểm tra hoạt động

### Test API endpoints:

```bash
# Test calorie prediction
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"content": "Tôi ăn 1 bát phở bò"}'

# Test meal plan recommendation
curl -X POST "http://127.0.0.1:8000/recommend_meal_plan" \
  -H "Content-Type: application/json" \
  -d '{"question": "Tôi muốn giảm cân, sáng nên ăn gì?"}'

# Test obesity prediction
curl -X POST "http://127.0.0.1:8000/obesity/predict" \
  -H "Content-Type: application/json" \
  -d '{"Gender": "Female", "Age": 25.0, "Height": 1.65, "Weight": 70.0, "family_history_with_overweight": "yes", "FAVC": "no", "FCVC": 2.0, "NCP": 3.0, "CAEC": "Sometimes", "SMOKE": "no", "CH2O": 2.5, "SCC": "no", "FAF": 1.0, "TUE": 1.0, "CALC": "no", "MTRANS": "Public_Transportation"}'
```

### Xem API documentation:

Mở trình duyệt và truy cập: `http://127.0.0.1:8000/docs`

## 📁 Cấu trúc dự án sau khi setup

```
nutrition-ai-app/
├── api/                    # FastAPI application
├── services/               # Core inference services
├── models/                 # Trained models (tự động tạo)
│   ├── meal_plan/         # Meal plan recommendation models
│   └── obesity/           # Obesity prediction models
├── utils/                  # Utility functions
├── scripts/                # Training scripts
├── data/                   # Datasets
├── logs/                   # Application logs
├── requirements.txt        # Dependencies
├── run_server.py          # Server startup
└── README.md              # Main documentation
```

## 🔧 Troubleshooting

### Lỗi thường gặp:

1. **ModuleNotFoundError**: Đảm bảo đã kích hoạt virtual environment
2. **FileNotFoundError**: Chạy `python scripts/setup_models.py` để train models
3. **Memory Error**: Tăng RAM hoặc giảm batch size trong training scripts
4. **UnicodeEncodeError**: Đảm bảo terminal hỗ trợ UTF-8

### Kiểm tra logs:

```bash
# Xem logs
tail -f logs/meal_plan_requests.log
tail -f logs/llm_interactions.log
```

## 🚀 Production Deployment

Để deploy lên production:

1. Cài đặt dependencies: `pip install -r requirements.txt`
2. Train models: `python scripts/setup_models.py`
3. Cấu hình environment variables
4. Chạy với Gunicorn hoặc uWSGI

```bash
# Ví dụ với Gunicorn
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker api.main:app --bind 0.0.0.0:8000
```

## 📞 Hỗ trợ

Nếu gặp vấn đề, vui lòng:
1. Kiểm tra logs trong thư mục `logs/`
2. Đảm bảo đã chạy `python scripts/setup_models.py`
3. Kiểm tra Python version và dependencies
4. Tạo issue trên GitHub repository
