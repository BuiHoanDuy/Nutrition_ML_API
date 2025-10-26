# tạo virtualenv

python -m venv .venv
source .venv/bin/activate # windows: .venv\Scripts\activate

# 🍎 Nutrition AI – Food NLP Pipeline

### ⚙️ Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

- Run step by step
python data/translate_food_items.py
python scripts/clean_data.py
python scripts/build_food_master.py
python scripts/fuzzy_match.py     # test fuzzy search
python scripts/parser_rule_based.py
python scripts/generate_ner_dataset.py
python services/calorie_model_train.py
python services/ner_train_phobert.py
python services/infer.py

-- Models saved at
services/models/
├── calorie_from_macro_rf.pkl
└── ner_phobert/

-- Run server for testing API
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload

## API Endpoints (Postman Examples)

### 1. `/predict`

**Mục đích**: Phân tích câu nói để tìm món ăn và tính toán dinh dưỡng.
- **URL**: `http://127.0.0.1:8000/predict`
- **Method**: `POST`
- **Body (raw/json)**:
```json
{
    "content": "Tôi ăn 300g bột gạo"
}
```
# Run the obesity predict
C:\Users\Admin\Documents\Study\AI\Project1\nutrition-ai-app\.venv\Scripts\python.exe -m uvicorn obesity.api:app --reload

Example successful response:
```

{
"success": true,
"result": {
"food": "Bột gạo tí",
"food_key": "bot gao ti",
"quantity": 300.0,
"unit": "g",
"nutrition": {
"calories": 1077.0,
"protein_g": 19.8,
"carbs_g": 246.6,
"fat_g": 1.2,
"fiber_g": 1.2,
"water_g": 30.0,
"ash_g": 1.2
},
"found_in_master": true
}
}

```

If the food isn't found the API returns a helpful message and parsed output:
```

{ "success": true, "result": { "found_in_master": false, "message": "Không tìm thấy món ăn.", "parsed": { ... } } }

```

```
````
