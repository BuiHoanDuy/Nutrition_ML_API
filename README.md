# tạo virtualenv
python -m venv .venv
source .venv/bin/activate    # windows: .venv\Scripts\activate

pip install --upgrade pip
pip install jupyterlab pandas numpy scikit-learn datasets transformers torch fastapi uvicorn pydantic sqlalchemy alembic psycopg2-binary joblib xgboost lightgbm prophet sentencepiece tokenizers sentence-transformers
# thêm thư viện tuỳ chọn: rasa, implicit, surprise, torchvision nếu cần


```
nutrition-ai/
├─ data/
│  ├─ ner/            # train.json, valid.json (NER)
│  ├─ intent/         # intent.csv
│  ├─ sentiment/      # sentiment.csv
│  ├─ users/          # user_profiles.csv, logs.csv
│  ├─ food_db.csv     # food metadata (name, kcal, protein, carb, fat)
│  └─ wearable/       # sensor csv nếu có
├─ notebooks/
│  ├─ 01_ner_train.ipynb
│  ├─ 02_intent_train.ipynb
│  ├─ 03_sentiment_train.ipynb
│  ├─ 04_calorie_regression.ipynb
│  ├─ 05_recommender.ipynb
│  └─ 06_timeseries.ipynb
├─ services/
│  ├─ api/            # FastAPI app
│  └─ models/         # saved models (.pt, .pkl)
├─ scripts/
│  ├─ preprocess.py
│  └─ train_ner.py
├─ Dockerfile
└─ README.md

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