# Models, Algorithms and Technologies Report

This document summarizes the main machine learning and NLP models, algorithms, and supporting technologies used in the project. It lists where each model/algorithm appears in the codebase and points to saved artifacts.

---

## 1) Named-Entity Recognition (NER)

- Purpose: Extract food items and quantities from user text (token-level NER labels like B-FOOD, I-FOOD, B-QUANTITY, I-QUANTITY).
- Model: PhoBERT (vinai/phobert-base) fine-tuned for token classification.
- Framework: Hugging Face Transformers (PyTorch backend).
- Key code:
  - `services/ner_train_phobert.py` — loads tokenizer and model via `AutoTokenizer` and `AutoModelForTokenClassification`, defines `LABELS`, uses `Trainer` + `TrainingArguments`.
  - Data loaded from: `data/ner_generated.jsonl` via `datasets.load_dataset("json")`.
- Saved artifacts:
  - `services/models/ner_phobert/` (contains `config.json`, `vocab.txt`, tokenizer files, `checkpoint-*` folders).
- Evidence (examples):
  - `AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=True)` in `services/ner_train_phobert.py`.
  - `AutoModelForTokenClassification.from_pretrained("vinai/phobert-base", num_labels=...)` in `services/ner_train_phobert.py`.

---

## 2) Calories Regression (tabular ML)

- Purpose: Predict Calories from macros (Protein, Carbohydrates, Fat, Fiber, Sugars).
- Model: RandomForestRegressor (scikit-learn).
- Framework: scikit-learn.
- Key code:
  - `services/calorie_model_train.py` — builds features from `data/cleaned_logs.csv`, trains `RandomForestRegressor(n_estimators=200, random_state=42)`, computes RMSE.
- Saved artifacts:
  - `services/models/calorie_from_macro_rf.pkl` (joblib dump of trained RandomForest model).
- Evidence (examples):
  - `from sklearn.ensemble import RandomForestRegressor` and `model = RandomForestRegressor(n_estimators=200, random_state=42)` in `services/calorie_model_train.py`.

---

## 3) Fuzzy Matching & Normalization

- Purpose: Map free-text food names from user input to canonical `food_key` in `data/food_master.csv`.
- Algorithm / Library: rapidfuzz (token/string fuzzy matching, scorer WRatio).
- Key code:
  - `scripts/fuzzy_match.py` — `normalize_text` (remove diacritics using `unicodedata`), builds `choices` from `food_master`, uses `rapidfuzz.process.extractOne(..., scorer=fuzz.WRatio)` with a threshold (default 80). Falls back to token-level matching.
- Evidence: `from rapidfuzz import process, fuzz` and usage in `scripts/fuzzy_match.py`.

---

## 4) Rule-based Parsing (quantity & unit extraction)

- Purpose: Extract quantity, unit, and the remaining food name from user sentences.
- Technique: Regex-based rule parser.
- Key code:
  - `scripts/parser_rule_based.py` — defines a list of common Vietnamese unit words, uses regex to find `(\d+(?:\.\d+)?)` quantity and optional unit, removes matched parts and returns `{"quantity":..., "unit":..., "food":...}`.
- Evidence: `parse_input` function in `scripts/parser_rule_based.py`.

---

## 5) Inference Pipeline / Integration

- Purpose: Combine rule-based parsing, fuzzy match, and tabular model or lookup for final calorie estimation.
- Key code:
  - `services/infer.py` — workflow: `parse_input(text)` -> `map_food(...)` -> if found, lookup in `food_master.csv` and return calories; otherwise return not found. Loads `calorie_from_macro_rf.pkl` via joblib for possible estimations.
- Evidence: `from scripts.fuzzy_match import map_food`, `from scripts.parser_rule_based import parse_input`, and `rf = joblib.load("services/models/calorie_from_macro_rf.pkl")` in `services/infer.py`.

---

## 6) Supporting technologies & libraries

- Core languages & runtime: Python.
- Libraries (explicit or in `requirements.txt` / README):
  - `pandas`, `numpy` — data manipulation.
  - `scikit-learn` — RandomForest and evaluation metrics.
  - `joblib` — model persistence for sklearn models.
  - `rapidfuzz` — fuzzy string matching.
  - `unicodedata` / `unidecode`-like handling — normalization.
  - `transformers` (Hugging Face), `datasets` (Hugging Face), `torch` — for PhoBERT NER training/inference.
  - `sentencepiece`, `tokenizers` — tokenizer support for transformer models.
  - `fastapi`, `uvicorn`, `pydantic` — project mentions an API folder and README points to a FastAPI app (likely for serving inference).
- Dev / notebooks: Jupyter notebooks present for experiments (in `notebooks/`).

---

## 7) Suggested next actions (optional)

- Add a small README section with example inference commands and sample inputs/outputs.
- Create a `api/` FastAPI endpoint (if not present) for unified inference: accept raw text -> returns parsed food, matched food_key, and calorie estimate.
- Add tests (pytest) for `parse_input`, `map_food`, and `infer` (happy path + 1-2 edge cases).
- Consider improving fuzzy-match by using vector embeddings (sentence-transformers) for semantic matching if many synonyms or noisy inputs.

---

## 8) Where to find things (quick map)

- NER training: `services/ner_train_phobert.py`
- NER saved model: `services/models/ner_phobert/`
- Calorie regression training: `services/calorie_model_train.py`
- Calorie model artifact: `services/models/calorie_from_macro_rf.pkl`
- Fuzzy match helper: `scripts/fuzzy_match.py`
- Rule-based parser: `scripts/parser_rule_based.py`
- Inference glue: `services/infer.py`
- Data: `data/food_master.csv`, `data/cleaned_logs.csv`, `data/ner_generated.jsonl`

---

If bạn muốn tôi cập nhật report (ví dụ thêm chi tiết cấu hình hyperparameters, kích thước vocabulary, hoặc extract một vài dòng từ `services/models/ner_phobert/config.json`), hãy nói rõ. Tôi sẽ chuyển todo `Create MD report file` sang completed sau khi bạn xác nhận tôi có thể hoàn tất.