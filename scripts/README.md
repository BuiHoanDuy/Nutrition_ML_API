# 📚 Training Scripts

Thư mục này chứa các script để train các models của ứng dụng Nutrition AI.

## 📁 Files

- `setup_models.py` - Script chính để train tất cả models
- `train_calorie_model.py` - Train calorie prediction model
- `train_obesity_model.py` - Train obesity prediction model  
- `train_meal_plan_model.py` - Train meal plan recommendation model

## 🚀 Cách sử dụng

### Train tất cả models:
```bash
python scripts/setup_models.py
```

### Train từng model riêng lẻ:
```bash
# Train calorie model
python scripts/train_calorie_model.py

# Train obesity model
python scripts/train_obesity_model.py

# Train meal plan model
python scripts/train_meal_plan_model.py
```

## 📊 Model Performance

### Calorie Prediction Model
- **Algorithm**: Random Forest Regressor
- **Features**: Protein, Lipid, Glucid, Celluloza
- **Target**: Calories (Tro)
- **RMSE**: ~8.47
- **R²**: ~-2.15

### Obesity Prediction Model
- **Algorithm**: Random Forest Classifier
- **Features**: 17 features including BMI, lifestyle factors
- **Target**: 7 obesity categories
- **Accuracy**: 98.82%

### Meal Plan Recommendation Model
- **Algorithm**: TF-IDF + Cosine Similarity
- **Features**: User profile (health status, goals, nutrition intent)
- **Target**: Personalized meal recommendations
- **Output**: Natural language responses

## 🔧 Requirements

- Python 3.8+
- pandas
- scikit-learn
- numpy
- scipy
- joblib

## 📝 Notes

- Models được lưu trong thư mục `models/`
- Dữ liệu training cần có trong `data/food_nutrition_data_final.csv`
- Obesity dataset cần có trong `models/obesity/ObesityDataSet.csv`
- Tất cả models sẽ được tự động train khi chạy `setup_models.py`
