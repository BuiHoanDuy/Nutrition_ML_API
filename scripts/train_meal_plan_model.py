"""
Train meal plan recommendation model.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse
import joblib
from pathlib import Path

def train_meal_plan_model():
    """Train meal plan recommendation model."""
    print("[INFO] Loading meal plan data...")
    
    # Load data
    data_path = Path(__file__).parent.parent / "data" / "food_nutrition_data_final.csv"
    df = pd.read_csv(data_path, encoding='utf-8-sig')
    
    print(f"[INFO] Dataset shape: {df.shape}")
    
    # Create sample meal plans (in real scenario, this would be actual meal plan data)
    print("[INFO] Creating sample meal plans...")
    
    # Sample meal plans based on different health conditions and goals
    meal_plans_data = []
    
    # Define sample meal plans
    sample_plans = [
        {
            'tình_trạng_sức_khỏe': 'bình thường',
            'mục_tiêu': 'giảm cân',
            'intent_dinh_dưỡng': 'ăn ít tinh bột',
            'bữa_sáng': 'Cháo yến mạch với sữa tách béo và chuối',
            'bữa_trưa': 'Salad rau xanh với ức gà nướng',
            'bữa_tối': 'Cá hồi áp chảo với rau luộc',
            'bữa_phụ': 'Hạt hạnh nhân và táo',
            'tổng_calo': 1200,
            'ghi_chú': 'Tập trung vào protein nạc và rau xanh'
        },
        {
            'tình_trạng_sức_khỏe': 'bình thường',
            'mục_tiêu': 'tăng cơ',
            'intent_dinh_dưỡng': 'ăn nhiều protein',
            'bữa_sáng': 'Trứng ốp la với bánh mì đen và sữa',
            'bữa_trưa': 'Thịt bò nướng với khoai tây và rau',
            'bữa_tối': 'Ức gà nướng với cơm gạo lứt',
            'bữa_phụ': 'Whey protein và chuối',
            'tổng_calo': 2000,
            'ghi_chú': 'Tăng cường protein cho phát triển cơ bắp'
        },
        {
            'tình_trạng_sức_khỏe': 'tiểu đường',
            'mục_tiêu': 'ổn định đường huyết',
            'intent_dinh_dưỡng': 'ổn định đường huyết',
            'bữa_sáng': 'Cháo yến mạch với hạt chia và quả mọng',
            'bữa_trưa': 'Cá hồi nướng với rau xanh',
            'bữa_tối': 'Thịt gà nướng với rau củ',
            'bữa_phụ': 'Hạt óc chó và dưa chuột',
            'tổng_calo': 1500,
            'ghi_chú': 'Kiểm soát carbohydrate và đường'
        },
        {
            'tình_trạng_sức_khỏe': 'béo phì',
            'mục_tiêu': 'giảm cân',
            'intent_dinh_dưỡng': 'ăn ít tinh bột',
            'bữa_sáng': 'Smoothie rau xanh với protein powder',
            'bữa_trưa': 'Salad cá ngừ với dầu oliu',
            'bữa_tối': 'Thịt bò nướng với rau luộc',
            'bữa_phụ': 'Hạt hạnh nhân',
            'tổng_calo': 1000,
            'ghi_chú': 'Chế độ ăn ít calo, nhiều protein'
        },
        {
            'tình_trạng_sức_khỏe': 'bình thường',
            'mục_tiêu': 'giữ cân',
            'intent_dinh_dưỡng': 'ăn lành mạnh',
            'bữa_sáng': 'Bánh mì đen với bơ và trứng',
            'bữa_trưa': 'Cơm gạo lứt với thịt gà và rau',
            'bữa_tối': 'Cá nướng với rau củ',
            'bữa_phụ': 'Sữa chua và quả tươi',
            'tổng_calo': 1800,
            'ghi_chú': 'Chế độ ăn cân bằng dinh dưỡng'
        }
    ]
    
    # Create DataFrame
    meal_plans_df = pd.DataFrame(sample_plans)
    
    print(f"[INFO] Created {len(meal_plans_df)} sample meal plans")
    
    # Feature Engineering
    print("[INFO] Processing user profile features with OneHotEncoder...")
    user_features = ['tình_trạng_sức_khỏe', 'mục_tiêu', 'intent_dinh_dưỡng']
    meal_plans_df[user_features] = meal_plans_df[user_features].fillna('không có')
    
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    user_features_encoded = encoder.fit_transform(meal_plans_df[user_features])
    
    print("[INFO] Processing meal plan content with TF-IDF Vectorizer...")
    meal_cols = ['bữa_sáng', 'bữa_trưa', 'bữa_tối', 'bữa_phụ']
    meal_plans_df[meal_cols] = meal_plans_df[meal_cols].fillna('')
    
    # Combine all meal items into a single string for each row
    meal_plans_df['full_meal_plan'] = meal_plans_df[meal_cols].apply(lambda row: ' '.join(row), axis=1)
    
    vectorizer = TfidfVectorizer(analyzer='word', stop_words=['và', 'cùng', 'với'])
    meal_features_tfidf = vectorizer.fit_transform(meal_plans_df['full_meal_plan'])
    
    # Save artifacts
    print("[INFO] Saving processed data and models...")
    
    model_dir = Path(__file__).parent.parent / "models" / "meal_plan"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the main DataFrame for lookup
    meal_plans_df.to_csv(model_dir / "meal_plans_data.csv", index=False, encoding='utf-8-sig')
    
    # Save the fitted models/encoders
    joblib.dump(encoder, model_dir / "user_feature_encoder.pkl")
    joblib.dump(vectorizer, model_dir / "meal_tfidf_vectorizer.pkl")
    
    # Save the processed feature matrices
    scipy.sparse.save_npz(model_dir / "user_features_encoded.npz", user_features_encoded)
    scipy.sparse.save_npz(model_dir / "meal_features_tfidf.npz", meal_features_tfidf)
    
    print(f"[SUCCESS] Successfully processed and saved artifacts to {model_dir}")
    
    return meal_plans_df, encoder, vectorizer

if __name__ == "__main__":
    train_meal_plan_model()
