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
    
    # Load real meal plan dataset
    data_path = Path(__file__).parent.parent / "data" / "dataset_thuc_don_nguyen_ngay.csv"
    meal_plans_df = pd.read_csv(data_path, encoding='utf-8-sig')
    
    print(f"[INFO] Dataset shape: {meal_plans_df.shape}")
    try:
        print(f"[INFO] Columns: {meal_plans_df.columns.tolist()}")
    except UnicodeEncodeError:
        print("[INFO] Columns loaded successfully")
    
    # Display sample data
    print(f"[INFO] Sample data:")
    try:
        print(meal_plans_df.head(3))
    except UnicodeEncodeError:
        print("[INFO] Sample data loaded successfully")
    
    print(f"[INFO] Loaded {len(meal_plans_df)} real meal plans from dataset")
    
    # Feature Engineering
    print("[INFO] Processing user profile features with OneHotEncoder...")
    user_features = ['tình_trạng_sức_khỏe', 'mục_tiêu', 'intent_dinh_dưỡng']
    meal_plans_df[user_features] = meal_plans_df[user_features].fillna('không có')
    
    # Display unique values for each feature
    for i, feature in enumerate(user_features):
        unique_vals = meal_plans_df[feature].unique()
        print(f"[INFO] Feature {i+1} unique values: {len(unique_vals)} values loaded")
    
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    user_features_encoded = encoder.fit_transform(meal_plans_df[user_features])
    
    print("[INFO] Processing meal plan content with TF-IDF Vectorizer...")
    meal_cols = ['bữa_sáng', 'bữa_trưa', 'bữa_tối', 'bữa_phụ']
    meal_plans_df[meal_cols] = meal_plans_df[meal_cols].fillna('')
    
    # Combine all meal items into a single string for each row
    meal_plans_df['full_meal_plan'] = meal_plans_df[meal_cols].apply(lambda row: ' '.join(row), axis=1)
    
    # Add user question to meal plan content for better matching
    meal_plans_df['full_meal_plan'] = meal_plans_df['câu_hỏi_người_dùng'] + ' ' + meal_plans_df['full_meal_plan']
    
    vectorizer = TfidfVectorizer(
        analyzer='word', 
        stop_words=['và', 'cùng', 'với', 'tôi', 'muốn', 'nên', 'ăn', 'gì', 'trong', 'ngày'],
        max_features=1000,
        ngram_range=(1, 2)
    )
    meal_features_tfidf = vectorizer.fit_transform(meal_plans_df['full_meal_plan'])
    
    print(f"[INFO] TF-IDF matrix shape: {meal_features_tfidf.shape}")
    print(f"[INFO] User features encoded shape: {user_features_encoded.shape}")
    
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
