"""
Train a content-based filtering recommender for meal plans.

This script reads the meal plan dataset, processes user features (health status, goals)
and meal descriptions, and saves the processed data and models for inference.

The approach is as follows:
1. Load the dataset, handling potential encoding issues.
2. Preprocess user-related categorical features using One-Hot Encoding.
3. Combine meal items (breakfast, lunch, dinner, snacks) into a single text document for each row.
4. Vectorize these meal documents using TF-IDF to create a numerical representation of each meal plan.
5. Save the original data, the processed user features, the TF-IDF matrix of meals,
   and the fitted encoder/vectorizer objects for later use in an inference service.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
import joblib
from pathlib import Path
import scipy.sparse

def main():
    """
    Main function to run the training and preprocessing pipeline.
    """
    # Define paths
    HERE = Path(__file__).parent.parent
    DATA_PATH = HERE / "data" / "dataset_thuc_don_nguyen_ngay.csv"
    MODEL_DIR = HERE / "services" / "models" / "meal_recommender"

    # Create model directory if it doesn't exist
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading and cleaning data...")
    try:
        # Load the CSV file, ensuring UTF-8 encoding is used for Vietnamese characters.
        df = pd.read_csv(DATA_PATH, encoding='utf-8')

    except FileNotFoundError:
        print(f"[ERROR] Error: Data file not found at {DATA_PATH}")
        return
    except Exception as e:
        print(f"[ERROR] An error occurred while reading the CSV file: {e}")
        return

    # --- Feature Engineering ---

    # 1. User Profile Features (Categorical)
    print("[INFO] Processing user profile features with OneHotEncoder...")
    user_features = ['tình_trạng_sức_khỏe', 'mục_tiêu', 'intent_dinh_dưỡng']
    df[user_features] = df[user_features].fillna('không có') # Handle missing values

    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    user_features_encoded = encoder.fit_transform(df[user_features])

    # 2. Meal Plan Content Features (Text)
    print("[INFO] Processing meal plan content with TF-IDF Vectorizer...")
    meal_cols = ['bữa_sáng', 'bữa_trưa', 'bữa_tối', 'bữa_phụ']
    df[meal_cols] = df[meal_cols].fillna('') # Handle missing meals
    
    # Combine all meal items into a single string for each row
    df['full_meal_plan'] = df[meal_cols].apply(lambda row: ' '.join(row), axis=1)

    vectorizer = TfidfVectorizer(analyzer='word', stop_words=['và', 'cùng', 'với']) # Simple stop words
    meal_features_tfidf = vectorizer.fit_transform(df['full_meal_plan'])

    # --- Save Artifacts for Inference ---
    print("[INFO] Saving processed data and models...")

    # Save the main DataFrame for lookup
    df.to_csv(MODEL_DIR / "meal_plans_data.csv", index=False, encoding='utf-8-sig')

    # Save the fitted models/encoders
    joblib.dump(encoder, MODEL_DIR / "user_feature_encoder.pkl")
    joblib.dump(vectorizer, MODEL_DIR / "meal_tfidf_vectorizer.pkl")

    # Save the processed feature matrices
    scipy.sparse.save_npz(MODEL_DIR / "user_features_encoded.npz", user_features_encoded)
    scipy.sparse.save_npz(MODEL_DIR / "meal_features_tfidf.npz", meal_features_tfidf)

    print(f"[SUCCESS] Successfully processed and saved artifacts to {MODEL_DIR}")

if __name__ == "__main__":
    main()
