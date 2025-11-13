
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse
import joblib
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch

def train_meal_plan_model():
    """Train meal plan recommendation model."""
    print("[INFO] Loading meal plan data...")
    
    # Load real meal plan dataset
    data_path = Path(__file__).parent.parent / "data" / "Dataset_Thucdon.csv"
    meal_plans_df = pd.read_csv(data_path, encoding='utf-8')
    
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
    
    # --- Feature Engineering ---
    print("[INFO] Performing feature engineering...")

    # 1. Standardize column names (from "Tên Cột" to "ten_cot")
    meal_plans_df = meal_plans_df.rename(columns={
        'Chế độ ăn': 'che_do_an',
        'Tình trạng sức khỏe': 'tinh_trang_suc_khoe',
        'Mục tiêu': 'muc_tieu',
        'Bữa sáng': 'bua_sang',
        'Bữa trưa': 'bua_trua',
        'Bữa tối': 'bua_toi',
        'Bữa phụ': 'bua_phu'
    })
    print("[INFO] Standardized column names.")
    # --- NEW: Clean whitespace from key categorical columns BEFORE encoding and embedding ---
    print("[INFO] Cleaning whitespace from categorical features...")
    for col in ['che_do_an', 'tinh_trang_suc_khoe', 'muc_tieu']:
        if col in meal_plans_df.columns:
            meal_plans_df[col] = meal_plans_df[col].str.strip()
    print("[INFO] Whitespace cleaning complete.")

    # --- One-Hot Encode User Profile Features ---
    print("[INFO] Processing user profile features with OneHotEncoder...")
    # Rename columns to match new data structure
    user_features = ['tinh_trang_suc_khoe', 'muc_tieu', 'che_do_an']
    # --- CRITICAL FIX: Consistent data cleaning ---
    # 1. Fill actual NaN values with 'không có'.
    meal_plans_df[user_features] = meal_plans_df[user_features].fillna('không có')
    # 2. Strip whitespace from all entries. This might create empty strings ''.
    # 3. Replace any resulting empty strings with 'không có'.
    for col in user_features:
        meal_plans_df[col] = meal_plans_df[col].astype(str).str.strip()
        meal_plans_df[col] = meal_plans_df[col].replace('', 'không có', regex=False)
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    user_features_encoded = encoder.fit_transform(meal_plans_df[user_features])
    
    # --- Generate PhoBERT Embeddings for Meal Content ---
    print("[INFO] Processing meal plan content with PhoBERT...")
    meal_cols = ['bua_sang', 'bua_trua', 'bua_toi', 'bua_phu']
    
    # Create 'full_meal_plan' by combining all meal columns, filling NaNs with empty strings
    meal_plans_df['full_meal_plan'] = meal_plans_df[meal_cols].fillna('').apply(lambda row: ' '.join(row), axis=1)
    
    # Create the text for PhoBERT embedding.
    # Combine structured features and the meal plan itself.
    # This creates a rich context for the embedding.
    meal_plans_df['embedding_text'] = meal_plans_df['che_do_an'] + ' ' + \
                                      meal_plans_df['tinh_trang_suc_khoe'] + ' ' + \
                                      meal_plans_df['muc_tieu'] + ' ' + \
                                      meal_plans_df['full_meal_plan']
    
    # --- PhoBERT Embedding Generation ---
    phobert_model_name = "vinai/phobert-base"
    print(f"[INFO] Loading PhoBERT model: {phobert_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(phobert_model_name)
    phobert_model = AutoModel.from_pretrained(phobert_model_name)

    # Function to get embeddings
    def get_phobert_embeddings(texts, batch_size=32):
        all_embeddings = []
        print(f"[INFO] Generating embeddings for {len(texts)} texts...")
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=256)
            with torch.no_grad():
                outputs = phobert_model(**inputs)
            # Use mean pooling of the last hidden state
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            all_embeddings.append(embeddings)
            print(f"  - Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        return np.vstack(all_embeddings)

    meal_plan_texts = meal_plans_df['embedding_text'].tolist()
    meal_features_phobert = get_phobert_embeddings(meal_plan_texts)
    
    # No longer a sparse matrix, but a dense numpy array
    print(f"[INFO] PhoBERT embeddings matrix shape: {meal_features_phobert.shape}")
    print(f"[INFO] User features encoded shape: {user_features_encoded.shape}")
    
    # Save artifacts
    print("[INFO] Saving processed data and models...")
    
    model_dir = Path(__file__).parent.parent / "models" / "meal_plan"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the main DataFrame for lookup
    # Rename columns back for consistency if needed, but saving with new names is fine
    output_df = meal_plans_df.rename(columns={
        'che_do_an': 'Chế độ ăn',
        'tinh_trang_suc_khoe': 'Tình trạng sức khỏe',
        'muc_tieu': 'Mục tiêu',
        'bua_sang': 'Bữa sáng',
        'bua_trua': 'Bữa trưa',
        'bua_toi': 'Bữa tối',
        'bua_phu': 'Bữa phụ'
    })
    output_df.to_csv(model_dir / "meal_plans_data.csv", index=False, encoding='utf-8-sig')
    
    # Save the fitted models/encoders
    # We don't save the PhoBERT model itself, but we will load it on-the-fly during inference.
    # We save the tokenizer to ensure consistency.
    joblib.dump(encoder, model_dir / "user_feature_encoder.pkl")
    tokenizer.save_pretrained(model_dir / "phobert_tokenizer")
    phobert_model.save_pretrained(model_dir / "phobert_model") # Thêm dòng này
    
    # Save the processed feature matrices
    scipy.sparse.save_npz(model_dir / "user_features_encoded.npz", user_features_encoded)
    np.save(model_dir / "meal_features_phobert.npy", meal_features_phobert)
    
    print(f"[SUCCESS] Successfully processed and saved artifacts to {model_dir}")
    
    return meal_plans_df, encoder, None # No vectorizer to return

if __name__ == "__main__":
    train_meal_plan_model()
