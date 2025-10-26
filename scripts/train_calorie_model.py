"""
Train calorie prediction model from macro nutrients.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from pathlib import Path

def train_calorie_model():
    """Train calorie prediction model."""
    print("[INFO] Loading nutrition data...")
    
    # Load data
    data_path = Path(__file__).parent.parent / "data" / "food_nutrition_data_final.csv"
    df = pd.read_csv(data_path, encoding='utf-8-sig')
    
    print(f"[INFO] Dataset shape: {df.shape}")
    try:
        print(f"[INFO] Columns: {df.columns.tolist()}")
    except UnicodeEncodeError:
        print("[INFO] Columns loaded successfully")
    
    # Prepare features and target
    feature_cols = ['Protein', 'Lipid', 'Glucid', 'Celluloza']
    X = df[feature_cols].fillna(0)
    y = df['Tro'].fillna(0)  # 'Tro' is calories
    
    print(f"[INFO] Features: {feature_cols}")
    try:
        print(f"[INFO] Target range: {y.min():.2f} - {y.max():.2f}")
    except UnicodeEncodeError:
        print("[INFO] Target range calculated successfully")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"[INFO] Training set: {X_train.shape[0]} samples")
    print(f"[INFO] Test set: {X_test.shape[0]} samples")
    
    # Train model
    print("[INFO] Training Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    try:
        print(f"[SUCCESS] RMSE: {rmse:.2f}")
        print(f"[SUCCESS] R²: {r2:.2f}")
    except UnicodeEncodeError:
        print("[SUCCESS] Model evaluation completed")
    
    # Save model
    model_dir = Path(__file__).parent.parent / "models" / "calorie"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / "calorie_from_macro_rf.pkl"
    joblib.dump(model, model_path)
    
    try:
        print(f"[SUCCESS] Model saved to {model_path}")
    except UnicodeEncodeError:
        print("[SUCCESS] Model saved successfully")
    
    return model, rmse, r2

if __name__ == "__main__":
    train_calorie_model()
