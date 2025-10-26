"""
Train obesity prediction model.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from pathlib import Path

def train_obesity_model():
    """Train obesity prediction model."""
    print("[INFO] Loading obesity dataset...")
    
    # Load data
    data_path = Path(__file__).parent.parent / "data" / "ObesityDataSet.csv"
    df = pd.read_csv(data_path, encoding='utf-8')
    
    print(f"[INFO] Dataset shape: {df.shape}")
    print(f"[INFO] Columns: {df.columns.tolist()}")
    
    # Calculate BMI if not present
    if 'BMI_computed' not in df.columns:
        df['BMI_computed'] = df['Weight'] / (df['Height'] ** 2)
        print("[INFO] Calculated BMI_computed column")
    
    # Prepare features and target
    target_col = 'NObeyesdad'
    feature_cols = [col for col in df.columns if col != target_col]
    
    X = df[feature_cols]
    y = df[target_col]
    
    print(f"[INFO] Features: {feature_cols}")
    print(f"[INFO] Target classes: {y.unique()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"[INFO] Training set: {X_train.shape[0]} samples")
    print(f"[INFO] Test set: {X_test.shape[0]} samples")
    
    # Prepare data for training
    X_train_processed = X_train.copy()
    X_test_processed = X_test.copy()
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = X_train.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        le = LabelEncoder()
        X_train_processed[col] = le.fit_transform(X_train[col])
        X_test_processed[col] = le.transform(X_test[col])
        label_encoders[col] = le
        print(f"[INFO] Encoded column: {col}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_processed)
    X_test_scaled = scaler.transform(X_test_processed)
    
    # Encode target
    target_le = LabelEncoder()
    y_train_encoded = target_le.fit_transform(y_train)
    y_test_encoded = target_le.transform(y_test)
    
    print(f"[INFO] Target encoding: {dict(zip(target_le.classes_, target_le.transform(target_le.classes_)))}")
    
    # Train model
    print("[INFO] Training Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train_encoded)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test_encoded, y_pred)
    
    print(f"[SUCCESS] Accuracy: {accuracy:.4f}")
    print("\n[INFO] Classification Report:")
    print(classification_report(y_test_encoded, y_pred, target_names=target_le.classes_))
    
    # Save model and preprocessing objects
    model_dict = {
        'model': model,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'target_le': target_le,
        'features': feature_cols
    }
    
    model_dir = Path(__file__).parent.parent / "models" / "obesity"
    model_path = model_dir / "final_model.pkl"
    joblib.dump(model_dict, model_path)
    
    print(f"[SUCCESS] Model saved to {model_path}")
    
    return model, accuracy

if __name__ == "__main__":
    train_obesity_model()
