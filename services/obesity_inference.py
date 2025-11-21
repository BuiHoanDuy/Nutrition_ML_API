"""
Obesity inference service.
"""

from __future__ import annotations

import pandas as pd
import joblib
from pathlib import Path

# Load the trained model
MODEL_DIR = Path(__file__).parent.parent / "models" / "obesity"
MODEL_PATH = MODEL_DIR / "final_model.pkl"


def _normalize_category_value(value) -> str:
    if isinstance(value, str):
        return value.strip().lower()
    return str(value).lower()


def _prepare_features(input_df: pd.DataFrame, model_dict: dict) -> pd.DataFrame:
    df = input_df.copy()
    if "BMI" not in df.columns:
        df["BMI"] = df["Weight"] / (df["Height"] ** 2)

    numeric_cols = model_dict["numeric_columns"]
    numeric_stats = model_dict["numeric_stats"]
    categorical_mapping = model_dict["categorical_mapping"]

    row = {}

    # Scale numeric features using stored stats
    for col in numeric_cols:
        value = float(df.iloc[0][col])
        mean = numeric_stats[col]["mean"]
        std = numeric_stats[col]["std"] or 1.0
        row[col] = (value - mean) / std

    # One-hot encode categorical features
    for field, columns in categorical_mapping.items():
        if not columns:
            continue
        raw_value = df.iloc[0].get(field)
        normalized_value = _normalize_category_value(raw_value)

        matched_column = None
        for col_name in columns:
            suffix = col_name.split(f"{field}_", 1)[-1]
            if suffix.lower() == normalized_value:
                matched_column = col_name
                break

        if matched_column is None:
            raise ValueError(f"Giá trị '{raw_value}' không hợp lệ cho '{field}'.")

        for col_name in columns:
            row[col_name] = 1 if col_name == matched_column else 0

    features = model_dict["feature_columns"]
    feature_df = pd.DataFrame([row])
    # Ensure every feature column exists in the expected order
    for col in features:
        if col not in feature_df.columns:
            feature_df[col] = 0

    return feature_df[features]


def predict_obesity(input_df: pd.DataFrame) -> list[str]:
    """
    Predict obesity level based on input features.
    """
    try:
        model_dict = joblib.load(MODEL_PATH)
        model = model_dict["model"]
        target_le = model_dict["target_le"]

        processed_features = _prepare_features(input_df, model_dict)
        prediction = model.predict(processed_features)
        prediction_labels = target_le.inverse_transform(prediction)
        return prediction_labels.tolist()
    except Exception as e:
        raise Exception(f"Error in obesity prediction: {str(e)}")
