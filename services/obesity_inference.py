"""
Obesity inference service.
"""

import pandas as pd
import joblib
from pathlib import Path

# Load the trained model
MODEL_DIR = Path(__file__).parent.parent / "models" / "obesity"
MODEL_PATH = MODEL_DIR / "final_model.pkl"

def predict_obesity(input_df: pd.DataFrame) -> list:
    """
    Predict obesity level based on input features.
    
    Args:
        input_df: DataFrame with input features
        
    Returns:
        List of predictions
    """
    try:
        # Load the model dictionary
        model_dict = joblib.load(MODEL_PATH)
        
        # Extract components
        model = model_dict['model']
        scaler = model_dict['scaler']
        label_encoders = model_dict['label_encoders']
        target_le = model_dict['target_le']
        features = model_dict['features']
        
        # Calculate BMI if not present
        if 'BMI_computed' not in input_df.columns:
            input_df = input_df.copy()
            input_df['BMI_computed'] = input_df['Weight'] / (input_df['Height'] ** 2)
        
        # Prepare input data
        input_data = input_df[features].copy()
        
        # Apply label encoding to categorical features
        for col in input_data.columns:
            if col in label_encoders:
                input_data[col] = label_encoders[col].transform(input_data[col])
        
        # Apply scaling
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)
        
        # Convert back to original labels
        prediction_labels = target_le.inverse_transform(prediction)
        
        return prediction_labels.tolist()
    except Exception as e:
        raise Exception(f"Error in obesity prediction: {str(e)}")
